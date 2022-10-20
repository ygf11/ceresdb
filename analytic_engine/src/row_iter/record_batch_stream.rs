// Copyright 2022 CeresDB Project Authors. Licensed under Apache-2.0.

use std::{ops::Bound, sync::Arc};

use arrow::array::BooleanArray;
use common_types::{
    projected_schema::ProjectedSchema, record_batch::RecordBatchWithKey, SequenceNumber,
};
use common_util::define_result;
use datafusion::{
    error::DataFusionError,
    logical_plan::{combine_filters, ToDFSchema},
    physical_expr::{create_physical_expr, execution_props::ExecutionProps},
    physical_plan::PhysicalExpr,
};
use futures::stream::{self, Stream, StreamExt};
use log::{error, info, trace};
use object_store::ObjectStoreRef;
use snafu::{Backtrace, OptionExt, ResultExt, Snafu};
use table_engine::{
    predicate::{filter_record_batch::RecordBatchFilter, Predicate},
    table::TableId,
};

use crate::{
    memtable::{MemTableRef, ScanContext, ScanRequest},
    space::SpaceId,
    sst::{
        factory::{FactoryRef as SstFactoryRef, SstReaderOptions},
        file::FileHandle,
    },
    table::sst_util,
};

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display(
        "No sst reader found, sst_reader_options:{:?}.\nBacktrace:\n{}",
        options,
        backtrace
    ))]
    SstReaderNotFound {
        options: SstReaderOptions,
        backtrace: Backtrace,
    },

    #[snafu(display("Fail to read sst meta, err:{}", source))]
    ReadSstMeta { source: crate::sst::reader::Error },

    #[snafu(display("Fail to read sst data, err:{}", source))]
    ReadSstData { source: crate::sst::reader::Error },

    #[snafu(display("Fail to scan memtable, err:{}", source))]
    ScanMemtable { source: crate::memtable::Error },
}

define_result!(Error);

const REBUILD_FILTERED_RECORD_BATCH_MAGNIFICATION: usize = 2;

// TODO(yingwen): Can we move sequence to RecordBatchWithKey and remove this
// struct? But what is the sequence after merge?
#[derive(Debug)]
pub struct SequencedRecordBatch {
    pub record_batch: RecordBatchWithKey,
    pub sequence: SequenceNumber,
}

impl SequencedRecordBatch {
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.record_batch.num_rows()
    }
}

pub type SequencedRecordBatchStream = Box<
    dyn Stream<
            Item = std::result::Result<
                SequencedRecordBatch,
                Box<dyn std::error::Error + Send + Sync>,
            >,
        > + Send
        + Unpin,
>;

fn filter_batch(
    mut sequenced_record_batch: SequencedRecordBatch,
    predicate: Arc<dyn PhysicalExpr>,
) -> Option<SequencedRecordBatch> {
    // 报错了 要插类型转换
    info!("filter batch, batch:{:?}", sequenced_record_batch.record_batch.as_arrow_record_batch());
    // info!("filter batch, batch:{:?}", sequenced_record_batch.record_batch.as_arrow_record_batch().column(0));
    // info!("filter batch, batch:{:?}", sequenced_record_batch.record_batch.as_arrow_record_batch().column(1));
    let filter_array = predicate
        .evaluate(sequenced_record_batch.record_batch.as_arrow_record_batch())
        .map(|v| v.into_array(sequenced_record_batch.num_rows()))
        .unwrap();
    let filtered_array = filter_array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    sequenced_record_batch
        .record_batch
        .select_data_v2(filtered_array);
    Some(sequenced_record_batch)
}

/// Filter the `sequenced_record_batch` according to the `filter` if necessary.
/// Returns the original batch if only a small proportion of the batch is
/// filtered out.
/// The `selected_rows_buf` is for reuse.
fn maybe_filter_record_batch(
    mut sequenced_record_batch: SequencedRecordBatch,
    filter: &RecordBatchFilter,
    selected_rows_buf: &mut Vec<bool>,
) -> Option<SequencedRecordBatch> {
    if filter.is_empty() {
        return Some(sequenced_record_batch);
    }

    // The filter requires the `selected_rows_buf.len() >=
    // sequenced_record_batch.num_rows()`.
    selected_rows_buf.resize(sequenced_record_batch.num_rows(), true);
    let num_selected_rows = filter.filter(
        &sequenced_record_batch.record_batch,
        selected_rows_buf.as_mut_slice(),
    );

    trace!(
        "filter record batch, selected_rows:{}, origin_rows:{}",
        num_selected_rows,
        sequenced_record_batch.num_rows()
    );

    // No row is selected.
    if num_selected_rows == 0 {
        return None;
    }

    if num_selected_rows
        > sequenced_record_batch.num_rows() / REBUILD_FILTERED_RECORD_BATCH_MAGNIFICATION
    {
        // just use the original record batch because only a small proportion is
        // filtered out.
        return Some(sequenced_record_batch);
    }

    // select the rows according to the filter result.
    if let Err(e) = sequenced_record_batch
        .record_batch
        .select_data(selected_rows_buf.as_slice())
    {
        error!(
            "Fail to select record batch, data:{:?}, selected_rows:{:?}, err:{}",
            sequenced_record_batch, selected_rows_buf, e,
        );
    }

    Some(sequenced_record_batch)
}

/// Filter the sequenced record batch stream by applying the `predicate`.
/// However, the output record batches is not ensured to meet the requirements
/// of the `predicate`.
pub fn filter_stream(
    origin_stream: SequencedRecordBatchStream,
    predicate: &Predicate,
    projected_schema: ProjectedSchema,
) -> SequencedRecordBatchStream {
    info!("111111111111111111111111111111111111111111111111111111111111111111111111111111111");
    info!("filter record batch, predicate:{:?}", predicate,);
    if predicate.exprs().is_empty() {
        return origin_stream;
    }

    let filter = combine_filters(predicate.exprs());
    let input_schema = projected_schema.to_projected_arrow_schema();
    let input_df_schema = input_schema.clone().to_dfschema().unwrap();
    let execution_props = ExecutionProps::new();
    let predicate = create_physical_expr(
        &filter.unwrap(),
        &input_df_schema,
        input_schema.as_ref(),
        &execution_props,
    )
    .unwrap();
    info!("filter record batch, physical_expr:{:#?}, projected_schema:{:#?}", predicate, projected_schema.to_projected_arrow_schema());
    info!("filter record batch, physical_expr:{:#?}, schema:{:#?}", predicate, projected_schema.to_record_schema().to_arrow_schema_ref());

    let stream = origin_stream.filter_map(move |sequence_record_batch| {
        let v = match sequence_record_batch {
            Ok(v) => filter_batch(v, predicate.clone()).map(Ok),
            // Ok(v) => maybe_filter_record_batch(v, &filter, &mut select_row_buf).map(Ok),
            Err(e) => Some(Err(e)),
        };

        futures::future::ready(v)
    });

    Box::new(stream)
}

/// Build filtered (by `predicate`) [SequencedRecordBatchStream] from a
/// memtable.
pub fn filtered_stream_from_memtable(
    projected_schema: ProjectedSchema,
    need_dedup: bool,
    memtable: &MemTableRef,
    reverse: bool,
    predicate: &Predicate,
) -> Result<SequencedRecordBatchStream> {
    stream_from_memtable(projected_schema.clone(), need_dedup, memtable, reverse)
        .map(|origin_stream| filter_stream(origin_stream, predicate, projected_schema))
}

/// Build [SequencedRecordBatchStream] from a memtable.
pub fn stream_from_memtable(
    projected_schema: ProjectedSchema,
    need_dedup: bool,
    memtable: &MemTableRef,
    reverse: bool,
) -> Result<SequencedRecordBatchStream> {
    let scan_ctx = ScanContext::default();
    let max_seq = memtable.last_sequence();
    let scan_req = ScanRequest {
        start_user_key: Bound::Unbounded,
        end_user_key: Bound::Unbounded,
        sequence: max_seq,
        projected_schema,
        need_dedup,
        reverse,
    };

    let iter = memtable.scan(scan_ctx, scan_req).context(ScanMemtable)?;
    let stream = stream::iter(iter).map(move |v| {
        v.map(|record_batch| SequencedRecordBatch {
            record_batch,
            sequence: max_seq,
        })
        .map_err(|e| Box::new(e) as _)
    });

    Ok(Box::new(stream))
}

/// Build the filtered by `sst_read_options.predicate`
/// [SequencedRecordBatchStream] from a sst.
pub async fn filtered_stream_from_sst_file(
    space_id: SpaceId,
    table_id: TableId,
    sst_file: &FileHandle,
    sst_factory: &SstFactoryRef,
    sst_reader_options: &SstReaderOptions,
    store: &ObjectStoreRef,
) -> Result<SequencedRecordBatchStream> {
    stream_from_sst_file(
        space_id,
        table_id,
        sst_file,
        sst_factory,
        sst_reader_options,
        store,
    )
    .await
    .map(|origin_stream| {
        filter_stream(
            origin_stream,
            sst_reader_options.predicate.as_ref(),
            sst_reader_options.projected_schema.clone(),
        )
    })
}

/// Build the [SequencedRecordBatchStream] from a sst.
pub async fn stream_from_sst_file(
    space_id: SpaceId,
    table_id: TableId,
    sst_file: &FileHandle,
    sst_factory: &SstFactoryRef,
    sst_reader_options: &SstReaderOptions,
    store: &ObjectStoreRef,
) -> Result<SequencedRecordBatchStream> {
    sst_file.read_meter().mark();
    let path = sst_util::new_sst_file_path(space_id, table_id, sst_file.id());
    let mut sst_reader = sst_factory
        .new_sst_reader(sst_reader_options, &path, store)
        .with_context(|| SstReaderNotFound {
            options: sst_reader_options.clone(),
        })?;
    let meta = sst_reader.meta_data().await.context(ReadSstMeta)?;
    let max_seq = meta.max_sequence;
    let sst_stream = sst_reader.read().await.context(ReadSstData)?;

    let stream = Box::new(sst_stream.map(move |v| {
        v.map(|record_batch| SequencedRecordBatch {
            record_batch,
            sequence: max_seq,
        })
        .map_err(|e| Box::new(e) as _)
    }));

    Ok(stream)
}

#[cfg(test)]
pub mod tests {
    use common_types::{row::Row, schema::Schema};

    use super::*;
    use crate::row_iter;

    /// Build [SequencedRecordBatchStream] from the sequenced rows.
    pub fn build_sequenced_record_batch_stream(
        schema: &Schema,
        batches: Vec<(SequenceNumber, Vec<Row>)>,
    ) -> Vec<SequencedRecordBatchStream> {
        batches
            .into_iter()
            .map(|(seq, rows)| {
                let batch = SequencedRecordBatch {
                    record_batch: row_iter::tests::build_record_batch_with_key(
                        schema.clone(),
                        rows,
                    ),
                    sequence: seq,
                };
                Box::new(stream::iter(vec![Ok(batch)])) as SequencedRecordBatchStream
            })
            .collect()
    }
}
