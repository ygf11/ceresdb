// Copyright 2022 CeresDB Project Authors. Licensed under Apache-2.0.

//! Interpreter for insert statement

use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    ops::IndexMut,
    sync::Arc,
};

use arrow_deps::{
    arrow::{
        array::ArrayRef,
        datatypes::{
            DataType as ArrowDataType, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
        },
        record_batch::RecordBatch,
    },
    datafusion::{
        common::{DFField, DFSchema},
        error::Result as DataFusionResult,
        logical_expr::{
            expr_visitor::{ExpressionVisitor, Recursion},
            ColumnarValue as DfColumnarValue, Expr as LogicalExpr,
        },
        physical_expr::{create_physical_expr, execution_props::ExecutionProps},
    },
    datafusion_expr::expr_visitor::ExprVisitable,
};
use async_trait::async_trait;
use common_types::{
    column::{ColumnBlock, ColumnBlockBuilder},
    column_schema::ColumnId,
    datum::Datum,
    hash::hash64,
    row::RowGroup,
    schema::Schema,
};
use common_util::{
    codec::{compact::MemCompactEncoder, Encoder},
    panic,
};
use df_operator::functions::ColumnarValue;
use snafu::{ResultExt, Snafu};
use sql::plan::InsertPlan;
use table_engine::table::{TableRef, WriteRequest};

use crate::{
    context::Context,
    interpreter::{Insert, Interpreter, InterpreterPtr, Output, Result},
};

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Failed to write table, err:{}", source))]
    WriteTable { source: table_engine::table::Error },

    #[snafu(display("Failed to encode tsid, err:{}", source))]
    EncodeTsid {
        source: common_util::codec::compact::Error,
    },
}

pub struct InsertInterpreter {
    ctx: Context,
    plan: InsertPlan,
}

impl InsertInterpreter {
    pub fn create(ctx: Context, plan: InsertPlan) -> InterpreterPtr {
        Box::new(Self { ctx, plan })
    }
}

#[async_trait]
impl Interpreter for InsertInterpreter {
    async fn execute(mut self: Box<Self>) -> Result<Output> {
        // Generate tsid if needed.
        self.maybe_generate_tsid()?;

        let InsertPlan {
            table,
            mut rows,
            need_fill_column_idxes,
        } = self.plan;

        // Generate default value if nedded.
        fill_default_values(table.clone(), &mut rows, &need_fill_column_idxes).unwrap();

        // Context is unused now
        let _ctx = self.ctx;

        let request = WriteRequest { row_group: rows };

        let num_rows = table
            .write(request)
            .await
            .context(WriteTable)
            .context(Insert)?;

        Ok(Output::AffectedRows(num_rows))
    }
}

impl InsertInterpreter {
    fn maybe_generate_tsid(&mut self) -> Result<()> {
        let schema = self.plan.rows.schema();
        let tsid_idx = schema.index_of_tsid();

        if let Some(idx) = tsid_idx {
            // Vec of (`index of tag`, `column id of tag`).
            let tag_idx_column_ids: Vec<_> = schema
                .columns()
                .iter()
                .enumerate()
                .filter_map(|(i, column)| {
                    if column.is_tag {
                        Some((i, column.id))
                    } else {
                        None
                    }
                })
                .collect();

            let mut hash_bytes = Vec::new();
            for i in 0..self.plan.rows.num_rows() {
                let row = self.plan.rows.get_row_mut(i).unwrap();

                let mut tsid_builder = TsidBuilder::new(&mut hash_bytes);

                for (idx, column_id) in &tag_idx_column_ids {
                    tsid_builder.maybe_write_datum(*column_id, &row[*idx])?;
                }

                let tsid = tsid_builder.finish();
                row[idx] = Datum::UInt64(tsid);
            }
        }
        Ok(())
    }
}

struct TsidBuilder<'a> {
    encoder: MemCompactEncoder,
    hash_bytes: &'a mut Vec<u8>,
}

impl<'a> TsidBuilder<'a> {
    fn new(hash_bytes: &'a mut Vec<u8>) -> Self {
        // Clear the bytes buffer.
        hash_bytes.clear();

        Self {
            encoder: MemCompactEncoder,
            hash_bytes,
        }
    }

    fn maybe_write_datum(&mut self, column_id: ColumnId, datum: &Datum) -> Result<()> {
        // Null datum will be ignored, so tsid remains unchanged after adding a null
        // column.
        if datum.is_null() {
            return Ok(());
        }

        // Write column id first.
        self.encoder
            .encode(self.hash_bytes, &Datum::UInt64(u64::from(column_id)))
            .context(EncodeTsid)
            .context(Insert)?;
        // Write datum.
        self.encoder
            .encode(self.hash_bytes, datum)
            .context(EncodeTsid)
            .context(Insert)?;
        Ok(())
    }

    fn finish(self) -> u64 {
        hash64(self.hash_bytes)
    }
}

fn fill_default_values(
    table: TableRef,
    rows: &mut RowGroup,
    need_fill_column_idxes: &[usize],
) -> Result<()> {
    let df_fields = table
        .schema()
        .columns()
        .iter()
        .map(|column| {
            DFField::new(
                None,
                &column.name,
                column.data_type.to_arrow_data_type(),
                column.is_nullable,
            )
        })
        .collect::<Vec<_>>();
    let mut cached_columns_map: HashMap<usize, DfColumnarValue> =
        HashMap::with_capacity(need_fill_column_idxes.len());

    // determine execute order
    let mut reorder_need_fill_column_idxes = (0..table.schema().num_columns())
        .filter(|idx| !need_fill_column_idxes.contains(idx))
        .collect::<Vec<_>>();
    for column_idx in need_fill_column_idxes.iter() {
        visit_and_save_order(
            *column_idx,
            &table.schema(),
            &mut reorder_need_fill_column_idxes,
        );
    }

    let non_default_value_num = table.schema().num_columns() - need_fill_column_idxes.len();
    for column_idx in reorder_need_fill_column_idxes
        .iter()
        .skip(non_default_value_num)
    {
        // 1. Build physical expr
        // 2. Get input fields
        // 3. Get input columns
        // 4. calculate default value
        // 5. fill into row group and save column
        if let Some(expr) = &table.schema().column(*column_idx).default_value {
            let required_column_idxes = find_columns_by_expr(expr)
                .iter()
                .map(|column_name| table.schema().index_of(column_name))
                .collect::<Option<Vec<usize>>>()
                .unwrap();

            // Build DFSchema and ArrowSchema
            let input_fields = df_fields
                .iter()
                .enumerate()
                .filter(|(idx, _)| required_column_idxes.contains(idx))
                .map(|(_, field)| field.clone())
                .collect::<Vec<_>>();

            let input_df_schema =
                DFSchema::new_with_metadata(input_fields, HashMap::new()).unwrap();
            let input_arrow_schema: Arc<ArrowSchema> = Arc::new(input_df_schema.clone().into());
            // Build input batch
            let input_batch = build_and_cached_input(
                &required_column_idxes,
                rows,
                &mut cached_columns_map,
                input_arrow_schema.clone(),
            );

            let physical_expr = create_physical_expr(
                expr,
                &input_df_schema,
                input_arrow_schema.as_ref(),
                &ExecutionProps::default(),
            )
            .unwrap();

            let output = physical_expr.evaluate(&input_batch).unwrap();
            cached_columns_map.insert(*column_idx, output.clone());

            fill_column_to_row_group(*column_idx, &output, rows);
        }
    }

    Ok(())
}

#[derive(Default)]
struct ColumnCollector {
    /// column use by the given expr
    columns: Vec<String>,
}

impl ExpressionVisitor for ColumnCollector {
    fn pre_visit(mut self, expr: &LogicalExpr) -> DataFusionResult<Recursion<Self>>
    where
        Self: ExpressionVisitor,
    {
        if let LogicalExpr::Column(column) = expr {
            self.columns.push(column.name.clone())
        }
        Ok(Recursion::Continue(self))
    }
}

fn find_columns_by_expr(expr: &LogicalExpr) -> Vec<String> {
    let ColumnCollector { columns } = expr.accept(ColumnCollector::default()).unwrap();

    columns
}

fn build_and_cached_input(
    input_column_idexes: &[usize],
    row_groups: &RowGroup,
    cached_column_map: &mut HashMap<usize, DfColumnarValue>,
    schema: ArrowSchemaRef,
) -> RecordBatch {
    let mut input_arrays = Vec::with_capacity(input_column_idexes.len());
    for idx in input_column_idexes.iter() {
        if let Some(column_value) = cached_column_map.get(idx) {
            input_arrays.push(column_value.clone().into_array(row_groups.num_rows()))
        } else {
            // Get from row group
            let data_type = row_groups.schema().column(*idx).data_type;
            let iter = row_groups.iter_column(*idx);
            let mut builder = ColumnBlockBuilder::with_capacity(&data_type, iter.size_hint().0);

            for datum in iter {
                builder.append(datum.clone()).unwrap();
            }

            let input_column = builder.build().to_arrow_array_ref();
            cached_column_map.insert(*idx, DfColumnarValue::Array(input_column.clone()));
            input_arrays.push(input_column);
        }
    }
    RecordBatch::try_new(schema, input_arrays).unwrap()
}

fn fill_column_to_row_group(column_idx: usize, column: &DfColumnarValue, rows: &mut RowGroup) {
    match column {
        DfColumnarValue::Array(array) => {
            for row_idx in 0..rows.num_rows() {
                let column_block = ColumnBlock::try_cast_arrow_array_ref(array).unwrap();
                let datum = column_block.datum(row_idx);
                rows.get_row_mut(row_idx)
                    .map(|row| std::mem::replace(row.index_mut(column_idx), datum.clone()));
            }
        }
        DfColumnarValue::Scalar(scalar) => {
            if let Some(datum) = Datum::from_scalar_value(scalar) {
                for row_idx in 0..rows.num_rows() {
                    rows.get_row_mut(row_idx)
                        .map(|row| std::mem::replace(row.index_mut(column_idx), datum.clone()));
                }
            }
        }
    }
}

fn visit_and_save_order(
    column_idx: usize,
    schema: &Schema,
    finished_column_idxes: &mut Vec<usize>,
) {
    if finished_column_idxes.contains(&column_idx) {
        return;
    }

    if let Some(expr) = &schema.column(column_idx).default_value {
        let required_column_idxes = find_columns_by_expr(expr)
            .iter()
            .map(|column_name| schema.index_of(column_name))
            .collect::<Option<Vec<usize>>>()
            .unwrap();

        for required_idx in required_column_idxes.iter() {
            visit_and_save_order(*required_idx, schema, finished_column_idxes);
        }

        finished_column_idxes.push(column_idx);
    }
}
