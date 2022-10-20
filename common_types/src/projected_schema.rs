// Copyright 2022 CeresDB Project Authors. Licensed under Apache-2.0.

//! Projected schema

use std::{fmt, sync::Arc};

use snafu::{ensure, Backtrace, ResultExt, Snafu};

use crate::{
    column_schema::{ColumnSchema, ReadOp},
    datum::Datum,
    row::Row,
    schema::{ArrowSchemaRef, RecordSchema, RecordSchemaWithKey, Schema},
};


#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display(
        "Invalid projection index, index:{}.\nBacktrace:\n{}",
        index,
        backtrace
    ))]
    InvalidProjectionIndex { index: usize, backtrace: Backtrace },

    #[snafu(display("Incompatible column schema for read, err:{}", source))]
    IncompatReadColumn {
        source: crate::column_schema::CompatError,
    },

    #[snafu(display("Failed to build projected schema, err:{}", source))]
    BuildProjectedSchema { source: crate::schema::Error },

    #[snafu(display(
        "Missing not null column for read, name:{}.\nBacktrace:\n{}",
        name,
        backtrace
    ))]
    MissingReadColumn { name: String, backtrace: Backtrace },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct RowProjector {
    schema_with_key: RecordSchemaWithKey,
    source_schema: Schema,
    /// The Vec stores the column index in source, and `None` means this column
    /// is not in source but required by reader, and need to filled by null.
    /// The length of Vec is the same as the number of columns reader intended
    /// to read.
    source_projection: Vec<Option<usize>>,
}

impl RowProjector {
    /// The projected indexes of existed columns in the source schema.
    pub fn existed_source_projection(&self) -> Vec<usize> {
        self.source_projection
            .iter()
            .filter_map(|index| *index)
            .collect()
    }

    /// The projected indexes of all columns(existed and not exist) in the
    /// source schema.
    pub fn source_projection(&self) -> &[Option<usize>] {
        &self.source_projection
    }

    pub fn schema_with_key(&self) -> &RecordSchemaWithKey {
        &self.schema_with_key
    }

    /// Project the row.
    ///
    /// REQUIRE: The schema of row is the same as source schema.
    pub fn project_row(&self, row: &Row, mut datums_buffer: Vec<Datum>) -> Row {
        assert_eq!(self.source_schema.num_columns(), row.num_columns());

        datums_buffer.reserve(self.schema_with_key.num_columns());

        for p in &self.source_projection {
            let datum = match p {
                Some(index_in_source) => row[*index_in_source].clone(),
                None => Datum::Null,
            };

            datums_buffer.push(datum);
        }

        Row::from_datums(datums_buffer)
    }
}

#[derive(Clone)]
pub struct ProjectedSchema(Arc<ProjectedSchemaInner>);

impl fmt::Debug for ProjectedSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectedSchema")
            .field("original_schema", &self.0.original_schema)
            .field("projection", &self.0.projection)
            .finish()
    }
}

impl ProjectedSchema {
    pub fn no_projection(schema: Schema) -> Self {
        let inner = ProjectedSchemaInner::no_projection(schema);
        Self(Arc::new(inner))
    }

    pub fn new(schema: Schema, projection: Option<Vec<usize>>) -> Result<Self> {
        let inner = ProjectedSchemaInner::new(schema, projection)?;
        Ok(Self(Arc::new(inner)))
    }

    pub fn is_all_projection(&self) -> bool {
        self.0.is_all_projection()
    }

    /// Returns the [RowProjector] to project the rows with source schema to
    /// rows with [RecordSchemaWithKey].
    ///
    /// REQUIRE: The key columns are the same as this schema.
    #[inline]
    pub fn try_project_with_key(&self, source_schema: &Schema) -> Result<RowProjector> {
        self.0.try_project_with_key(source_schema)
    }

    // Returns the record schema after projection with key.
    pub fn to_record_schema_with_key(&self) -> RecordSchemaWithKey {
        self.0.schema_with_key.clone()
    }

    pub(crate) fn as_record_schema_with_key(&self) -> &RecordSchemaWithKey {
        &self.0.schema_with_key
    }

    // Returns the record schema after projection.
    pub fn to_record_schema(&self) -> RecordSchema {
        self.0.record_schema.clone()
    }

    /// Returns the arrow schema after projection.
    pub fn to_projected_arrow_schema(&self) -> ArrowSchemaRef {
        self.0.record_schema.to_arrow_schema_ref()
    }
}

/// Schema with projection informations
struct ProjectedSchemaInner {
    /// The schema before projection that the reader intended to read, may
    /// differ from current schema of the table.
    original_schema: Schema,
    /// Index of the projected columns in `self.schema`, `None` if
    /// all columns are needed.
    projection: Option<Vec<usize>>,

    /// The record schema from `self.schema` with key columns after projection.
    schema_with_key: RecordSchemaWithKey,
    /// The record schema from `self.schema` after projection.
    record_schema: RecordSchema,
}

impl ProjectedSchemaInner {
    fn no_projection(schema: Schema) -> Self {
        let schema_with_key = schema.to_record_schema_with_key();
        let record_schema = schema.to_record_schema();

        Self {
            original_schema: schema,
            projection: None,
            schema_with_key,
            record_schema,
        }
    }

    fn new(schema: Schema, projection: Option<Vec<usize>>) -> Result<Self> {
        if let Some(p) = &projection {
            // Projection is provided, validate the projection is valid. This is necessary
            // to avoid panic when creating RecordSchema and
            // RecordSchemaWithKey.
            if let Some(max_idx) = p.iter().max() {
                ensure!(
                    *max_idx < schema.num_columns(),
                    InvalidProjectionIndex { index: *max_idx }
                );
            }

            let schema_with_key = schema.project_record_schema_with_key(p);
            let record_schema = schema.project_record_schema(p);

            Ok(Self {
                original_schema: schema,
                projection,
                schema_with_key,
                record_schema,
            })
        } else {
            Ok(Self::no_projection(schema))
        }
    }

    /// Selecting all the columns is the all projection.
    fn is_all_projection(&self) -> bool {
        self.projection.is_none()
    }

    // TODO(yingwen): We can fill missing not null column with default value instead
    //  of returning error.
    fn try_project_with_key(&self, source_schema: &Schema) -> Result<RowProjector> {
        debug_assert_eq!(
            self.schema_with_key.key_columns(),
            source_schema.key_columns()
        );
        // We consider the two schema is equal if they have same version.
        if self.original_schema.version() == source_schema.version() {
            debug_assert_eq!(self.original_schema, *source_schema);
        }

        let mut source_projection = Vec::with_capacity(self.schema_with_key.num_columns());
        // For each column in `schema_with_key`
        for column_schema in self.schema_with_key.columns() {
            self.try_project_column(column_schema, source_schema, &mut source_projection)?;
        }

        Ok(RowProjector {
            schema_with_key: self.schema_with_key.clone(),
            source_schema: source_schema.clone(),
            source_projection,
        })
    }

    fn try_project_column(
        &self,
        column: &ColumnSchema,
        source_schema: &Schema,
        source_projection: &mut Vec<Option<usize>>,
    ) -> Result<()> {
        match source_schema.index_of(&column.name) {
            Some(source_idx) => {
                // Column is in source
                if self.original_schema.version() == source_schema.version() {
                    // Same version, just use that column in source
                    source_projection.push(Some(source_idx));
                } else {
                    // Different version, need to check column schema
                    let source_column = source_schema.column(source_idx);
                    // TODO(yingwen): Data type is not checked here because we do not support alter
                    // data type now.
                    match column
                        .compatible_for_read(source_column)
                        .context(IncompatReadColumn)?
                    {
                        ReadOp::Exact => {
                            source_projection.push(Some(source_idx));
                        }
                        ReadOp::FillNull => {
                            source_projection.push(None);
                        }
                    }
                }
            }
            None => {
                // Column is not in source
                ensure!(column.is_nullable, MissingReadColumn { name: &column.name });
                // Column is nullable, fill this column by null
                source_projection.push(None);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{projected_schema::ProjectedSchema, tests::build_schema};

    #[test]
    fn test_projected_schema() {
        let schema = build_schema();
        assert!(schema.num_columns() > 1);
        let projection: Vec<usize> = (0..schema.num_columns() - 1).collect();
        let projected_schema = ProjectedSchema::new(schema.clone(), Some(projection)).unwrap();
        assert_eq!(
            projected_schema.0.schema_with_key.num_columns(),
            schema.num_columns() - 1
        );
        assert!(!projected_schema.is_all_projection());
    }
}
