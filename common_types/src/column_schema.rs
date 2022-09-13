// Copyright 2022 CeresDB Project Authors. Licensed under Apache-2.0.

//! Schema of column

use std::{collections::BTreeMap, convert::TryFrom, str::FromStr};

use arrow_deps::{
    arrow::datatypes::{DataType, Field},
    datafusion_expr::Expr as LogicalExpr,
};
use proto::common as common_pb;
use snafu::{ensure, Backtrace, OptionExt, ResultExt, Snafu};

use crate::datum::DatumKind;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display(
        "Unsupported arrow data type, type:{}.\nBacktrace:\n{}",
        data_type,
        backtrace
    ))]
    UnsupportedDataType {
        data_type: DataType,
        backtrace: Backtrace,
    },

    #[snafu(display("Invalid tag type:{}.\nBacktrace:\n{}", data_type, backtrace))]
    InvalidTagType {
        data_type: DataType,
        backtrace: Backtrace,
    },

    #[snafu(display(
        "Arrow field meta data is missing, field name:{}.\nBacktrace:\n{}",
        field_name,
        backtrace
    ))]
    ArrowFieldMetaDataMissing {
        field_name: String,
        backtrace: Backtrace,
    },

    #[snafu(display(
        "Arrow field meta key is not found, key:{:?}.\nBacktrace:\n{}",
        key,
        backtrace
    ))]
    ArrowFieldMetaKeyNotFound {
        key: ArrowFieldMetaKey,
        backtrace: Backtrace,
    },

    #[snafu(display(
        "Arrow field meta value is invalid, key:{:?}, raw_value:{}, err:{}.\nBacktrace:\n{}",
        key,
        raw_value,
        source,
        backtrace
    ))]
    InvalidArrowFieldMetaValue {
        key: ArrowFieldMetaKey,
        raw_value: String,
        source: Box<dyn std::error::Error + Send + Sync>,
        backtrace: Backtrace,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Error of compatibility check
#[derive(Debug, Snafu)]
pub enum CompatError {
    #[snafu(display(
        "Incompatible data type of column, name:{}, expect:{:?}, given:{:?}.\nBacktrace:\n{}",
        name,
        expect,
        given,
        backtrace,
    ))]
    IncompatDataType {
        name: String,
        expect: DatumKind,
        given: DatumKind,
        backtrace: Backtrace,
    },

    #[snafu(display("Column is not nullable, name:{}.\nBacktrace:\n{}", name, backtrace))]
    NotNullable { name: String, backtrace: Backtrace },
}

/// Id of column
pub type ColumnId = u32;

/// A ColumnId used to indicate that the column id is uninitialized
pub const COLUMN_ID_UNINIT: ColumnId = 0;

/// Read operation of a column
#[derive(Debug)]
pub enum ReadOp {
    /// Use the column exactly
    Exact,
    /// Fill the column by null
    FillNull,
}

/// Meta data of the arrow field.
#[derive(Clone, Debug, Default)]
struct ArrowFieldMeta {
    id: u32,
    is_tag: bool,
    comment: String,
}

#[derive(Copy, Clone, Debug)]
pub enum ArrowFieldMetaKey {
    Id,
    IsTag,
    Comment,
}

impl ArrowFieldMetaKey {
    fn as_str(&self) -> &str {
        match self {
            ArrowFieldMetaKey::Id => "field::id",
            ArrowFieldMetaKey::IsTag => "field::is_tag",
            ArrowFieldMetaKey::Comment => "field::comment",
        }
    }
}

impl ToString for ArrowFieldMetaKey {
    fn to_string(&self) -> String {
        self.as_str().to_string()
    }
}

/// Schema of column
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnSchema {
    /// Id of column
    pub id: ColumnId,
    /// Column name
    pub name: String,
    /// Data type of the column
    pub data_type: DatumKind,
    /// Is nullable
    pub is_nullable: bool,
    /// Is tag, tag is just a hint for a column, there is no restriction that a
    /// tag column must be a part of primary key
    pub is_tag: bool,
    /// Comment of the column
    pub comment: String,
    /// Column name in response
    pub escaped_name: String,
    /// Default value expr
    pub default_value: Option<LogicalExpr>,
}

impl ColumnSchema {
    /// Check whether a type is valid tag type.
    pub fn is_valid_tag_type(typ: DatumKind) -> bool {
        match typ {
            DatumKind::Null => false,
            DatumKind::Timestamp => true,
            DatumKind::Double => false,
            DatumKind::Float => false,
            DatumKind::Varbinary => true,
            DatumKind::String => true,
            DatumKind::UInt64 => true,
            DatumKind::UInt32 => true,
            DatumKind::UInt16 => true,
            DatumKind::UInt8 => true,
            DatumKind::Int64 => true,
            DatumKind::Int32 => true,
            DatumKind::Int16 => true,
            DatumKind::Int8 => true,
            DatumKind::Boolean => true,
        }
    }

    /// Convert `self` to [proto::common::ColumnSchema]
    ///
    /// The `is_key` is needed because it is maintained by
    /// [crate::schema::Schema]
    pub fn to_pb(&self) -> common_pb::ColumnSchema {
        let mut column_schema = common_pb::ColumnSchema::new();
        column_schema.set_name(self.name.clone());
        column_schema.set_data_type(self.data_type.into());
        column_schema.set_is_nullable(self.is_nullable);
        column_schema.set_id(self.id);
        column_schema.set_is_tag(self.is_tag);
        column_schema.set_comment(self.comment.clone());

        column_schema
    }

    /// Convert `self` to [arrow_deps::arrow::datatypes::Field]
    pub fn to_arrow_field(&self) -> Field {
        From::from(self)
    }

    /// Returns Ok if column with `writer_schema` can write to column with the
    /// same schema as `self`.
    pub fn compatible_for_write(
        &self,
        writer_schema: &ColumnSchema,
    ) -> std::result::Result<(), CompatError> {
        ensure!(
            self.data_type == writer_schema.data_type,
            IncompatDataType {
                name: &self.name,
                expect: writer_schema.data_type,
                given: self.data_type,
            }
        );

        // This column is not nullable but writer is nullable
        ensure!(
            self.is_nullable || !writer_schema.is_nullable,
            NotNullable { name: &self.name }
        );

        Ok(())
    }

    /// Returns `Ok` if the source schema can read by this schema, now we won't
    /// validate data type of column
    pub fn compatible_for_read(
        &self,
        source_schema: &ColumnSchema,
    ) -> std::result::Result<ReadOp, CompatError> {
        if self.is_nullable {
            // Column is nullable
            if self.id == source_schema.id {
                // Same column
                Ok(ReadOp::Exact)
            } else {
                // Not the same column, maybe dropped, fill by null.
                Ok(ReadOp::FillNull)
            }
        } else {
            // Column is not null. We consider the old column was dropped if they have
            // different column id and also try to fill by null, so we
            // also check column id.
            ensure!(
                self.id == source_schema.id && !source_schema.is_nullable,
                NotNullable {
                    name: &source_schema.name,
                }
            );

            Ok(ReadOp::Exact)
        }
    }
}

impl From<common_pb::ColumnSchema> for ColumnSchema {
    fn from(column_schema: common_pb::ColumnSchema) -> Self {
        let escaped_name = column_schema.name.escape_debug().to_string();
        Self {
            id: column_schema.id,
            name: column_schema.name,
            data_type: DatumKind::from(column_schema.data_type),
            is_nullable: column_schema.is_nullable,
            is_tag: column_schema.is_tag,
            comment: column_schema.comment,
            escaped_name,
            default_value: None,
        }
    }
}

impl TryFrom<&Field> for ColumnSchema {
    type Error = Error;

    fn try_from(field: &Field) -> Result<Self> {
        let ArrowFieldMeta {
            id,
            is_tag,
            comment,
        } = field
            .metadata()
            .map(decode_arrow_field_meta_data)
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            id,
            name: field.name().clone(),
            data_type: DatumKind::from_data_type(field.data_type()).context(
                UnsupportedDataType {
                    data_type: field.data_type().clone(),
                },
            )?,
            is_nullable: field.is_nullable(),
            is_tag,
            comment,
            escaped_name: field.name().escape_debug().to_string(),
            default_value: None,
        })
    }
}

impl From<&ColumnSchema> for Field {
    fn from(col_schema: &ColumnSchema) -> Self {
        let metadata = encode_arrow_field_meta_data(col_schema);
        let mut field = Field::new(
            &col_schema.name,
            col_schema.data_type.into(),
            col_schema.is_nullable,
        );
        field.set_metadata(Some(metadata));

        field
    }
}

fn parse_arrow_field_meta_value<T>(
    meta: &BTreeMap<String, String>,
    key: ArrowFieldMetaKey,
) -> Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let raw_value = meta
        .get(key.as_str())
        .context(ArrowFieldMetaKeyNotFound { key })?;
    T::from_str(raw_value.as_str())
        .map_err(|e| Box::new(e) as _)
        .context(InvalidArrowFieldMetaValue { key, raw_value })
}

fn decode_arrow_field_meta_data(meta: &BTreeMap<String, String>) -> Result<ArrowFieldMeta> {
    Ok(ArrowFieldMeta {
        id: parse_arrow_field_meta_value(meta, ArrowFieldMetaKey::Id)?,
        is_tag: parse_arrow_field_meta_value(meta, ArrowFieldMetaKey::IsTag)?,
        comment: parse_arrow_field_meta_value(meta, ArrowFieldMetaKey::Comment)?,
    })
}

fn encode_arrow_field_meta_data(col_schema: &ColumnSchema) -> BTreeMap<String, String> {
    let mut meta = BTreeMap::new();

    meta.insert(ArrowFieldMetaKey::Id.to_string(), col_schema.id.to_string());
    meta.insert(
        ArrowFieldMetaKey::IsTag.to_string(),
        col_schema.is_tag.to_string(),
    );
    meta.insert(
        ArrowFieldMetaKey::Comment.to_string(),
        col_schema.comment.clone(),
    );

    meta
}

/// ColumnSchema builder
#[must_use]
pub struct Builder {
    id: ColumnId,
    name: String,
    data_type: DatumKind,
    is_nullable: bool,
    is_tag: bool,
    comment: String,
    default_value: Option<LogicalExpr>,
}

impl Builder {
    /// Create a new builder
    pub fn new(name: String, data_type: DatumKind) -> Self {
        Self {
            id: COLUMN_ID_UNINIT,
            name,
            data_type,
            is_nullable: false,
            is_tag: false,
            comment: String::new(),
            default_value: None,
        }
    }

    pub fn id(mut self, id: ColumnId) -> Self {
        self.id = id;
        self
    }

    /// Set this column is nullable, default is true (not nullable).
    pub fn is_nullable(mut self, is_nullable: bool) -> Self {
        self.is_nullable = is_nullable;
        self
    }

    /// Set this column is tag, default is false (not a tag).
    pub fn is_tag(mut self, is_tag: bool) -> Self {
        self.is_tag = is_tag;
        self
    }

    pub fn comment(mut self, comment: String) -> Self {
        self.comment = comment;
        self
    }

    pub fn default_value(mut self, default_value: Option<LogicalExpr>) -> Self {
        self.default_value = default_value;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.is_tag {
            ensure!(
                ColumnSchema::is_valid_tag_type(self.data_type),
                InvalidTagType {
                    data_type: self.data_type
                }
            );
        }

        Ok(())
    }

    pub fn build(self) -> Result<ColumnSchema> {
        self.validate()?;
        let escaped_name = self.name.escape_debug().to_string();
        Ok(ColumnSchema {
            id: self.id,
            name: self.name,
            data_type: self.data_type,
            is_nullable: self.is_nullable,
            is_tag: self.is_tag,
            comment: self.comment,
            escaped_name,
            default_value: self.default_value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a column schema for test, each field is filled with non-default
    /// value
    fn new_test_column_schema() -> ColumnSchema {
        Builder::new("test_column_schema".to_string(), DatumKind::Boolean)
            .id(18)
            .is_nullable(true)
            .is_tag(true)
            .comment("Comment of this column".to_string())
            .build()
            .expect("should succeed to build column schema")
    }

    #[test]
    fn test_builder() {
        let lhs = new_test_column_schema();
        let rhs = ColumnSchema {
            id: 18,
            name: "test_column_schema".to_string(),
            data_type: DatumKind::Boolean,
            is_nullable: true,
            is_tag: true,
            comment: "Comment of this column".to_string(),
            escaped_name: "test_column_schema".escape_debug().to_string(),
            default_value: None,
        };

        assert_eq!(&lhs, &rhs);
    }

    #[test]
    fn test_pb_convert() {
        let column_schema = new_test_column_schema();
        let pb_schema = column_schema.to_pb();
        // Check pb specific fields
        assert!(pb_schema.is_tag);

        let schema_from_pb = ColumnSchema::from(pb_schema);
        assert_eq!(&schema_from_pb, &column_schema);
    }

    #[test]
    fn test_valid_tag_type() {
        let invalid_tag_types = vec![DatumKind::Null, DatumKind::Float, DatumKind::Double];

        for v in &DatumKind::VALUES {
            assert_eq!(
                ColumnSchema::is_valid_tag_type(*v),
                !invalid_tag_types.contains(v)
            );
        }
    }
}
