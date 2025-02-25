// Copyright 2022 CeresDB Project Authors. Licensed under Apache-2.0.

//! Contains common types

pub mod bytes;
pub mod column;
pub mod column_schema;
pub mod datum;
pub mod hash;
pub mod projected_schema;
pub mod record_batch;
pub mod request_id;
pub mod row;
pub mod schema;
pub mod string;
pub mod time;

/// Sequence number
pub type SequenceNumber = u64;
pub const MAX_SEQUENCE_NUMBER: u64 = u64::MAX;
pub const MIN_SEQUENCE_NUMBER: u64 = 0;

#[cfg(any(test, feature = "test"))]
pub mod tests;
