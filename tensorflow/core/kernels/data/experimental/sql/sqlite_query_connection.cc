/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/kernels/data/experimental/sql/sqlite_query_connection.h"

#include <vector>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace sql {

SqliteQueryConnection::SqliteQueryConnection() {}

SqliteQueryConnection::~SqliteQueryConnection() {
  if (db_ != nullptr) db_->Unref();
}

absl::Status SqliteQueryConnection::Open(const string& data_source_name,
                                         const string& query,
                                         const DataTypeVector& output_types) {
  if (db_ != nullptr) {
    return errors::FailedPrecondition(
        "Failed to open query connection: Connection already opened.");
  }
  TF_RETURN_IF_ERROR(Sqlite::Open(
      data_source_name, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, &db_));
  query_ = query;
  output_types_ = output_types;
  return absl::OkStatus();
}

absl::Status SqliteQueryConnection::Close() {
  stmt_ = SqliteStatement();
  db_->Unref();
  db_ = nullptr;
  return absl::OkStatus();
}

absl::Status SqliteQueryConnection::GetNext(IteratorContext* ctx,
                                            std::vector<Tensor>* out_tensors,
                                            bool* end_of_sequence) {
  if (!stmt_) TF_RETURN_IF_ERROR(PrepareQuery());
  TF_RETURN_IF_ERROR(stmt_.Step(end_of_sequence));
  if (!*end_of_sequence) {
    for (int i = 0; i < column_count_; i++) {
      DataType dt = output_types_[i];
      // TODO(mrry): Pass in the `IteratorContext::allocator()`.
      out_tensors->emplace_back(ctx->allocator({}), dt, TensorShape({}));
      FillTensorWithResultSetEntry(dt, i, &out_tensors->back());
    }
  }
  return absl::OkStatus();
}

absl::Status SqliteQueryConnection::PrepareQuery() {
  TF_RETURN_IF_ERROR(db_->Prepare(query_, &stmt_));
  int column_count = stmt_.ColumnCount();
  if (column_count != static_cast<int>(output_types_.size())) {
    stmt_ = SqliteStatement();
    return errors::InvalidArgument(tensorflow::strings::Printf(
        "The number of columns in query (%d) must match the number of "
        "elements in output_types (%zu).",
        column_count, output_types_.size()));
  }
  column_count_ = column_count;
  return absl::OkStatus();
}

void SqliteQueryConnection::FillTensorWithResultSetEntry(
    const DataType& data_type, int column_index, Tensor* tensor) {
#define CASE(T, M)                                                 \
  case DataTypeToEnum<T>::value:                                   \
    tensor->scalar<T>()() = static_cast<T>(stmt_.M(column_index)); \
    break;
#define INT_CASE(T) CASE(T, ColumnInt)
#define DOUBLE_CASE(T) CASE(T, ColumnDouble)
#define STRING_CASE(T) CASE(T, ColumnString)
  // clang-format off
  switch (data_type) {
    TF_CALL_int8(INT_CASE)
    TF_CALL_uint8(INT_CASE)
    TF_CALL_int16(INT_CASE)
    TF_CALL_uint16(INT_CASE)
    TF_CALL_int32(INT_CASE)
    TF_CALL_uint32(INT_CASE)
    TF_CALL_int64(INT_CASE)
    TF_CALL_uint64(INT_CASE)
    TF_CALL_float(DOUBLE_CASE)
    TF_CALL_double(DOUBLE_CASE)
    TF_CALL_tstring(STRING_CASE)
    case DT_BOOL:
      tensor->scalar<bool>()() = stmt_.ColumnInt(column_index) != 0;
      break;
    // Error preemptively thrown by SqlDatasetOp::MakeDataset in this case.
    default:
      LOG(ERROR)
          << "Use of unsupported TensorFlow data type by 'SqlQueryConnection': "
          << DataTypeString(data_type) << ".";
  }
  // clang-format on
}

}  // namespace sql
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
