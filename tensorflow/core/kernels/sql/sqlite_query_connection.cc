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
#include "tensorflow/core/kernels/sql/sqlite_query_connection.h"

#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {

namespace sql {

SqliteQueryConnection::SqliteQueryConnection() {}
SqliteQueryConnection::~SqliteQueryConnection() {}

Status SqliteQueryConnection::Open(const string& data_source_name,
                                   const string& query,
                                   const DataTypeVector& output_types) {
  if (db_ != nullptr) {
    return errors::FailedPrecondition(
        "Failed to open query connection: Connection already opeend.");
  }
  auto s = Sqlite::Open(data_source_name);
  if (s.ok()) {
    db_ = std::move(s.ValueOrDie());
    query_ = query;
    output_types_ = output_types;
  }
  return s.status();
}

Status SqliteQueryConnection::Close() {
  Status s;
  s.Update(stmt_.Close());
  s.Update(db_->Close());
  return s;
}

Status SqliteQueryConnection::GetNext(std::vector<Tensor>* out_tensors,
                                      bool* end_of_sequence) {
  if (!stmt_) {
    Status s = PrepareQuery();
    if (!s.ok()) {
      return s;
    }
  }
  Status s = stmt_.Step(end_of_sequence);
  if (!*end_of_sequence) {
    for (int i = 0; i < column_count_; i++) {
      DataType dt = output_types_[i];
      Tensor tensor(cpu_allocator(), dt, {});
      FillTensorWithResultSetEntry(dt, i, &tensor);
      out_tensors->emplace_back(std::move(tensor));
    }
  }
  return s;
}

Status SqliteQueryConnection::PrepareQuery() {
  stmt_ = db_->Prepare(query_);
  Status s = stmt_.status();
  if (s.ok()) {
    int column_count = stmt_.ColumnCount();
    if (column_count != output_types_.size()) {
      return errors::InvalidArgument(tensorflow::strings::Printf(
          "The number of columns in query (%d) must match the number of "
          "elements in output_types (%zu).",
          column_count, output_types_.size()));
    }
    column_count_ = column_count;
  }
  return s;
}

void SqliteQueryConnection::FillTensorWithResultSetEntry(
    const DataType& data_type, int column_index, Tensor* tensor) {
  switch (data_type) {
    case DT_STRING:
      tensor->scalar<string>()() = stmt_.ColumnString(column_index);
      break;
    case DT_INT8:
      tensor->scalar<int8>()() =
          static_cast<int8>(stmt_.ColumnInt(column_index));
      break;
    case DT_INT16:
      tensor->scalar<int16>()() =
          static_cast<int16>(stmt_.ColumnInt(column_index));
      break;
    case DT_INT32:
      tensor->scalar<int32>()() =
          static_cast<int32>(stmt_.ColumnInt(column_index));
      break;
    case DT_INT64:
      tensor->scalar<int64>()() = stmt_.ColumnInt(column_index);
      break;
    case DT_UINT8:
      tensor->scalar<uint8>()() =
          static_cast<uint8>(stmt_.ColumnInt(column_index));
      break;
    case DT_UINT16:
      tensor->scalar<uint16>()() =
          static_cast<uint16>(stmt_.ColumnInt(column_index));
      break;
    case DT_BOOL:
      tensor->scalar<bool>()() = stmt_.ColumnInt(column_index) != 0;
      break;
    case DT_FLOAT:
      tensor->scalar<float>()() =
          static_cast<float>(stmt_.ColumnDouble(column_index));
      break;
    case DT_DOUBLE:
      tensor->scalar<double>()() = stmt_.ColumnDouble(column_index);
      break;
      // Error preemptively thrown by SqlDatasetOp::MakeDataset in this case.
    default: {
      LOG(FATAL)
          << "Use of unsupported TensorFlow data type by 'SqlQueryConnection': "
          << DataTypeString(data_type) << ".";
    }
  }
}

}  // namespace sql

}  // namespace tensorflow
