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

// Returns a Status with the sqlite error message corresponding to the
// sqlite error number, `sqlite_err`.
static Status SqliteErrorToStatus(sqlite3* db, int sqlite_err) {
  if (sqlite_err == SQLITE_OK) {
    return Status::OK();
  } else {
    const char* err_msg = sqlite3_errmsg(db);
    // TODO(b/64276468) Be smart about the error code being returned
    return errors::Unknown(
        tensorflow::strings::Printf("Sqlite error: %s", err_msg));
  }
}

SqliteQueryConnection::SqliteQueryConnection(){};

SqliteQueryConnection::~SqliteQueryConnection() {
  Status s = Close();
  if (!s.ok()) {
    LOG(WARNING) << "Failed to close query connection: " << s;
  }
}

Status SqliteQueryConnection::Open(const string& data_source_name,
                                   const string& query,
                                   const DataTypeVector& output_types) {
  if (db_ != nullptr) {
    return errors::FailedPrecondition(
        "Failed to open query connection: Connection already opeend.");
  }
  int err = sqlite3_open(data_source_name.c_str(), &db_);
  Status s = SqliteErrorToStatus(db_, err);
  if (s.ok()) {
    query_ = query;
    output_types_ = output_types;
  }
  return s;
}

Status SqliteQueryConnection::Close() {
  int err = sqlite3_finalize(stmt_);
  if (err != SQLITE_OK) {
    return SqliteErrorToStatus(db_, err);
  }
  stmt_ = nullptr;
  err = sqlite3_close(db_);
  if (err != SQLITE_OK) {
    return SqliteErrorToStatus(db_, err);
  }
  db_ = nullptr;
  return Status::OK();
}

Status SqliteQueryConnection::GetNext(std::vector<Tensor>* out_tensors,
                                      bool* end_of_sequence) {
  if (stmt_ == nullptr) {
    Status s = ExecuteQuery();
    if (!s.ok()) {
      return s;
    }
  }
  int rc = sqlite3_step(stmt_);
  if (rc == SQLITE_ROW) {
    for (int i = 0; i < column_count_; i++) {
      // TODO(b/64276939) Support other tensorflow types. Interpret columns as
      // the types that the client specifies.
      DataType dt = output_types_[i];
      Tensor tensor(cpu_allocator(), dt, {});
      FillTensorWithResultSetEntry(dt, i, &tensor);
      out_tensors->emplace_back(std::move(tensor));
    }
    *end_of_sequence = false;
    return Status::OK();
  } else if (rc == SQLITE_DONE) {
    *end_of_sequence = true;
    return Status::OK();
  } else {
    return SqliteErrorToStatus(db_, rc);
  }
}

Status SqliteQueryConnection::ExecuteQuery() {
  int err = sqlite3_prepare_v2(db_, query_.c_str(), -1, &stmt_, nullptr);
  Status s = SqliteErrorToStatus(db_, err);
  if (s.ok()) {
    int column_count = sqlite3_column_count(stmt_);
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
    case DT_STRING: {
      const void* bytes = sqlite3_column_blob(stmt_, column_index);
      int num_bytes = sqlite3_column_bytes(stmt_, column_index);
      string value(reinterpret_cast<const char*>(bytes), num_bytes);
      tensor->scalar<string>()() = value;
      break;
    }
    case DT_INT32: {
      int32 value = sqlite3_column_int(stmt_, column_index);
      tensor->scalar<int32>()() = value;
      break;
    }
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
