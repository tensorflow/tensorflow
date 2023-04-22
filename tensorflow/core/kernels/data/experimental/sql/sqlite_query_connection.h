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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_SQL_SQLITE_QUERY_CONNECTION_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_SQL_SQLITE_QUERY_CONNECTION_H_

#include <memory>

#include "tensorflow/core/kernels/data/experimental/sql/query_connection.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace sql {

class SqliteQueryConnection : public QueryConnection {
 public:
  SqliteQueryConnection();
  ~SqliteQueryConnection() override;
  Status Open(const string& data_source_name, const string& query,
              const DataTypeVector& output_types) override;
  Status Close() override;
  Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) override;

 private:
  // Prepares the query string `query_`.
  Status PrepareQuery();
  // Fills `tensor` with the column_index_th element of the current row of
  // `stmt_`.
  void FillTensorWithResultSetEntry(const DataType& data_type, int column_index,
                                    Tensor* tensor);
  Sqlite* db_ = nullptr;
  SqliteStatement stmt_;
  int column_count_ = 0;
  string query_;
  DataTypeVector output_types_;
};

}  // namespace sql
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_SQL_SQLITE_QUERY_CONNECTION_H_
