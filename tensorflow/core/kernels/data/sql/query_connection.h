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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_SQL_QUERY_CONNECTION_H_
#define TENSORFLOW_CORE_KERNELS_DATA_SQL_QUERY_CONNECTION_H_

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

class IteratorContext;

namespace sql {
// This interface allows a user to connect to a database, execute a query, and
// iterate over the result set, putting the results into an output tensor.
// A subclass implementation is required for each type of database
// (e.g. sqlite3, mysql, etc.)
//
// Presently, a `QueryConnection` instance can only handle one query at a time.
// In a future extension, this class may be refactored so that it creates
// instances of a new class (named, say, `Statement`) which could have a
// one-to-one correspondence with queries. This would make `QueryConnection`
// more consistent with `Connection` classes of other database APIs.
// `QueryConnection` would then be renamed simply `Connection`.
//
// This class is not thread safe. Access to it is guarded by a mutex in
// `SqlDatasetOp::Dataset::Iterator`.
class QueryConnection {
 public:
  virtual ~QueryConnection() {}
  // Opens a connection to the database named by `data_source_name`. Prepares to
  // execute `query` against the database.
  //
  // The client must call `Close()` to release the connection resources, even
  // if `Open()` fails. `Close()` must be called before making another call
  // to `Open()`.
  virtual Status Open(const string& data_source_name, const string& query,
                      const DataTypeVector& output_types) = 0;
  // Closes an opened connection.
  virtual Status Close() = 0;
  // Retrieves the next row of the result set of the query from the most recent
  // call to `Open()`.
  //
  // If such a row exists, then the row will be stored in `*out_tensors`, and
  // `false` will be stored in `*end_of_sequence`.
  //
  // If there are no more rows in the result set, then instead `true` will be
  // stored in `*end_of_sequence`, and the content of `*out_tensors` will be
  // undefined.
  virtual Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) = 0;
};

}  // namespace sql
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_SQL_QUERY_CONNECTION_H_
