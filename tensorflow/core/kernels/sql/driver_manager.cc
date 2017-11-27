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
#include "tensorflow/core/kernels/sql/driver_manager.h"
#include "tensorflow/core/kernels/sql/sqlite_query_connection.h"

namespace tensorflow {

namespace sql {

std::unique_ptr<QueryConnection> DriverManager::CreateQueryConnection(
    const string& driver_name) {
  if (driver_name == "sqlite") {
    return std::unique_ptr<SqliteQueryConnection>(new SqliteQueryConnection());
  } else {  // TODO(b/64276826, b/64276995) Add support for other db types.
            // Change to registry pattern.
    return nullptr;
  }
}

}  // namespace sql

}  // namespace tensorflow
