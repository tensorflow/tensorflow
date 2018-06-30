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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_SQL_DRIVER_MANAGER_H_
#define TENSORFLOW_CORE_KERNELS_DATA_SQL_DRIVER_MANAGER_H_

#include "tensorflow/core/kernels/data/sql/query_connection.h"

namespace tensorflow {

namespace sql {

// A factory class for creating `QueryConnection` instances.
class DriverManager {
 public:
  // A factory method for creating `QueryConnection` instances.
  //
  // `driver_name` is the database type (e.g. 'sqlite'). `driver_name`
  // corresponds to a `QueryConnection` subclass. For example, if `driver_name`
  // == `sqlite`, then `CreateQueryConnection` will create a
  // `SqliteQueryConnection` instance.
  static std::unique_ptr<QueryConnection> CreateQueryConnection(
      const string& driver_name);
};

}  // namespace sql

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_SQL_DRIVER_MANAGER_H_
