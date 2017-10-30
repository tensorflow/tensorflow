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
#ifndef TENSORFLOW_CONTRIB_TENSORBOARD_DB_SCHEMA_H_
#define TENSORFLOW_CONTRIB_TENSORBOARD_DB_SCHEMA_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/db/sqlite.h"

namespace tensorflow {
namespace db {

/// \brief Creates TensorBoard SQLite tables and indexes.
///
/// If they are already created, this has no effect. If schema
/// migrations are necessary, they will be performed with logging.
Status SetupTensorboardSqliteDb(Sqlite* db);

}  // namespace db
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSORBOARD_DB_SCHEMA_H_
