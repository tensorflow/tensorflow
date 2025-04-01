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
#ifndef TENSORFLOW_CORE_SUMMARY_SUMMARY_DB_WRITER_H_
#define TENSORFLOW_CORE_SUMMARY_SUMMARY_DB_WRITER_H_

#include "absl/status/status.h"
#include "tensorflow/core/kernels/summary_interface.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

/// \brief Creates SQLite SummaryWriterInterface.
///
/// This can be used to write tensors from the execution graph directly
/// to a database. The schema must be created beforehand. Entries in
/// Users, Experiments, and Runs tables will be created automatically
/// if they don't already exist.
///
/// Please note that the type signature of this function may change in
/// the future if support for other DBs is added to core.
///
/// The result holds a new reference to db.
absl::Status CreateSummaryDbWriter(Sqlite* db, const string& experiment_name,
                                   const string& run_name,
                                   const string& user_name, Env* env,
                                   SummaryWriterInterface** result);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_SUMMARY_SUMMARY_DB_WRITER_H_
