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
#include "tensorflow/core/summary/schema.h"

#include <memory>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(SchemaTest, SmokeTestTensorboardSchema) {
  Sqlite* db;
  TF_ASSERT_OK(Sqlite::Open(":memory:", SQLITE_OPEN_READWRITE, &db));
  core::ScopedUnref unref_db(db);
  TF_ASSERT_OK(SetupTensorboardSqliteDb(db));
}

}  // namespace
}  // namespace tensorflow
