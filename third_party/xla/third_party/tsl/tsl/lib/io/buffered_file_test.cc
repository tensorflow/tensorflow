/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/lib/io/buffered_file.h"

#include <memory>
#include <utility>

#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace tsl {
namespace io {
namespace {

TEST(BufferedInputStream, Tell) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  std::unique_ptr<WritableFile> write_file;
  TF_ASSERT_OK(env->NewWritableFile(fname, &write_file));
  BufferedWritableFile file(std::move(write_file), 8);
  int64_t position;
  TF_ASSERT_OK(file.Append("foo"));
  TF_ASSERT_OK(file.Tell(&position));
  EXPECT_EQ(position, 3);
  TF_ASSERT_OK(file.Append("bar"));
  TF_ASSERT_OK(file.Tell(&position));
  EXPECT_EQ(position, 6);
  TF_ASSERT_OK(file.Append("baz"));
  TF_ASSERT_OK(file.Tell(&position));
  EXPECT_EQ(position, 9);
}

}  // anonymous namespace
}  // namespace io
}  // namespace tsl
