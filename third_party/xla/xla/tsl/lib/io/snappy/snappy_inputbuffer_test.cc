/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/lib/io/snappy/snappy_inputbuffer.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/tstring.h"

namespace tsl {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(SnappyInputBuffer, RejectsNegativeBytesToRead) {
  Env* env = Env::Default();
  const std::string fname =
      testing::TmpDir() + "/snappy_inputbuffer_negative_bytes";
  ASSERT_THAT(WriteStringToFile(env, fname, "snappy"), IsOk());
  std::unique_ptr<RandomAccessFile> file;
  ASSERT_THAT(env->NewRandomAccessFile(fname, &file), IsOk());
  io::SnappyInputBuffer in(file.get(), /*input_buffer_bytes=*/64,
                           /*output_buffer_bytes=*/64);

  tstring result;
  EXPECT_THAT(in.ReadNBytes(-1, &result),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Can't read a negative number of bytes")));
}

}  // namespace
}  // namespace tsl
