// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/dispatch_op_schema.h"

#include <cstddef>

#include <gtest/gtest.h>

namespace litert {
namespace internal {
namespace {

static constexpr size_t kBufferSize = 100;
static constexpr size_t kBufferOffset = 200;
static constexpr const char kName[] = "test_name";

TEST(DispatchOpSchemaTest, DispatchOpOptions) {
  DispatchOpOptions options = {
      kBufferSize,
      kBufferOffset,
      kName,
  };

  auto buffer = MakeDispatchOpOptions(options);
  ASSERT_GT(buffer.Size(), 0);

  auto parsed_options = GetDispatchOpOptions(buffer);
  ASSERT_EQ(parsed_options.bytecode_size, kBufferSize);
  ASSERT_EQ(parsed_options.bytecode_offset, kBufferOffset);
  ASSERT_EQ(parsed_options.name, kName);
}

TEST(DispatchOpSchemaTest, UpdateDispatchOpOptions) {
  DispatchOpOptions options = {
      kBufferSize,
      kBufferOffset,
      kName,
  };

  auto buffer = MakeDispatchOpOptions(options);
  ASSERT_GT(buffer.Size(), 0);

  static constexpr size_t kNewBufferSize = 1000;
  static constexpr size_t kNewBufferOffset = 2000;

  DispatchOpOptions new_options = {
      kNewBufferSize,
      kNewBufferOffset,
      kName,
  };

  ASSERT_TRUE(UpdateDispatchOpOptionsInPlace(new_options, buffer));

  auto parsed_options = GetDispatchOpOptions(buffer);
  ASSERT_EQ(parsed_options.bytecode_size, kNewBufferSize);
  ASSERT_EQ(parsed_options.bytecode_offset, kNewBufferOffset);
  ASSERT_EQ(parsed_options.name, kName);
}

}  // namespace
}  // namespace internal
}  // namespace litert
