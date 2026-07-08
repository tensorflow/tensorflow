/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/command_buffer.h"

#include <gtest/gtest.h>

namespace stream_executor {

class CommandBufferTest : public ::testing::Test {
 protected:
  using ResourceTypeId = CommandBuffer::ResourceTypeId;

  static ResourceTypeId GetNextResourceTypeId() {
    return CommandBuffer::GetNextResourceTypeId();
  }
};

TEST_F(CommandBufferTest, GetNextResourceTypeIdReturnsDifferentIds) {
  ResourceTypeId id1 = GetNextResourceTypeId();
  ResourceTypeId id2 = GetNextResourceTypeId();
  EXPECT_NE(id1, id2);
}

}  // namespace stream_executor
