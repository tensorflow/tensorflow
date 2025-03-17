/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/conditional_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xla/backends/cpu/runtime/resource_use.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(ConditionalThunkTest, BufferUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice branch_index_slice(&alloc, 0, sizeof(int32_t));
  BufferAllocation::Slice read_slice(&alloc, 10, 10);

  std::vector<ThunkSequence> branch_sequences(1);
  branch_sequences[0].push_back(
      std::make_unique<BufferUseThunk>(BufferUse::Read(read_slice)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, ConditionalThunk::Create({"conditional"}, branch_index_slice,
                                           std::move(branch_sequences)));

  EXPECT_EQ(thunk->buffer_uses().size(), 2);
  EXPECT_EQ(thunk->buffer_uses()[0], BufferUse::Read(branch_index_slice));
  EXPECT_EQ(thunk->buffer_uses()[1], BufferUse::Read(read_slice));
}

TEST(ConditionalThunkTest, ResourceUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice branch_index_slice(&alloc, 0, sizeof(int32_t));

  auto token = Resource::Create(Resource::kToken);

  std::vector<ThunkSequence> branch_sequences(1);
  branch_sequences[0].push_back(
      std::make_unique<ResourceUseThunk>(ResourceUse::Read(token)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, ConditionalThunk::Create({"conditional"}, branch_index_slice,
                                           std::move(branch_sequences)));

  EXPECT_EQ(thunk->resource_uses().size(), 1);
  EXPECT_EQ(thunk->resource_uses()[0], ResourceUse::Read(token));
}

}  // namespace
}  // namespace xla::cpu
