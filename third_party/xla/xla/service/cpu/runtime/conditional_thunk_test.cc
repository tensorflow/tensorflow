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

#include "xla/service/cpu/runtime/conditional_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

// A test-only thunk to create a Thunk with a specific buffer use.
class TestThunk : public Thunk {
 public:
  explicit TestThunk(BufferUse buffer_use)
      : Thunk(Kind::kKernel, {"test"}), buffer_use_(buffer_use) {}

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final {
    return absl::UnimplementedError("Unimplemented");
  }

  BufferUses buffer_uses() const final { return {buffer_use_}; }

 private:
  BufferUse buffer_use_;
};

TEST(ConditionalThunkTest, BufferUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice branch_index_slice(&alloc, 0, sizeof(int32_t));
  BufferAllocation::Slice read_slice(&alloc, 10, 10);

  std::vector<ThunkSequence> branch_sequences(1);
  branch_sequences[0].push_back(
      std::make_unique<TestThunk>(BufferUse::Read(read_slice)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, ConditionalThunk::Create({"conditional"}, branch_index_slice,
                                           std::move(branch_sequences)));

  EXPECT_EQ(thunk->buffer_uses().size(), 2);
  EXPECT_EQ(thunk->buffer_uses()[0], BufferUse::Read(branch_index_slice));
  EXPECT_EQ(thunk->buffer_uses()[1], BufferUse::Read(read_slice));
}

}  // namespace
}  // namespace xla::cpu
