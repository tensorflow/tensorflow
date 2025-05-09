/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/conditional_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

using ::testing::Pointee;
using ::testing::Property;

// A dummy `Thunk` that does nothing.
struct DummyThunk : public Thunk {
  DummyThunk() : Thunk(Thunk::Kind::kGemm, Thunk::ThunkInfo()) {}
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
};

TEST(ConditionalThunkTest, BufferUses) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice branch_index_slice(&alloc, 0, sizeof(int32_t));

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::make_unique<DummyThunk>());

  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks;
  branch_thunks.push_back(std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_sequence)));

  constexpr bool kBranchIndexIsBool = false;
  ConditionalThunk conditional_thunk(Thunk::ThunkInfo(), branch_index_slice,
                                     std::move(branch_thunks),
                                     kBranchIndexIsBool);

  EXPECT_EQ(conditional_thunk.branch_index_is_bool(), kBranchIndexIsBool);
  EXPECT_EQ(conditional_thunk.branch_index_buffer(), branch_index_slice);
  EXPECT_THAT(
      conditional_thunk.branch_thunks(),
      ElementsAre(Pointee(Property(
          &SequentialThunk::thunks,
          ElementsAre(Pointee(Property(&Thunk::kind, Thunk::Kind::kGemm)))))));
}
}  // namespace
}  // namespace xla::gpu
