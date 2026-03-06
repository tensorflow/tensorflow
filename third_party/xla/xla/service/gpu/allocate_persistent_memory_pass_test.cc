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

#include "xla/service/gpu/allocate_persistent_memory_pass.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOk;

class MockThunkPassBufferAllocator : public ThunkPassBufferAllocator {
 public:
  absl::StatusOr<BufferAllocation* absl_nonnull> NewEmptyAllocation(
      int64_t size) override {
    return absl::UnimplementedError("MockThunkPassBufferAllocator");
  }
};

class MockThunk : public Thunk {
 public:
  MockThunk() : Thunk(Kind::kKernel, ThunkInfo()) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }

  absl::Status AllocatePersistentBuffers(
      ThunkPassBufferAllocator& allocator) override {
    allocate_called_ = true;
    return absl::OkStatus();
  }

  bool allocate_called() const { return allocate_called_; }

 private:
  bool allocate_called_ = false;
};

TEST(AllocatePersistentMemoryPassTest, WalksNestedThunks) {
  ThunkSequence sequence;
  auto thunk1_ptr = std::make_unique<MockThunk>();
  MockThunk* thunk1 = thunk1_ptr.get();
  sequence.push_back(std::move(thunk1_ptr));

  ThunkSequence nested_sequence;
  auto thunk2_ptr = std::make_unique<MockThunk>();
  MockThunk* thunk2 = thunk2_ptr.get();
  nested_sequence.push_back(std::move(thunk2_ptr));
  auto nested_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(nested_sequence));

  sequence.push_back(std::move(nested_thunk));

  SequentialThunk root(Thunk::ThunkInfo(), std::move(sequence));

  MockThunkPassBufferAllocator allocator;
  EXPECT_THAT(AllocatePersistentMemoryPass::Run(root, allocator), IsOk());

  EXPECT_TRUE(thunk1->allocate_called());
  EXPECT_TRUE(thunk2->allocate_called());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
