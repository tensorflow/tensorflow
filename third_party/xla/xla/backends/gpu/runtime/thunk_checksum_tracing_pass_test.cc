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

#include "xla/backends/gpu/runtime/thunk_checksum_tracing_pass.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class FakeThunkPassBufferAllocator : public ThunkPassBufferAllocator {
 public:
  absl::StatusOr<BufferAllocation*> NewEmptyAllocation(int64_t size) override {
    if (CreatedAlloc()) {
      return absl::InvalidArgumentError("Expected only one allocation");
    }
    alloc_ = std::make_unique<BufferAllocation>(0, size, 0);
    return alloc_.get();
  }

  bool CreatedAlloc() { return alloc_ != nullptr; }

 private:
  std::unique_ptr<BufferAllocation> alloc_;
};

TEST(ThunkChecksumTracingPassTest, CreatesLogAlloc) {
  ThunkChecksumTracingPass pass;
  auto root_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::vector<std::unique_ptr<Thunk>>());
  DebugOptions debug_options;
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      pass.Run(root_thunk.get(), debug_options, device_info, allocator));
  EXPECT_FALSE(changed);
  EXPECT_TRUE(allocator.CreatedAlloc());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
