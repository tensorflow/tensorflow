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

#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

// A simple ThunkPass that adds a new thunk to the root thunk.
class TestPass : public ThunkPassInterface {
 public:
  absl::string_view name() const override { return "test-pass"; }
  absl::StatusOr<bool> Run(SequentialThunk* root_thunk,
                           const DebugOptions& debug_options,
                           const HloModule* hlo_module,
                           const se::DeviceDescription& device_info,
                           ThunkPassBufferAllocator& /*allocator*/) override {
    root_thunk->thunks().push_back(std::make_unique<SequentialThunk>(
        Thunk::ThunkInfo(), std::vector<std::unique_ptr<Thunk>>()));
    return true;
  }
};

class FakeThunkPassBufferAllocator : public ThunkPassBufferAllocator {
  absl::StatusOr<BufferAllocation* absl_nonnull> NewEmptyAllocation(
      int64_t size) override {
    return absl::UnimplementedError("NewEmptyAllocation is not implemented.");
  }
};

TEST(ThunkPassPipelineTest, PipelineRunsPass) {
  ThunkPassPipeline pipeline("test-pipeline");
  pipeline.AddPass(std::make_unique<TestPass>());

  auto root_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::vector<std::unique_ptr<Thunk>>());
  DebugOptions debug_options;
  se::DeviceDescription device_info;
  FakeThunkPassBufferAllocator allocator;

  EXPECT_EQ(root_thunk->thunks().size(), 0);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      pipeline.Run(root_thunk.get(), debug_options, /*hlo_module=*/nullptr,
                   device_info, allocator));
  EXPECT_TRUE(changed);
  EXPECT_EQ(root_thunk->thunks().size(), 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
