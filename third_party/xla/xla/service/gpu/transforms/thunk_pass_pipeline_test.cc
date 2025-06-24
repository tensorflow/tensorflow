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

#include "xla/service/gpu/transforms/thunk_pass_pipeline.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// A simple ThunkPass that adds a new thunk to the root thunk.
class TestPass : public ThunkPassInterface {
 public:
  absl::string_view name() const override { return "test-pass"; }
  absl::StatusOr<bool> Run(SequentialThunk* root_thunk,
                           const HloModuleConfig& hlo_module_config,
                           const se::DeviceDescription& device_info) override {
    root_thunk->thunks().push_back(std::make_unique<SequentialThunk>(
        Thunk::ThunkInfo(), std::vector<std::unique_ptr<Thunk>>()));
    return true;
  }
};

TEST(ThunkPassPipelineTest, PipelineRunsPass) {
  ThunkPassPipeline pipeline("test-pipeline");
  pipeline.AddPass(std::make_unique<TestPass>());

  auto root_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::vector<std::unique_ptr<Thunk>>());
  HloModuleConfig config;
  se::DeviceDescription device_info;

  EXPECT_EQ(root_thunk->thunks().size(), 0);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          pipeline.Run(root_thunk.get(), config, device_info));
  EXPECT_TRUE(changed);
  EXPECT_EQ(root_thunk->thunks().size(), 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
