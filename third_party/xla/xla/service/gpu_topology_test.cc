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

#include "xla/service/gpu_topology.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/target_config/target_config.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

TEST(GpuTopologyTest, GetGpuModel) {
  EXPECT_THAT(GetGpuModel("tesla_a100"),
              absl_testing::IsOkAndHolds(gpu::GpuModel::A100_SXM_40));
  EXPECT_THAT(GetGpuModel("nvidia_h100"),
              absl_testing::IsOkAndHolds(gpu::GpuModel::H100_SXM));
  EXPECT_THAT(GetGpuModel("umbriel_b200"),
              absl_testing::IsOkAndHolds(gpu::GpuModel::B200));
}

TEST(GpuTopologyTest, GetGpuModelInvalid) {
  EXPECT_THAT(GetGpuModel("invalid_gpu"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
