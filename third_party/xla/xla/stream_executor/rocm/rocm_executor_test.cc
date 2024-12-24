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

#include "xla/stream_executor/rocm/rocm_executor.h"

#include <memory>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
using testing::IsEmpty;
using testing::Not;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(RocmExecutorTest, CreateDeviceDescription) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceDescription> result,
                          RocmExecutor::CreateDeviceDescription(0));

  constexpr SemanticVersion kNullVersion{0, 0, 0};
  EXPECT_NE(result->runtime_version(), kNullVersion);
  EXPECT_NE(result->driver_version(), kNullVersion);
  EXPECT_NE(result->compile_time_toolkit_version(), kNullVersion);

  EXPECT_THAT(result->platform_version(), Not(IsEmpty()));
  EXPECT_THAT(result->name(), Not(IsEmpty()));
  EXPECT_THAT(result->model_str(), Not(IsEmpty()));
  EXPECT_THAT(result->device_vendor(), "Advanced Micro Devices, Inc");

  EXPECT_THAT(
      std::get_if<RocmComputeCapability>(&result->gpu_compute_capability())
          ->gcn_arch_name(),
      Not(IsEmpty()));
}

TEST(RocmExecutorTest, GetRocmKernel) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("ROCM"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          executor->LoadKernel(GetAddI32KernelSpec()));

  auto rocm_executor = dynamic_cast<RocmExecutor*>(executor);
  ASSERT_NE(rocm_executor, nullptr);
  EXPECT_THAT(rocm_executor->GetRocmKernel(kernel.get()),
              IsOkAndHolds(kernel.get()));

  rocm_executor->UnloadKernel(kernel.get());
  EXPECT_THAT(rocm_executor->GetRocmKernel(kernel.get()),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_THAT(rocm_executor->GetRocmKernel(nullptr),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace stream_executor::gpu
