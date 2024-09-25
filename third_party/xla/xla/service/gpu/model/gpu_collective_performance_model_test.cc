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


#include <gtest/gtest.h>
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

using GpuPerformanceWithCollectiveModelTest = HloTestBase;

TEST_F(GpuPerformanceWithCollectiveModelTest, TestNvmlLibraryLoading) {
#if GOOGLE_CUDA
  EXPECT_TRUE(GpuPerformanceWithCollectiveModel::InitNvml());
  // After successful init, we try to use one of the
  // nvml functions to see if the result is good.
  nvmlDevice_t nvml_device;
  nvmlReturn_t get_device_result =
      xla_nvmlDeviceGetHandleByIndex(0, &nvml_device);
  EXPECT_TRUE(get_device_result == NVML_SUCCESS);

  EXPECT_TRUE(GpuPerformanceWithCollectiveModel::InitNvml());

#endif  // GOOGLE_CUDA
}

}  // namespace
}  // namespace gpu
}  // namespace xla
