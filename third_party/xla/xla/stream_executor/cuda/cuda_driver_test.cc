/* Copyright 2020 The OpenXLA Authors.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_diagnostics.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

using ::tsl::testing::IsOkAndHolds;

namespace stream_executor {
namespace cuda {

void CheckCuda(CUresult result, const char* file, int line) {
  TF_CHECK_OK(cuda::ToStatus(result));
}

void CheckCuda(cudaError_t result, const char* file, int line) {
  if (result == cudaSuccess) {
    return;
  }
  const char* name = cudaGetErrorName(result);
  const char* message = cudaGetErrorString(result);
  LOG(FATAL) << file << "(" << line << "): " << name << ", " << message;
}

#define CHECK_CUDA(result) CheckCuda(result, __FILE__, __LINE__)

class CudaDriverTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { CHECK_CUDA(cuInit(0)); }
};

TEST_F(CudaDriverTest, DriverVersionParsingTest) {
  // Tests that the driver version can be right after 'Kernel Module',
  // or later as well.
  auto driver_version = Diagnostician::FindKernelModuleVersion(
      "... NVIDIA UNIX Open Kernel Module for x86_64  570.00  Release Build  "
      "...  Mon Aug 12 04:17:20 UTC 2024");
  TF_CHECK_OK(driver_version.status());
  EXPECT_EQ("570.0.0", cuda::DriverVersionToString(driver_version.value()));

  driver_version = Diagnostician::FindKernelModuleVersion(
      "... NVIDIA UNIX Open Kernel Module  571.00  Release Build  "
      "...  Mon Aug 12 04:17:20 UTC 2024");
  TF_CHECK_OK(driver_version.status());
  EXPECT_EQ("571.0.0", cuda::DriverVersionToString(driver_version.value()));
}

TEST_F(CudaDriverTest, GraphGetNodeCountTest) {
  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, 0));
  CUcontext context;
  CHECK_CUDA(cuCtxCreate(&context, 0, device));
  gpu::GpuGraphHandle graph;
  TF_CHECK_OK(gpu::GpuDriver::CreateGraph(&graph));
  absl::Cleanup cleanup(
      [graph] { TF_CHECK_OK(gpu::GpuDriver::DestroyGraph(graph)); });
  EXPECT_THAT(gpu::GpuDriver::GraphGetNodeCount(graph), IsOkAndHolds(0));
  gpu::GpuGraphNodeHandle node;
  TF_CHECK_OK(gpu::GpuDriver::GraphAddEmptyNode(&node, graph, {}));
  EXPECT_THAT(gpu::GpuDriver::GraphGetNodeCount(graph), IsOkAndHolds(1));
}

}  // namespace cuda

}  // namespace stream_executor
