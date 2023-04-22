/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_driver.h"

#include "absl/memory/memory.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/core/platform/test.h"

namespace stream_executor {
namespace gpu {

void CheckCuda(CUresult result, const char* file, int line) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name;
  cuGetErrorName(result, &name);
  const char* message;
  cuGetErrorString(result, &message);
  LOG(FATAL) << file << "(" << line << "): " << name << ", " << message;
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

TEST(CudaDriverTest, ScopedActivateContextTest) {
  CHECK_CUDA(cuInit(0));
  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, 0));
  CUcontext context0, context1;
  CHECK_CUDA(cuCtxCreate(&context0, 0, device));
  CHECK_CUDA(cuCtxCreate(&context1, 0, device));
  GpuContext se_context1(context1, /*id=*/101);
  {
    ScopedActivateContext scope(&se_context1);
    CUcontext c;
    CHECK_CUDA(cuCtxGetCurrent(&c));
    EXPECT_EQ(c, context1);
  }
  CHECK_CUDA(cuCtxSetCurrent(context0));
  // ScopedActivateContext must correctly set the CUDA context even if some
  // other code changes the context between the two scopes.
  {
    ScopedActivateContext scope(&se_context1);
    CUcontext c;
    CHECK_CUDA(cuCtxGetCurrent(&c));
    EXPECT_EQ(c, context1);
  }
}

}  // namespace gpu
}  // namespace stream_executor

#endif  // GOOGLE_CUDA
