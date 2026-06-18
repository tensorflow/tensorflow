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

#include "xla/backends/profiler/gpu/cuda_version_variants.h"

#include <algorithm>

#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"

namespace xla {
namespace profiler {
namespace cuda_versions {

namespace {

int GetCudaRuntimeVersion() {
  static int runtime_version = []() {
    int version = 0;
    cudaError_t status = cudaRuntimeGetVersion(&version);
    if (status == cudaSuccess) {
      LOG(INFO) << "CUDA runtime version: " << version;
      return version;
    }
    LOG(ERROR) << "Failed to get CUDA runtime version, error: "
               << cudaGetErrorName(status) << " - "
               << cudaGetErrorString(status);
    return 0;
  }();
  return runtime_version;
}

int GetCudaDriverVersion() {
  static int driver_version = []() {
    int version = 0;
    cudaError_t status = cudaDriverGetVersion(&version);
    if (status == cudaSuccess) {
      LOG(INFO) << "CUDA driver version: " << version;
      return version;
    }
    LOG(ERROR) << "Failed to get CUDA driver version, error: "
               << cudaGetErrorName(status) << " - "
               << cudaGetErrorString(status);
    return 0;
  }();
  return driver_version;
}

}  // namespace

int GetSafeCudaVersion() {
  static int safe_cuda_version = []() {
    return std::min(GetCudaRuntimeVersion(), GetCudaDriverVersion());
  }();
  return safe_cuda_version;
}

const CbidCategoryMap& EmptyCallbackIdCategories() {
  static const absl::NoDestructor<CbidCategoryMap> kCbidCategoryMap{};
  return *kCbidCategoryMap;
}

}  // namespace cuda_versions
}  // namespace profiler
}  // namespace xla
