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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_callbacks.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace xla {
namespace profiler {
namespace cuda_versions {
namespace test {

namespace {

TEST(CudaVersionVariantsTest, GetCudaGraphTracingResourceCbids) {
  int safe_cuda_version = GetSafeCudaVersion();
  absl::Span<const CUpti_CallbackIdResource> res_cbids =
      GetCudaGraphTracingResourceCbids();
  if (CUDA_VERSION >= 12000 && safe_cuda_version >= 12000) {
    EXPECT_THAT(res_cbids, testing::Not(testing::IsEmpty()))
        << ", CUDA_VERSION: " << CUDA_VERSION
        << ", safe_cuda_version: " << safe_cuda_version;
  } else {
    EXPECT_THAT(res_cbids, testing::IsEmpty())
        << ", CUDA_VERSION: " << CUDA_VERSION
        << ", safe_cuda_version: " << safe_cuda_version;
  }
}

TEST(CudaVersionVariantsTest, GetExtraCallbackIdCategories12000) {
  int safe_cuda_version = GetSafeCudaVersion();
  const CbidCategoryMap& map = GetExtraCallbackIdCategories12000();
  if (CUDA_VERSION >= 12000 && safe_cuda_version >= 12000) {
    EXPECT_THAT(map, testing::Not(testing::IsEmpty()))
        << ", CUDA_VERSION: " << CUDA_VERSION
        << ", safe_cuda_version: " << safe_cuda_version;

  } else {
    EXPECT_THAT(map, testing::IsEmpty())
        << ", CUDA_VERSION: " << CUDA_VERSION
        << ", safe_cuda_version: " << safe_cuda_version;
  }
}

TEST(CudaVersionVariantsTest, GetExtraCallbackIdCategories12080) {
  int safe_cuda_version = GetSafeCudaVersion();
  const CbidCategoryMap& map = GetExtraCallbackIdCategories12080();
  if (CUDA_VERSION >= 12080 && safe_cuda_version >= 12080) {
    EXPECT_THAT(map, testing::Not(testing::IsEmpty()))
        << ", CUDA_VERSION: " << CUDA_VERSION
        << ", safe_cuda_version: " << safe_cuda_version;

  } else {
    EXPECT_THAT(map, testing::IsEmpty())
        << ", CUDA_VERSION: " << CUDA_VERSION
        << ", safe_cuda_version: " << safe_cuda_version;
  }
}

}  // namespace

}  // namespace test
}  // namespace cuda_versions
}  // namespace profiler
}  // namespace xla
