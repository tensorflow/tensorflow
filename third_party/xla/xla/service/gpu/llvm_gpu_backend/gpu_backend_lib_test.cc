/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"

#include "xla/stream_executor/device_description.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {
namespace se = ::stream_executor;

TEST(UtilsTest, TestGetSmName) {
  se::CudaComputeCapability cc_hopper(9, 0);
  ASSERT_EQ(nvptx::GetSmName(cc_hopper), "sm_90a");
  // Do not default to sm90_a after Hopper, because it is not forward
  // compatible.
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ptx-compatibility
  se::CudaComputeCapability cc_next(10, 0);
  ASSERT_EQ(nvptx::GetSmName(cc_next), "sm_90");
}

using VersionPair = std::pair<nvptx::Version, nvptx::Version>;
using PtxVersionFromCudaVersionTest = ::testing::TestWithParam<VersionPair>;

TEST_P(PtxVersionFromCudaVersionTest, VerifyMapping) {
  EXPECT_EQ(nvptx::DetermineHighestSupportedPtxVersionFromCudaVersion(
                GetParam().first),
            GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(VersionTest, PtxVersionFromCudaVersionTest,
                         ::testing::ValuesIn<VersionPair>({
                             // CUDA 11
                             {{11, 0}, {7, 0}},
                             {{11, 1}, {7, 1}},
                             {{11, 2}, {7, 2}},
                             {{11, 3}, {7, 3}},
                             {{11, 4}, {7, 4}},
                             {{11, 5}, {7, 5}},
                             {{11, 6}, {7, 6}},
                             {{11, 7}, {7, 7}},
                             {{11, 8}, {7, 8}},
                             // CUDA 12
                             {{12, 0}, {8, 0}},
                             {{12, 1}, {8, 1}},
                             {{12, 2}, {8, 2}},
                             {{12, 3}, {8, 3}},
                             {{12, 4}, {8, 4}},
                             {{12, 5}, {8, 5}},
                             {{12, 6}, {8, 5}},
                         }),
                         [](::testing::TestParamInfo<VersionPair> data) {
                           nvptx::Version cuda_version = data.param.first;
                           return absl::StrCat("cuda", cuda_version.first,
                                               cuda_version.second);
                         });

}  // namespace
}  // namespace gpu
}  // namespace xla
