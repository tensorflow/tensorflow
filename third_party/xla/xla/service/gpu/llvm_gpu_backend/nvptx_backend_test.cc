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

#include "xla/service/gpu/llvm_gpu_backend/nvptx_backend.h"

#include <utility>

#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {
namespace se = ::stream_executor;

TEST(UtilsTest, TestGetSmName) {
  ASSERT_EQ(nvptx::GetSmName(se::CudaComputeCapability{9, 0}), "sm_90a");
  ASSERT_EQ(nvptx::GetSmName(se::CudaComputeCapability{10, 0}), "sm_100a");
  ASSERT_EQ(nvptx::GetSmName(se::CudaComputeCapability{10, 1}), "sm_101a");
  ASSERT_EQ(nvptx::GetSmName(se::CudaComputeCapability{10, 3}), "sm_103a");
  ASSERT_EQ(nvptx::GetSmName(se::CudaComputeCapability{12, 0}), "sm_120a");
  ASSERT_EQ(nvptx::GetSmName(se::CudaComputeCapability{12, 1}), "sm_121a");
  // Do not use the extension for a yet-unknown compute capability.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes-ptx-release-history
  ASSERT_EQ(nvptx::GetSmName(se::CudaComputeCapability{12, 9}), "sm_121");
}

using VersionPair = std::pair<se::SemanticVersion, se::SemanticVersion>;
using PtxVersionFromCudaVersionTest = ::testing::TestWithParam<VersionPair>;

TEST_P(PtxVersionFromCudaVersionTest, VerifyMapping) {
  EXPECT_EQ(nvptx::DetermineHighestSupportedPtxVersionFromCudaVersion(
                GetParam().first),
            GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(VersionTest, PtxVersionFromCudaVersionTest,
                         ::testing::ValuesIn<VersionPair>({
                             // CUDA 11
                             {{11, 0, 0}, {7, 0, 0}},
                             {{11, 1, 0}, {7, 1, 0}},
                             {{11, 2, 0}, {7, 2, 0}},
                             {{11, 3, 0}, {7, 3, 0}},
                             {{11, 4, 0}, {7, 4, 0}},
                             {{11, 5, 0}, {7, 5, 0}},
                             {{11, 6, 0}, {7, 6, 0}},
                             {{11, 7, 0}, {7, 7, 0}},
                             {{11, 8, 0}, {7, 8, 0}},
                             // CUDA 12
                             {{12, 0, 0}, {8, 0, 0}},
                             {{12, 1, 0}, {8, 1, 0}},
                             {{12, 2, 0}, {8, 2, 0}},
                             {{12, 3, 0}, {8, 3, 0}},
                             {{12, 4, 0}, {8, 4, 0}},
                             {{12, 5, 0}, {8, 5, 0}},
                             {{12, 6, 0}, {8, 5, 0}},
                             {{12, 8, 0}, {8, 7, 0}},
                             {{12, 9, 0}, {8, 8, 0}},
                         }),
                         [](::testing::TestParamInfo<VersionPair> data) {
                           se::SemanticVersion cuda_version = data.param.first;
                           return absl::StrCat("cuda_", cuda_version.major(),
                                               "_", cuda_version.minor());
                         });

}  // namespace
}  // namespace gpu
}  // namespace xla
