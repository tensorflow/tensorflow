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

#include "xla/backends/profiler/gpu/nvtx_with_cuda_kernels.h"

#include <vector>

#include <gtest/gtest.h>

namespace xla {
namespace profiler {
namespace test {

namespace {

// This test just verify the cuda kernels ares running well and generate correct
// output.
TEST(NvtxCudaKernelSanityTest, SimpleAddSub) {
  constexpr int kNumElements = 2048;
  std::vector<int> vec = SimpleAddSubWithNvtxTag(kNumElements);

  EXPECT_EQ(vec.size(), kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(vec[i], 0) << "index: " << i;
  }
}

}  // namespace

}  // namespace test
}  // namespace profiler
}  // namespace xla
