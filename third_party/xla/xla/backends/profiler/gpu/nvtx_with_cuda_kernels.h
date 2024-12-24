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

#ifndef XLA_BACKENDS_PROFILER_GPU_NVTX_WITH_CUDA_KERNELS_H_
#define XLA_BACKENDS_PROFILER_GPU_NVTX_WITH_CUDA_KERNELS_H_

#include <vector>

namespace xla {
namespace profiler {
namespace test {

// If runs correctly, the returned vector will only contain num_elements of 0.
std::vector<int> SimpleAddSubWithNvtxTag(int num_elements);

}  // namespace test
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_NVTX_WITH_CUDA_KERNELS_H_
