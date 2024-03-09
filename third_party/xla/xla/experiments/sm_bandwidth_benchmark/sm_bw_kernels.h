/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_EXPERIMENTS_SM_BANDWIDTH_BENCHMARK_SM_BW_KERNELS_H_
#define XLA_EXPERIMENTS_SM_BANDWIDTH_BENCHMARK_SM_BW_KERNELS_H_

namespace experiments {
namespace benchmark {

template <int chunks>
void BenchmarkDeviceCopy(float* in, float* out, int64_t size, int num_blocks,
                         int num_threads);

}  // namespace benchmark
}  // namespace experiments

#endif  // XLA_EXPERIMENTS_SM_BANDWIDTH_BENCHMARK_SM_BW_KERNELS_H_
