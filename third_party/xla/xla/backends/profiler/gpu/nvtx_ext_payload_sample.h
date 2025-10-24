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

#ifndef XLA_BACKENDS_PROFILER_GPU_NVTX_EXT_PAYLOAD_SAMPLE_H_
#define XLA_BACKENDS_PROFILER_GPU_NVTX_EXT_PAYLOAD_SAMPLE_H_

#include <cstddef>
#include <cstdint>

namespace xla {
namespace profiler {
namespace test {

enum ReduceOpType {
  Sum = 0,
  Avg = 1,
  NumOps = 2,
};

struct ReduceParams {
  uint64_t comm;
  size_t bytes;
  int root;
  ReduceOpType op;
};

void RegisteredSchemas();

void ReduceWithManualOffset(ReduceParams params);

void ReduceWithAutoOffset(ReduceParams params);

}  // namespace test
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_NVTX_EXT_PAYLOAD_SAMPLE_H_
