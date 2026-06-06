/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUDA_DRIVER_GRAPH_INTERFACE_H_
#define XLA_BACKENDS_PROFILER_GPU_CUDA_DRIVER_GRAPH_INTERFACE_H_

#include <cstdint>

#include "third_party/gpus/cuda/include/cuda.h"

namespace xla {
namespace profiler {

// A narrow-scoped interface wrapping raw CUDA Driver Graph ID query APIs.
// This interface decouples host-side metadata registries from physical GPU
// hardware and link-time dependencies on the CUDA driver library, allowing CPU
// unit testing.
class CudaDriverGraphInterface {
 public:
  virtual ~CudaDriverGraphInterface() = default;

  // Wraps cuGraphGetId.
  virtual CUresult GetGraphId(CUgraph graph, unsigned int* id) const = 0;

  // Wraps cuGraphNodeGetToolsId.
  virtual CUresult GetNodeToolsId(CUgraphNode node, uint64_t* id) const = 0;

  // Wraps cuGraphExecGetId.
  virtual CUresult GetExecId(CUgraphExec exec, unsigned int* id) const = 0;

  // Returns the default, production-configured singleton instance that invokes
  // the actual, native CUDA Driver APIs.
  static const CudaDriverGraphInterface* GetDefault();
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUDA_DRIVER_GRAPH_INTERFACE_H_
