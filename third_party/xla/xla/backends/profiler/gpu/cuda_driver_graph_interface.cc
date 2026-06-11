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

#include "xla/backends/profiler/gpu/cuda_driver_graph_interface.h"

#include <cstdint>

#include "absl/base/no_destructor.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace xla {
namespace profiler {
namespace {

// Default production implementation that delegates directly to native CUDA
// APIs.
class CudaDriverGraphImpl : public CudaDriverGraphInterface {
 public:
  CudaDriverGraphImpl() = default;
  ~CudaDriverGraphImpl() override = default;

  CUresult GetGraphId(CUgraph graph, unsigned int* id) const override {
#if CUDA_VERSION >= 13010
    return cuGraphGetId(graph, id);
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
  }

  CUresult GetNodeToolsId(CUgraphNode node, uint64_t* id) const override {
#if CUDA_VERSION >= 13010
    unsigned long long tools_id;  // NOLINT(runtime/int)
    CUresult status = cuGraphNodeGetToolsId(node, &tools_id);
    if (status == CUDA_SUCCESS) {
      *id = tools_id;
    }
    return status;
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
  }

  CUresult GetExecId(CUgraphExec exec, unsigned int* id) const override {
#if CUDA_VERSION >= 13010
    return cuGraphExecGetId(exec, id);
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
  }
};

}  // namespace

const CudaDriverGraphInterface* CudaDriverGraphInterface::GetDefault() {
  static const absl::NoDestructor<CudaDriverGraphImpl> default_impl;
  return default_impl.get();
}

}  // namespace profiler
}  // namespace xla
