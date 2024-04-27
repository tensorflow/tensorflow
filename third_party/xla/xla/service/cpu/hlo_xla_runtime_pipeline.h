/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_HLO_XLA_RUNTIME_PIPELINE_H_
#define XLA_SERVICE_CPU_HLO_XLA_RUNTIME_PIPELINE_H_

#include <string>
#include <vector>

#include "xla/runtime/compiler.h"
#include "xla/status.h"

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace xla {
namespace cpu {

struct HloXlaRuntimePipelineOptions {
  bool enable_tiling_and_fusion = false;
  bool enable_fusion_outlining = true;
  bool remove_copies_to_outparams = true;
  bool sparse_bufferization = true;
  bool enable_avx2 = true;
  // Accelerate sparse computations with CUDA threading.
  // This is an experimental feature, so off by default.
  int32_t xla_cpu_sparse_cuda_threads = 0;
  // Optional CPU name, similar to llc's -mcpu flag.
  std::string cpu_name = "";
  std::vector<int64_t> matmul_tile_sizes = {};
};

// Creates a pipeline that lowers modules from HLO to Linalg on buffers.
Status CreateHloXlaRuntimePipeline(xla::runtime::PassManager& passes,
                                   const HloXlaRuntimePipelineOptions& options);
Status CreateDefaultHloXlaRuntimePipeline(xla::runtime::PassManager& passes);

void RegisterHloXlaRuntimePipelineDialects(mlir::DialectRegistry& dialects);
}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_HLO_XLA_RUNTIME_PIPELINE_H_
