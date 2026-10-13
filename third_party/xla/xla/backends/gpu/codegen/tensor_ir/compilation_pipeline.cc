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

#include "xla/backends/gpu/codegen/tensor_ir/compilation_pipeline.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "tensor_ir/Compiler/CudaTile/Pipelines.h"
#include "tensor_ir/Compiler/CudaTile/TileIRAssembly.h"
#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/PassManager.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/path.h"

namespace xla::gpu::tensor_ir {

mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions GetPipelineOptions(
    const stream_executor::GpuComputeCapability& gpu_cc,
    llvm::ArrayRef<int32_t> tile_size, int64_t reduction_tile_size) {
  mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions options;
  options.tileSize.assign(tile_size.begin(), tile_size.end());
  options.reductionTileSize = reduction_tile_size;
  // Skip the TileAnalyzer candidate search when an explicit tile shape is
  // already provided (e.g. by the autotuner).
  options.maxCandidates = tile_size.empty() ? 1 : 0;

  if (const auto* cuda_cc = gpu_cc.cuda_compute_capability()) {
    options.computeCapability = cuda_cc->major * 10 + cuda_cc->minor;
  }
  return options;
}

void CreateTensorIrPipeline(
    mlir::OpPassManager* pm,
    const mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions& options) {
  mlir::nv_tensor_ir::buildTensorToCudaTileConversionPipeline(*pm, options);
}

void SetUpTileIrAssembler() {
  static absl::once_flag once;
  absl::call_once(once, [] {
    // The same search that finds `ptxas`: the CUDA toolkit roots first, `$PATH`
    // last. `tileiras` ships next to `ptxas` from CUDA 13.1 on; TensorIR itself
    // needs the 13.3 release or newer.
    absl::StatusOr<std::string> path = stream_executor::FindCudaExecutable(
        "tileiras", /*preferred_cuda_dir=*/"");
    if (!path.ok()) {
      LOG(WARNING) << "TensorIR: no Tile IR assembler (CUDA 13.3 or newer "
                      "ships one), falling back to Tile IR bytecode for the "
                      "driver to JIT: "
                   << path.status();
      return;
    }
    // `tileiras` finds `libnvvm` and `ptxas` relative to the resolved location
    // of its own executable (`/proc/self/exe`). When the toolkit is staged
    // through symlinks (bazel runfiles, hermetic CUDA) that resolves to a
    // location with no neighbours. `CUDA_HOME` anchors both lookups on the
    // staged layout instead: `<root>/bin/tileiras`,
    // `<root>/nvvm/lib64/libnvvm.so` and `<root>/bin/ptxas`. The rest of the
    // environment is a hermetic minimum.
    std::string root(tsl::io::Dirname(tsl::io::Dirname(*path)));
    std::vector<std::string> environment = {absl::StrCat("CUDA_HOME=", root)};
    for (const char* name : {"PATH", "TMPDIR"}) {
      if (const char* value = std::getenv(name); value != nullptr) {
        environment.push_back(absl::StrCat(name, "=", value));
      }
    }
    mlir::nv_tensor_ir::backend::cuda_tile::setTileIRAssembler(*path,
                                                               environment);
  });
}

}  // namespace xla::gpu::tensor_ir
