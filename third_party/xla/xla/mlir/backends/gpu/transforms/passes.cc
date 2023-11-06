/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/mlir/backends/gpu/transforms/passes.h"

#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/rt_ops.h"

namespace xla {
namespace gpu {

using namespace mlir;  // NOLINT

std::vector<std::vector<int64_t>> GetAllocationIndices(mlir::ModuleOp module) {
  std::vector<std::vector<int64_t>> res;

  SymbolTable sym_table(module);
  for (auto op : module.getOps<runtime::ExportOp>()) {
    unsigned ordinal = *op.ordinal();
    if (ordinal >= res.size()) res.resize(ordinal + 1);

    auto func = sym_table.lookup<func::FuncOp>(op.getFunctionRef());
    res[ordinal].resize(func.getNumArguments(), -1);

    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      auto idx = func.getArgAttrOfType<IntegerAttr>(i, "rt.allocation_index");
      if (idx) res[ordinal][i] = idx.getInt();
    }
  }

  return res;
}

void populateXlaGpuRuntimePasses(mlir::OpPassManager& pm,
                                 ThunkSequence* thunk_sequence,
                                 const GpuPipelineOpts& opts) {
  // Lower operations with registered IR emitters to Gpu launches.
  pm.addPass(createConvertLmhloToGpuLaunchPass(thunk_sequence));

  // Clean up IR before converting it to the runtime operations.
  pm.addPass(createCSEPass());

  // Convert global memrefs corresponding to constant arguments.
  pm.addPass(createConvertMemrefGetGlobalToArgPass());
  pm.addPass(createSymbolDCEPass());  // Clean up unused global constants.

  // Outline CUDA-Graph-compatible operations into graph capture functions.
  pm.addPass(
      createOutlineGpuGraphsPass(opts.command_types, opts.min_graph_size));
  if (opts.enable_concurrent_region) {
    // Concurrent regions create repeated-fork-join topology inside CUDA graphs,
    // which is not optimized by architectures prior to Ampere and may cause
    // regression. So we enable concurrent regions only on Ampere GPUs.
    if (auto cc = std::get_if<stream_executor::CudaComputeCapability>(
            &opts.compute_capability);
        !cc || cc->IsAtLeast(8, 0)) {
      pm.addPass(createAddConcurrentRegionsPass());
    } else {
      LOG(WARNING)
          << "Multi-stream execution disabled on non-ampere architectures";
    }
  }

  // Lower all Gpu operations to the XLA Gpu runtime custom calls.
  pm.addPass(createConvertLmhloGpuToGpuRuntimePass());
  pm.addPass(createConvertLmhloToGpuRuntimePass());
  pm.addPass(createConvertGpuToGpuRuntimePass());

  // Add performance tracing annotations.
  pm.addPass(createAddHloTraceAnnotationsPass());
}

}  // namespace gpu
}  // namespace xla
