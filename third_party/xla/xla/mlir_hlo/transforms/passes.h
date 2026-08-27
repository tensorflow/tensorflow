/* Copyright 2021 The OpenXLA Authors.

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

#ifndef MLIR_HLO_TRANSFORMS_PASSES_H
#define MLIR_HLO_TRANSFORMS_PASSES_H

#include <cstdint>
#include <functional>
#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class ModuleOp;
class MLIRContext;
class ConversionTarget;
class DialectRegistry;
class PassManager;
class TypeConverter;
class RewritePatternSet;

namespace func {
class FuncOp;
}  // namespace func

using BufferizeDialectsCallback = std::function<void(DialectRegistry&)>;
using BufferizePatternsCallback = std::function<void(
    ConversionTarget&, MLIRContext*, TypeConverter*, RewritePatternSet*)>;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "transforms/passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createFinalBufferizePass(
    uint64_t alignment, BufferizeDialectsCallback dc = {},
    BufferizePatternsCallback pc = {});

// Creates a TileLoopsPass with tiles sizes provided through `tile_sizes`
// and unroll factors provided through `unroll_factors`.
inline std::unique_ptr<Pass> createTileLoopsPass(
    ArrayRef<int64_t> tileSizes, ArrayRef<int64_t> unrollFactors) {
  TileLoopsPassOptions options;
  options.tile_sizes_ =
      SmallVector<int64_t>(tileSizes.begin(), tileSizes.end());
  options.unroll_factors_ =
      SmallVector<int64_t>(unrollFactors.begin(), unrollFactors.end());
  return createTileLoopsPass(options);
}

namespace hlo {
using mlir::createAllocToArgPass;
using mlir::createGenericHostToLLVMPass;
using mlir::createUnbufferizePass;

inline std::unique_ptr<Pass> createOneShotBufferizePass() {
  return mlir::createOneShotBufferize();
}

#define GEN_PASS_REGISTRATION
#include "transforms/passes.h.inc"

}  // namespace hlo
}  // namespace mlir

#endif  // MLIR_HLO_TRANSFORMS_PASSES_H
