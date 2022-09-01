/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_DIALECT_LHLO_TRANSFORMS_PASSES_H
#define MLIR_HLO_DIALECT_LHLO_TRANSFORMS_PASSES_H

#include <memory>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;
namespace func {
class FuncOp;
}  // namespace func
namespace lmhlo {
class FusionOp;
}  // namespace lmhlo

namespace lmhlo {

#define GEN_PASS_DECL_LHLOFUSELINALGPASS
#include "mlir-hlo/Dialect/lhlo/transforms/lmhlo_passes.h.inc"

// Lowers from LHLO dialect to Affine dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLhloLegalizeToAffinePass();

// Lowers from LHLO dialect to GPU dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToGpuPass();

// Fuses linalg ops obtained after LHLO lowering. To enable fusion,
// operations are first tiled.
//
// When 'use_parallel_loops' is set, the tiling will use scf.parallel
// operations. Otherwise, scf.for operations are used.
//
// 'tile_sizes' provides the tile sizes to use for tiling. If the linalg
// operation has more dimensions than tile sizes provided, 1 is used as
// default.
std::unique_ptr<OperationPass<func::FuncOp>> createLhloFuseLinalgPass(
    bool useParallelLoops = false, llvm::ArrayRef<unsigned> tileSizes = {});

// Lowers from LHLO dialect to parallel loops.
std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeLhloToParallelLoopsPass();

// Legalizes tensor load ops that are inserted during mhlo to lmhlo conversion.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeToTensorOpPass();

// Input inline fusion pass for fusion codegen
std::unique_ptr<OperationPass<func::FuncOp>> createInputInlineFusionPass();

}  // namespace lmhlo

}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_LHLO_TRANSFORMS_PASSES_H
