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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_LHLO_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_LHLO_TRANSFORMS_PASSES_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class FuncOp;
class FunctionPass;
class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;
namespace lmhlo {
class FusionOp;
}

namespace lmhlo {

// Lowers from LHLO dialect to Affine dialect.
std::unique_ptr<OperationPass<FuncOp>> createLhloLegalizeToAffinePass();

// Lowers from LHLO dialect to GPU dialect.
std::unique_ptr<FunctionPass> createLegalizeToGpuPass();

// Fuses linalg ops obtained after LHLO lowering. To enable fusion,
// operations are first tiled.
//
// When 'use_parallel_loops' is set, the tiling will use scf.parallel
// operations. Otherwise, scf.for operations are used.
//
// 'tile_sizes' provides the tile sizes to use for tiling. If the linalg
// operation has more dimensions than tile sizes provided, 1 is used as
// default.
std::unique_ptr<FunctionPass> createLhloFuseLinalgPass(
    bool use_parallel_loops = false, llvm::ArrayRef<unsigned> tile_sizes = {});

// Lowers from LHLO dialect to parallel loops.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeLhloToParallelLoopsPass();

// Legalizes tensor load ops that are inserted during mhlo to lmhlo conversion.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeToTensorOpPass();

// fuse lmhlo ops to kLoop/kInput fusion patterns
std::unique_ptr<OperationPass<FuncOp>> createLhloFusionPass(
    int max_num_arguments_per_kernel = 64);

// inline lmhlo.Fusion
std::unique_ptr<OperationPass<FuncOp>> createLhloFusionInlinerPass();

// Lowers the roots of lmhlo.fusion to parallel loops
std::unique_ptr<OperationPass<FuncOp>>
createLhloLegalizeRootsToParallelLoopsPass();

// Input inline fusion pass for fusion codegen
std::unique_ptr<FunctionPass> createInputInlineFusionPass();

}  // namespace lmhlo

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_LHLO_TRANSFORMS_PASSES_H_
