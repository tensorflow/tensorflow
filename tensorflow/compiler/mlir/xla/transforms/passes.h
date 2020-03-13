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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project

namespace mlir {

class FuncOp;
class ModuleOp;
class Operation;
template <typename T>
class OpPassBase;
class Pass;

namespace xla_hlo {

/// Lowers from TF dialect to HLO dialect. When allow_partial_conversion is
/// false, emits an error if there is any operation that can't be legalized.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeTFPass(
    bool allow_partial_conversion = false);

/// Lowers from TF dialect's control flow to HLO dialect's control flow.
std::unique_ptr<OpPassBase<ModuleOp>> createLegalizeTFControlFlowPass();

/// Converts the provided Operation as well as all nested operations into HLO
/// dialect using the conversion patterns registered by the HLO dialect. When
/// allow_partial_conversion is false, emits an error if there is any operation
/// that can't be legalized.
LogicalResult legalizeTF(Operation* op, bool allow_partial_conversion = false);

/// Lowers HLO control flow ops to the Standard dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeControlFlowPass();

/// Lowers from HLO dialect to Standard dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToStdPass();

// Lowers from HLO dialect to LHLO dialect allocating/deallocating temporary
// buffers if necessary.
std::unique_ptr<OpPassBase<ModuleOp>> createLegalizeToLhloPass();

// Lowers from HLO dialect to Linalg dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeHloToLinalgPass();

}  // namespace xla_hlo

namespace xla_lhlo {

// Lowers from LHLO dialect to Affine dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToAffinePass();

// Lowers from LHLO dialect to Linalg dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeLhloToLinalgPass();

// Lowers from LHLO dialect to GPU dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToGpuPass();

// Fuses linalg ops obtained after LHLO lowering. To enable fusion,
// operations are first tiled.
//
// When 'use_parallel_loops' is set, the tiling will use loop.parallel
// operations. Otherwise, loop.for operations are used.
//
// 'tile_sizes' provides the tile sizes to use for tiling. If the linalg
// operation has more dimensions than tile sizes provided, 1 is used as
// default.
std::unique_ptr<OpPassBase<FuncOp>> createLhloFuseLinalg(
    bool use_parallel_loops = false, ArrayRef<unsigned> tile_sizes = {});

// Removes unnecessary LHLO copies which copy from the allocated buffers to the
// block arguments. The block arguments are used instead of all uses of these
// buffers. The buffers are freed. This pass only works in regions that contain
// a single block.
std::unique_ptr<Pass> createLhloCopyRemovalPass();

// Lowers from LHLO dialect to parallel loops.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeLhloToParallelLoopsPass();

}  // namespace xla_lhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
