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
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {

class FuncOp;
class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;

namespace xla_hlo {

/// Lowers from TF dialect to HLO dialect. When allow_partial_conversion is
/// false, emits an error if there is any operation that can't be legalized.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeTFPass(
    bool allow_partial_conversion = false, bool legalize_chlo = true);

/// Lowers from TF dialect to HLO dialect using tf2xla op kernels for the
/// specified device type.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeTfWithTf2XlaPass(
    llvm::StringRef device_type);

/// Lowers from TF dialect's control flow to HLO dialect's control flow.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeTFControlFlowPass();

/// Converts the provided Operation as well as all nested operations into HLO
/// dialect using the conversion patterns registered by the HLO dialect. When
/// allow_partial_conversion is false, emits an error if there is any operation
/// that can't be legalized.
LogicalResult legalizeTF(Operation* op, bool allow_partial_conversion = false,
                         bool legalize_chlo = true);

/// Lowers HLO control flow ops to the Standard dialect.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeControlFlowPass();

/// Lowers from HLO dialect to Standard dialect.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeToStdPass();

// Lowers from HLO dialect to LHLO dialect allocating/deallocating temporary
// buffers if necessary.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToLhloPass();

// Lowers from HLO dialect to Linalg dialect.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeHloToLinalgPass();

// Transforms unranked HLO operations to ranked ones where possible.
std::unique_ptr<OperationPass<FuncOp>> createTransformUnrankedHloPass();

// Sinks constants implicitly captured in control flow regions. This is
// necessary to export to XLA.
std::unique_ptr<OperationPass<FuncOp>> createSinkConstantsToControlFlowPass();

// fuse xla_hlo ops to kLoop/kInput fusion patterns
std::unique_ptr<OperationPass<FuncOp>> createXlaHloFusionPass();

}  // namespace xla_hlo

namespace xla_lhlo {

// Lowers from LHLO dialect to Affine dialect.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeToAffinePass();

// Lowers from LHLO dialect to Linalg dialect.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeLhloToLinalgPass();

// Lowers from LHLO dialect to GPU dialect.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeToGpuPass();

// Fuses linalg ops obtained after LHLO lowering. To enable fusion,
// operations are first tiled.
//
// When 'use_parallel_loops' is set, the tiling will use scf.parallel
// operations. Otherwise, scf.for operations are used.
//
// 'tile_sizes' provides the tile sizes to use for tiling. If the linalg
// operation has more dimensions than tile sizes provided, 1 is used as
// default.
std::unique_ptr<OperationPass<FuncOp>> createLhloFuseLinalg(
    bool use_parallel_loops = false, ArrayRef<unsigned> tile_sizes = {});

// Removes unnecessary LHLO copies which copy from the allocated buffers to the
// block arguments. The block arguments are used instead of all uses of these
// buffers. The buffers are freed. This pass only works in regions that contain
// a single block.
std::unique_ptr<Pass> createLhloCopyRemovalPass();

// Lowers from LHLO dialect to parallel loops.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeLhloToParallelLoopsPass();

}  // namespace xla_lhlo

namespace xla {

/// Moves alloc nodes (and their associated dealloc nodes - if any) into the
/// right positions. If there is no associated dealloc node for a given alloc
/// node, this pass will automatically insert a proper dealloc node in the right
/// place. The intended use case of this pass is to store SSA values into
/// buffers using load/store operations. For this purpose, you need to know
/// proper positions to place the required allocs and deallocs.
/// 1) Note that the function signatures and all types for which buffers should
/// be allocated need to be converted in advance.
/// 2) All required alloc nodes have the be inserted in advance.
/// 3) Note that the current implementation does not support loops.
/// Refer to the class mlir::xla::BufferAssignmentLegalizer for more
/// information.
std::unique_ptr<OperationPass<FuncOp>> createBufferAssignmentPass();

}  // namespace xla
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
