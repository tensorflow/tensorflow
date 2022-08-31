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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASSES_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASSES_H

#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gml_st {

/// Pass to fuse producers into `gml_st.materialize` ops.
std::unique_ptr<OperationPass<func::FuncOp>> createDeprecatedFusionPass();

/// Pass to tile operations.
std::unique_ptr<OperationPass<func::FuncOp>> createDeprecatedTilingPass();
std::unique_ptr<OperationPass<func::FuncOp>> createDeprecatedTilingPass(
    const SmallVector<SmallVector<int64_t>>& tileSizes);
std::unique_ptr<OperationPass<func::FuncOp>> createDeprecatedTilingPass(
    const std::string& tileSizes);

/// Pass to tile ops using TilingInterface.
std::unique_ptr<OperationPass<func::FuncOp>> createTilingPass(
    StringRef tilingTarget = "", bool distribute = true,
    ArrayRef<int64_t> tileSizes = {});

/// Pass to fuse producers into a tiled consumer.
std::unique_ptr<OperationPass<func::FuncOp>> createFusionPass(
    StringRef producer = "", StringRef consumer = "");

/// Pass to compose set operations.
std::unique_ptr<OperationPass<func::FuncOp>> createComposeSetOpsPass();

/// Pass to collapse (or uncollapse) materialize operations.
std::unique_ptr<OperationPass<func::FuncOp>> createCollapseMaterializeOpsPass(
    bool reverse = false);

/// Create a pass to convert `gml_st.loop` to `scf.for` and `scf.parallel`
/// loops and memref.load/memref.store accesses.
std::unique_ptr<OperationPass<func::FuncOp>> createGmlStToScfPass();

// Pass to bufferize `linalg.tiled_loop` including the operations contained in
// its body.
std::unique_ptr<OperationPass<func::FuncOp>> CreateTiledLoopBufferizePass();

/// Pass to vectorize linalg.generic ops tiled to gml_st.parallel and gml_st.for
/// loops.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeGmlStLoopsPass();

#define GEN_PASS_REGISTRATION
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASSES_H
