/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_REWRITERS_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_REWRITERS_H

#include <functional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace bufferization {
class BufferizeTypeConverter;
}  // namespace bufferization
class MLIRContext;
class RewritePatternSet;

namespace gml_st {

using OpFilterFn = llvm::function_ref<LogicalResult(Operation *)>;

/// Options to use to control tiling.
struct TilingOptions {
  using TileSizeComputationFn =
      std::function<SmallVector<Value>(OpBuilder &, Operation *)>;

  /// Function to materialize the tile sizes for a given operation. This allows
  /// to infer tile sizes statically, e.g. based on an operation's rank, and
  /// also dynamically based, e.g. based on a tensor's shape at runtime.
  TileSizeComputationFn tileSizeComputationFn = nullptr;

  /// If `true`, generate a `gml_st.parallel` loop nest.
  bool distribute = true;
};

/// Populate pattern to bufferize `linalg.tiled_loop`.
void populateTiledLoopBufferizePattern(
    MLIRContext *context,
    mlir::bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns);

/// Populate tiling patterns.
void populateTilingPatterns(MLIRContext *context, OpFilterFn filterFn,
                            const TilingOptions &opts,
                            RewritePatternSet *patterns);

/// Populate fusion patterns.
void populateFusionPatterns(MLIRContext *context, OpFilterFn filterFn,
                            RewritePatternSet *patterns);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_REWRITERS_H
