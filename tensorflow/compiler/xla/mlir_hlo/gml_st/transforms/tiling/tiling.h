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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_H

#include <functional>
#include <string>

#include "gml_st/interfaces/tiling_interface.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gml_st {

struct TilingResult {
  Operation *tiledOp = nullptr;
  Operation *loop = nullptr;
};

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

  // Distribution label to add to the gml_st.parallel op
  std::string distributionLabel = "";

  /// Convenience function to set the `tileSizeComputationFn` to a
  /// function that computes tile sizes from an input vector parameter.
  void setTileSizeComputationFn(ArrayRef<int64_t> ts);
};

/// Create tiled operation based on the specified tiling options. The result is
/// equivalent to original op.
FailureOr<TilingResult> tile(const TilingOptions &options,
                             PatternRewriter &rewriter, TilingInterface op);

/// Populate tiling patterns.
void populateTilingPatterns(
    MLIRContext *context,
    llvm::function_ref<LogicalResult(TilingInterface)> filterFn,
    const TilingOptions &opts, RewritePatternSet *patterns);

/// Cleans up attributes from applying above tiling patterns.
void removeTilingLabels(Operation *op);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_H
