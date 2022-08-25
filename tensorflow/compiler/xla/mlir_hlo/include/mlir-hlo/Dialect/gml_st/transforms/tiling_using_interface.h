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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_USING_INTERFACE_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_USING_INTERFACE_H

#include <functional>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gml_st {

/// Options to use to control tiling.
struct GmlStTilingOptions {
  using TileSizeComputationFunction =
      std::function<SmallVector<Value>(OpBuilder &, Operation *)>;

  /// Computation function that returns the tile sizes for each operation.
  TileSizeComputationFunction tileSizeComputationFunction = nullptr;
};

struct GmlStTilingResult {
  TilingInterface tiledOp;
  gml_st::ForOp loop;
};

/// Pattern to tile an op that implements the `TilingInterface` using
/// `gml_st.for` for iterating over the tiles.
struct TileToGmlStLoops : public OpInterfaceRewritePattern<TilingInterface> {
  TileToGmlStLoops(MLIRContext *context, StringRef tilingTarget,
                   GmlStTilingOptions options, PatternBenefit benefit = 1);

  FailureOr<GmlStTilingResult> returningMatchAndRewrite(
      TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override;

 private:
  StringRef tilingTarget;
  GmlStTilingOptions options;
};

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TILING_USING_INTERFACE_H
