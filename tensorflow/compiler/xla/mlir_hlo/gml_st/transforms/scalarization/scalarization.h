/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_GML_ST_TRANSFORMS_SCALARIZATION_SCALARIZATION_H
#define MLIR_HLO_GML_ST_TRANSFORMS_SCALARIZATION_SCALARIZATION_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir {
namespace gml_st {

/// Rewrites thlo.concatenate, returns `failure` if IR was not changed.
LogicalResult scalarizeConcatenateOp(thlo::ConcatenateOp concatenateOp,
                                     PatternRewriter &rewriter);

/// Rewrites thlo.dynamic_broadcast_in_dim, returns `failure` if IR was not
/// changed.
LogicalResult scalarizeDynamicBroadcastInDimOp(
    thlo::DynamicBroadcastInDimOp broadcastOp, PatternRewriter &rewriter);

/// Rewrites thlo.gather, returns `failure` if IR was not changed.
LogicalResult scalarizeGatherOp(thlo::GatherOp gatherOp,
                                PatternRewriter &rewriter);

/// Rewrites LinalgOp interface ops, returns `failure` if IR was not changed.
LogicalResult scalarizeLinalgOp(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter);

/// Rewrites thlo.reverse, returns `failure` if IR was not changed.
LogicalResult scalarizeReverseOp(thlo::ReverseOp reverseOp,
                                 PatternRewriter &rewriter);

/// Rewrites thlo.scatter, returns `failure` if IR was not changed.
LogicalResult scalarizeScatterOp(thlo::ScatterOp scatterOp,
                                 PatternRewriter &rewriter);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_SCALARIZATION_SCALARIZATION_H
