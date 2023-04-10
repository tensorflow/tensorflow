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

#ifndef MLIR_HLO_GML_ST_UTILS_LINALG_UTILS_H
#define MLIR_HLO_GML_ST_UTILS_LINALG_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::gml_st {

// Helper functions to match Linalg ops that implement simple reductions,
// bcasts, and cwise ops.

struct SimpleBcastReduction {
  Operation *bcast;
  Operation *reduction;
  Value operand;
};

bool isSimpleBcastReduction(Operation *op, int64_t *dimension = nullptr,
                            SimpleBcastReduction *chain = nullptr);

// The Conv2D is transformable into a matmul, if it has the following shape
//
// linalg.conv_2d_nhwc_hwcf
//   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
//   ins(%input, %kernel : tensor<1x(N+L-1)xKx1xf32>, tensor<LxKx1xMxf32>)
//   outs(%fill : tensor<1xNx1xM>) -> tensor<1xNx1xMxf32>
//
// in that case we can tile w.r.t. L to bring it to the following form
//
// linalg.conv_2d_nhwc_hwcf
//   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
//   ins(%input, %kernel : tensor<1xNxKx1xf32>, tensor<1xKx1xMxf32>)
//   outs(%fill : tensor<1xNx1xM>) -> tensor<1xNx1xMxf32>
bool isTransformableIntoMatmul(linalg::Conv2DNhwcHwcfOp convOp);

// linalg.conv_2d_nhwc_hwcf
//   {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
//   ins(%input, %kernel : tensor<1xNxKx1xf32>, tensor<1xKx1xMxf32>)
//   outs(%fill : tensor<1xNx1xM>) -> tensor<1xNx1xMxf32>
//
//  into
//
// linalg.matmul
//   ins(%lhs, %rhs : tensor<NxKxf32>, tensor<KxMxf32>)
//   outs(%fill : tensor<NxM>) -> tensor<1xNx1xMxf32>
FailureOr<linalg::MatmulOp> convertConvToMatmul(linalg::Conv2DNhwcHwcfOp convOp,
                                                PatternRewriter &rewriter);

// Converts linalg.batch_matmul into linalg.matmul.
FailureOr<linalg::MatmulOp> convertBatchMatmulToMatmul(
    linalg::BatchMatmulOp batchMatmulOp, PatternRewriter &rewriter);

// Converts linalg.matvec into linalg.dot.
FailureOr<linalg::DotOp> convertMatvecToDotOp(PatternRewriter &rewriter,
                                              linalg::MatvecOp matvecOp);

// Converts linalg.dot into linalg.reduce(linalg.map).
FailureOr<linalg::ReduceOp> convertDotOpToReduce(linalg::DotOp dotOp,
                                                 PatternRewriter &rewriter);

}  // namespace mlir::gml_st

#endif  // MLIR_HLO_GML_ST_UTILS_LINALG_UTILS_H
