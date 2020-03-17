/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_BATCHMATMUL_TO_EINSUM_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_BATCHMATMUL_TO_EINSUM_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

// Replace TF BatchMatMul by TF Einsum op
template <typename BatchMatMulOpType>
class ConvertTFBatchMatMulToEinsumOp
    : public OpRewritePattern<BatchMatMulOpType> {
  using OpRewritePattern<BatchMatMulOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      BatchMatMulOpType op,
      PatternRewriter& rewriter) const override;  // NOLINT
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_BATCHMATMUL_TO_EINSUM_H_
