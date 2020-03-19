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
// This pass identifies patterns for certain Einsum Ops and replaces them
// with other equivalent TF Ops.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_EINSUM_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_EINSUM_H_

#include <cstdint>
#include <initializer_list>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Matchers.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

// TF.Einsum provides fully general tensor contractions. For a few select
// cases, we can convert this op to other TF Ops, which in later passes
// properly convert to TF Lite ops.
struct ConvertTFEinsumOp : public OpRewritePattern<TF::EinsumOp> {
 public:
  explicit ConvertTFEinsumOp(MLIRContext* context)
      : OpRewritePattern<TF::EinsumOp>(context) {}

  LogicalResult matchAndRewrite(TF::EinsumOp op,
                                PatternRewriter& rewriter) const override;
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_EINSUM_H_
