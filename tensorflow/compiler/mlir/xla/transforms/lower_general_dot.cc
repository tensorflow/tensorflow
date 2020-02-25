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

// This file implements logic for lowering XLA general dot to a regular dot.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

using mlir::DenseIntElementsAttr;
using mlir::ElementsAttr;
using mlir::FunctionPass;
using mlir::MLIRContext;
using mlir::OpRewritePattern;
using mlir::OwningRewritePatternList;
using mlir::PassRegistration;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::Value;

namespace {

Value TransposeReshape(Value arg, mlir::Location loc,
                       llvm::ArrayRef<int64_t> left_dims,
                       llvm::ArrayRef<int64_t> right_dims,
                       llvm::ArrayRef<int64_t> arg_shape,
                       PatternRewriter *rewriter) {
  auto element_type = mlir::getElementTypeOrSelf(arg.getType());

  int64_t left_size = 1;
  for (auto dim : left_dims) {
    left_size *= arg_shape[dim];
  }

  int64_t right_size = 1;
  for (auto dim : right_dims) {
    right_size *= arg_shape[dim];
  }

  // Generate the transpose permutation attribute.
  llvm::SmallVector<int64_t, 5> transpose_permutation(left_dims.begin(),
                                                      left_dims.end());
  transpose_permutation.append(right_dims.begin(), right_dims.end());

  mlir::TensorType transpose_permutation_type = RankedTensorType::get(
      {static_cast<int64_t>(transpose_permutation.size())},
      rewriter->getIntegerType(64));

  auto transpose_permutation_attr =
      DenseIntElementsAttr::get(transpose_permutation_type,
                                llvm::makeArrayRef(transpose_permutation))
          .cast<DenseIntElementsAttr>();

  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 5> transposed_shape;
  for (auto val : transpose_permutation) {
    transposed_shape.push_back(arg_shape[val]);
  }
  auto transpose_type = RankedTensorType::get(transposed_shape, element_type);
  auto transpose_result = rewriter->create<mlir::xla_hlo::TransposeOp>(
      loc, transpose_type, arg, transpose_permutation_attr);

  // Return the final result.
  auto reshaped_type =
      RankedTensorType::get({left_size, right_size}, element_type);
  return rewriter->create<mlir::xla_hlo::ReshapeOp>(loc, reshaped_type,
                                                    transpose_result);
}

Value ProcessDotArg(Value arg, mlir::Location loc,
                    ElementsAttr contract_dims_attr, bool outer_dims_first,
                    PatternRewriter *rewriter) {
  auto shape = arg.getType().cast<mlir::ShapedType>().getShape();

  llvm::SmallVector<bool, 5> is_outer_dim;
  is_outer_dim.resize(shape.size(), true);

  // Compute the contract dimension ordering.
  llvm::SmallVector<int64_t, 5> contract_dims;
  for (auto dim : contract_dims_attr.getValues<int64_t>()) {
    contract_dims.push_back(dim);
    is_outer_dim[dim] = false;
  }

  // Compute the outer dimension orderings.
  llvm::SmallVector<int64_t, 5> outer_dims;
  for (auto it : llvm::enumerate(is_outer_dim)) {
    if (it.value()) {
      outer_dims.push_back(it.index());
    }
  }

  if (outer_dims_first) {
    return TransposeReshape(arg, loc, outer_dims, contract_dims, shape,
                            rewriter);
  }

  return TransposeReshape(arg, loc, contract_dims, outer_dims, shape, rewriter);
}

struct GeneralDotConvert
    : public OpRewritePattern<mlir::xla_hlo::DotGeneralOp> {
  // Attempts to lower a General Dot operator to a standard Dot operator.
  // General dots include batching dimensions and can have collapsing
  // dimensions along any axis. Inserting correctly arrange transpose and
  // reshape operators organizes the tensors and allows the General Dot to be
  // replaced with the standard Dot operator.
  //
  // Note: This requires an empty list of batch dimensions.

  explicit GeneralDotConvert(MLIRContext *context)
      : OpRewritePattern(context) {}

  PatternMatchResult matchAndRewrite(mlir::xla_hlo::DotGeneralOp op,
                                     PatternRewriter &rewriter) const override {
    auto dot_element_type = mlir::getElementTypeOrSelf(op);

    auto dot_numbers = op.dot_dimension_numbers();
    if (dot_numbers.lhs_batching_dimensions().getNumElements() != 0 ||
        dot_numbers.rhs_batching_dimensions().getNumElements() != 0) {
      return matchFailure();
    }

    auto lhs = ProcessDotArg(op.lhs(), op.getLoc(),
                             dot_numbers.lhs_contracting_dimensions(),
                             /*outer_dims_first=*/true, &rewriter);

    auto rhs = ProcessDotArg(op.rhs(), op.getLoc(),
                             dot_numbers.rhs_contracting_dimensions(),
                             /*outer_dims_first=*/false, &rewriter);

    // Dot resulting shape.
    auto lhs_shape = lhs.getType().cast<mlir::ShapedType>().getShape();
    auto rhs_shape = rhs.getType().cast<mlir::ShapedType>().getShape();
    auto new_dot_type =
        RankedTensorType::get({lhs_shape[0], rhs_shape[1]}, dot_element_type);

    auto new_dot_op = rewriter.create<mlir::xla_hlo::DotOp>(
        op.getLoc(), new_dot_type, lhs, rhs, *(op.precision_config()));

    rewriter.replaceOpWithNewOp<mlir::xla_hlo::ReshapeOp>(op, op.getType(),
                                                          new_dot_op);
    return matchSuccess();
  }
};

struct LegalizeGeneralDot : public FunctionPass<LegalizeGeneralDot> {
  /// Lower all general dots that can be represented as a non-batched matmul.
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    mlir::xla_hlo::PopulateGeneralDotOpLoweringPatterns(&patterns,
                                                        &getContext());
    applyPatternsGreedily(getFunction(), patterns);
  }
};

}  // namespace

void mlir::xla_hlo::PopulateGeneralDotOpLoweringPatterns(
    OwningRewritePatternList *patterns, MLIRContext *ctx) {
  patterns->insert<GeneralDotConvert>(ctx);
}

static PassRegistration<LegalizeGeneralDot> legalize_pass(
    "test-xla-lower-general-dot",
    "Tests lowering general dot to a non-batched dot when possible");
