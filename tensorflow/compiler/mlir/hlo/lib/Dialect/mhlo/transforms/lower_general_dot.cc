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

// This file implements logic for lowering MHLO general dot to a regular dot.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::DenseIntElementsAttr;
using mlir::ElementsAttr;
using mlir::failure;
using mlir::FunctionPass;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpRewritePattern;
using mlir::OwningRewritePatternList;
using mlir::PassWrapper;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::success;
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
  auto transpose_result = rewriter->create<mlir::mhlo::TransposeOp>(
      loc, transpose_type, arg, transpose_permutation_attr);

  // Return the final result.
  auto reshaped_type =
      RankedTensorType::get({left_size, right_size}, element_type);
  return rewriter->create<mlir::mhlo::ReshapeOp>(loc, reshaped_type,
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

struct GeneralDotConvert : public OpRewritePattern<mlir::mhlo::DotGeneralOp> {
  // Attempts to lower a General Dot operator to a standard Dot operator.
  // General dots include batching dimensions and can have collapsing
  // dimensions along any axis. Inserting correctly arrange transpose and
  // reshape operators organizes the tensors and allows the General Dot to be
  // replaced with the standard Dot operator.
  //
  // Note: This requires an empty list of batch dimensions.

  explicit GeneralDotConvert(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(mlir::mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto dot_element_type = mlir::getElementTypeOrSelf(op);

    auto dot_numbers = op.dot_dimension_numbers();
    if (dot_numbers.lhs_batching_dimensions().getNumElements() != 0 ||
        dot_numbers.rhs_batching_dimensions().getNumElements() != 0) {
      return failure();
    }

    auto lhs = ProcessDotArg(op.lhs(), op.getLoc(),
                             dot_numbers.lhs_contracting_dimensions(),
                             /*outer_dims_first=*/true, &rewriter);

    auto rhs = ProcessDotArg(op.rhs(), op.getLoc(),
                             dot_numbers.rhs_contracting_dimensions(),
                             /*outer_dims_first=*/false, &rewriter);

    // Accept only static shaped types.
    auto lhs_shape_type = lhs.getType().dyn_cast_or_null<mlir::ShapedType>();
    auto rhs_shape_type = rhs.getType().dyn_cast_or_null<mlir::ShapedType>();
    if (!lhs_shape_type || !rhs_shape_type) return failure();
    if (!lhs_shape_type.hasStaticShape() || !rhs_shape_type.hasStaticShape())
      return failure();

    // Dot resulting shape.
    auto lhs_shape = lhs_shape_type.getShape();
    auto rhs_shape = rhs_shape_type.getShape();
    auto new_dot_type =
        RankedTensorType::get({lhs_shape[0], rhs_shape[1]}, dot_element_type);

    mlir::ArrayAttr precision_config;
    if (op.precision_config()) precision_config = *op.precision_config();
    auto new_dot_op = rewriter.create<mlir::mhlo::DotOp>(
        op.getLoc(), new_dot_type, lhs, rhs, precision_config);

    rewriter.replaceOpWithNewOp<mlir::mhlo::ReshapeOp>(op, op.getType(),
                                                       new_dot_op);
    return success();
  }
};

struct LegalizeGeneralDotPass
    : public PassWrapper<LegalizeGeneralDotPass, FunctionPass> {
  /// Lower all general dots that can be represented as a non-batched matmul.
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    mlir::mhlo::PopulateGeneralDotOpLoweringPatterns(&patterns, &getContext());
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

}  // namespace

void mlir::mhlo::PopulateGeneralDotOpLoweringPatterns(
    OwningRewritePatternList *patterns, MLIRContext *ctx) {
  patterns->insert<GeneralDotConvert>(ctx);
}

std::unique_ptr<::mlir::Pass> mlir::mhlo::createLegalizeGeneralDotPass() {
  return std::make_unique<LegalizeGeneralDotPass>();
}
