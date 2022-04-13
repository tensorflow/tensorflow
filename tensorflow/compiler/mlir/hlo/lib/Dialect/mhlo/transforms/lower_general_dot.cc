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

#include <sys/types.h>

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

Value TransposeReshape(Value arg, Location loc,
                       llvm::ArrayRef<int64_t> left_dims,
                       llvm::ArrayRef<int64_t> right_dims,
                       llvm::ArrayRef<int64_t> arg_shape,
                       PatternRewriter &rewriter) {
  auto element_type = getElementTypeOrSelf(arg.getType());

  int64_t left_size = 1;
  for (auto dim : left_dims) {
    left_size = (ShapedType::isDynamic(arg_shape[dim]) || left_size < 0)
                    ? ShapedType::kDynamicSize
                    : left_size * arg_shape[dim];
  }

  int64_t right_size = 1;
  for (auto dim : right_dims) {
    right_size = (ShapedType::isDynamic(arg_shape[dim]) || right_size < 0)
                     ? ShapedType::kDynamicSize
                     : right_size * arg_shape[dim];
  }

  // Generate the transpose permutation attribute.
  llvm::SmallVector<int64_t, 5> transpose_permutation(left_dims.begin(),
                                                      left_dims.end());
  transpose_permutation.append(right_dims.begin(), right_dims.end());

  TensorType transpose_permutation_type = RankedTensorType::get(
      {static_cast<int64_t>(transpose_permutation.size())},
      rewriter.getIntegerType(64));

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
  Value transpose_result = rewriter.create<TransposeOp>(
      loc, transpose_type, arg, transpose_permutation_attr);

  // If there are only a single pair of contracting dimensions and the output
  // rank is two we can skip a needless reshape.
  if (transpose_type.getRank() == 2 && left_dims.size() == 1 &&
      right_dims.size() == 1)
    return transpose_result;

  // Return the final result.
  auto reshaped_type =
      RankedTensorType::get({left_size, right_size}, element_type);

  if (reshaped_type.hasStaticShape()) {
    return rewriter.create<mhlo::ReshapeOp>(loc, reshaped_type,
                                            transpose_result);
  }

  SmallVector<Value> reshape_dims;
  auto multiply_dynamic_dims = [&](llvm::ArrayRef<int64_t> dims) -> Value {
    Value dynamic_size = rewriter.create<mhlo::GetDimensionSizeOp>(
        loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
        rewriter.getI64IntegerAttr(dims.front()));

    for (auto idx : dims.drop_front()) {
      Value dim = rewriter.create<mhlo::GetDimensionSizeOp>(
          loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
          rewriter.getI64IntegerAttr(idx));
      dynamic_size = rewriter.create<mhlo::MulOp>(loc, dynamic_size, dim);
    }
    return dynamic_size;
  };

  if (left_size < 0) {
    reshape_dims.push_back(multiply_dynamic_dims(left_dims));
  } else {
    reshape_dims.push_back(
        rewriter.create<ConstOp>(loc, rewriter.getI32TensorAttr(left_size)));
  }

  if (right_size < 0) {
    reshape_dims.push_back(multiply_dynamic_dims(right_dims));
  } else {
    reshape_dims.push_back(
        rewriter.create<ConstOp>(loc, rewriter.getI32TensorAttr(right_size)));
  }

  Value reshape_dims_tensor = rewriter.create<mhlo::ConcatenateOp>(
      loc, RankedTensorType::get({2}, rewriter.getI32Type()), reshape_dims,
      rewriter.getI64IntegerAttr(0));

  return rewriter.create<DynamicReshapeOp>(loc, reshaped_type, transpose_result,
                                           reshape_dims_tensor);
}

Value ProcessDotArg(Value arg, Location loc,
                    ArrayRef<int64_t> contract_dims_attr, bool outer_dims_first,
                    PatternRewriter &rewriter) {
  auto shape = arg.getType().cast<ShapedType>().getShape();

  llvm::SmallVector<bool, 5> is_outer_dim;
  is_outer_dim.resize(shape.size(), true);

  // Compute the contract dimension ordering.
  llvm::SmallVector<int64_t, 5> contract_dims;
  for (auto dim : contract_dims_attr) {
    contract_dims.push_back(dim);
    is_outer_dim[dim] = false;
  }

  // Compute the outer dimension orderings.
  llvm::SmallVector<int64_t, 5> outer_dims;
  for (const auto &it : llvm::enumerate(is_outer_dim)) {
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

struct GeneralDotConvert : public OpRewritePattern<DotGeneralOp> {
  // Attempts to lower a General Dot operator to a standard Dot operator.
  // General dots include batching dimensions and can have collapsing
  // dimensions along any axis. Inserting correctly arrange transpose and
  // reshape operators organizes the tensors and allows the General Dot to be
  // replaced with the standard Dot operator.
  //
  // Note: This requires an empty list of batch dimensions.

  explicit GeneralDotConvert(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto dot_numbers = op.dot_dimension_numbers();
    if (!dot_numbers.getLhsBatchingDimensions().empty() ||
        !dot_numbers.getRhsBatchingDimensions().empty()) {
      return failure();
    }

    auto lhs_contracting_dims = dot_numbers.getLhsContractingDimensions();
    auto rhs_contracting_dims = dot_numbers.getRhsContractingDimensions();

    auto lhs = op.lhs();
    auto rhs = op.rhs();

    RankedTensorType lhs_ty = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhs_ty = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhs_ty || !rhs_ty) return failure();

    lhs = ProcessDotArg(op.lhs(), op.getLoc(),
                        dot_numbers.getLhsContractingDimensions(),
                        /*outer_dims_first=*/true, rewriter);

    rhs = ProcessDotArg(op.rhs(), op.getLoc(),
                        dot_numbers.getRhsContractingDimensions(),
                        /*outer_dims_first=*/false, rewriter);

    // Accept only static shaped types.
    auto lhs_shape_type = lhs.getType().dyn_cast_or_null<ShapedType>();
    auto rhs_shape_type = rhs.getType().dyn_cast_or_null<ShapedType>();
    if (!lhs_shape_type || !rhs_shape_type) return failure();

    ArrayAttr precision_config;
    if (op.precision_config()) precision_config = *op.precision_config();
    Value new_dot_op =
        rewriter.create<DotOp>(op.getLoc(), lhs, rhs, precision_config);
    if (lhs_contracting_dims.size() == (lhs_ty.getRank() - 1) &&
        rhs_contracting_dims.size() == (rhs_ty.getRank() - 1)) {
      rewriter.replaceOp(op, new_dot_op);
      return success();
    }

    ShapedType result_ty = op.getType().cast<ShapedType>();

    // We can avoid all the computation below if we know the static shape.
    if (result_ty.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, result_ty, new_dot_op);
      return success();
    }

    llvm::SmallVector<int64_t> static_dims;
    llvm::SmallVector<Value> dyn_dims;

    auto getDynamicDims = [&](Value arg,
                              llvm::ArrayRef<int64_t> contracting_dims) {
      RankedTensorType ty = arg.getType().cast<RankedTensorType>();
      int index = 0;
      for (auto contracting_dim : contracting_dims) {
        for (; index < contracting_dim; index++) {
          static_dims.push_back(ty.getDimSize(index));
          dyn_dims.push_back(rewriter.create<mhlo::GetDimensionSizeOp>(
              loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
              rewriter.getI64IntegerAttr(index)));
        }
        index++;
      }

      for (; index < ty.getRank(); index++) {
        static_dims.push_back(ty.getDimSize(index));
        dyn_dims.push_back(rewriter.create<mhlo::GetDimensionSizeOp>(
            loc, RankedTensorType::get({1}, rewriter.getI32Type()), arg,
            rewriter.getI64IntegerAttr(index)));
      }
    };

    getDynamicDims(op.lhs(), lhs_contracting_dims);
    getDynamicDims(op.rhs(), rhs_contracting_dims);

    Value reshape_dims_tensor = rewriter.create<mhlo::ConcatenateOp>(
        loc,
        RankedTensorType::get({static_cast<int64_t>(dyn_dims.size())},
                              rewriter.getI32Type()),
        dyn_dims, rewriter.getI64IntegerAttr(0));

    Value result = rewriter.create<DynamicReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(static_dims, result_ty.getElementType()),
        new_dot_op, reshape_dims_tensor);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LegalizeGeneralDotPass
    : public LegalizeGeneralDotPassBase<LegalizeGeneralDotPass> {
  /// Lower all general dots that can be represented as a non-batched matmul.
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    PopulateGeneralDotOpLoweringPatterns(&patterns, &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

void mlir::mhlo::PopulateGeneralDotOpLoweringPatterns(
    RewritePatternSet *patterns, MLIRContext *ctx) {
  patterns->add<GeneralDotConvert>(ctx);
}

std::unique_ptr<::mlir::Pass> mlir::mhlo::createLegalizeGeneralDotPass() {
  return std::make_unique<LegalizeGeneralDotPass>();
}
