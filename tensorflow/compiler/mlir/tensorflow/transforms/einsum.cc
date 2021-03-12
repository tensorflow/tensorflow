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

#include "tensorflow/compiler/mlir/tensorflow/transforms/einsum.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"
#include "mlir/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verification_utils.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

namespace {

TF::TransposeOp createTransposeOp(Value value, Location loc,
                                  llvm::ArrayRef<int32_t> permutation,
                                  PatternRewriter* rewriter) {
  auto value_type = value.getType().cast<RankedTensorType>();
  auto shape = value_type.getShape();
  auto perm_type = RankedTensorType::get(
      {static_cast<int32_t>(permutation.size())}, rewriter->getIntegerType(32));
  auto perm_attr = DenseElementsAttr::get(perm_type, permutation);
  auto perm_op = rewriter->create<ConstantOp>(loc, perm_type, perm_attr);
  SmallVector<int64_t, 4> transposed_shape(shape.begin(), shape.end());
  for (int i = 0, end = shape.size(); i < end; ++i) {
    transposed_shape[i] = shape[permutation[i]];
  }
  auto transposed_type =
      RankedTensorType::get(transposed_shape, value_type.getElementType());
  return rewriter->create<TF::TransposeOp>(loc, transposed_type, value,
                                           perm_op);
}

TF::ReshapeOp createReshapeOp(Value value, ArrayRef<int64_t> shape,
                              Type element_type, Location loc,
                              PatternRewriter* rewriter) {
  int64_t shape_rank = shape.size();
  auto shape_spec_type =
      RankedTensorType::get({shape_rank}, rewriter->getIntegerType(64));
  Type resultType = RankedTensorType::get(shape, element_type);
  auto constant_attr = DenseElementsAttr::get(shape_spec_type, shape);
  auto shape_tensor =
      rewriter->create<ConstantOp>(loc, shape_spec_type, constant_attr);
  return rewriter->create<TF::ReshapeOp>(loc, resultType, /*tensor=*/value,
                                         /*shape=*/shape_tensor);
}

struct EinsumDimensionNumbers {
  // Each field contains the list of dimensions appearing only in the specifed
  // arguments of the einsum op with natural ordering. For example `rhs_out`
  // contains the dimensions appearing in the RHS and the OUTPUT of the einsum
  // but not in the LHS.
  std::vector<int64_t> lhs;
  std::vector<int64_t> rhs;
  std::vector<std::tuple<int64_t, int64_t>> lhs_rhs;
  std::vector<std::tuple<int64_t, int64_t>> lhs_out;
  std::vector<std::tuple<int64_t, int64_t>> rhs_out;
  std::vector<std::tuple<int64_t, int64_t, int64_t>> lhs_rhs_out;
};

llvm::Optional<llvm::SmallDenseMap<char, int64_t>> EquationToMap(
    llvm::StringRef equation) {
  llvm::SmallDenseMap<char, int64_t> map;
  for (int64_t i = 0; i < equation.size(); ++i) {
    if (!std::isalpha(equation[i])) {
      // Unsupported character in the equation.
      return llvm::None;
    }
    if (map.count(equation[i])) {
      // Duplicate character in the equation.
      return llvm::None;
    }
    map.try_emplace(equation[i], i);
  }
  return map;
}

llvm::Optional<EinsumDimensionNumbers> GetEinsumDimensionNumbers(
    llvm::StringRef equation) {
  llvm::StringRef lhs_rhs;
  llvm::StringRef out;
  std::tie(lhs_rhs, out) = equation.split("->");
  if (lhs_rhs.empty() || out.empty()) return llvm::None;

  llvm::StringRef lhs;
  llvm::StringRef rhs;
  std::tie(lhs, rhs) = lhs_rhs.split(',');
  if (lhs.empty() || rhs.empty()) return llvm::None;

  auto lhs_map_or = EquationToMap(lhs);
  if (!lhs_map_or.hasValue()) return llvm::None;
  auto lhs_map = lhs_map_or.getValue();

  auto rhs_map_or = EquationToMap(rhs);
  if (!rhs_map_or.hasValue()) return llvm::None;
  auto rhs_map = rhs_map_or.getValue();

  auto out_map_or = EquationToMap(out);
  if (!out_map_or.hasValue()) return llvm::None;
  auto out_map = out_map_or.getValue();

  EinsumDimensionNumbers dnums;
  for (int64_t i = 0, e = lhs.size(); i < e; ++i) {
    auto rhs_index = rhs_map.find(lhs[i]);
    auto out_index = out_map.find(lhs[i]);
    if (rhs_index == rhs_map.end() && out_index == out_map.end()) {
      dnums.lhs.emplace_back(i);
    } else if (rhs_index == rhs_map.end()) {
      dnums.lhs_out.emplace_back(i, out_index->second);
    } else if (out_index == out_map.end()) {
      dnums.lhs_rhs.emplace_back(i, rhs_index->second);
    } else {
      dnums.lhs_rhs_out.emplace_back(i, rhs_index->second, out_index->second);
    }
  }
  for (int64_t i = 0, e = rhs.size(); i < e; ++i) {
    auto lhs_index = lhs_map.find(rhs[i]);
    auto out_index = out_map.find(rhs[i]);
    if (lhs_index == lhs_map.end()) {
      if (out_index == out_map.end()) {
        dnums.rhs.emplace_back(i);
      } else {
        dnums.rhs_out.emplace_back(i, out_index->second);
      }
    }
  }
  for (int64_t i = 0, e = out.size(); i < e; ++i) {
    auto lhs_index = lhs_map.find(out[i]);
    auto rhs_index = rhs_map.find(out[i]);
    if (lhs_index == lhs_map.end() && rhs_index == rhs_map.end()) {
      // out only isn't supported
      return llvm::None;
    }
  }
  return dnums;
}

std::vector<int64_t> inverseTransposeVector(
    llvm::ArrayRef<int64_t> input, llvm::ArrayRef<int32_t> permutation) {
  std::vector<int64_t> output(input.size());
  for (int64_t i = 0; i < input.size(); ++i) {
    output[permutation[i]] = input[i];
  }
  return output;
}

// Computes the transpositions required to convert dnums to one supported by
// tf.BatchMatmulV2 and returns the new set of dimension numbers with them.
LogicalResult transposeForBatchMatmul(
    const Location& loc, EinsumDimensionNumbers& dnums, Value* lhs, Value* rhs,
    std::vector<int32_t>* out_inverse_transpose, PatternRewriter* rewriter) {
  std::vector<int32_t> lhs_transpose;
  std::vector<int32_t> rhs_transpose;
  std::vector<int32_t> out_transpose;
  lhs_transpose.reserve(dnums.lhs_rhs_out.size() + dnums.lhs_out.size() +
                        dnums.lhs_rhs.size());
  rhs_transpose.reserve(dnums.lhs_rhs_out.size() + dnums.rhs_out.size() +
                        dnums.lhs_rhs.size());
  out_transpose.reserve(dnums.lhs_rhs_out.size() + dnums.lhs_out.size() +
                        dnums.rhs_out.size());
  for (int64_t i = 0, e = dnums.lhs_rhs_out.size(); i < e; ++i) {
    lhs_transpose.push_back(std::get<0>(dnums.lhs_rhs_out[i]));
    rhs_transpose.push_back(std::get<1>(dnums.lhs_rhs_out[i]));
    out_transpose.push_back(std::get<2>(dnums.lhs_rhs_out[i]));
    dnums.lhs_rhs_out[i] = std::make_tuple(i, i, i);
  }

  for (int64_t i = 0, e = dnums.lhs_out.size(); i < e; ++i) {
    lhs_transpose.push_back(std::get<0>(dnums.lhs_out[i]));
    out_transpose.push_back(std::get<1>(dnums.lhs_out[i]));
    dnums.lhs_out[i] =
        std::make_tuple(lhs_transpose.size() - 1, out_transpose.size() - 1);
  }
  for (int64_t i = 0, e = dnums.lhs_rhs.size(); i < e; ++i) {
    lhs_transpose.push_back(std::get<0>(dnums.lhs_rhs[i]));
    rhs_transpose.push_back(std::get<1>(dnums.lhs_rhs[i]));
    dnums.lhs_rhs[i] =
        std::make_tuple(lhs_transpose.size() - 1, rhs_transpose.size() - 1);
  }
  for (int64_t i = 0, e = dnums.rhs_out.size(); i < e; ++i) {
    rhs_transpose.push_back(std::get<0>(dnums.rhs_out[i]));
    out_transpose.push_back(std::get<1>(dnums.rhs_out[i]));
    dnums.rhs_out[i] =
        std::make_tuple(rhs_transpose.size() - 1, out_transpose.size() - 1);
  }

  out_inverse_transpose->resize(out_transpose.size());
  for (int64_t i = 0, e = out_transpose.size(); i < e; ++i) {
    out_inverse_transpose->at(out_transpose[i]) = i;
  }

  *lhs = createTransposeOp(*lhs, loc, lhs_transpose, rewriter);
  *rhs = createTransposeOp(*rhs, loc, rhs_transpose, rewriter);
  return success();
}

// Reshapes LHS and RHS to have B0,...,Bn,L,C and B0,...,Bn,C,R shape
// respectively while assuming that the initial shape for them is
// B0,...,Bn,L0,...,Ln,C0,...,Cn and B0,...,Bn,C0,...,Cn,R0,...,Rn respectively.
LogicalResult reshapeForBatchMatmul(const Location& loc,
                                    EinsumDimensionNumbers& dnums, Value* lhs,
                                    Value* rhs, std::vector<int64_t>* out_shape,
                                    PatternRewriter* rewriter) {
  RankedTensorType lhs_type = lhs->getType().cast<RankedTensorType>();
  RankedTensorType rhs_type = rhs->getType().cast<RankedTensorType>();

  std::vector<int64_t> lhs_shape;
  std::vector<int64_t> rhs_shape;
  lhs_shape.reserve(dnums.lhs_rhs_out.size() + dnums.lhs_out.size() + 1);
  rhs_shape.reserve(dnums.lhs_rhs_out.size() + 2);
  for (auto i : dnums.lhs_rhs_out) {
    int64_t b = lhs_type.getShape()[std::get<0>(i)];
    lhs_shape.push_back(b);
    rhs_shape.push_back(b);
    out_shape->push_back(b);
  }

  if (dnums.lhs_out.empty()) {
    lhs_shape.push_back(1);
    out_shape->push_back(1);
    dnums.lhs_out.emplace_back(lhs_shape.size() - 1, out_shape->size() - 1);
  } else if (dnums.lhs_rhs_out.empty()) {
    for (auto i : dnums.lhs_out) {
      int64_t b = lhs_type.getShape()[std::get<0>(i)];
      lhs_shape.push_back(b);
      out_shape->push_back(b);
    }
  } else {
    int64_t lhs_out_size = 1;
    for (auto i : dnums.lhs_out) {
      lhs_out_size *= lhs_type.getShape()[std::get<0>(i)];
    }
    lhs_shape.push_back(lhs_out_size);
    out_shape->push_back(lhs_out_size);
  }

  int64_t lhs_rhs_size = 1;
  for (auto i : dnums.lhs_rhs) {
    lhs_rhs_size *= lhs_type.getShape()[std::get<0>(i)];
  }
  lhs_shape.push_back(lhs_rhs_size);
  rhs_shape.push_back(lhs_rhs_size);

  int64_t rhs_size = 1;
  for (auto i : dnums.rhs_out) {
    rhs_size *= rhs_type.getShape()[std::get<0>(i)];
  }
  rhs_shape.push_back(rhs_size);
  out_shape->push_back(rhs_size);

  if (failed(VerifyShapeOfReshapeOp(lhs_shape)) ||
      failed(VerifyShapeOfReshapeOp(rhs_shape)))
    return failure();

  *lhs = createReshapeOp(*lhs, lhs_shape, lhs_type.getElementType(), loc,
                         rewriter);
  *rhs = createReshapeOp(*rhs, rhs_shape, rhs_type.getElementType(), loc,
                         rewriter);

  dnums.lhs_rhs.assign(
      {std::make_tuple(dnums.lhs_rhs_out.size() + dnums.lhs_out.size(),
                       dnums.lhs_rhs_out.size())});
  dnums.rhs_out.assign(
      {std::make_tuple(dnums.lhs_rhs_out.size() + dnums.lhs_out.size(),
                       dnums.lhs_rhs_out.size() + dnums.lhs_out.size())});
  return success();
}

LogicalResult rewriteToBatchMatmul(TF::EinsumOp op,
                                   EinsumDimensionNumbers dnums,
                                   PatternRewriter& rewriter) {
  if (!dnums.lhs.empty() || !dnums.rhs.empty()) return failure();

  auto inputs = op.inputs();
  if (inputs.size() != 2) return failure();
  Value lhs = inputs.front();
  Value rhs = inputs.back();

  RankedTensorType original_type =
      op.getResult().getType().dyn_cast_or_null<RankedTensorType>();
  if (!original_type) return failure();

  std::vector<int32_t> out_transpose;
  if (failed(transposeForBatchMatmul(op.getLoc(), dnums, &lhs, &rhs,
                                     &out_transpose, &rewriter)))
    return failure();

  std::vector<int64_t> matmul_shape;
  if (failed(reshapeForBatchMatmul(op.getLoc(), dnums, &lhs, &rhs,
                                   &matmul_shape, &rewriter)))
    return failure();

  std::vector<int64_t> reshape_shape =
      inverseTransposeVector(original_type.getShape(), out_transpose);
  if (failed(VerifyShapeOfReshapeOp(reshape_shape))) return failure();

  auto matmul_type =
      RankedTensorType::get(matmul_shape, original_type.getElementType());
  Value out = rewriter.create<TF::BatchMatMulV2Op>(
      op.getLoc(), matmul_type, lhs, rhs, rewriter.getBoolAttr(false),
      rewriter.getBoolAttr(false));

  out = createReshapeOp(out, reshape_shape, original_type.getElementType(),
                        op.getLoc(), &rewriter);
  out = createTransposeOp(out, op.getLoc(), out_transpose, &rewriter);

  rewriter.replaceOp(op, out);
  return success();
}

}  // namespace

LogicalResult ConvertTFEinsumOp::matchAndRewrite(
    TF::EinsumOp op, PatternRewriter& rewriter) const {
  const auto dnums_or = GetEinsumDimensionNumbers(op.equation());
  if (!dnums_or.hasValue()) return failure();
  const auto& dnums = dnums_or.getValue();

  RankedTensorType lhs =
      op.getOperand(0).getType().dyn_cast_or_null<RankedTensorType>();
  RankedTensorType rhs =
      op.getOperand(1).getType().dyn_cast_or_null<RankedTensorType>();
  if (!lhs || !rhs) return failure();

  return rewriteToBatchMatmul(op, dnums, rewriter);
}

// Transform Einsum to other TF Ops for the supported variants.
struct TransformEinsumPass
    : public PassWrapper<TransformEinsumPass, FunctionPass> {
  void runOnFunction() override;
};

void TransformEinsumPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();

  patterns.insert<ConvertTFEinsumOp>(&getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

static PassRegistration<TransformEinsumPass> pass(
    "tf-einsum", "Transform Einsum to other TF Ops for the supported variants");

}  // namespace TF
}  // namespace mlir
