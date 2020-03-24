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

#include <climits>
#include <cstdint>

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"
#include "mlir/Analysis/LoopAnalysis.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/OpImplementation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/Functional.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

namespace {

// All supported Einsum equations.
enum EinsumEquation {
  BatchMatMul,
  FourDMatrixDotProd,
  ThreeDReshapeTail,
  FourDBatchMatMul,
  UnsupportedEquation
};

// Tokens for parsing the given equation string.
enum EquationToken {
  A,
  B,
  C,
  D,
  E,
  COMMA,
  ARROW,
};
constexpr int kNumSupportedEquationVariables = 5;  // A - E for now.

bool tokenizeEquation(const llvm::StringRef& equation,
                      std::vector<EquationToken>* tokens) {
  std::map<char, EquationToken> label_axis_mapping;
  int index = 0;
  int variable_count = 0;
  llvm::Regex r("[[:alpha:]]");
  while (index < equation.size()) {
    if (r.match(equation.substr(index, 1))) {
      const char ltr = equation[index];
      auto itr = label_axis_mapping.find(ltr);
      if (itr == label_axis_mapping.end() &&
          variable_count < kNumSupportedEquationVariables) {
        label_axis_mapping[ltr] = EquationToken(variable_count);
        tokens->push_back(EquationToken(variable_count));
        variable_count++;
      } else if (itr != label_axis_mapping.end()) {
        tokens->push_back(itr->second);
      } else {
        // Ran out of equation variables.
        return false;
      }
    } else if (equation.substr(index, 1).contains(",")) {
      tokens->push_back(COMMA);
    } else if ((index < (equation.size() - 1)) &&
               (equation.substr(index, 2).contains("->"))) {
      tokens->push_back(ARROW);
      index++;
    } else {
      // Unallowed character encountered.
      return false;
    }
    index++;
  }
  return true;
}

EinsumEquation parseEquation(const std::vector<EquationToken>& eqn) {
  auto is_equal = [](const std::vector<EquationToken>& eqn1,
                     const std::initializer_list<EquationToken>& eqn2) {
    return std::equal(eqn1.begin(), eqn1.end(), eqn2.begin(), eqn2.end());
  };
  // IJK,IKM->IJM
  if (is_equal(eqn, {A, B, C, COMMA, A, C, D, ARROW, A, B, D})) {
    return EinsumEquation::BatchMatMul;
  }
  // BFND,NDH->BFH
  if (is_equal(eqn, {A, B, C, D, COMMA, C, D, E, ARROW, A, B, E})) {
    return EinsumEquation::FourDMatrixDotProd;
  }
  // BFNH,BTNH->BNFT
  if (is_equal(eqn, {A, B, C, D, COMMA, A, E, C, D, ARROW, A, C, B, E})) {
    return EinsumEquation::FourDBatchMatMul;
  }
  // BFD,DNH->BFNH
  if (is_equal(eqn, {A, B, C, COMMA, C, D, E, ARROW, A, B, D, E})) {
    return EinsumEquation::ThreeDReshapeTail;
  }
  return EinsumEquation::UnsupportedEquation;
}

EinsumEquation tokenizeAndParse(const llvm::StringRef& equation) {
  std::vector<EquationToken> tokens;
  if (tokenizeEquation(equation, &tokens)) {
    return parseEquation(tokens);
  }
  return EinsumEquation::UnsupportedEquation;
}

TF::TransposeOp createTransposeOp(Value value, Location loc,
                                  llvm::ArrayRef<int32_t> permutation,
                                  PatternRewriter* rewriter) {
  auto value_type = value.getType().cast<RankedTensorType>();
  auto shape = value_type.getShape();
  auto perm_type = RankedTensorType::get(
      {static_cast<int32_t>(permutation.size())}, rewriter->getIntegerType(32));
  auto perm_attr = DenseElementsAttr::get(perm_type, permutation);
  auto perm_op = rewriter->create<ConstantOp>(loc, perm_type, perm_attr);
  std::vector<int64_t> transposed_shape(shape.begin(), shape.end());
  for (int i = 0; i < shape.size(); ++i) {
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

}  // namespace

LogicalResult ConvertTFEinsumOp::matchAndRewrite(
    TF::EinsumOp op, PatternRewriter& rewriter) const {
  Type output_type = op.getResult().getType();
  Value lhs = op.getOperand(0);
  Value rhs = op.getOperand(1);
  Location loc = op.getLoc();

  if (!lhs.getType().isa<RankedTensorType>()) {
    // LHS must be a ranked tensor type
    return failure();
  }
  if (!rhs.getType().isa<RankedTensorType>()) {
    // RHS must be a ranked tensor type
    return failure();
  }

  auto lhs_type = lhs.getType().cast<RankedTensorType>();
  auto rhs_type = rhs.getType().cast<RankedTensorType>();
  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();

  // Currently only support static shapes.
  if (!(lhs_type.hasStaticShape() && rhs_type.hasStaticShape())) {
    return failure();
  }

  // Currently support use cases of LHS, RHS dims = 3 or 4
  const int dims_lhs = lhs_shape.size();
  const int dims_rhs = rhs_shape.size();
  if (dims_rhs < 3 || dims_rhs > 4 || dims_lhs < 3 || dims_lhs > 4) {
    return failure();
  }

  EinsumEquation einsum_eqn = tokenizeAndParse(op.equation());
  if (einsum_eqn == EinsumEquation::BatchMatMul) {
    // Case "IJK,IKM->IJM"
    auto bmm_op = rewriter.create<TF::BatchMatMulV2Op>(
        loc, ArrayRef<Type>{output_type}, lhs, rhs, rewriter.getBoolAttr(false),
        rewriter.getBoolAttr(false));
    rewriter.replaceOp(op, bmm_op.getResult());
    return success();
  }
  if (einsum_eqn == EinsumEquation::ThreeDReshapeTail) {
    // Case "BFD,DNH->BFNH"
    auto lhs_type = lhs.getType().cast<RankedTensorType>();
    auto lhs_shape = lhs_type.getShape();
    const int lhs_dim0 = lhs_shape[0];
    const int lhs_dim1 = lhs_shape[1];
    // Reshape RHS
    auto rhs_type = rhs.getType().cast<RankedTensorType>();
    auto rhs_shape = rhs_type.getShape();
    auto rhs_element_type = rhs_type.getElementType();
    const int rhs_dim0 = rhs_shape[0];
    const int rhs_dim1 = rhs_shape[1];
    const int rhs_dim2 = rhs_shape[2];
    auto reshaped_rhs = createReshapeOp(rhs, {rhs_dim0, rhs_dim1 * rhs_dim2},
                                        rhs_element_type, loc, &rewriter);

    std::vector<int64_t> bmm_shape = {lhs_dim0, lhs_dim1, rhs_dim1 * rhs_dim2};
    auto bmm_type = RankedTensorType::get(bmm_shape, rhs_type.getElementType());
    auto bmm_op = rewriter.create<TF::BatchMatMulV2Op>(
        loc, ArrayRef<Type>{bmm_type}, lhs, reshaped_rhs,
        rewriter.getBoolAttr(false), rewriter.getBoolAttr(false));
    auto bmm_element_type = bmm_type.getElementType();
    auto final_reshape =
        createReshapeOp(bmm_op, {lhs_dim0, lhs_dim1, rhs_dim1, rhs_dim2},
                        bmm_element_type, loc, &rewriter);
    rewriter.replaceOp(op, {final_reshape.getResult()});
    return success();
  }
  if (einsum_eqn == EinsumEquation::FourDMatrixDotProd) {
    // Case "BFND,NDH->BFH"
    // Reshape LHS
    auto lhs_element_type = lhs_type.getElementType();
    const int lhs_dim0 = lhs_shape[0];
    const int lhs_dim1 = lhs_shape[1];
    const int lhs_dim2 = lhs_shape[2];
    const int lhs_dim3 = lhs_shape[3];
    auto reshaped_lhs =
        createReshapeOp(lhs, {lhs_dim0, lhs_dim1, lhs_dim2 * lhs_dim3},
                        lhs_element_type, loc, &rewriter);
    // Reshape RHS
    auto rhs_element_type = rhs_type.getElementType();
    const int rhs_dim0 = rhs_shape[0];
    const int rhs_dim1 = rhs_shape[1];
    const int rhs_dim2 = rhs_shape[2];
    auto reshaped_rhs = createReshapeOp(rhs, {rhs_dim0 * rhs_dim1, rhs_dim2},
                                        rhs_element_type, loc, &rewriter);
    auto bmm_op = rewriter.create<TF::BatchMatMulV2Op>(
        loc, ArrayRef<Type>{output_type}, reshaped_lhs, reshaped_rhs,
        rewriter.getBoolAttr(false), rewriter.getBoolAttr(false));
    rewriter.replaceOp(op, {bmm_op.getResult()});
    return success();
  }
  if (einsum_eqn == EinsumEquation::FourDBatchMatMul) {
    // Case "BFNH,BTNH->BNFT"
    // Transpose LHS
    lhs = createTransposeOp(lhs, loc, {0, 2, 1, 3}, &rewriter);
    // Transpose RHS
    rhs = createTransposeOp(rhs, loc, {0, 2, 3, 1}, &rewriter);
    auto bmm_op = rewriter.create<TF::BatchMatMulV2Op>(
        loc, ArrayRef<Type>{output_type}, lhs, rhs, rewriter.getBoolAttr(false),
        rewriter.getBoolAttr(false));
    rewriter.replaceOp(op, {bmm_op.getResult()});
    return success();
  }
  return failure();
}

// Transform Einsum to other TF Ops for the supported variants.
struct TransformEinsumPass : public FunctionPass<TransformEinsumPass> {
  void runOnFunction() override;
};

void TransformEinsumPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();

  patterns.insert<ConvertTFEinsumOp>(&getContext());
  applyPatternsGreedily(func, patterns);
}

static PassRegistration<TransformEinsumPass> pass(
    "tf-einsum", "Transform Einsum to other TF Ops for the supported variants");

}  // namespace TF
}  // namespace mlir
