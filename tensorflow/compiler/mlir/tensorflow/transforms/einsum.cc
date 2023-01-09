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
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verification_utils.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace mlir {
namespace TF {

namespace {

// Creates ConstOp for int32_t value.
ConstOp createI32ConstOp(int32_t value, Location loc,
                         PatternRewriter* rewriter) {
  auto int_attr = IntegerAttr::get(rewriter->getIntegerType(32), value);
  return rewriter->create<ConstOp>(loc, int_attr);
}

// Creates ConstantOp for array of int32_t.
arith::ConstantOp createI32ConstantOp(llvm::ArrayRef<int32_t> values,
                                      Location loc, PatternRewriter* rewriter) {
  auto values_type = RankedTensorType::get(
      {static_cast<int32_t>(values.size())}, rewriter->getIntegerType(32));
  auto constant_attr = rewriter->getI32TensorAttr(values);
  return rewriter->create<arith::ConstantOp>(loc, values_type, constant_attr);
}

// Creates ConstantOp for array of int64_t.
arith::ConstantOp createI64ConstantOp(llvm::ArrayRef<int64_t> values,
                                      Location loc, PatternRewriter* rewriter) {
  auto values_type = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, rewriter->getIntegerType(64));
  auto constant_attr = rewriter->getI64TensorAttr(values);
  return rewriter->create<arith::ConstantOp>(loc, values_type, constant_attr);
}

TF::TransposeOp createTransposeOp(Value value, Location loc,
                                  llvm::ArrayRef<int32_t> permutation,
                                  PatternRewriter* rewriter) {
  auto perm_op = createI32ConstantOp(permutation, loc, rewriter);
  auto value_type = value.getType().cast<RankedTensorType>();
  auto shape = value_type.getShape();
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
  auto shape_tensor = createI64ConstantOp(
      tensorflow::ConvertMlirShapeToTF(shape), loc, rewriter);
  Type resultType = RankedTensorType::get(shape, element_type);
  return rewriter->create<TF::ReshapeOp>(loc, resultType, /*tensor=*/value,
                                         /*shape=*/shape_tensor);
}

// Creates ReshapeOp with runtime calcuation of required shape to support
// dynamic shapes. The shape is calculated by Shape and UnsortedSegmentProd op.
// `reshape_segids` and `num_reshape_segids` for UnsortedSegmentProd is
// calculated in `reshapeForBatchMatmul`.
TF::ReshapeOp createReshapeOpForDynamic(Value value, ArrayRef<int64_t> shape,
                                        ArrayRef<int32_t> reshape_segids,
                                        int32_t num_reshape_segids,
                                        Location loc,
                                        PatternRewriter* rewriter) {
  // Build ShapeOp
  auto input_shape =
      rewriter->create<TF::ShapeOp>(loc, value, rewriter->getBoolAttr(true));

  // Build UnsortedSegmentProdOp
  Type segProdresultType =
      RankedTensorType::get(num_reshape_segids, rewriter->getIntegerType(32));
  auto segids_tensor = createI32ConstantOp(reshape_segids, loc, rewriter);
  auto num_reshape_segids_tensor =
      createI32ConstOp(num_reshape_segids, loc, rewriter);
  auto segprod = rewriter->create<TF::UnsortedSegmentProdOp>(
      loc, segProdresultType, input_shape->getResults()[0], segids_tensor,
      num_reshape_segids_tensor);

  // Build ReshapeOp with the result of UnsortedSegmentProdOp.
  Type out_tensor_type =
      RankedTensorType::get(shape, getElementTypeOrSelf(value.getType()));
  return rewriter->create<TF::ReshapeOp>(loc, out_tensor_type,
                                         /*tensor=*/value,
                                         /*shape=*/segprod->getResults()[0]);
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

TF::ReshapeOp createOutputReshapeOpForDynamic(
    Value value, ArrayRef<int64_t> shape, Value org_lhs, Value org_rhs,
    EinsumDimensionNumbers& dnums, Location loc, PatternRewriter* rewriter) {
  BoolAttr true_attr = rewriter->getBoolAttr(true);
  // Build ShapeOp
  auto shape_lhs = rewriter->create<TF::ShapeOp>(loc, org_lhs, true_attr);
  auto shape_rhs = rewriter->create<TF::ShapeOp>(loc, org_rhs, true_attr);

  std::vector<int32_t> bl_index;  // Indexes of B0,...,Bn and L0,...,Ln
  bl_index.reserve(dnums.lhs_rhs_out.size() + dnums.lhs_out.size());
  for (auto i : dnums.lhs_rhs_out) {
    bl_index.push_back(std::get<0>(i));
  }
  for (auto i : dnums.lhs_out) {
    bl_index.push_back(std::get<0>(i));
  }
  std::vector<int32_t> r_index;  // Indexes of R0,...,Rn
  r_index.reserve(dnums.rhs_out.size());
  for (auto i : dnums.rhs_out) {
    r_index.push_back(std::get<0>(i));
  }

  auto lhs_index_tensor = createI32ConstantOp(bl_index, loc, rewriter);
  auto gather_lhs = rewriter->create<TF::GatherOp>(
      loc,
      RankedTensorType::get({static_cast<int>(bl_index.size())},
                            rewriter->getIntegerType(32)),
      shape_lhs->getResults()[0], lhs_index_tensor->getResults()[0], true_attr);
  auto rhs_index_tensor = createI32ConstantOp(r_index, loc, rewriter);
  auto gather_rhs = rewriter->create<TF::GatherOp>(
      loc,
      RankedTensorType::get({static_cast<int>(r_index.size())},
                            rewriter->getIntegerType(32)),
      shape_rhs->getResults()[0], rhs_index_tensor->getResults()[0], true_attr);
  Value zero_value = createI32ConstOp(0, loc, rewriter);
  auto concat_out_shape = rewriter->create<TF::ConcatOp>(
      loc,
      RankedTensorType::get({static_cast<int>(bl_index.size()) +
                             static_cast<int>(r_index.size())},
                            rewriter->getIntegerType(32)),
      zero_value,
      ArrayRef<Value>(
          {gather_lhs->getResults()[0], gather_rhs->getResults()[0]}));

  // Build ReshapeOp with the calculated output shape.
  Type out_type =
      RankedTensorType::get(shape, getElementTypeOrSelf(value.getType()));
  return rewriter->create<TF::ReshapeOp>(
      loc, out_type,
      /*tensor=*/value,
      /*shape=*/concat_out_shape->getResults()[0]);
}

llvm::Optional<llvm::SmallDenseMap<char, int64_t>> EquationToMap(
    llvm::StringRef equation) {
  llvm::SmallDenseMap<char, int64_t> map;
  for (int64_t i = 0; i < equation.size(); ++i) {
    if (!std::isalpha(equation[i])) {
      // Unsupported character in the equation.
      return std::nullopt;
    }
    if (map.count(equation[i])) {
      // Duplicate character in the equation.
      return std::nullopt;
    }
    map.try_emplace(equation[i], i);
  }
  return map;
}

llvm::Optional<llvm::SetVector<char>> GetAvailableLabels(
    llvm::StringRef lhs, llvm::StringRef rhs, int* lhs_named_label_count,
    int* rhs_named_label_count) {
  llvm::SetVector<char> labels;
  for (int i = 0; i < 26; ++i) {
    labels.insert('a' + i);
    labels.insert('A' + i);
  }

  auto is_start_of_ellipsis = [](StringRef equation, int start_index) {
    if (equation.size() < (start_index + 3)) return false;

    if (equation.substr(start_index, 3) != "...") return false;
    return true;
  };

  int lhs_count = 0;
  const int lhs_size = lhs.size();
  for (int i = 0; i < lhs_size; ++i) {
    const char label = lhs[i];
    if (std::isalpha(label)) {
      labels.remove(label);
      ++lhs_count;
    } else if (label == '.') {
      if (!is_start_of_ellipsis(lhs, i)) return std::nullopt;
      i += 2;
    } else {
      // Unsupported character in the equation.
      return std::nullopt;
    }
  }
  *lhs_named_label_count = lhs_count;

  int rhs_count = 0;
  const int rhs_size = rhs.size();
  for (int i = 0; i < rhs_size; ++i) {
    const char label = rhs[i];
    if (std::isalpha(label)) {
      labels.remove(label);
      ++rhs_count;
    } else if (label == '.') {
      if (!is_start_of_ellipsis(rhs, i)) return std::nullopt;
      i += 2;
    } else {
      // Unsupported character in the equation.
      return std::nullopt;
    }
  }

  *rhs_named_label_count = rhs_count;
  return labels;
}

// Generate new unnamed labels for the expression.
// For example, if we have GenerateLabels(2, {'b', 'c', 'd'}) for "...xy"
// We will have "dcxy" for the ellipsis expression since it's rank 4,
// we will have dcbxy if it's rank 5.
std::string GenerateLabels(int count,
                           const llvm::SetVector<char>& available_labels) {
  std::string new_labels(count, 0);
  for (int i = 0; i < count; ++i) {
    new_labels[count - 1 - i] = available_labels[i];
  }

  return new_labels;
}

std::tuple<std::string, std::string, std::string> FlattenEllipsis(
    llvm::StringRef lhs, int lhs_named_label_count, llvm::StringRef rhs,
    int rhs_named_label_count, llvm::StringRef out, RankedTensorType lhs_ty,
    RankedTensorType rhs_ty, const llvm::SetVector<char>& available_labels) {
  std::string new_labels;
  std::string new_lhs;
  for (int i = 0; i < lhs.size(); ++i) {
    const char label = lhs[i];
    if (std::isalpha(label)) {
      new_lhs.push_back(label);
    } else {
      // Encounter ellipsis: generate unnamed labels then insert to the new
      // labels.
      new_labels = GenerateLabels(lhs_ty.getRank() - lhs_named_label_count,
                                  available_labels);
      new_lhs.append(new_labels);
      i += 2;
    }
  }

  std::string new_rhs, new_rhs_labels;
  for (int i = 0; i < rhs.size(); ++i) {
    const char label = rhs[i];
    if (std::isalpha(label)) {
      new_rhs.push_back(label);
    } else {
      // Encounter ellipsis: generate unnamed labels then insert to the new
      // labels.
      new_rhs_labels = GenerateLabels(rhs_ty.getRank() - rhs_named_label_count,
                                      available_labels);
      new_rhs.append(new_rhs_labels);
      i += 2;
      if (new_rhs_labels.size() > new_labels.size()) {
        new_labels = new_rhs_labels;
      }
    }
  }

  // Deal with the output next.
  std::string new_output;
  for (int i = 0; i < out.size(); ++i) {
    const char label = out[i];
    if (std::isalpha(label)) {
      new_output.push_back(label);
    } else {
      // Encounter ellipsis: we will just insert the generated labels to the new
      // output label.
      new_output.append(new_labels);
      i += 2;
    }
  }

  return std::make_tuple(new_lhs, new_rhs, new_output);
}

llvm::Optional<EinsumDimensionNumbers> GetEinsumDimensionNumbers(
    llvm::StringRef equation, RankedTensorType lhs_ty,
    RankedTensorType rhs_ty) {
  llvm::StringRef lhs_rhs;
  llvm::StringRef out;
  std::tie(lhs_rhs, out) = equation.split("->");
  if (lhs_rhs.empty() || out.empty()) return std::nullopt;

  llvm::StringRef lhs;
  llvm::StringRef rhs;
  std::tie(lhs, rhs) = lhs_rhs.split(',');
  if (lhs.empty() || rhs.empty()) return std::nullopt;

  // Try to flatten the "..." if possible.
  int lhs_named_label, rhs_named_label;
  auto available_labels =
      GetAvailableLabels(lhs, rhs, &lhs_named_label, &rhs_named_label);
  if (!available_labels.has_value()) return std::nullopt;

  auto flattended_labels =
      FlattenEllipsis(lhs, lhs_named_label, rhs, rhs_named_label, out, lhs_ty,
                      rhs_ty, available_labels.value());

  lhs = std::get<0>(flattended_labels);
  rhs = std::get<1>(flattended_labels);
  out = std::get<2>(flattended_labels);

  auto lhs_map_or = EquationToMap(lhs);
  if (!lhs_map_or.has_value()) return std::nullopt;
  auto lhs_map = lhs_map_or.value();

  auto rhs_map_or = EquationToMap(rhs);
  if (!rhs_map_or.has_value()) return std::nullopt;
  auto rhs_map = rhs_map_or.value();

  auto out_map_or = EquationToMap(out);
  if (!out_map_or.has_value()) return std::nullopt;
  auto out_map = out_map_or.value();

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
      return std::nullopt;
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
// Transposed LHS shape will be B0,...,Bn,L0,...,Ln,C0,...,Cn and,
// transposed RHS shape will be B0,...,Bn,C0,...,Cn,R0,...,Rn respectively.
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
  // Generate transpose matrix for B0,...,Bn
  for (int64_t i = 0, e = dnums.lhs_rhs_out.size(); i < e; ++i) {
    lhs_transpose.push_back(std::get<0>(dnums.lhs_rhs_out[i]));
    rhs_transpose.push_back(std::get<1>(dnums.lhs_rhs_out[i]));
    out_transpose.push_back(std::get<2>(dnums.lhs_rhs_out[i]));
    dnums.lhs_rhs_out[i] = std::make_tuple(i, i, i);
  }

  // Generate transpose matrix for L0,...,Ln
  for (int64_t i = 0, e = dnums.lhs_out.size(); i < e; ++i) {
    lhs_transpose.push_back(std::get<0>(dnums.lhs_out[i]));
    out_transpose.push_back(std::get<1>(dnums.lhs_out[i]));
    dnums.lhs_out[i] =
        std::make_tuple(lhs_transpose.size() - 1, out_transpose.size() - 1);
  }
  // Generate transpose matrix for C0,...,Cn
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

template <int I>
inline int64_t ProdShapeWithIndexInTuple(
    ArrayRef<int64_t> shape,
    const std::vector<std::tuple<int64_t, int64_t>>& index_tuples) {
  int64_t prod_shape = 1;
  for (auto index_tuple : index_tuples) {
    const int64_t shape_i = shape[std::get<I>(index_tuple)];
    if (ShapedType::isDynamic(shape_i)) return ShapedType::kDynamic;
    prod_shape *= shape_i;
  }
  return prod_shape;
}

// Reshapes LHS and RHS to have B0,...,Bn,L,C and B0,...,Bn,C,R shape
// respectively while assuming that the initial shape for them is
// B0,...,Bn,L0,...,Ln,C0,...,Cn and B0,...,Bn,C0,...,Cn,R0,...,Rn respectively.
LogicalResult reshapeForBatchMatmul(const Location& loc,
                                    EinsumDimensionNumbers& dnums, Value* lhs,
                                    Value* rhs,
                                    SmallVectorImpl<int64_t>* out_shape,
                                    PatternRewriter* rewriter) {
  RankedTensorType lhs_type = lhs->getType().cast<RankedTensorType>();
  RankedTensorType rhs_type = rhs->getType().cast<RankedTensorType>();

  int32_t num_lhs_reshape_segids = 0;
  int32_t num_rhs_reshape_segids = 0;
  std::vector<int32_t> lhs_reshape_segids;
  int lhs_rank =
      dnums.lhs_rhs_out.size() + dnums.lhs_out.size() + dnums.lhs_rhs.size();
  lhs_reshape_segids.resize(lhs_rank);
  std::vector<int32_t> rhs_reshape_segids;
  int rhs_rank =
      dnums.lhs_rhs_out.size() + dnums.rhs_out.size() + dnums.lhs_rhs.size();
  rhs_reshape_segids.resize(rhs_rank);

  // Labels exist in all lhs, rhs and output are the batch labels B0,...,Bn.
  std::vector<int64_t> lhs_shape;
  std::vector<int64_t> rhs_shape;
  lhs_shape.reserve(dnums.lhs_rhs_out.size() + dnums.lhs_out.size() + 1);
  rhs_shape.reserve(dnums.lhs_rhs_out.size() + 2);
  for (auto i : dnums.lhs_rhs_out) {
    const int64_t b1 = lhs_type.getShape()[std::get<0>(i)];
    lhs_shape.push_back(b1);
    const int64_t b2 = rhs_type.getShape()[std::get<1>(i)];
    rhs_shape.push_back(b2);

    lhs_reshape_segids.at(std::get<0>(i)) = num_lhs_reshape_segids++;
    rhs_reshape_segids.at(std::get<1>(i)) = num_rhs_reshape_segids++;
  }
  if (!OpTrait::util::getBroadcastedShape(lhs_shape, rhs_shape, *out_shape)) {
    return failure();
  }

  // Calculates dimension for the label L from L0,...,Ln in lhs.
  if (dnums.lhs_out.empty()) {
    lhs_shape.push_back(1);
    out_shape->push_back(1);
    dnums.lhs_out.emplace_back(lhs_shape.size() - 1, out_shape->size() - 1);
    ++num_lhs_reshape_segids;
  } else if (dnums.lhs_rhs_out.empty()) {
    // If there is not batch labels B0,...,Bn, it is safe to use L0,...,Ln as
    // the batch labels in lhs, the rhs will be broadcasted.
    for (auto i : dnums.lhs_out) {
      const int64_t b = lhs_type.getShape()[std::get<0>(i)];
      lhs_shape.push_back(b);
      out_shape->push_back(b);

      lhs_reshape_segids.at(std::get<0>(i)) = num_lhs_reshape_segids++;
    }
  } else {
    const int64_t lhs_out_size =
        ProdShapeWithIndexInTuple<0>(lhs_type.getShape(), dnums.lhs_out);
    lhs_shape.push_back(lhs_out_size);
    out_shape->push_back(lhs_out_size);

    for (auto i : dnums.lhs_out) {
      lhs_reshape_segids.at(std::get<0>(i)) = num_lhs_reshape_segids;
    }
    ++num_lhs_reshape_segids;
  }

  // Calculates dimension for the common label C from labels C0,...,Cn that
  // exist in both lhs and rhs.
  const int64_t lhs_size =
      ProdShapeWithIndexInTuple<0>(lhs_type.getShape(), dnums.lhs_rhs);
  const int64_t rhs_size =
      ProdShapeWithIndexInTuple<1>(rhs_type.getShape(), dnums.lhs_rhs);
  lhs_shape.push_back(lhs_size);
  rhs_shape.push_back(rhs_size);

  for (auto i : dnums.lhs_rhs) {
    lhs_reshape_segids.at(std::get<0>(i)) = num_lhs_reshape_segids;
    rhs_reshape_segids.at(std::get<1>(i)) = num_rhs_reshape_segids;
  }
  ++num_lhs_reshape_segids;
  ++num_rhs_reshape_segids;

  // Calculates dimension for the label R from R0,...,Rn in rhs.
  const int64_t rhs_out_size =
      ProdShapeWithIndexInTuple<0>(rhs_type.getShape(), dnums.rhs_out);
  rhs_shape.push_back(rhs_out_size);
  out_shape->push_back(rhs_out_size);

  for (auto i : dnums.rhs_out) {
    rhs_reshape_segids.at(std::get<0>(i)) = num_rhs_reshape_segids;
  }
  ++num_rhs_reshape_segids;

  // If LHS requires reshapes.
  if (lhs_rank != num_lhs_reshape_segids) {
    if (succeeded(VerifyShapeOfReshapeOp(lhs_shape))) {
      *lhs = createReshapeOp(*lhs, lhs_shape, lhs_type.getElementType(), loc,
                             rewriter);
    } else {
      // Check if lhs LHS shape can be calculated with SegmentProd. It requires
      // to have at least 1 common index in lhs_out and lhs_rhs.
      if (dnums.lhs_out.empty() || dnums.lhs_rhs.empty()) return failure();
      *lhs = createReshapeOpForDynamic(*lhs, lhs_shape, lhs_reshape_segids,
                                       num_lhs_reshape_segids, loc, rewriter);
    }
  }
  // If RHS requires reshapes.
  if (rhs_rank != num_rhs_reshape_segids) {
    if (succeeded(VerifyShapeOfReshapeOp(rhs_shape))) {
      *rhs = createReshapeOp(*rhs, rhs_shape, rhs_type.getElementType(), loc,
                             rewriter);
    } else {
      // Check if lhs RHS shape can be calculated with SegmentProd. It requires
      // to have at least 1 common index in rhs_out and lhs_rhs.
      if (dnums.rhs_out.empty() || dnums.lhs_rhs.empty()) return failure();
      *rhs = createReshapeOpForDynamic(*rhs, rhs_shape, rhs_reshape_segids,
                                       num_rhs_reshape_segids, loc, rewriter);
    }
  }

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

  auto inputs = op.getInputs();
  if (inputs.size() != 2) return failure();
  Value lhs = inputs.front();
  Value rhs = inputs.back();
  // Back original values for the later output shape calculation in
  // `createOutputReshapeOpForDynamic`.
  Value original_lhs = lhs;
  Value original_rhs = rhs;
  EinsumDimensionNumbers original_dnums = dnums;

  RankedTensorType original_type =
      op.getResult().getType().dyn_cast_or_null<RankedTensorType>();
  if (!original_type) return failure();

  std::vector<int32_t> out_transpose;
  if (failed(transposeForBatchMatmul(op.getLoc(), dnums, &lhs, &rhs,
                                     &out_transpose, &rewriter)))
    return failure();

  llvm::SmallVector<int64_t, 4> matmul_shape;
  if (failed(reshapeForBatchMatmul(op.getLoc(), dnums, &lhs, &rhs,
                                   &matmul_shape, &rewriter)))
    return failure();

  std::vector<int64_t> reshape_shape =
      inverseTransposeVector(original_type.getShape(), out_transpose);

  auto matmul_type =
      RankedTensorType::get(matmul_shape, original_type.getElementType());
  Value out = rewriter.create<TF::BatchMatMulV2Op>(
      op.getLoc(), matmul_type, lhs, rhs, rewriter.getBoolAttr(false),
      rewriter.getBoolAttr(false));

  bool out_reshape_need = (reshape_shape.size() != matmul_shape.size() ||
                           original_type.getRank() != matmul_shape.size());
  // Always add reshape for concrete output shapes.
  if (succeeded(VerifyShapeOfReshapeOp(reshape_shape))) {
    out = createReshapeOp(out, reshape_shape, original_type.getElementType(),
                          op.getLoc(), &rewriter);
  } else if (out_reshape_need) {
    out = createOutputReshapeOpForDynamic(out, reshape_shape, original_lhs,
                                          original_rhs, original_dnums,
                                          op.getLoc(), &rewriter);
  }
  out = createTransposeOp(out, op.getLoc(), out_transpose, &rewriter);

  rewriter.replaceOp(op, out);
  return success();
}

#define GEN_PASS_DEF_TRANSFORMEINSUMPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Transform Einsum to other TF Ops for the supported variants.
struct TransformEinsumPass
    : public impl::TransformEinsumPassBase<TransformEinsumPass> {
  void runOnOperation() override;
};

void TransformEinsumPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  patterns.add<ConvertTFEinsumOp>(&getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

LogicalResult ConvertTFEinsumOp::matchAndRewrite(
    TF::EinsumOp op, PatternRewriter& rewriter) const {
  RankedTensorType lhs =
      op.getOperand(0).getType().dyn_cast_or_null<RankedTensorType>();
  RankedTensorType rhs =
      op.getOperand(1).getType().dyn_cast_or_null<RankedTensorType>();
  if (!lhs || !rhs) {
    return failure();
  }

  // TODO(b/162328998) Better support Einsum with dynamic input. Currently, one
  // dynamic dimension is always supported. If there are two or more dynamic
  // dimensions, it is supported if they only exist in a single component
  // among: L0,...,Ln R0,...,Rn or C0,...,Cn.
  if (const auto dnums_or =
          GetEinsumDimensionNumbers(op.getEquation(), lhs, rhs))
    return rewriteToBatchMatmul(op, dnums_or.value(), rewriter);
  return rewriter.notifyMatchFailure(op, "unsupported einsum lowering");
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTransformEinsumPass() {
  return std::make_unique<TransformEinsumPass>();
}

}  // namespace TF
}  // namespace mlir
