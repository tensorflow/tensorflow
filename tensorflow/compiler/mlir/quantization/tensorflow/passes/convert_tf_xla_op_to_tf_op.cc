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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace quant {
namespace {

using ::mlir::tf_quant::Create1DConstValue;
using ::mlir::tf_quant::CreateScalarConstValue;

class ConvertTfXlaOpToTfOpPass
    : public PassWrapper<ConvertTfXlaOpToTfOpPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTfXlaOpToTfOpPass)

  ConvertTfXlaOpToTfOpPass() = default;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-convert-tf-xla-op-to-tf-op";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply converting Tensorflow Xla ops to non-xla ops.";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, arith::ArithDialect>();
  }

  void runOnOperation() override;
};

// Generate an einsum equation from the given DotDimensionNumber.
std::string CreateEinsumEquation(
    const xla::DotDimensionNumbers& dot_dimension_numbers, const int lhs_rank,
    const int rhs_rank) {
  // Prepare necessary indices.
  absl::flat_hash_set<int64_t> lhs_batch_idx, rhs_batch_idx;
  absl::flat_hash_set<int64_t> lhs_contract_idx, rhs_contract_idx;
  lhs_batch_idx.insert(dot_dimension_numbers.lhs_batch_dimensions().begin(),
                       dot_dimension_numbers.lhs_batch_dimensions().end());
  lhs_contract_idx.insert(
      dot_dimension_numbers.lhs_contracting_dimensions().begin(),
      dot_dimension_numbers.lhs_contracting_dimensions().end());
  rhs_batch_idx.insert(dot_dimension_numbers.rhs_batch_dimensions().begin(),
                       dot_dimension_numbers.rhs_batch_dimensions().end());
  rhs_contract_idx.insert(
      dot_dimension_numbers.rhs_contracting_dimensions().begin(),
      dot_dimension_numbers.rhs_contracting_dimensions().end());

  // Generate equation.
  std::string lhs_eq = "";
  std::string rhs_eq = "";
  std::string out_eq = "";
  char c = 'a';
  std::vector<char> lhs_batch_dims;
  std::vector<char> lhs_contract_dims;
  for (int i = 0; i < lhs_rank; i++) {
    absl::StrAppend(&lhs_eq, std::string(1, c));
    if (lhs_batch_idx.contains(i)) {
      lhs_batch_dims.push_back(c);
    } else if (lhs_contract_idx.contains(i)) {
      lhs_contract_dims.push_back(c);
    }
    c++;
  }

  int batch_trace_idx = 0;
  int contract_trace_idx = 0;
  const bool rhs_only_batch = lhs_batch_dims.empty();
  for (int i = 0; i < rhs_rank; i++) {
    if (rhs_batch_idx.contains(i)) {
      if (rhs_only_batch) {
        rhs_eq.push_back(c);
        lhs_batch_dims.push_back(c);
        c++;
      } else {
        rhs_eq.push_back(lhs_batch_dims[batch_trace_idx]);
        batch_trace_idx++;
      }
    } else if (rhs_contract_idx.contains(i)) {
      absl::StrAppend(&rhs_eq,
                      std::string(1, lhs_contract_dims[contract_trace_idx]));
      contract_trace_idx++;
    } else {
      rhs_eq += c;
      c++;
    }
  }

  // Create out_eq by merging lhs and rhs.
  // In XlaDotv2 style - batch dim - leftover from lhs - leftover from rhs.
  for (const char c : lhs_batch_dims) {
    absl::StrAppend(&out_eq, std::string(1, c));
  }
  for (const char c : lhs_eq) {
    if (!absl::StrContains(out_eq, c) && !absl::StrContains(rhs_eq, c)) {
      absl::StrAppend(&out_eq, std::string(1, c));
    }
  }
  for (const char c : rhs_eq) {
    if (!absl::StrContains(out_eq, c) && !absl::StrContains(lhs_eq, c)) {
      absl::StrAppend(&out_eq, std::string(1, c));
    }
  }

  return absl::StrCat(lhs_eq, ",", rhs_eq, "->", out_eq);
}

Value CreateEinsumOpFromXlaDotV2Op(OpBuilder& builder, const Location loc,
                                   Value lhs, Value rhs, Value output,
                                   StringAttr dot_dimension_numbers_str) {
  xla::DotDimensionNumbers dot_dimension_numbers;
  dot_dimension_numbers.ParseFromString(dot_dimension_numbers_str.str());
  SmallVector<Value> input_arguments = {lhs, rhs};
  const int lhs_rank = mlir::cast<ShapedType>(lhs.getType()).getShape().size();
  const int rhs_rank = mlir::cast<ShapedType>(rhs.getType()).getShape().size();

  const std::string einsum_equation =
      CreateEinsumEquation(dot_dimension_numbers, lhs_rank, rhs_rank);

  return builder.create<TF::EinsumOp>(loc, output.getType(), input_arguments,
                                      builder.getStringAttr(einsum_equation));
}

// Restores the collapsed dimensions to the `tensor_type`. `collapsed_dims`
// designate the dimension indices that were collapsed to produce `tensor_type`.
// The restored dimensions' sizes are 1, according to the semantics of
// `XlaGatherOp (https://www.tensorflow.org/xla/operation_semantics#gather). The
// resulting type's shape has `tensor_type.size() + collapsed_dims.size()`
// dimensions.
RankedTensorType RestoreCollapsedDimensions(
    const RankedTensorType tensor_type,
    const absl::flat_hash_set<int64_t>& collapsed_dims) {
  ArrayRef<int64_t> original_tensor_shape = tensor_type.getShape();
  const int output_tensor_rank =
      original_tensor_shape.size() + collapsed_dims.size();
  auto shape_itr = tensor_type.getShape().begin();

  // Populate the dimensions of the output shape, including the restored
  // dimensions.
  SmallVector<int64_t> output_shape(output_tensor_rank);
  for (int i = 0; i < output_tensor_rank; i++) {
    if (collapsed_dims.contains(i)) {
      // The collapsed dimension's size should have been 1, so it restores the
      // dimension with size 1.
      output_shape[i] = 1;
    } else {
      output_shape[i] = *shape_itr;
      shape_itr++;
    }
  }

  return RankedTensorType::get(output_shape, tensor_type.getElementType());
}

// Determines the output type of the `SliceOp` when it is being inserted in
// place of a `XlaGatherOp`. When the dimensions of `xla_gather_op_output_type`
// is known, the `collapsed_dims` are restored. `xla_gather_op_output_type` is
// the result of collapsing the `collapsed_dims`, but the `SliceOp`'s output
// should not have the dimensions collapsed already. Returns
// `xla_gather_op_output_type` unchanged if the rank is unknown.
//
// Examples:
//   * If `xla_gather_op_output_type` == tensor<*xf32>, then it returns:
//     tensor<*xf32>.
//   * If `xla_gather_op_output_type` == tensor<3x5xi32> and `collapsed_dims` ==
//     {0}, then it returns: tensor<1x3x5xi32>.
//   * If `xla_gather_op_output_type` == tensor<3x5xf32> and `collapsed_dims` ==
//     {1, 3}, then it returns: tensor<3x1x5x1xf32>.
Type GetSliceOpOutputType(Type xla_gather_op_output_type,
                          const absl::flat_hash_set<int64_t>& collapsed_dims) {
  if (auto ranked_output_type =
          mlir::dyn_cast<RankedTensorType>(xla_gather_op_output_type);
      ranked_output_type) {
    return RestoreCollapsedDimensions(ranked_output_type, collapsed_dims);
  }

  return xla_gather_op_output_type;
}

// TODO (b/275225582): Supports Xla Gather op in general case.
bool IsXlaGatherWithoutBatch(Value operand, Value start_indices) {
  auto operand_type = mlir::dyn_cast_or_null<ShapedType>(operand.getType());
  auto start_indices_type =
      mlir::dyn_cast_or_null<ShapedType>(start_indices.getType());
  if (start_indices_type == nullptr || operand_type == nullptr) return false;
  return start_indices_type.getShape().size() == 1;
}

Value CreateSliceAndReshapeOpFromXlaGatherOpWithoutBatch(
    OpBuilder& builder, const Location loc, Value operand, Value start_indices,
    Value slice_sizes, Value output, StringAttr dimension_numbers_str) {
  // Reads dimension numbers.
  xla::GatherDimensionNumbers dimension_numbers;
  dimension_numbers.ParseFromString(dimension_numbers_str.str());

  // Construct full start_indices with given start_indices and
  // start_index_map.
  const ArrayRef<int64_t> operand_shape =
      mlir::cast<ShapedType>(operand.getType()).getShape();
  const int64_t operand_rank = operand_shape.size();

  // Fills zeros if start_index is not given in start_indices.
  Value empty_start_indices = builder.create<TF::FillOp>(
      loc, RankedTensorType::get({operand_rank}, builder.getI64Type()),
      /*shape=*/Create1DConstValue<int64_t>(builder, loc, {operand_rank}),
      /*value=*/CreateScalarConstValue<int64_t>(builder, loc, 0));

  // Converts start_index_map proto to tensor.
  const int64_t index_map_size = dimension_numbers.start_index_map().size();
  SmallVector<int64_t> indices(index_map_size);
  for (int64_t i = 0; i < index_map_size; i++) {
    indices[i] = dimension_numbers.start_index_map()[i];
  }

  // Fill elements from start_indices with start_index_map
  Value scattered_start_indices = builder.create<TF::TensorScatterUpdateOp>(
      loc, empty_start_indices,
      /*indices=*/
      builder.create<TF::ReshapeOp>(
          loc, RankedTensorType::get({index_map_size, 1}, builder.getI64Type()),
          Create1DConstValue<int64_t>(builder, loc, indices),
          Create1DConstValue<int64_t>(builder, loc, {index_map_size, 1})),
      /*value=*/
      builder.create<TF::CastOp>(
          loc,
          RankedTensorType::get(
              mlir::cast<ShapedType>(start_indices.getType()).getShape(),
              builder.getI64Type()),
          start_indices));

  absl::flat_hash_set<int64_t> collapsed_dims;
  collapsed_dims.insert(dimension_numbers.collapsed_slice_dims().begin(),
                        dimension_numbers.collapsed_slice_dims().end());

  // Slice operand by constructed start_indices and slice_sizes.
  auto slice_op = builder.create<TF::SliceOp>(
      loc, GetSliceOpOutputType(output.getType(), collapsed_dims), operand,
      /*start_indices=*/scattered_start_indices,
      /*slice_sizes=*/
      builder.create<TF::CastOp>(
          loc,
          RankedTensorType::get(
              mlir::cast<ShapedType>(slice_sizes.getType()).getShape(),
              builder.getI64Type()),
          slice_sizes));

  // Collapses dimensions by reshaping.
  SmallVector<int64_t> new_shape(operand_rank - collapsed_dims.size());
  for (int64_t i = 0, j = 0; i < operand_rank; i++) {
    if (!collapsed_dims.contains(i)) {
      new_shape[j++] = operand_shape[i];
    }
  }
  if (!new_shape.empty()) new_shape[0] = -1;
  return builder.create<TF::ReshapeOp>(
      loc, output.getType(), slice_op,
      Create1DConstValue(builder, loc, new_shape));
}

bool IsPrecisionEmpty(StringAttr prec_str) {
  xla::PrecisionConfig prec;
  prec.ParseFromString(prec_str.str());
  return !prec.operand_precision_size();
}

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/convert_tf_xla_op_to_tf_op.inc"

void ConvertTfXlaOpToTfOpPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  auto func = getOperation();

  // The pattern includes
  // - Converting XlaDotV2Op to EinsumOp
  // - Converting XlaGatherOp to SliceOp
  RewritePatternSet patterns(ctx);
  populateWithGenerated(patterns);

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    func.emitError() << "quant-converting-tf-xla-op-to-tf-op failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertTfXlaOpToTfOpPass() {
  return std::make_unique<ConvertTfXlaOpToTfOpPass>();
}

static PassRegistration<ConvertTfXlaOpToTfOpPass> pass;

}  // namespace quant
}  // namespace mlir
