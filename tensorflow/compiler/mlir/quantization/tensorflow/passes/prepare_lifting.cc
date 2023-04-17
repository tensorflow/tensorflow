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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/remove_identity_op_pattern.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/einsum.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace mlir {
namespace quant {
namespace {

class PrepareLiftingPass
    : public PassWrapper<PrepareLiftingPass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareLiftingPass)

  PrepareLiftingPass() = default;

  explicit PrepareLiftingPass(OpSet op_set) { op_set_ = op_set; }

  PrepareLiftingPass(const PrepareLiftingPass& other) {
    op_set_ = other.op_set_;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-prepare-lifting";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply graph optimizations such as fusing and constant folding to "
           "prepare lifting.";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, arith::ArithDialect>();
  }

  void runOnOperation() override;

 private:
  Option<OpSet> op_set_{
      *this, "target-opset", llvm::cl::init(OpSet::TF),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};
};

// Check if given indices in `val1` has same number of elements as given
// indices in `val2`.
bool HasEqualElementSize(Value val1, Value val2, ArrayRef<int> val1_indices,
                         ArrayRef<int> val2_indices) {
  ShapedType val1_shape = val1.getType().cast<ShapedType>();
  ShapedType val2_shape = val2.getType().cast<ShapedType>();
  if (!val1_shape.hasRank() || !val2_shape.hasRank()) return false;

  int val1_result = 1;
  int val2_result = 1;
  for (auto idx : val1_indices) {
    if (idx < 0) idx = idx + val1_shape.getRank();
    if (idx >= val1_shape.getRank() || val1_shape.isDynamicDim(idx)) {
      return false;
    }
    val1_result *= val1_shape.getDimSize(idx);
  }

  for (auto idx : val2_indices) {
    if (idx < 0) idx = idx + val2_shape.getRank();
    if (idx >= val2_shape.getRank() || val2_shape.isDynamicDim(idx)) {
      return false;
    }
    val2_result *= val2_shape.getDimSize(idx);
  }

  return val1_result == val2_result;
}

// Matches convolution op with "NHWC" data format or matmul op with false adj_y.
// The list of supported ops in this function is:
// - Conv2DOp
// - Conv3DOp
// - DepthwiseConv2dNativeOp
// - MatMulOp
// - BatchMatMulV2Op
LogicalResult MatchSupportedAffineOp(Operation* op, Value& binding_output,
                                     Value& binding_input,
                                     Value& binding_weight) {
  bool is_supported_affine_op = false;
  if (llvm::isa<TF::Conv2DOp, TF::Conv3DOp, TF::DepthwiseConv2dNativeOp>(op)) {
    if (const auto data_format = op->getAttrOfType<StringAttr>("data_format")) {
      is_supported_affine_op = data_format.getValue().equals("NHWC") ||
                               data_format.getValue().equals("NDHWC");
    }
  } else if (llvm::isa<TF::MatMulOp, TF::BatchMatMulV2Op>(op)) {
    if (const auto adj_y = op->getAttrOfType<BoolAttr>("adj_y")) {
      is_supported_affine_op = !adj_y.getValue();
    }
  }

  if (!is_supported_affine_op) return failure();

  // Bind input, output and weight to the given values.
  binding_output = op->getResult(0);
  binding_input = op->getOperand(0);
  binding_weight = op->getOperand(1);
  return success();
}

// Makes the 1D value broadcastable with the `rhs_shape`.
Value MakeOneDimValueBroadcastable(OpBuilder& builder, Location loc,
                                   Value value, ShapedType rhs_shape) {
  ShapedType value_shape = value.getType().dyn_cast_or_null<ShapedType>();
  if (!value_shape || value_shape.getRank() != 1 ||
      !value_shape.hasStaticShape() || !rhs_shape.hasStaticShape()) {
    return {};
  }

  int64_t num_elements = value_shape.getNumElements();
  SmallVector<int64_t> new_shape;
  for (auto idx : llvm::reverse(llvm::seq<int32_t>(0, rhs_shape.getRank()))) {
    const int64_t rhs_dim = rhs_shape.getDimSize(idx);
    if (num_elements % rhs_dim != 0) {
      return {};
    }
    new_shape.push_back(rhs_dim);
    num_elements = num_elements / rhs_dim;
    if (num_elements == 1) break;
  }
  absl::c_reverse(new_shape);

  auto reshape_op = builder.create<TF::ReshapeOp>(
      loc, value, Create1DConstValue(builder, loc, new_shape));
  return ConstantFoldOpIfPossible(reshape_op).front();
}

// Checks if a value can be symetrically quantized.
bool CanBeSymmetricallyQuantized(Value weight) {
  auto dq_op = weight.getDefiningOp<quantfork::DequantizeCastOp>();
  if (!dq_op) return true;

  auto qtype = dq_op.getArg().getType().cast<TensorType>().getElementType();
  if (auto uniform_type = llvm::dyn_cast_or_null<UniformQuantizedType>(qtype)) {
    return uniform_type.getZeroPoint() == 0;
  } else if (auto per_axis_type =
                 llvm::dyn_cast_or_null<UniformQuantizedPerAxisType>(qtype)) {
    return absl::c_all_of(per_axis_type.getZeroPoints(),
                          [](int64_t x) { return x == 0; });
  }
  return false;
}

// Multiplies two 1D arrays with broadcasting support.
template <typename T>
SmallVector<T> MultiplyTwoArrays(ArrayRef<T> a, ArrayRef<T> b) {
  auto get_value_at = [](ArrayRef<T> v, size_t i) -> T {
    if (v.size() == 1) return v.front();
    return v[i];
  };

  size_t max_size = std::max(a.size(), b.size());
  SmallVector<T> result(max_size);
  for (size_t i : llvm::seq<size_t>(0, max_size)) {
    result[i] = get_value_at(a, i) * get_value_at(b, i);
  }
  return result;
}

// Multiplies the value followed by a FakeQuant op and adjusts the quantization
// params. This funtion only supports symetrically quantized values.
Value MultiplyFakeQuantValue(OpBuilder& builder, Location loc, Value value,
                             Value multiplier) {
  auto dq_op = value.getDefiningOp<quantfork::DequantizeCastOp>();
  if (!dq_op) {
    auto mul_op = builder.create<TF::MulOp>(loc, value, multiplier);
    return ConstantFoldOpIfPossible(mul_op).front();
  }
  auto q_op = dq_op.getArg().getDefiningOp<quantfork::QuantizeCastOp>();
  if (!q_op) return {};

  Value float_value = q_op.getArg();
  Value new_value = builder.create<TF::MulOp>(loc, float_value, multiplier);
  auto new_value_type = new_value.getType().cast<TensorType>();

  // Get multiplier value in double.
  DenseFPElementsAttr multiplier_attr;
  if (!matchPattern(multiplier, m_Constant(&multiplier_attr)) ||
      multiplier_attr.getType().cast<ShapedType>().getRank() > 1) {
    return {};
  }
  std::vector<double> multiplier_values;
  absl::c_transform(multiplier_attr, std::back_inserter(multiplier_values),
                    [](auto v) { return FloatAttr::getValueAsDouble(v); });
  ArrayRef<double> multiplier_array(multiplier_values.data(),
                                    multiplier_values.size());

  // Multiply the quantization parameters by the multiplier.
  QuantizedType new_qtype;
  auto element_type = q_op.getType().cast<TensorType>().getElementType();
  if (auto uniform_type = llvm::dyn_cast<UniformQuantizedType>(element_type)) {
    if (multiplier_attr.isSplat()) {
      double new_scale = multiplier_array.front() * uniform_type.getScale();
      new_qtype = UniformQuantizedType::get(
          uniform_type.getFlags(), uniform_type.getStorageType(),
          uniform_type.getExpressedType(), new_scale,
          uniform_type.getZeroPoint(), uniform_type.getStorageTypeMin(),
          uniform_type.getStorageTypeMax());
    } else {
      auto new_scales =
          MultiplyTwoArrays(multiplier_array, {uniform_type.getScale()});
      int32_t quantized_dim = new_value_type.getRank() - 1;
      auto new_zero_points =
          SmallVector<int64_t>(new_scales.size(), uniform_type.getZeroPoint());
      new_qtype = UniformQuantizedPerAxisType::get(
          uniform_type.getFlags(), uniform_type.getStorageType(),
          uniform_type.getExpressedType(), new_scales, new_zero_points,
          quantized_dim, uniform_type.getStorageTypeMin(),
          uniform_type.getStorageTypeMax());
    }
  } else if (auto per_axis_type =
                 llvm::dyn_cast_or_null<UniformQuantizedPerAxisType>(
                     element_type)) {
    auto new_scales =
        MultiplyTwoArrays(multiplier_array, per_axis_type.getScales());
    new_qtype = UniformQuantizedPerAxisType::get(
        per_axis_type.getFlags(), per_axis_type.getStorageType(),
        per_axis_type.getExpressedType(), new_scales,
        per_axis_type.getZeroPoints(), per_axis_type.getQuantizedDimension(),
        per_axis_type.getStorageTypeMin(), per_axis_type.getStorageTypeMax());
  }

  auto quantize = builder.create<quantfork::QuantizeCastOp>(
      q_op.getLoc(), new_value_type.clone(new_qtype), new_value);
  auto dequantize = builder.create<quantfork::DequantizeCastOp>(
      dq_op.getLoc(), new_value_type, quantize.getResult());
  return ConstantFoldOpIfPossible(dequantize).front();
}

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
  const int lhs_rank =
      lhs.getType().template cast<ShapedType>().getShape().size();
  const int rhs_rank =
      rhs.getType().template cast<ShapedType>().getShape().size();

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
          xla_gather_op_output_type.dyn_cast<RankedTensorType>();
      ranked_output_type) {
    return RestoreCollapsedDimensions(ranked_output_type, collapsed_dims);
  }

  return xla_gather_op_output_type;
}

// TODO (b/275225582): Supports Xla Gather op in general case.
bool IsXlaGatherWithoutBatch(Value operand, Value start_indices) {
  auto operand_type = operand.getType().dyn_cast_or_null<ShapedType>();
  auto start_indices_type =
      start_indices.getType().dyn_cast_or_null<ShapedType>();
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
      operand.getType().cast<ShapedType>().getShape();
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
              start_indices.getType().template cast<ShapedType>().getShape(),
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
              slice_sizes.getType().template cast<ShapedType>().getShape(),
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

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/prepare_lifting.inc"

void PrepareLiftingPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  auto func = getOperation();

  // The pattern includes decomposing batch normalization ops, fusing add/mul
  // with a constant operand to a preceding affine operation.
  RewritePatternSet patterns(ctx);
  populateWithGenerated(patterns);
  patterns.add<RemoveIdentity>(ctx);
  if (op_set_ != OpSet::XLA) {
    // Convert Einsum into BatchMatMul for non-XLA opsets.
    // For the uniform opset, it is requested to maintain the BatchMatmul logic.
    // For the TF opset, since we need to test the effect we remain it as a
    // future work.
    patterns.add<TF::ConvertTFEinsumOp>(ctx);
  }

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError() << "quant-internal-prepare-lifting failed.";
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareLiftingPass(
    const OpSet target_opset) {
  return std::make_unique<PrepareLiftingPass>(target_opset);
}

static PassRegistration<PrepareLiftingPass> pass;

}  // namespace quant
}  // namespace mlir
