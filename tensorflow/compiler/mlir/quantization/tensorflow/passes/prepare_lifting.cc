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

#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/remove_identity_op_pattern.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/einsum.h"

namespace mlir {
namespace quant {
namespace {

class PrepareLiftingPass
    : public PassWrapper<PrepareLiftingPass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareLiftingPass)

  PrepareLiftingPass() = default;

  explicit PrepareLiftingPass(const OpSet op_set) : op_set_(op_set) {}

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
  OpSet op_set_;
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
  llvm::SmallVector<int64_t> new_shape;
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
