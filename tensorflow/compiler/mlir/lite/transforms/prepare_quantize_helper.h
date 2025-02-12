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

// Transform pass for LSTMs.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_QUANTIZE_HELPER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_QUANTIZE_HELPER_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/tools/optimize/operator_property.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/FakeQuantSupport.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/common/uniform_quantized_types.h"
#include "tensorflow/core/framework/types.pb.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass for LSTM.
//
namespace mlir {
namespace TFL {

constexpr double power_of_two_scale = 32768.0;

// Same with the ordering of //tensorflow/compiler/mlir/lite/ir/tfl_ops.td
constexpr const char* intermediate_attributes[] = {
    "input_to_input_intermediate", "input_to_forget_intermediate",
    "input_to_cell_intermediate", "input_to_output_intermediate",
    "effective_hidden_scale_intermediate"};

// Calculates the minimum power of two that is not less than the value.
double PowerOfTwoBound(double value);

tensorflow::DataType GetQuantizedInferenceType(bool is_signed,
                                               int activation_number_of_bits);

// Returns the element type of LSTM's intermediate tensor designated by the
// index.
template <typename LstmOp>
inline QuantizedType GetIntermediateElementType(LstmOp op, int tensor_index) {
  if (tensor_index < 0 || tensor_index > 4) return nullptr;
  TypeAttr attr = op->template getAttrOfType<TypeAttr>(
      intermediate_attributes[tensor_index]);
  if (!attr) {
    return nullptr;
  }
  return QuantizedType::getQuantizedElementType(attr.getValue());
}

namespace operator_property = ::tflite::optimize::operator_property;
using Q = quantfork::QuantizeCastOp;
using DQ = quantfork::DequantizeCastOp;

template <typename LstmOp>
LogicalResult GetLstmProperty(LstmOp op,
                              operator_property::OpVariant* lstm_variant,
                              operator_property::OperatorProperty* op_property,
                              int activation_number_of_bits = 8) {
  if (llvm::isa<TFL::LSTMOp>(op.getOperation())) {
    lstm_variant->op_code = tflite::BuiltinOperator_LSTM;
  } else if (llvm::isa<TFL::UnidirectionalSequenceLSTMOp>(op.getOperation())) {
    lstm_variant->op_code =
        tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM;
  } else {
    op.emitError("ConvertLstmStatsToQDQs pass only supports LSTMs.");
    return failure();
  }
  lstm_variant->use_projection =
      !mlir::isa<NoneType>(op.getProjectionWeights().getType());
  lstm_variant->use_peephole =
      !mlir::isa<NoneType>(op.getCellToOutputWeights().getType());
  lstm_variant->use_layer_norm =
      !mlir::isa<NoneType>(op.getForgetLayerNormCoefficients().getType());

  *op_property = operator_property::GetOperatorProperty(
      *lstm_variant, activation_number_of_bits);

  // TODO(b/176258587) move this to operator_property.cc if this is needed in
  // other components, too.
  bool use_cifg = mlir::isa<NoneType>(op.getInputToInputWeights().getType());
  if (use_cifg) {
    const absl::flat_hash_set<int> cifg_non_inputs = {1, 5, 9, 12, 20};
    const int cifg_non_intermediate = 0;
    op_property->inputs.erase(
        std::remove_if(
            op_property->inputs.begin(), op_property->inputs.end(),
            [&](std::pair<int, operator_property::TensorProperty> input) {
              return cifg_non_inputs.find(input.first) != cifg_non_inputs.end();
            }),
        op_property->inputs.end());
    op_property->intermediates.erase(
        std::remove_if(op_property->intermediates.begin(),
                       op_property->intermediates.end(),
                       [&](std::pair<int, operator_property::TensorProperty>
                               intermediate) {
                         return intermediate.first == cifg_non_intermediate;
                       }),
        op_property->intermediates.end());
  }
  return success();
}

template <typename SourceOp>
class PrepareLstmOutputScale : public OpRewritePattern<SourceOp> {
 public:
  explicit PrepareLstmOutputScale(MLIRContext* context)
      : OpRewritePattern<SourceOp>(context) {}
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter& rewriter) const override {
    operator_property::OpVariant lstm_variant;
    operator_property::OperatorProperty lstm_property;

    if (failed(GetLstmProperty(op, &lstm_variant, &lstm_property))) {
      return failure();
    }
    if (lstm_property.restrict_scale.size() != 1) {
      op.emitError() << "The LSTM's operator property expects exactly one "
                     << "restrict scale requirement. Got "
                     << lstm_property.restrict_scale.size()
                     << " restrict scale requirements.";
      return failure();
    }

    // Use same scale for input and output specified in restrict_scale.
    const std::vector<int>& tensors = lstm_property.restrict_scale[0];
    if (tensors.size() != 2) {
      op.emitError(
          "Unexpected restricted_scale from operator property."
          " Should only have a pair of indices.");
      return failure();
    }
    return processRestrictScale(op, tensors[0], tensors[1], rewriter);
  }

 private:
  // For LSTM's recurrent input activation and output, they are quantized with
  // the collective range of both tensors, because theoretically the input
  // activation value for the very first inference is not reflected in the
  // output and the input activation is not captured.
  LogicalResult processRestrictScale(SourceOp op, int input_index,
                                     int output_index,
                                     PatternRewriter& rewriter) const {
    assert(output_index == 0);
    if (!op.getResult().hasOneUse()) {
      op.emitError()
          << "output " << output_index
          << " should have only one use, which should be quant.stats.";
      return failure();
    }

    llvm::SmallVector<quantfork::StatisticsOp, 2> stats_ops = {
        llvm::dyn_cast_or_null<quantfork::StatisticsOp>(
            op.getOperand(input_index).getDefiningOp()),
        llvm::dyn_cast_or_null<quantfork::StatisticsOp>(
            *op.getResult().getUsers().begin()),
    };

    if (!stats_ops[0] || !stats_ops[1]) {
      return failure();  // Already converted to Q-DQ pair.
    }

    llvm::SmallVector<llvm::APFloat, 4> min_max_values;

    for (auto& stats_op : stats_ops) {
      auto values =
          mlir::dyn_cast<DenseFPElementsAttr>(stats_op.getLayerStats())
              .getValues<llvm::APFloat>();
      min_max_values.insert(min_max_values.end(), values.begin(), values.end());
    }

    // min and max values of two stats are already the same.
    if (min_max_values[0] == min_max_values[2] &&
        min_max_values[1] == min_max_values[3]) {
      return failure();
    }

    mlir::ElementsAttr layer_stats = mlir::DenseFPElementsAttr::get(
        mlir::RankedTensorType::get({2}, rewriter.getF32Type()),
        {llvm::minimum(min_max_values[0], min_max_values[2]),
         llvm::maximum(min_max_values[1], min_max_values[3])});
    mlir::ElementsAttr axis_stats;
    mlir::IntegerAttr axis;
    for (auto& stats_op : stats_ops) {
      rewriter.setInsertionPointAfter(stats_op);
      rewriter.replaceOpWithNewOp<quantfork::StatisticsOp>(
          stats_op, stats_op.getArg(), layer_stats, axis_stats, axis);
    }
    return success();
  }
};

template <typename SourceOp>
class ConvertOpStatsToQDQs : public OpRewritePattern<SourceOp> {
 public:
  explicit ConvertOpStatsToQDQs(MLIRContext* context,
                                const quant::QuantizationSpecs& quant_specs,
                                PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(context, benefit),
        quant_specs_(quant_specs) {}

 protected:
  quant::QuantizationSpecs quant_specs_;

  LogicalResult processInputs(
      SourceOp op, const operator_property::OpVariant& op_variant,
      const operator_property::OperatorProperty& op_property,
      PatternRewriter& rewriter) const {
    for (auto& enumerated_inputs : op_property.inputs) {
      int index = enumerated_inputs.first;
      auto& tensor_property = enumerated_inputs.second;

      Value input = op.getOperand(index);

      if (input.getDefiningOp() == nullptr) continue;

      // TODO(b/172517537): make this work with non-PTQ case.
      if (llvm::isa<func::ConstantOp, arith::ConstantOp, TFL::ConstOp>(
              input.getDefiningOp())) {
        // Tensors with derived scale are biases, and handled in propagation.
        if (tensor_property.use_derived_scale) continue;
        // For weights, use quantization scale inferred from the values.
        if (failed(processConstantOp(op, input.getDefiningOp(), index,
                                     tensor_property, rewriter))) {
          return failure();
        }
      } else {
        if (auto stats_op = llvm::dyn_cast<quantfork::StatisticsOp>(
                input.getDefiningOp())) {
          if (failed(replaceStatsOp(op, stats_op, index, tensor_property,
                                    rewriter))) {
            return failure();
          }
        } else if (!llvm::isa<DQ>(input.getDefiningOp()) &&
                   !llvm::isa<SameScalesOpInterface, FixedOutputRangeInterface>(
                       input.getDefiningOp())) {
          // Continue if StatisticsOp is already converted to Q-DQ pair, or
          // stats op is not immediately available to the input because either
          // it's connected to ops with same scale requirements or it has
          // fixed output range.
          // TODO(b/172517537): make this work with non-PTQ case.
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult processConstantOp(
      SourceOp op, Operation* const_op, int input_index,
      const operator_property::TensorProperty& tensor_property,
      PatternRewriter& rewriter) const {
    // Non-float tensors are neither weights nor require quantization.
    auto type = mlir::dyn_cast<ShapedType>(const_op->getResult(0).getType());
    if (!type || !mlir::isa<FloatType>(type.getElementType())) return success();

    DenseFPElementsAttr attr;
    if (!matchPattern(const_op->getResult(0), m_Constant(&attr))) {
      const_op->emitError("Not a constant op.");
      return failure();
    }

    UniformQuantizedType quant_type = nullptr;
    // When the number of bits is 10 (instead of 16), quantize the tensor to
    // [-512, 512], instead of [-32767, 32767].
    // For now this behavior is specific for SVDF, where 6 bits are reserved for
    // the reduce operation after element-wise multiplication between state and
    // time weights.
    if (tensor_property.number_of_bits == 10) {
      SmallVector<double, 4> mins(1, std::numeric_limits<double>::max());
      SmallVector<double, 4> maxs(1, std::numeric_limits<double>::min());
      // Computes the effective min/max values of the attribute values.
      quant::ExtractMinMaxFromAttr(attr, /*dim_size=*/1, /*slice_size=*/1,
                                   /*symmetric=*/true, mins, maxs);
      double scale = maxs[0] / -llvm::minIntN(tensor_property.number_of_bits);
      quant_type = UniformQuantizedType::getChecked(
          const_op->getLoc(), quant::QuantizationFlags::Signed,
          rewriter.getIntegerType(16), attr.getType().getElementType(), scale,
          /*zeroPoint=*/0, llvm::minIntN(10), -llvm::minIntN(10));
    } else {
      quant_type = mlir::dyn_cast<quant::UniformQuantizedType>(
          quant::GetUniformQuantizedTypeForWeight(
              attr, /*symmetric=*/true,
              /*num_bits=*/tensor_property.number_of_bits,
              /*is_signed=*/true,
              /*narrow_range=*/true, quant_specs_.legacy_float_scale));
    }
    if (!quant_type) {
      const_op->emitError("Failed to get quantized type");
      return failure();
    }

    // TODO(b/172517537): duplicate the constant when the bias is shared.
    Type expressed_type = const_op->getResult(0).getType();
    Type cast_type = quant_type.castFromExpressedType(expressed_type);
    rewriter.setInsertionPointAfter(const_op);
    auto q = rewriter.create<Q>(const_op->getLoc(), cast_type,
                                const_op->getResult(0));
    auto dq = rewriter.create<DQ>(const_op->getLoc(), expressed_type, q);
    op.setOperand(input_index, dq.getResult());
    return success();
  }

  LogicalResult replaceStatsOp(
      SourceOp op, quantfork::StatisticsOp stats_op, int input_index,
      const operator_property::TensorProperty& tensor_property,
      PatternRewriter& rewriter) const {
    if (tensor_property.state_tensor && !stats_op.getResult().hasOneUse()) {
      // TODO(b/172517537): check if other tensors should go through this
      // check too.
      op.emitError() << "Input tensor [" << input_index
                     << "] is a state tensor, but has more than one use.";
      return failure();
    }
    auto stats = mlir::dyn_cast<DenseFPElementsAttr>(stats_op.getLayerStats());
    if (!stats || stats.getNumElements() != 2) {
      stats_op.emitError("Stats should have 2 values.");
      return failure();
    }
    quant::QuantizedType quant_type;
    double min = FloatAttr::getValueAsDouble(stats.getValues<APFloat>()[0]);
    double max = FloatAttr::getValueAsDouble(stats.getValues<APFloat>()[1]);
    // Make sure the range includes zero.
    min = std::min(min, 0.0);
    max = std::max(max, 0.0);
    Type expressed = getElementTypeOrSelf(stats_op.getType());

    if (tensor_property.extend_to_power_of_two) {
      if (tensor_property.number_of_bits != 16) {
        op.emitError(
            "extended power of 2 scale is only supported for 16-bit"
            " quantization.");
        return failure();
      }

      double bound = PowerOfTwoBound(std::max(std::abs(min), std::abs(max)));
      // Set flags to 1 for signed type.
      quant_type = UniformQuantizedType::getChecked(
          op.getLoc(), quant::QuantizationFlags::Signed,
          rewriter.getIntegerType(tensor_property.number_of_bits), expressed,
          /*scale=*/bound / -llvm::minIntN(tensor_property.number_of_bits),
          /*zeroPoint=*/0, llvm::minIntN(tensor_property.number_of_bits),
          llvm::maxIntN(tensor_property.number_of_bits));
    } else {
      // int16 uses range [-32767, 32767]
      if (tensor_property.number_of_bits == 16) {
        max = std::max(std::abs(min), std::abs(max));
        min = -max;
        quant_type = quantfork::fakeQuantAttrsToType(
            op.getLoc(), tensor_property.number_of_bits, min, max,
            /*narrowRange=*/true, expressed,
            /*isSigned=*/true);
      } else {
        quant_type = quantfork::fakeQuantAttrsToType(
            op.getLoc(), tensor_property.number_of_bits, min, max,
            /*narrowRange=*/false, expressed,
            /*isSigned=*/true);
      }
      if (quant_specs_.legacy_float_scale) {
        quant_type = quant::DownCastScale(quant_type, min, max, op.getLoc());
      }
    }
    rewriter.setInsertionPointAfter(stats_op);
    Type result_type = quant_type.castFromExpressedType(stats_op.getType());
    auto q =
        rewriter.create<Q>(stats_op.getLoc(), result_type, stats_op.getArg());
    rewriter.replaceOpWithNewOp<DQ>(stats_op, stats_op.getType(), q);
    return success();
  }
};

// Quantize LSTM according to its quantization recipe.
template <typename SourceOp>
class ConvertLstmStatsToQDQs : public ConvertOpStatsToQDQs<SourceOp> {
 public:
  ConvertLstmStatsToQDQs(MLIRContext* context,
                         const quant::QuantizationSpecs& quant_specs)
      : ConvertOpStatsToQDQs<SourceOp>(context, quant_specs),
        activation_number_of_bits_(quant_specs.GetQuantizationTypeWidth()) {}
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter& rewriter) const override {
    operator_property::OpVariant lstm_variant;
    operator_property::OperatorProperty lstm_property;
    if (failed(GetLstmProperty(op, &lstm_variant, &lstm_property,
                               activation_number_of_bits_))) {
      return failure();
    }

    if (failed(processIntermediates(op, lstm_variant, lstm_property)) ||
        failed(ConvertOpStatsToQDQs<SourceOp>::processInputs(
            op, lstm_variant, lstm_property, rewriter))) {
      return failure();
    }

    return success();
  }

 private:
  LogicalResult processIntermediates(
      SourceOp op, const operator_property::OpVariant& lstm_variant,
      const operator_property::OperatorProperty& lstm_property) const {
    for (auto& enumerated_intermediates : lstm_property.intermediates) {
      int index = enumerated_intermediates.first;
      auto& tensor_property = enumerated_intermediates.second;
      // intermediate tensors 0, 1, 2, 3 are only used with layer normalization.
      if (!lstm_variant.use_layer_norm && index != 4) {
        continue;
      }

      TypeAttr attr =
          op->template getAttrOfType<TypeAttr>(intermediate_attributes[index]);
      auto quant_type = GetIntermediateElementType<SourceOp>(op, index);
      if (!quant_type) {
        // intermediate tensor 4 is optional, unless the LSTM uses projection.
        if (index == 4 && !lstm_variant.use_projection) {
          return success();
        }
        op.emitError() << intermediate_attributes[index]
                       << " is not quantized.";
        return failure();
      }
      auto calibrated_type =
          mlir::dyn_cast<quant::CalibratedQuantizedType>(quant_type);
      if (!calibrated_type) {
        int num_storage_bits = quant_type.getStorageTypeIntegralWidth();
        if (tensor_property.number_of_bits != num_storage_bits) {
          op.emitError() << intermediate_attributes[index]
                         << " is expected to be quantized with "
                         << tensor_property.number_of_bits << " bits, but got "
                         << num_storage_bits << " bits instead.";
          return failure();
        }
        continue;  // skip if it is already quantized.
      }
      quant::UniformQuantizedType qtype;
      if (tensor_property.number_of_bits == 8) {
        qtype = quantfork::fakeQuantAttrsToType(
            op.getLoc(), tensor_property.number_of_bits,
            calibrated_type.getMin(), calibrated_type.getMax(),
            /*narrowRange=*/false, calibrated_type.getExpressedType(),
            /*isSigned=*/this->quant_specs_.IsSignedInferenceType());
        if (this->quant_specs_.legacy_float_scale) {
          qtype = mlir::cast<UniformQuantizedType>(
              quant::DownCastScale(qtype, calibrated_type.getMin(),
                                   calibrated_type.getMax(), op.getLoc()));
        }
      } else if (tensor_property.number_of_bits == 16) {
        double max = std::max(std::abs(calibrated_type.getMin()),
                              std::abs(calibrated_type.getMax()));
        qtype = quantfork::fakeQuantAttrsToType(
            op.getLoc(), tensor_property.number_of_bits, -max, max,
            /*narrowRange=*/true, calibrated_type.getExpressedType(),
            /*isSigned=*/true);
      } else {
        op.emitError() << "Unsupported quantization bits: "
                       << tensor_property.number_of_bits;
        return failure();
      }
      op->setAttr(intermediate_attributes[index],
                  TypeAttr::get(qtype.castFromExpressedType(
                      qtype.castToExpressedType(attr.getValue()))));
    }
    return success();
  }

  int activation_number_of_bits_;
};

// Returns a function that returns the quantized type of a bias input.
// The scale of bias is a multiplication of given scale and scales from the
// quantization type of other operands.
inline quant::AccumulatorScaleFunc GetUniformQuantizedTypeForBiasWithScale(
    double scale) {
  return [=](const std::vector<quant::QuantParams>& quant_params,
             const int adjusted_quant_dim,
             const bool legacy_float_scale) -> quant::QuantParams {
    if (auto qtype = mlir::dyn_cast_or_null<UniformQuantizedType>(
            quant::GetUniformQuantizedTypeForBias(
                quant_params, legacy_float_scale, adjusted_quant_dim))) {
      return quant::UniformQuantizedType::get(
          qtype.getFlags(), qtype.getStorageType(), qtype.getExpressedType(),
          qtype.getScale() * scale, qtype.getZeroPoint(),
          qtype.getStorageTypeMin(), qtype.getStorageTypeMax());
    }
    return {};
  };
}

// Returns quantization spec for LSTMs based on their operator properties.
template <typename LstmOp>
std::unique_ptr<quant::OpQuantSpec> GetLstmOpQuantSpec(LstmOp op) {
  operator_property::OpVariant lstm_variant;
  operator_property::OperatorProperty lstm_property;
  if (failed(GetLstmProperty(op, &lstm_variant, &lstm_property))) {
    return nullptr;
  }

  auto spec = std::make_unique<quant::OpQuantSpec>();

  for (const auto& enumerated_inputs : lstm_property.inputs) {
    int index = enumerated_inputs.first;
    auto& tensor_property = enumerated_inputs.second;
    if (tensor_property.use_derived_scale) {
      double scale = 1.0;
      for (int tensor_index :
           tensor_property.derived_scale.intermediate_tensors) {
        auto quant_type = GetIntermediateElementType<LstmOp>(op, tensor_index);
        if (!quant_type ||
            !mlir::isa<quant::UniformQuantizedType>(quant_type)) {
          op->emitError() << "While processing derived scale, intermediate "
                          << intermediate_attributes[tensor_index]
                          << " is not quantized.";
          return nullptr;
        }
        scale *=
            mlir::dyn_cast<quant::UniformQuantizedType>(quant_type).getScale();
      }
      for (float factor : tensor_property.derived_scale.factors) {
        scale *= factor;
      }
      spec->biases_params.emplace(
          index,
          std::make_pair(tensor_property.derived_scale.input_tensors,
                         GetUniformQuantizedTypeForBiasWithScale(scale)));
    }
  }
  return spec;
}

class ConvertSvdfStatsToQDQs : public ConvertOpStatsToQDQs<TFL::SVDFOp> {
 public:
  explicit ConvertSvdfStatsToQDQs(
      MLIRContext* context, const quant::QuantizationSpecs& quant_specs_param)
      : ConvertOpStatsToQDQs<TFL::SVDFOp>(context, quant_specs_param) {}
  LogicalResult matchAndRewrite(TFL::SVDFOp op,
                                PatternRewriter& rewriter) const override {
    operator_property::OpVariant op_variant;
    op_variant.op_code = tflite::BuiltinOperator_SVDF;
    auto op_property = operator_property::GetOperatorProperty(op_variant);
    return ConvertOpStatsToQDQs<TFL::SVDFOp>::processInputs(
        op, op_variant, op_property, rewriter);
  }
};

class PropagateTransposedPerAxisQuantDim
    : public OpRewritePattern<TFL::TransposeOp> {
 public:
  explicit PropagateTransposedPerAxisQuantDim(MLIRContext* context)
      : OpRewritePattern<TFL::TransposeOp>(context) {}
  LogicalResult matchAndRewrite(TFL::TransposeOp transpose_op,
                                PatternRewriter& rewriter) const override {
    // Check if the quantization is per-axis
    auto dq_op = dyn_cast_or_null<quantfork::DequantizeCastOp>(
        transpose_op.getOperand(0).getDefiningOp());
    if (!dq_op) return failure();
    auto q_op = dyn_cast_or_null<quantfork::QuantizeCastOp>(
        dq_op.getOperand().getDefiningOp());
    if (!q_op) return failure();
    auto qtype =
        mlir::cast<TensorType>(dq_op.getArg().getType()).getElementType();
    auto aqtype = dyn_cast_or_null<quant::UniformQuantizedPerAxisType>(qtype);
    if (!aqtype) return failure();

    // Return if the result of TransposeOp is already quantized
    if (!transpose_op.getResult().hasOneUse()) return failure();
    auto next_op = *transpose_op.getResult().getUsers().begin();
    if (dyn_cast_or_null<quantfork::QuantizeCastOp>(next_op)) return failure();

    auto input_type = mlir::cast<ShapedType>(transpose_op.getInput().getType());
    auto perm_type = mlir::cast<ShapedType>(transpose_op.getPerm().getType());
    if (input_type.hasStaticShape() && perm_type.hasStaticShape()) {
      if (perm_type.getNumElements() != input_type.getRank()) {
        return transpose_op.emitOpError(
            "perm tensor elements size is not equal to input tensor rank");
      }
    }

    // Get permutation axes of the TransposeOp
    DenseIntElementsAttr perm;
    if (!matchPattern(transpose_op.getPerm(), m_Constant(&perm))) {
      return failure();
    }

    SmallVector<int64_t, 4> axes;
    for (const auto& axis_int : perm.getValues<APInt>()) {
      int64_t axis = axis_int.getSExtValue();
      if (axis < 0) {
        axis += input_type.getRank();
      }
      if (axis < 0 || (input_type.hasRank() && axis >= input_type.getRank())) {
        return transpose_op.emitOpError("perm must be in [-rank, rank)");
      }
      if (std::count(axes.begin(), axes.end(), axis) > 0) {
        return transpose_op.emitOpError("perm cannot have duplicated axis");
      }
      axes.push_back(axis);
    }

    // Find what the quantized dimension has been transposed to
    int new_out_quant_dim = -1;
    for (int i = 0; i < axes.size(); ++i) {
      if (axes[i] == aqtype.getQuantizedDimension()) {
        new_out_quant_dim = i;
        break;
      }
    }
    if (new_out_quant_dim == -1) {
      return transpose_op.emitOpError(
          "new quantization dimension not found in perm");
    }

    // Insert a QDQ pair with the new quantized dimension after TransposeOp
    auto new_qtype = quant::CreateI8F32UniformQuantizedPerAxisType(
        transpose_op.getLoc(), *rewriter.getContext(), aqtype.getScales(),
        aqtype.getZeroPoints(), new_out_quant_dim, /*narrow_range=*/true);
    auto new_tensor_type = RankedTensorType::getChecked(
        transpose_op.getLoc(), transpose_op.getType().getShape(), new_qtype);
    rewriter.setInsertionPointAfter(transpose_op);
    auto new_q_op = rewriter.create<quantfork::QuantizeCastOp>(
        transpose_op.getLoc(), new_tensor_type, q_op.getArg());
    auto new_dq_op = rewriter.create<quantfork::DequantizeCastOp>(
        new_q_op.getLoc(), transpose_op.getResult().getType(),
        new_q_op.getResult());
    transpose_op.getResult().replaceAllUsesWith(new_dq_op.getResult());
    new_q_op.setOperand(transpose_op.getResult());

    return success();
  }
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_QUANTIZE_HELPER_H_
