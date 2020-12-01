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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_QUANTIZE_LSTM
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_QUANTIZE_LSTM

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/operator_property.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass for LSTM.
//
namespace mlir {
namespace TFL {

// Calculates the minimum power of two that is not less than the value.
inline double power_of_two_bound(double value) {
  return std::pow(2, std::ceil(std::log2(value)));
}

namespace operator_property = ::tflite::optimize::operator_property;

// Quantize recurrent input of LSTM with 16 bits.
template <typename SourceOp, typename Q, typename DQ>
struct ConvertLstmStatsToQDQs : public OpRewritePattern<SourceOp> {
 public:
  explicit ConvertLstmStatsToQDQs(MLIRContext* context)
      : OpRewritePattern<SourceOp>(context, /*benefit=*/2) {}
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter& rewriter) const override {
    operator_property::OpVariant lstm_variant;
    if (llvm::isa<TFL::LSTMOp>(op.getOperation())) {
      lstm_variant.op_code = tflite::BuiltinOperator_LSTM;
    } else if (llvm::isa<TFL::UnidirectionalSequenceLSTMOp>(
                   op.getOperation())) {
      lstm_variant.op_code =
          tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM;
    } else {
      op.emitError("ConvertLstmStatsToQDQs pass only supports LSTMs.");
      return failure();
    }
    lstm_variant.use_projection =
        !op.projection_weights().getType().template isa<NoneType>();
    lstm_variant.use_peephole =
        !op.cell_to_output_weights().getType().template isa<NoneType>();
    lstm_variant.use_peephole =
        !op.cell_to_output_weights().getType().template isa<NoneType>();
    lstm_variant.use_layer_norm =
        !op.forget_layer_norm_coefficients().getType().template isa<NoneType>();

    auto lstm_property = operator_property::GetOperatorProperty(lstm_variant);

    // Same with the ordering of //tensorflow/compiler/mlir/lite/ir/tfl_ops.td
    const std::vector<std::string> intermediate_attributes = {
        "input_to_input_intermediate", "input_to_forget_intermediate",
        "input_to_cell_intermediate", "input_to_output_intermediate",
        "effective_hidden_scale_intermediate"};

    for (auto& enumerated_intermediates : lstm_property.intermediates) {
      int index = enumerated_intermediates.first;
      auto& tensor_property = enumerated_intermediates.second;
      // intermediate tensors 0, 1, 2, 3 are only used with layer normalization.
      if (!lstm_variant.use_layer_norm && index != 4) {
        continue;
      }
      // intermediate tensor 4 is only used with projection.
      if (!lstm_variant.use_projection && index == 4) {
        continue;
      }
      TypeAttr attr =
          op.template getAttrOfType<TypeAttr>(intermediate_attributes[index]);

      if (!attr) {
        op.emitError()
            << op.getOperationName()
            << " requires quantization values for intermediate tensor "
            << intermediate_attributes[index];
        return failure();
      }
      auto quantized_type =
          QuantizedType::getQuantizedElementType(attr.getValue());
      if (!quantized_type) {
        op.emitError() << intermediate_attributes[index]
                       << " is not quantized.";
        return failure();
      }
      auto calibrated_type =
          quantized_type.dyn_cast<quant::CalibratedQuantizedType>();
      if (!calibrated_type) {
        int num_storage_bits = quantized_type.getStorageTypeIntegralWidth();
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
        qtype = quant::fakeQuantAttrsToType(
            op.getLoc(), tensor_property.number_of_bits,
            calibrated_type.getMin(), calibrated_type.getMax(),
            /*narrowRange=*/false, calibrated_type.getExpressedType(),
            /*isSigned=*/false);
      } else if (tensor_property.number_of_bits == 16) {
        double max = std::max(std::abs(calibrated_type.getMin()),
                              std::abs(calibrated_type.getMax()));
        qtype = quant::fakeQuantAttrsToType(
            op.getLoc(), tensor_property.number_of_bits, -max, max,
            /*narrowRange=*/true, calibrated_type.getExpressedType(),
            /*isSigned=*/true);
      } else {
        op.emitError() << "Unsupported quantization bits: "
                       << tensor_property.number_of_bits;
        return failure();
      }

      op.setAttr(intermediate_attributes[index],
                 TypeAttr::get(qtype.castFromExpressedType(
                     qtype.castToExpressedType(attr.getValue()))));
    }

    quant::StatisticsOp stats_op = llvm::dyn_cast_or_null<quant::StatisticsOp>(
        op.input_cell_state().getDefiningOp());
    // Recurrent input is be used within an LSTM, and thus should have one use.
    if (!stats_op || !stats_op.getResult().hasOneUse()) {
      return failure();
    }
    auto stats = stats_op.layerStats().dyn_cast<DenseFPElementsAttr>();
    if (!stats) {
      return failure();
    }

    double max = std::max(
        std::abs(FloatAttr::getValueAsDouble(stats.getValue<APFloat>({0}))),
        std::abs(FloatAttr::getValueAsDouble(stats.getValue<APFloat>({1}))));
    double bound = power_of_two_bound(max);
    Type expressed = stats_op.getType().cast<ShapedType>().getElementType();
    // Set flags to 1 for signed type.
    quant::QuantizedType quant_type = UniformQuantizedType::getChecked(
        quant::QuantizationFlags::Signed,
        IntegerType::get(16, expressed.getContext()), expressed,
        /*scale=*/bound / 32768.0, /*zeroPoint=*/0, llvm::minIntN(16),
        llvm::maxIntN(16), op.getLoc());

    rewriter.setInsertionPointAfter(stats_op);
    Type result_type = quant_type.castFromExpressedType(stats_op.getType());
    auto q = rewriter.create<Q>(stats_op.getLoc(), result_type, stats_op.arg());
    rewriter.replaceOpWithNewOp<DQ>(stats_op, stats_op.getType(), q);
    return success();
  }
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PREPARE_QUANTIZE_LSTM
