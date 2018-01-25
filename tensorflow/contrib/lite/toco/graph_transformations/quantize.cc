/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool SupportsQuantization(const Operator& op) {
  auto type = op.type;
  if (type == OperatorType::kTensorFlowUnsupported) {
    auto* unsupported = static_cast<const TensorFlowUnsupportedOperator*>(&op);
    return unsupported->quantized;
  }
  return type == OperatorType::kConv || type == OperatorType::kDepthwiseConv ||
         type == OperatorType::kFullyConnected ||
         type == OperatorType::kConcatenation ||
         type == OperatorType::kL2Normalization || type == OperatorType::kAdd ||
         type == OperatorType::kAveragePool || type == OperatorType::kMaxPool ||
         type == OperatorType::kLogistic || type == OperatorType::kSoftmax ||
         type == OperatorType::kSqueeze || type == OperatorType::kPad ||
         type == OperatorType::kTensorFlowReshape ||
         type == OperatorType::kMul || type == OperatorType::kSpaceToDepth ||
         type == OperatorType::kDepthToSpace;
}

template <ArrayDataType A>
std::unique_ptr<GenericBuffer> QuantizeBuffer(
    const GenericBuffer& buffer,
    const QuantizationParams& quantization_params) {
  const auto inverse_scale = 1. / quantization_params.scale;
  CHECK(buffer.type == ArrayDataType::kFloat);
  const auto& float_buffer =
      static_cast<const Buffer<ArrayDataType::kFloat>&>(buffer);
  auto* quantized_buffer = new Buffer<A>;
  quantized_buffer->data.resize(float_buffer.data.size());
  const auto qmin = static_cast<int32>(std::numeric_limits<DataType<A>>::min());
  const auto qmax = static_cast<int32>(std::numeric_limits<DataType<A>>::max());
  for (std::size_t i = 0; i < float_buffer.data.size(); i++) {
    const float src_val = float_buffer.data[i];
    double scaled_val;  // Astonishingly, using 'float' degrades accuracy just
                        // enough to make a few tests fail!
    if (quantization_params.scale == 0) {
      CHECK_EQ(src_val, 0) << "The quantization scale for this array is 0, "
                           << "so all its values should be 0.";
      scaled_val = quantization_params.zero_point;
    } else {
      scaled_val = quantization_params.zero_point + inverse_scale * src_val;
    }
    const auto rounded_val = static_cast<int32>(std::round(scaled_val));
    const auto clamped_val = std::min(qmax, std::max(qmin, rounded_val));
    quantized_buffer->data[i] = static_cast<DataType<A>>(clamped_val);
  }
  return std::unique_ptr<GenericBuffer>(quantized_buffer);
}

template <ArrayDataType A>
void QuantizeArray(GraphTransformation* transformation, Model* model,
                   const string& name,
                   const QuantizationParams& quantization_params) {
  auto& array = model->GetArray(name);
  CHECK(array.data_type == ArrayDataType::kFloat);
  CHECK(!array.quantization_params);
  array.GetOrCreateQuantizationParams() = quantization_params;
  if (array.buffer) {
    array.buffer = QuantizeBuffer<A>(*array.buffer, quantization_params);
  }
  array.data_type = A;
  transformation->AddMessageF("Quantized array %s", name);
}

void QuantizeArray(GraphTransformation* transformation, Model* model,
                   const string& name, ArrayDataType quantized_data_type,
                   const QuantizationParams& quantization_params) {
  switch (quantized_data_type) {
    case ArrayDataType::kUint8:
      return QuantizeArray<ArrayDataType::kUint8>(transformation, model, name,
                                                  quantization_params);
    case ArrayDataType::kInt32:
      return QuantizeArray<ArrayDataType::kInt32>(transformation, model, name,
                                                  quantization_params);
    default:
      LOG(FATAL) << "Unhandled case.";
  }
}

const MinMax& GetOrComputeMinMax(Model* model, const string& array_name) {
  auto& array = model->GetArray(array_name);
  // Normally we should have a MinMax recorded on this Array,
  // so we just use it.
  if (array.minmax != nullptr) {
    return *array.minmax;
  }

  // We don't have a MinMax. That's bad news: we need
  // the graph to provide MinMax info for all arrays in order
  // for inference to reproduce faithfully the same quantization
  // error as the training process had.
  //
  // But we still want to support a fallback for constant arrays,
  // just using the plain min and max computed from array elements.
  // We should hopefully never rely on that in production, as that
  // will not give very good accuracy as that typically won't be
  // exactly what the training process used. But it will be useful
  // to allow easily trying out quantization even if the graph
  // lacks some minmax information.
  if (array.buffer != nullptr) {
    LOG(WARNING)
        << "Constant array " << array_name
        << " lacks MinMax information. To make up for that, we will now compute"
        << " the MinMax from actual array elements. That will result in"
        << " quantization parameters that probably do not match whichever "
           "arithmetic"
        << " was used during training, and thus will probably be a cause of "
           "poor"
        << " inference accuracy.";
    CHECK(array.buffer->type == ArrayDataType::kFloat);
    const auto& data = array.GetBuffer<ArrayDataType::kFloat>().data;
    // We always want [min, max] to contain 0.
    float min = 0.f;
    float max = 0.f;
    for (auto val : data) {
      min = std::min(min, val);
      max = std::max(max, val);
    }
    auto& minmax = array.GetOrCreateMinMax();
    minmax.min = min;
    minmax.max = max;
    return minmax;
  }

  LOG(FATAL) << "Array " << array_name
             << " does not have MinMax information, "
                "and is not a constant array. Cannot "
                "proceed with quantization.";
}

bool ChooseQuantizationForOperatorInput(
    GraphTransformation* transformation, Model* model, const Operator& op,
    std::size_t input_index, ArrayDataType* quantized_data_type,
    QuantizationParams* quantization_params) {
  const auto& input = op.inputs[input_index];
  auto& array = model->GetArray(input);
  if (array.data_type != ArrayDataType::kFloat) {
    return false;
  }
  if (op.type == OperatorType::kConv ||
      op.type == OperatorType::kDepthwiseConv ||
      op.type == OperatorType::kFullyConnected) {
    if (input_index == 2) {
      // Quantization of bias vector.
      // We need both of the mandatory inputs (input activations and weights) to
      // have
      // been already quantized.
      const auto& input_activations = model->GetArray(op.inputs[0]);
      const auto& input_weights = model->GetArray(op.inputs[1]);
      if (!input_activations.quantization_params ||
          !input_weights.quantization_params) {
        return false;
      }
      const auto input_activations_scale =
          input_activations.quantization_params->scale;
      const auto input_weights_scale = input_weights.quantization_params->scale;
      quantization_params->scale =
          input_activations_scale * input_weights_scale;
      quantization_params->zero_point = 0;
      *quantized_data_type = ArrayDataType::kInt32;
      transformation->AddMessageF(
          "Input array %s is a bias vector. Choosing quantization params "
          "accordingly.",
          input);
      return true;
    }
  }

  const MinMax& minmax = GetOrComputeMinMax(model, input);
  GetQuantizationParamsFromMinMax<ArrayDataType::kUint8>(model->flags, minmax,
                                                         quantization_params);
  transformation->AddMessageF(
      "For input array %s with min=%g"
      ", max=%g"
      ", chose to quantize as uint8 with zero_point=%d"
      ", scale=%g",
      input, minmax.min, minmax.max, quantization_params->zero_point,
      quantization_params->scale);
  *quantized_data_type = ArrayDataType::kUint8;
  return true;
}

bool IsExactlyRepresentable(double real_value, ArrayDataType data_type,
                            const QuantizationParams& quantization_params) {
  const double scaled_value =
      quantization_params.zero_point + real_value / quantization_params.scale;
  const double fractional_scaled_value =
      scaled_value - std::round(scaled_value);
  if (std::abs(fractional_scaled_value) > 1e-12) {
    return false;
  }
  const double rounded_scaled_value = std::round(scaled_value);
  if (data_type == ArrayDataType::kUint8) {
    if (rounded_scaled_value < 0 || rounded_scaled_value > 255) {
      return false;
    }
  }
  return true;
}

bool ChooseHardcodedQuantizationForOperatorOutput(
    const Operator& op, ArrayDataType* quantized_data_type,
    QuantizationParams* quantization_params) {
  if (op.type == OperatorType::kL2Normalization) {
    // L2Normalization has range: [-1, 1].
    // 0 should be exactly representable, as values will typically be centered
    // around 0, with many values near 0.
    *quantized_data_type = ArrayDataType::kUint8;
    quantization_params->zero_point = 128;
    quantization_params->scale = 1. / 128.;
    CHECK(
        IsExactlyRepresentable(0., *quantized_data_type, *quantization_params));
    return true;
  }
  if ((op.type == OperatorType::kLogistic) ||
      (op.type == OperatorType::kSoftmax)) {
    // Logistic and Softmax have range: [0, 1].
    //
    // For Logistic, 0.5 should be exactly representable, as implementations
    // will typically exploit the symmetry logistic(-x) = 1 - logistic(x), and
    // the glueing of the two halves of the graph will only be seamless if we
    // are accurately representing logistic(0) == 0.5.
    *quantized_data_type = ArrayDataType::kUint8;
    quantization_params->zero_point = 0;
    quantization_params->scale = 1. / 256.;
    CHECK(IsExactlyRepresentable(0.5, *quantized_data_type,
                                 *quantization_params));
    return true;
  }
  return false;
}

bool ChooseQuantizationForOperatorOutput(
    GraphTransformation* transformation, Model* model, const Operator& op,
    std::size_t output_index, ArrayDataType* quantized_data_type,
    QuantizationParams* quantization_params) {
  const auto& output = op.outputs[output_index];
  auto& array = model->GetArray(output);
  if (array.data_type != ArrayDataType::kFloat) {
    return false;
  }
  if (ChooseHardcodedQuantizationForOperatorOutput(op, quantized_data_type,
                                                   quantization_params)) {
    transformation->AddMessageF(
        "Output array %s is produced by a %s operator. Choosing fixed "
        "quantization params accordingly.",
        output, OperatorTypeName(op.type));
    return true;
  }
  if ((op.type == OperatorType::kDepthToSpace) ||
      (op.type == OperatorType::kSpaceToDepth)) {
    // DepthToSpace and SpaceToDepth should preserve the quantization parameters
    // of the input array, as these are simple reshape operations.
    const auto& input_quantization_params =
        model->GetArray(op.inputs[0]).GetQuantizationParams();
    *quantized_data_type = ArrayDataType::kUint8;
    quantization_params->zero_point = input_quantization_params.zero_point;
    quantization_params->scale = input_quantization_params.scale;

    transformation->AddMessageF(
        "Output array %s is produced by a %s operator. Copying quantization "
        "params from input array.",
        output, OperatorTypeName(op.type));
    return true;
  }
  const MinMax& minmax = GetOrComputeMinMax(model, output);
  GetQuantizationParamsFromMinMax<ArrayDataType::kUint8>(model->flags, minmax,
                                                         quantization_params);
  *quantized_data_type = ArrayDataType::kUint8;
  transformation->AddMessageF(
      "For output array %s with min=%g, max=%g"
      ", chose to quantize as uint8 with zero_point=%d"
      ", scale=%g",
      output, minmax.min, minmax.max, quantization_params->zero_point,
      quantization_params->scale);

  return true;
}
}  // namespace

bool Quantize::Run(Model* model, std::size_t op_index) {
  // Our general "quantization" graph transformation consists in replacing
  //   QuantizedInputArrays[] ->
  //     DequantizeOperators[] ->
  //       FloatInputArrays[] ->
  //         Operator ->
  //           FloatOutputArray
  // by
  //   QuantizedInputArrays[] ->
  //     Operator ->
  //       QuantizedOutputArray ->
  //         DequantizeOperator ->
  //           FloatOutputArray
  //
  // In other words, this is pushing Dequantize operators to the right of
  // other operators.
  //

  auto& op = *model->operators[op_index];
  if (op.type == OperatorType::kDequantize ||
      op.type == OperatorType::kFakeQuant) {
    return false;
  }

  // Our assumption here is that the input arrays are already quantized -
  // that is typically the case in models operating on an input bitmap
  // image, and MakeInitialDequantizeOp should have already resolved
  // the handling of the input image as an initial Dequantize op.
  //
  // Thus we are building around the assumption that the graph always starts
  // with a quantized input array, and only after some Dequantize op do we have
  // float arrays. The problem of quantizing the graph thus becomes a problem of
  // pushing Dequantize ops to the right of other ops.
  //
  // Let us just guard this assumption by the following assertion:
  for (const auto& input : op.inputs) {
    if (IsInputArray(*model, input)) {
      const auto& input_array = model->GetArray(input);
      CHECK(input_array.quantization_params);
    }
  }
  if (!SupportsQuantization(op)) {
    LOG(FATAL) << "Unimplemented: this graph contains an operator of type "
               << HelpfulOperatorTypeName(op)
               << " for which the quantized form is not yet implemented. "
                  "Sorry, and patches welcome (that's a relatively fun patch "
                  "to write, mostly providing the actual quantized arithmetic "
                  "code for this op).";
  }

  for (const auto& input : op.inputs) {
    const auto& array = model->GetArray(input);
    if (array.data_type == ArrayDataType::kFloat) {
      if (!array.minmax && !array.buffer) {
        LOG(ERROR) << "Can't quantize input array " << input
                   << " because it lacks min/max info";
        return false;
      }
      const auto* other_op = GetOpWithOutput(*model, input);
      if (other_op && other_op->type != OperatorType::kDequantize) {
        AddMessageF(
            "Not quantizing %s for now, because its input array %s is not "
            "produced by a Dequantize op, "
            "which means that we should yield and let other ops "
            "get quantized first",
            LogName(op), input);
        return false;
      }
    }
  }

  bool changed = false;

  // Quantize inputs, remove any Dequantize op on the inputs side
  for (std::size_t input_index = 0; input_index < op.inputs.size();
       input_index++) {
    ArrayDataType quantized_data_type;
    QuantizationParams quantization_params;
    if (ChooseQuantizationForOperatorInput(this, model, op, input_index,
                                           &quantized_data_type,
                                           &quantization_params)) {
      changed = true;
      const auto& input = op.inputs[input_index];
      if (IsConstantParameterArray(*model, input)) {
        QuantizeArray(this, model, input, quantized_data_type,
                      quantization_params);
      } else {
        auto dequantize_it = FindOpWithOutput(*model, input);
        CHECK(dequantize_it != model->operators.end());
        auto* dequantize_op = dequantize_it->get();
        CHECK(dequantize_op->type == OperatorType::kDequantize);
        op.inputs[input_index] = dequantize_op->inputs[0];
        // Check if the output of that Dequantize op was not used by any
        // other operator. We will then erase that Dequantize op.
        if (!CountOpsWithInput(*model, dequantize_op->outputs[0])) {
          // If any of the model's output_arrays was pointing to the
          // Dequantize op's output, let it point to the Dequantize op's
          // input instead.
          for (int i = 0; i < model->flags.output_arrays_size(); i++) {
            if (model->flags.output_arrays(i) == dequantize_op->outputs[0]) {
              model->flags.set_output_arrays(i, dequantize_op->inputs[0]);
            }
          }
          model->EraseArray(dequantize_op->outputs[0]);
          model->operators.erase(dequantize_it);
        }
      }
    }
  }

  // Quantize outputs, add Dequantize ops as needed on the outputs side
  for (std::size_t output_index = 0; output_index < op.outputs.size();
       output_index++) {
    ArrayDataType quantized_data_type;
    QuantizationParams quantization_params;
    if (ChooseQuantizationForOperatorOutput(this, model, op, output_index,
                                            &quantized_data_type,
                                            &quantization_params)) {
      changed = true;
      const auto& output = op.outputs[output_index];
      QuantizeArray(this, model, output, quantized_data_type,
                    quantization_params);
      const auto& dequantized_output =
          AvailableArrayName(*model, output + "_dequantized");
      const auto& output_array = model->GetArray(output);
      const auto& output_minmax = output_array.GetMinMax();
      auto& dequantized_output_array =
          model->GetOrCreateArray(dequantized_output);
      dequantized_output_array.data_type = ArrayDataType::kFloat;
      auto& dequantized_output_minmax =
          dequantized_output_array.GetOrCreateMinMax();
      dequantized_output_minmax.min = output_minmax.min;
      dequantized_output_minmax.max = output_minmax.max;
      for (const auto& other_op : model->operators) {
        for (auto& other_op_input : other_op->inputs) {
          if (other_op_input == output) {
            other_op_input = dequantized_output;
          }
        }
      }
      auto* dequantize_op = new DequantizeOperator;
      dequantize_op->inputs = {output};
      dequantize_op->outputs = {dequantized_output};
      for (int i = 0; i < model->flags.output_arrays_size(); i++) {
        if (model->flags.output_arrays(i) == output) {
          model->flags.set_output_arrays(i, dequantized_output);
        }
      }
      const auto op_it = FindOp(*model, &op);
      model->operators.emplace(op_it + 1, dequantize_op);
    }
  }

  return changed;
}

}  // namespace toco
