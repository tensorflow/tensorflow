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

#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <any>
#include <limits>
#include <string>
#include <vector>

#include <fp16.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {
namespace {

// Creates a node that consumes output from the given node. Because output need
// to stay the same, newly created node will inherit the output from the given
// node, which will in turn get newly created copy of output. This is necessary
// to preserve reference consistency if another node was pointing at that
// output:
//   node(output)
// will turn into:
//   node(copy(output)) <- passthrough_node(output)
absl::Status NewPassthroughNode(GraphFloat32* graph, Node* node,
                                const Value* output, Node** passthru_node) {
  *passthru_node = graph->NewNode();
  // Make copies for every output in the original node.
  RETURN_IF_ERROR(graph->SetProducer((*passthru_node)->id, output->id));
  Value* copy_output = graph->NewValue();
  RETURN_IF_ERROR(graph->SetProducer(node->id, copy_output->id));
  RETURN_IF_ERROR(graph->AddConsumer((*passthru_node)->id, copy_output->id));
  copy_output->tensor = output->tensor;
  copy_output->tensor.ref = -1;
  return absl::OkStatus();
}

}  // namespace

absl::Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                                    TfLiteNode** tflite_node,
                                    TfLiteRegistration** registration) {
  if (context->GetNodeAndRegistration(context, node_id, tflite_node,
                                      registration) != kTfLiteOk) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Couldn't get node and registration info for op: ", node_id));
  }
  return absl::OkStatus();
}

DataType ToDataType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return DataType::FLOAT32;
    case kTfLiteInt32:
      return DataType::INT32;
    case kTfLiteInt64:
      return DataType::INT64;
    case kTfLiteInt8:
      return DataType::INT8;
    case kTfLiteUInt8:
      return DataType::UINT8;
    default:
      return DataType::UNKNOWN;
  }
}

absl::Status ExtractTensorShape(const TfLiteTensor& tflite_tensor, BHWC* bhwc) {
  const TfLiteIntArray* dims = tflite_tensor.dims;
  switch (dims->size) {
    case 1:
      // B layout
      *bhwc = BHWC(dims->data[0], 1, 1, 1);
      return absl::OkStatus();
    case 2:
      // BC layout
      *bhwc = BHWC(dims->data[0], 1, 1, dims->data[1]);
      return absl::OkStatus();
    case 3:
      // BWC layout
      *bhwc = BHWC(dims->data[0], 1, dims->data[1], dims->data[2]);
      return absl::OkStatus();
    case 4:
      // BHWC layout
      *bhwc = BHWC(dims->data[0], dims->data[1], dims->data[2], dims->data[3]);
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Tensor \"", tflite_tensor.name ? tflite_tensor.name : "nullptr",
          "\" has bad input dims size: ", dims->size, "."));
  }
}

absl::Status ExtractAxisFromIndex(const TfLiteTensor& tflite_tensor, int index,
                                  Axis* axis) {
  const TfLiteIntArray* dims = tflite_tensor.dims;
  if (index == -1) {
    index = dims->size - 1;
  }
  if (index < 0 || index >= dims->size) {
    return absl::OutOfRangeError("Index for axis out of range");
  }
  std::vector<Axis> index_to_axis;
  switch (dims->size) {
    case 1:
      // B layout
      index_to_axis = {Axis::BATCH};
      break;
    case 2:
      // BC layout
      index_to_axis = {Axis::BATCH, Axis::CHANNELS};
      break;
    case 3:
      // BWC layout
      index_to_axis = {Axis::BATCH, Axis::WIDTH, Axis::CHANNELS};
      break;
    case 4:
      // BHWC layout
      index_to_axis = {Axis::BATCH, Axis::HEIGHT, Axis::WIDTH, Axis::CHANNELS};
      break;
    default:
      return absl::UnavailableError("Unknown layout.");
  }
  *axis = index_to_axis[index];
  return absl::OkStatus();
}

absl::Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                            TensorRef<BHWC>* tensor_ref) {
  tensor_ref->type = ToDataType(tflite_tensor.type);
  return ExtractTensorShape(tflite_tensor, &tensor_ref->shape);
}

absl::Status PopulateQuantParams(const TfLiteTensor& tensor,
                                 QuantizationParams* quant_params) {
  const TfLiteQuantization& quant = tensor.quantization;
  if (quant.type != TfLiteQuantizationType::kTfLiteAffineQuantization) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tensor not quantized: ", std::string(tensor.name)));
  }
  const TfLiteAffineQuantization* params =
      static_cast<const TfLiteAffineQuantization*>(quant.params);
  if (params->scale->size > 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non-constant per-channel quantized tensor: ",
                     std::string(tensor.name)));
  }
  const float scale = params->scale->data[0];
  const float zero_point = static_cast<float>(params->zero_point->data[0]);

  float qmin_value = 0;
  float qmax_value = 0;
  if (tensor.type == kTfLiteUInt8) {
    qmin_value = static_cast<float>(std::numeric_limits<uint8_t>::min());
    qmax_value = static_cast<float>(std::numeric_limits<uint8_t>::max());
  } else if (tensor.type == kTfLiteInt8) {
    qmin_value = static_cast<float>(std::numeric_limits<int8_t>::min());
    qmax_value = static_cast<float>(std::numeric_limits<int8_t>::max());
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Type invalid for quantized tensor: ", std::string(tensor.name)));
  }
  quant_params->min = scale * (static_cast<float>(qmin_value) - zero_point);
  quant_params->max = scale * (static_cast<float>(qmax_value) - zero_point);
  quant_params->scale = scale;

  return absl::OkStatus();
}

int GetNumberOfRuntimeInputsForNode(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node) {
  int number_of_runtime_inputs = 0;
  for (int i = 0; i < NumInputs(tflite_node); i++) {
    const TfLiteTensor* tensor =
        GetOptionalInputTensor(context, tflite_node, i);
    if (tensor != nullptr && !IsConstantTensor(tensor)) {
      number_of_runtime_inputs++;
    }
  }
  return number_of_runtime_inputs;
}

int GetNumberOfConstInputsForNode(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node) {
  return NumInputs(tflite_node) -
         GetNumberOfRuntimeInputsForNode(context, tflite_node);
}

absl::Status CheckInputsOutputs(const TfLiteContext* context,
                                const TfLiteNode* tflite_node,
                                int runtime_inputs, int outputs) {
  const int runtime_inputs_from_model =
      GetNumberOfRuntimeInputsForNode(context, tflite_node);
  if (runtime_inputs_from_model != runtime_inputs) {
    return absl::InternalError(absl::StrCat(
        "Expected ", runtime_inputs, " runtime input tensor(s), but node has ",
        runtime_inputs_from_model, " runtime input(s)."));
  }
  const int outputs_from_model = NumOutputs(tflite_node);
  if (outputs_from_model != outputs) {
    return absl::InternalError(absl::StrCat("Expected ", outputs,
                                            " output tensor(s), but node has ",
                                            outputs_from_model, " output(s)."));
  }
  return absl::OkStatus();
}

absl::Status CheckInputsConstsOutputs(const TfLiteContext* context,
                                      const TfLiteNode* tflite_node,
                                      int runtime_inputs, int const_inputs,
                                      int outputs) {
  const int const_inputs_from_model =
      GetNumberOfConstInputsForNode(context, tflite_node);
  if (const_inputs_from_model != const_inputs) {
    return absl::InternalError(absl::StrCat(
        "Expected ", const_inputs, " const input tensor(s), but node has ",
        const_inputs_from_model, " const input(s)."));
  }
  return CheckInputsOutputs(context, tflite_node, runtime_inputs, outputs);
}

void ConvertFloat16ToFloat32(size_t num_elements, const uint16_t* src,
                             float* dst) {
  for (size_t i = 0; i < num_elements; i++) {
    *dst++ = fp16_ieee_to_fp32_value(*src++);
  }
}

template <>
absl::Status CreateVectorCopyData<float>(const TfLiteTensor& tensor,
                                         float* tensor_data) {
  switch (tensor.type) {
    case kTfLiteFloat32:
      std::memcpy(tensor_data, tensor.data.f, tensor.bytes);
      break;
    case kTfLiteFloat16:
      ConvertFloat16ToFloat32(
          NumElements(&tensor),
          reinterpret_cast<uint16_t const*>(tensor.data.f16), tensor_data);
      break;
    case kTfLiteInt8:
      DequantizeConstantTensor(tensor, tensor.data.int8, tensor_data);
      break;
    case kTfLiteUInt8:
      DequantizeConstantTensor(tensor, tensor.data.uint8, tensor_data);
      break;
    case kTfLiteInt32:
      DequantizeConstantTensor(tensor, tensor.data.i32, tensor_data);
      break;
    default:
      return absl::InvalidArgumentError(
          "Unsupported data type for float32 tensor");
  }
  return absl::OkStatus();
}

const std::string GetDimensionString(const TfLiteIntArray* dimensions) {
  return absl::StrJoin(TfLiteIntArrayView(dimensions), "x");
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Scalar* shape) {
  if (dimensions->size < 0) {
    return absl::InvalidArgumentError("Invalid Scalar dimensions");
  }
  for (int i = 0; i < dimensions->size; ++i) {
    if (dimensions->data[i] != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          GetDimensionString(dimensions), "  cannot be reduced to scalar."));
    }
  }
  shape->v = 1;
  return absl::OkStatus();
}

absl::Status CheckIfLinearConvertible(const TfLiteIntArray* dimensions) {
  if (dimensions->size <= 0) {
    return absl::InvalidArgumentError("Dimension is empty.");
  }
  for (int i = 0; i < dimensions->size - 1; ++i) {
    if (dimensions->data[i] != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          GetDimensionString(dimensions), "  cannot be reduced to linear."));
    }
  }
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Linear* shape) {
  RETURN_IF_ERROR(CheckIfLinearConvertible(dimensions));
  shape->v = dimensions->data[dimensions->size - 1];
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HWC* shape) {
  if (dimensions->size == 3) {
    shape->h = dimensions->data[0];
    shape->w = dimensions->data[1];
    shape->c = dimensions->data[2];
    return absl::OkStatus();
  }
  if (dimensions->size == 4) {
    if (dimensions->data[0] != 1) {
      return absl::UnimplementedError("Batch size is not equal to 1.");
    }
    shape->h = dimensions->data[1];
    shape->w = dimensions->data[2];
    shape->c = dimensions->data[3];
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Expected a 3D tensor of shape HxWxC or a 4D tensor of "
                   "shape 1xHxWxC but got ",
                   GetDimensionString(dimensions)));
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HW* shape) {
  if (dimensions->size != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a 2D tensor of shape HxW but got ",
                     GetDimensionString(dimensions)));
  }
  shape->h = dimensions->data[0];
  shape->w = dimensions->data[1];
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, OHWI* shape) {
  if (dimensions->size != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a 4D tensor of shape OxHxWxI but got ",
                     GetDimensionString(dimensions)));
  }
  shape->o = dimensions->data[0];
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->i = dimensions->data[3];
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, BHWC* shape) {
  if (dimensions->size != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a 4D tensor of shape BxHxWxC but got ",
                     GetDimensionString(dimensions)));
  }
  shape->b = dimensions->data[0];
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->c = dimensions->data[3];
  return absl::OkStatus();
}

absl::Status IsActivationSupported(TfLiteFusedActivation fused_activation) {
  switch (fused_activation) {
    case kTfLiteActNone:
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6:
    case kTfLiteActTanh:
    case kTfLiteActSigmoid:
      return absl::OkStatus();
    case kTfLiteActSignBit:
      return absl::UnimplementedError(
          "TfLiteFusedActivation.kTfLiteActSignBit");

      // Do not add default; we want compilation error rather than run-time
      // error.
  }
}

// If there is fused activation present, then there will be another node created
// that will have identical output as the given node. New operation node will
// depend on the given node output.
absl::Status MaybeFuseActivation(TfLiteFusedActivation fused_activation,
                                 GraphFloat32* graph, Node* node) {
  const auto outputs = graph->FindOutputs(node->id);
  if (outputs.size() != 1) {
    return absl::InternalError("Number of outputs != 1");
  }
  switch (fused_activation) {
    case kTfLiteActNone:
      // Nothing to do here
      return absl::OkStatus();
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6: {
      ReLUAttributes attr;
      attr.clip = fused_activation == kTfLiteActRelu
                      ? 0.0f
                      : (fused_activation == kTfLiteActReluN1To1 ? 1.0f : 6.0f);
      Node* activation_node;
      RETURN_IF_ERROR(
          NewPassthroughNode(graph, node, outputs[0], &activation_node));
      activation_node->operation.type = ToString(OperationType::RELU);
      activation_node->operation.attributes = attr;
      return absl::OkStatus();
    }
    case kTfLiteActTanh: {
      Node* activation_node;
      RETURN_IF_ERROR(
          NewPassthroughNode(graph, node, outputs[0], &activation_node));
      activation_node->operation.type = ToString(OperationType::TANH);
      return absl::OkStatus();
    }
    case kTfLiteActSigmoid: {
      Node* activation_node;
      RETURN_IF_ERROR(
          NewPassthroughNode(graph, node, outputs[0], &activation_node));
      activation_node->operation.type = ToString(OperationType::SIGMOID);
      return absl::OkStatus();
    } break;
    default:
      return absl::NotFoundError(
          absl::StrCat("Unsupported fused activation: ", fused_activation));
  }
}

}  // namespace gpu
}  // namespace tflite
