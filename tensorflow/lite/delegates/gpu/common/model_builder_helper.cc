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

#include <string>

#include <fp16.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace gpu {

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
      *bhwc = BHWC(dims->data[0], 1, 1, 1);
      return absl::OkStatus();
    case 2:
      *bhwc = BHWC(dims->data[0], 1, 1, dims->data[1]);
      return absl::OkStatus();
    case 3:
      *bhwc = BHWC(dims->data[0], 1, dims->data[1], dims->data[2]);
      return absl::OkStatus();
    case 4:
      *bhwc = BHWC(dims->data[0], dims->data[1], dims->data[2], dims->data[3]);
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Tensor \"", tflite_tensor.name ? tflite_tensor.name : "nullptr",
          "\" has bad input dims size: ", dims->size, "."));
  }
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
  for (int i = 0; i < tflite_node->inputs->size; i++) {
    if (!IsConstantTensor(&context->tensors[tflite_node->inputs->data[i]])) {
      number_of_runtime_inputs++;
    }
  }
  return number_of_runtime_inputs;
}

int GetNumberOfConstInputsForNode(const TfLiteContext* context,
                                  const TfLiteNode* tflite_node) {
  return tflite_node->inputs->size -
         GetNumberOfRuntimeInputsForNode(context, tflite_node);
}

int GetNumberOfRuntimeOutputsForNode(const TfLiteContext* context,
                                     const TfLiteNode* tflite_node) {
  int number_of_runtime_outputs = 0;
  for (int i = 0; i < tflite_node->outputs->size; i++) {
    if (!IsConstantTensor(&context->tensors[tflite_node->outputs->data[i]])) {
      number_of_runtime_outputs++;
    }
  }
  return number_of_runtime_outputs;
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
  const int runtime_outputs =
      GetNumberOfRuntimeOutputsForNode(context, tflite_node);
  if (runtime_outputs != outputs) {
    return absl::InternalError(absl::StrCat("Expected ", outputs,
                                            " output tensor(s), but node has ",
                                            runtime_outputs, " output(s)."));
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

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, Linear* shape) {
  if (dimensions->size <= 0) {
    return absl::InvalidArgumentError("Dimension is empty.");
  }
  for (int i = 0; i < dimensions->size - 1; ++i) {
    if (dimensions->data[i] != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          GetDimensionString(dimensions), "  cannot be reduced to linear."));
    }
  }
  shape->v = dimensions->data[dimensions->size - 1];
  return absl::OkStatus();
}

absl::Status SetAllDimensions(const TfLiteIntArray* dimensions, HWC* shape) {
  if (dimensions->size != 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a 4D tensor of shape 1xHxWxC but got ",
                     GetDimensionString(dimensions)));
  }
  if (dimensions->data[0] != 1) {
    return absl::UnimplementedError("Batch size is not equal to 1.");
  }
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->c = dimensions->data[3];
  return absl::OkStatus();
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

}  // namespace gpu
}  // namespace tflite
