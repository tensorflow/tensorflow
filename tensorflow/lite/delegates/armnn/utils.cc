/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/armnn/utils.h"

#include "tensorflow/lite/delegates/armnn/macros.h"

namespace tflite {
namespace delegate {
namespace arm {
bool AsUnsignedVector(const int32_t* data, size_t size,
                      std::vector<unsigned int>& unsignedVec) {
  unsignedVec.clear();
  if (data == nullptr) {
    return true;
  }

  for (unsigned int i = 0; i < size; ++i) {
    int val = data[i];
    RETURN_FALSE_IF(val < 0);
    unsignedVec.push_back(static_cast<unsigned int>(val));
  }
  return true;
}

std::vector<TfLiteTensor*> GetInputs(const TfLiteContext* context,
                                     const TfLiteNode* node) {
  size_t inputCount = node->inputs->size;
  std::vector<TfLiteTensor*> result(inputCount);
  for (size_t i = 0; i < inputCount; ++i) {
    uint32_t inputId = node->inputs->data[i];
    result[i] = &context->tensors[inputId];
  }
  return result;
}

std::vector<TfLiteTensor*> GetOutputs(const TfLiteContext* context,
                                      const TfLiteNode* node) {
  size_t outputCount = node->outputs->size;
  std::vector<TfLiteTensor*> result(outputCount);
  for (size_t i = 0; i < outputCount; ++i) {
    uint32_t outputId = node->outputs->data[i];
    result[i] = &context->tensors[outputId];
  }
  return result;
}

bool GetInputsInfo(const TfLiteContext* context, const TfLiteNode* node,
                   std::vector<armnn::TensorInfo>& inputsInfo) {
  auto inputs = GetInputs(context, node);
  for (const auto& input : inputs) {
    // Skip invalid tensors
    if (input == nullptr) {
      continue;
    }

    auto inputInfo = ToTensorInfo(input);
    RETURN_ON_FALSE(inputInfo.has_value());
    inputsInfo.push_back(inputInfo.value());
  }

  return true;
}

bool GetOutputsInfo(const TfLiteContext* context, const TfLiteNode* node,
                    std::vector<armnn::TensorInfo>& outputsInfo) {
  auto outputs = GetOutputs(context, node);
  for (const auto& output : outputs) {
    // Skip invalid tensors
    if (output == nullptr) {
      continue;
    }

    auto outputInfo = ToTensorInfo(output);
    RETURN_ON_FALSE(outputInfo.has_value());
    outputsInfo.push_back(outputInfo.value());
  }

  return true;
}

armnn::Optional<std::pair<float, int32_t>> ToScaleOffset(
    TfLiteQuantization qinfo) {
  float scale = 0;
  int32_t offset = 0;

  if (qinfo.type == kTfLiteAffineQuantization) {
    TfLiteAffineQuantization* quantization_params =
        static_cast<TfLiteAffineQuantization*>(qinfo.params);
    if (quantization_params->scale->size > 1) {
      return armnn::EmptyOptional();
    } else {
      scale = quantization_params->scale->data[0];
      offset = quantization_params->zero_point->data[0];
    }
  }

  return std::make_pair(scale, offset);
}

armnn::Optional<::armnn::DataType> ToDataType(TfLiteType type) {
  armnn::Optional<::armnn::DataType> dtype = armnn::EmptyOptional();
  switch (type) {
    case kTfLiteFloat32:
      dtype = armnn::DataType::Float32;
      break;
    case kTfLiteFloat16:
      dtype = armnn::DataType::Float16;
      break;
    case kTfLiteInt32:
      dtype = armnn::DataType::Signed32;
      break;
    case kTfLiteUInt8:
      dtype = armnn::DataType::QuantisedAsymm8;
      break;
    default:
      break;
  }

  return dtype;
}

armnn::Optional<::armnn::TensorShape> ToTensorShape(
    const TfLiteIntArray* shape) {
  std::vector<unsigned int> shapeVec;
  bool status = AsUnsignedVector(shape->data, shape->size, shapeVec);
  if (status) {
    return armnn::TensorShape(shapeVec.size(), shapeVec.data());
  } else {
    return armnn::EmptyOptional();
  }
}

armnn::Optional<armnn::TensorInfo> ToTensorInfo(const TfLiteTensor* tensor) {
  // Get data type
  auto data_type = ToDataType(tensor->type);
  RETURN_EMPTY_ON_INVALID_OPTIONAL(data_type.has_value());

  // Get tensor shape
  auto shape = ToTensorShape(tensor->dims);
  RETURN_EMPTY_ON_INVALID_OPTIONAL(shape);

  // Get quantization info
  auto qinfo = ToScaleOffset(tensor->quantization);
  RETURN_EMPTY_ON_INVALID_OPTIONAL(ToScaleOffset);

  // Create tensor info
  // two statements (on purpose) for easier debugging:
  armnn::TensorInfo result(shape.value(), data_type.value(),
                           qinfo.value().first, qinfo.value().second);
  return result;
}
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
