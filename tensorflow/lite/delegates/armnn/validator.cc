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
#include "tensorflow/lite/delegates/armnn/validator.h"

#include "tensorflow/lite/delegates/armnn/descriptor_helpers.h"
#include "tensorflow/lite/delegates/armnn/macros.h"
#include "tensorflow/lite/delegates/armnn/utils.h"

namespace tflite {
namespace delegate {
namespace arm {

bool OperationValidator::IsConvolution2dSupported(
    const TfLiteContext* context, const TfLiteNode* node, int version,
    const armnn::ILayerSupport& validator, std::vector<std::string>* failures) {
  std::vector<armnn::TensorInfo> inputsInfo;
  RETURN_ON_FALSE(GetInputsInfo(context, node, inputsInfo));
  RETURN_FALSE_IF(inputsInfo.size() < 2);
  std::vector<armnn::TensorInfo> outputsInfo;
  RETURN_ON_FALSE(GetOutputsInfo(context, node, outputsInfo));
  RETURN_FALSE_IF(outputsInfo.size() != 1);

  const auto builtin = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  const auto& input = inputsInfo[0];
  const auto& filter = inputsInfo[1];
  const auto& output = outputsInfo[0];
  const bool hasBias = (inputsInfo.size() > 2);

  armnn::Convolution2dDescriptor desc;
  RETURN_ON_FALSE(ToConv2dDescriptor(builtin, version, input.GetShape(),
                                     filter.GetShape(), hasBias, desc));

  armnn::Optional<armnn::TensorInfo> bias;
  if (hasBias) {
    bias = armnn::Optional<armnn::TensorInfo>(inputsInfo[2]);
  }

  return validator.IsConvolution2dSupported(input, output, desc, filter, bias);
}

bool OperationValidator::IsDepthwiseConvolutionSupported(
    const TfLiteContext* context, const TfLiteNode* node, int version,
    const armnn::ILayerSupport& validator, std::vector<std::string>* failures) {
  std::vector<armnn::TensorInfo> inputsInfo;
  RETURN_ON_FALSE(GetInputsInfo(context, node, inputsInfo));
  RETURN_FALSE_IF(inputsInfo.size() < 2);
  std::vector<armnn::TensorInfo> outputsInfo;
  RETURN_ON_FALSE(GetOutputsInfo(context, node, outputsInfo));
  RETURN_FALSE_IF(outputsInfo.size() != 1);

  const auto builtin =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  const auto& input = inputsInfo[0];
  const auto& filter = inputsInfo[1];
  const auto& output = outputsInfo[0];
  const bool hasBias = (inputsInfo.size() > 2);

  armnn::DepthwiseConvolution2dDescriptor desc;
  RETURN_ON_FALSE(ToDepthwiseConvDescriptor(builtin, version, input.GetShape(),
                                            filter.GetShape(), hasBias, desc));

  armnn::Optional<armnn::TensorInfo> bias;
  if (hasBias) {
    bias = armnn::Optional<armnn::TensorInfo>(inputsInfo[2]);
  }

  return validator.IsDepthwiseConvolutionSupported(input, output, desc, filter,
                                                   bias);
}

bool OperationValidator::IsPooling2dSupported(
    const TfLiteContext* context, const TfLiteNode* node, int builtin_code,
    int version, const armnn::ILayerSupport& validator,
    std::vector<std::string>* failures) {
  std::vector<armnn::TensorInfo> inputsInfo;
  RETURN_ON_FALSE(GetInputsInfo(context, node, inputsInfo));
  RETURN_FALSE_IF(inputsInfo.size() != 1);
  std::vector<armnn::TensorInfo> outputsInfo;
  RETURN_ON_FALSE(GetOutputsInfo(context, node, outputsInfo));
  RETURN_FALSE_IF(outputsInfo.size() != 1);

  const auto builtin = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  const auto& input = inputsInfo[0];
  const auto& output = outputsInfo[0];

  armnn::Pooling2dDescriptor desc;
  RETURN_ON_FALSE(ToPool2dDescriptor(builtin, version, builtin_code,
                                     input.GetShape(), desc));

  return validator.IsPooling2dSupported(input, output, desc);
}

bool OperationValidator::IsSoftmaxSupported(
    const TfLiteContext* context, const TfLiteNode* node, int version,
    const armnn::ILayerSupport& validator, std::vector<std::string>* failures) {
  std::vector<armnn::TensorInfo> inputsInfo;
  RETURN_ON_FALSE(GetInputsInfo(context, node, inputsInfo));
  RETURN_FALSE_IF(inputsInfo.size() != 1);
  std::vector<armnn::TensorInfo> outputsInfo;
  RETURN_ON_FALSE(GetOutputsInfo(context, node, outputsInfo));
  RETURN_FALSE_IF(outputsInfo.size() != 1);

  const auto builtin =
      reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  const auto& input = inputsInfo[0];
  const auto& output = outputsInfo[0];

  armnn::SoftmaxDescriptor desc;
  RETURN_ON_FALSE(ToSoftmaxDescriptor(builtin, version, desc));

  return validator.IsSoftmaxSupported(input, output, desc);
}

bool OperationValidator::IsSqueezeSupported(
    const TfLiteContext* context, const TfLiteNode* node, int version,
    const armnn::ILayerSupport& validator, std::vector<std::string>* failures) {
  std::vector<armnn::TensorInfo> inputsInfo;
  RETURN_ON_FALSE(GetInputsInfo(context, node, inputsInfo));
  RETURN_FALSE_IF(inputsInfo.size() != 1);
  std::vector<armnn::TensorInfo> outputsInfo;
  RETURN_ON_FALSE(GetOutputsInfo(context, node, outputsInfo));
  RETURN_FALSE_IF(outputsInfo.size() != 1);

  const auto builtin =
      reinterpret_cast<TfLiteSqueezeParams*>(node->builtin_data);
  const auto& input = inputsInfo[0];
  const auto& output = outputsInfo[0];

  armnn::ReshapeDescriptor desc;
  RETURN_ON_FALSE(
      ToSqueezeDescriptor(builtin, version, input.GetShape(), desc));

  return validator.IsReshapeSupported(input, desc);
}
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
