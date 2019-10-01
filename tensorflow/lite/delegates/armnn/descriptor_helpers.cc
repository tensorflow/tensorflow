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
#include "tensorflow/lite/delegates/armnn/descriptor_helpers.h"

#include <algorithm>

#include "tensorflow/lite/delegates/armnn/macros.h"
#include "tensorflow/lite/delegates/armnn/utils.h"

namespace tflite {
namespace delegate {
namespace arm {
namespace {
// Compute output shape of squeeze layer
bool GetSqueezedTensorShape(const std::vector<uint32_t>& squeezeDimsIn,
                            const armnn::TensorShape& inputTensorShape,
                            armnn::TensorShape& squeezedTensorShape) {
  std::vector<uint32_t> squeezeDims = squeezeDimsIn;
  static const uint32_t dimensionSequence[] = {0, 1, 2, 3};

  // We support tensor up to rank 4
  if (inputTensorShape.GetNumDimensions() > 4) {
    return false;
  }

  // Leave as is
  if (squeezeDims.empty()) {
    squeezeDims.assign(dimensionSequence,
                       dimensionSequence + inputTensorShape.GetNumDimensions());
  }

  std::vector<uint32_t> outputDims;
  for (unsigned int i = 0; i < inputTensorShape.GetNumDimensions(); i++) {
    bool skipSqueeze = (std::find(squeezeDims.begin(), squeezeDims.end(), i) ==
                        squeezeDims.end());
    auto currentDimension = inputTensorShape[i];
    if (skipSqueeze || currentDimension != 1) {
      outputDims.push_back(currentDimension);
    }
  }

  if (outputDims.size() > 4) {
    return false;
  }

  squeezedTensorShape = armnn::TensorShape(
      static_cast<unsigned int>(outputDims.size()), outputDims.data());

  return true;
}

// Calculate ArmNN padding information without dilation
void CalcPadding(uint32_t inputSize, uint32_t filterSize, uint32_t stride,
                 uint32_t dilation, uint32_t& paddingFront,
                 uint32_t& paddingBack, TfLitePadding padding) {
  paddingFront = 0;
  paddingBack = 0;
  if (padding == kTfLitePaddingSame) {
    uint32_t outputSize = (inputSize + stride - 1) / stride;
    uint32_t dilatedSize = filterSize + (dilation - 1) * (filterSize - 1);
    uint32_t temp = (outputSize - 1) * stride + dilatedSize;
    if (temp > inputSize) {
      paddingFront = (temp - inputSize) / 2;
      paddingBack = (temp - inputSize) - paddingFront;
    }
  }
}
}  // namespace

bool ToActivationDescriptor(TfLiteFusedActivation activation,
                            armnn::ActivationDescriptor& desc) {
  switch (activation) {
    case kTfLiteActRelu: {
      desc.m_Function = armnn::ActivationFunction::ReLu;
      break;
    }
    case kTfLiteActRelu1: {
      desc.m_Function = armnn::ActivationFunction::BoundedReLu;
      desc.m_A = 1.0f;
      desc.m_B = 0.0f;
      break;
    }
    case kTfLiteActRelu6: {
      desc.m_Function = armnn::ActivationFunction::BoundedReLu;
      desc.m_A = 6.0f;
      desc.m_B = 0.0f;
      break;
    }
    case kTfLiteActTanh: {
      desc.m_Function = armnn::ActivationFunction::TanH;
      desc.m_A = 1.0f;
      desc.m_B = 1.0f;
      break;
    }
    case kTfLiteActSigmoid: {
      desc.m_Function = armnn::ActivationFunction::Sigmoid;
      break;
    }
    default: { return false; }
  }
  return true;
}

bool ToConv2dDescriptor(const TfLiteConvParams* params, int version,
                        const armnn::TensorShape& inputShape,
                        const armnn::TensorShape& filterShape, bool hasBias,
                        armnn::Convolution2dDescriptor& desc) {
  RETURN_FALSE_IF(params == nullptr);

  desc.m_BiasEnabled = hasBias;
  desc.m_StrideX = params->stride_width;
  desc.m_StrideY = params->stride_height;
  desc.m_DataLayout = armnn::DataLayout::NHWC;
  desc.m_DilationX = params->dilation_width_factor;
  desc.m_DilationY = params->dilation_height_factor;

  // assuming input is NHWC
  unsigned int inputHeight = inputShape[1];
  unsigned int inputWidth = inputShape[2];

  // assuming the filter is OHWI : Output, H, W, Input
  // which is essentially the same as NHWC
  unsigned int filterHeight = filterShape[1];
  unsigned int filterWidth = filterShape[2];

  // Calculate padding
  CalcPadding(inputHeight, filterHeight, desc.m_StrideY, desc.m_DilationY,
              desc.m_PadTop, desc.m_PadBottom, params->padding);
  CalcPadding(inputWidth, filterWidth, desc.m_StrideX, desc.m_DilationX,
              desc.m_PadLeft, desc.m_PadRight, params->padding);

  return true;
}

bool ToDepthwiseConvDescriptor(const TfLiteDepthwiseConvParams* params,
                               int version,
                               const armnn::TensorShape& inputShape,
                               const armnn::TensorShape& filterShape,
                               bool hasBias,
                               armnn::DepthwiseConvolution2dDescriptor& desc) {
  RETURN_FALSE_IF(params == nullptr);

  desc.m_BiasEnabled = hasBias;
  desc.m_StrideX = params->stride_width;
  desc.m_StrideY = params->stride_height;
  desc.m_DataLayout = armnn::DataLayout::NHWC;
  desc.m_DilationX = 1;
  desc.m_DilationY = 1;

  if (version >= 2) {
    desc.m_DilationX = params->dilation_width_factor;
    desc.m_DilationY = params->dilation_height_factor;
  }

  // assuming input is NHWC
  const unsigned int inputHeight = inputShape[1];
  const unsigned int inputWidth = inputShape[2];

  // assuming the filter is OHWI : Output, H, W, Input
  // which is essentially the same as NHWC
  const unsigned int filterHeight = filterShape[1];
  const unsigned int filterWidth = filterShape[2];

  // Calculate padding
  CalcPadding(inputHeight, filterHeight, desc.m_StrideY, desc.m_DilationY,
              desc.m_PadTop, desc.m_PadBottom, params->padding);
  CalcPadding(inputWidth, filterWidth, desc.m_StrideX, desc.m_DilationX,
              desc.m_PadLeft, desc.m_PadRight, params->padding);

  return true;
}

bool ToPool2dDescriptor(const TfLitePoolParams* params, int builtin_code,
                        int version, const armnn::TensorShape& inputShape,
                        armnn::Pooling2dDescriptor& desc) {
  RETURN_FALSE_IF(params == nullptr);

  switch (builtin_code) {
    case kTfLiteBuiltinAveragePool2d: {
      desc.m_PoolType = armnn::PoolingAlgorithm::Average;
      break;
    }
    case kTfLiteBuiltinL2Pool2d: {
      desc.m_PoolType = armnn::PoolingAlgorithm::L2;
      break;
    }
    case kTfLiteBuiltinMaxPool2d: {
      desc.m_PoolType = armnn::PoolingAlgorithm::Max;
      break;
    }
    default: { return false; }
  }

  desc.m_StrideX = params->stride_width;
  desc.m_StrideY = params->stride_height;
  desc.m_PoolWidth = params->filter_width;
  desc.m_PoolHeight = params->filter_height;
  desc.m_PaddingMethod = armnn::PaddingMethod::Exclude;
  desc.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
  desc.m_DataLayout = armnn::DataLayout::NHWC;

  // assuming input is NHWC
  unsigned int inputHeight = inputShape[1];
  unsigned int inputWidth = inputShape[2];

  CalcPadding(inputHeight, desc.m_PoolHeight, desc.m_StrideY, 1u, desc.m_PadTop,
              desc.m_PadBottom, params->padding);
  CalcPadding(inputWidth, desc.m_PoolWidth, desc.m_StrideX, 1u, desc.m_PadLeft,
              desc.m_PadRight, params->padding);

  return true;
}

bool ToSoftmaxDescriptor(const TfLiteSoftmaxParams* params, int version,
                         armnn::SoftmaxDescriptor& desc) {
  RETURN_FALSE_IF(params == nullptr);

  desc.m_Beta = params->beta;
  return true;
}

bool ToSqueezeDescriptor(const TfLiteSqueezeParams* params, int version,
                         armnn::TensorShape inputShape,
                         armnn::ReshapeDescriptor& desc) {
  RETURN_FALSE_IF(params == nullptr);

  std::vector<unsigned int> squeezedDims;
  RETURN_ON_FALSE(AsUnsignedVector(params->squeeze_dims,
                                   params->num_squeeze_dims, squeezedDims));
  RETURN_ON_FALSE(
      GetSqueezedTensorShape(squeezedDims, inputShape, desc.m_TargetShape));
  return true;
}
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
