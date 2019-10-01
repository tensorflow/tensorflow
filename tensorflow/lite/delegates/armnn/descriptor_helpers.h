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
#ifndef TENSORFLOW_LITE_DELEGATES_ARMNN_DESCRIPTOR_HELPERS_H_
#define TENSORFLOW_LITE_DELEGATES_ARMNN_DESCRIPTOR_HELPERS_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"

#include "armnn/Descriptors.hpp"
#include "armnn/Types.hpp"

namespace tflite {
namespace delegate {
namespace arm {
// Converts TFLite activation to an ArmNN descriptor
bool ToActivationDescriptor(TfLiteFusedActivation activation, int version,
                            armnn::ActivationDescriptor& desc);

// Converts TFLite convolution parameters to an ArmNN descriptor
bool ToConv2dDescriptor(const TfLiteConvParams* params, int version,
                        const armnn::TensorShape& inputShape,
                        const armnn::TensorShape& filterShape, bool hasBias,
                        armnn::Convolution2dDescriptor& desc);

// Converts TFLite depthwise convolution parameters to an ArmNN descriptor
bool ToDepthwiseConvDescriptor(const TfLiteDepthwiseConvParams* params,
                               int version,
                               const armnn::TensorShape& inputShape,
                               const armnn::TensorShape& filterShape,
                               bool hasBias,
                               armnn::DepthwiseConvolution2dDescriptor& desc);

// Converts TFLite convolution parameters to an ArmNN descriptor
bool ToPool2dDescriptor(const TfLitePoolParams* params, int builtin_code,
                        int version, const armnn::TensorShape& inputShape,
                        armnn::Pooling2dDescriptor& desc);

// Converts TFLite softmax parameters to an ArmNN descriptor
bool ToSoftmaxDescriptor(const TfLiteSoftmaxParams* params, int version,
                         armnn::SoftmaxDescriptor& desc);

// Converts TFLite squeeze parameters to an ArmNN descriptor
bool ToSqueezeDescriptor(const TfLiteSqueezeParams* params, int version,
                         armnn::TensorShape inputShape,
                         armnn::ReshapeDescriptor& desc);
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_ARMNN_DESCRIPTOR_HELPERS_H_
