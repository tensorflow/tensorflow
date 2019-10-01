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
#ifndef TENSORFLOW_LITE_DELEGATES_ARMNN_VALIDATOR_H_
#define TENSORFLOW_LITE_DELEGATES_ARMNN_VALIDATOR_H_

#include <string>
#include <vector>

#include "tensorflow/lite/c/c_api_internal.h"

#include "armnn/ILayerSupport.hpp"

namespace tflite {
namespace delegate {
namespace arm {

// Operation validator interface
class OperationValidator {
 public:
  // Validate support of a convolution layer on a given backend
  static bool IsConvolution2dSupported(const TfLiteContext* context,
                                       const TfLiteNode* node, int version,
                                       const armnn::ILayerSupport& validator,
                                       std::vector<std::string>* failures);
  // Validate support of a depthwise convolution layer on a given backend
  static bool IsDepthwiseConvolutionSupported(
      const TfLiteContext* context, const TfLiteNode* node, int version,
      const armnn::ILayerSupport& validator,
      std::vector<std::string>* failures);
  // Validate support of a pooling layer on a given backend
  static bool IsPooling2dSupported(const TfLiteContext* context,
                                   const TfLiteNode* node, int builtin_code,
                                   int version,
                                   const armnn::ILayerSupport& validator,
                                   std::vector<std::string>* failures);
  // Validate support of a softmax layer on a given backend
  static bool IsSoftmaxSupported(const TfLiteContext* context,
                                 const TfLiteNode* node, int version,
                                 const armnn::ILayerSupport& validator,
                                 std::vector<std::string>* failures);
  // Validate support of a squeeze layer on a given backend
  static bool IsSqueezeSupported(const TfLiteContext* context,
                                 const TfLiteNode* node, int version,
                                 const armnn::ILayerSupport& validator,
                                 std::vector<std::string>* failures);
};
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_ARMNN_VALIDATOR_H_
