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
#ifndef TENSORFLOW_LITE_DELEGATES_ARMNN_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_ARMNN_UTILS_H_

#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"

#include "armnn/Optional.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/Types.hpp"

namespace tflite {
namespace delegate {
namespace arm {
// Get raw array as an std::vector
bool AsUnsignedVector(const int32_t* data, size_t size,
                      std::vector<unsigned int>& unsignedVec);

// Extract TfLite input tensors of a node
std::vector<TfLiteTensor*> GetInputs(const TfLiteContext* context,
                                     const TfLiteNode* node);

// Extract TfLite output tensors of a node
std::vector<TfLiteTensor*> GetOutputs(const TfLiteContext* context,
                                      const TfLiteNode* node);

// Extract TensorInfo of the TfLite input tensors
bool GetInputsInfo(const TfLiteContext* context, const TfLiteNode* node,
                   std::vector<armnn::TensorInfo>& inputsInfo);

// Extract TensorInfo of the TfLite output tensors
bool GetOutputsInfo(const TfLiteContext* context, const TfLiteNode* node,
                    std::vector<armnn::TensorInfo>& outputsInfo);

// Translate a TfLite Tensor to an ArmNN TensorInfo
armnn::Optional<armnn::TensorInfo> ToTensorInfo(const TfLiteTensor* tensor);

// Translate TfLite Quantization information to ArmNN scale, offset
armnn::Optional<std::pair<float, int32_t>> ToScaleOffset(
    TfLiteQuantization qinfo);

// Translate TfLite data type to ArmNN DataType
armnn::Optional<armnn::DataType> ToDataType(TfLiteType type);

// Translate TfLite tensor shape to ArmNN TensorShape
armnn::Optional<armnn::TensorShape> ToTensorShape(const TfLiteIntArray* shape);
}  // namespace arm
}  // namespace delegate
}  // namespace tflite
#endif  // TENSORFLOW_LITE_DELEGATES_ARMNN_UTILS_H_
