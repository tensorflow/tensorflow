/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_DELEGATES_EAGER_UTIL_H_
#define TENSORFLOW_CONTRIB_LITE_DELEGATES_EAGER_UTIL_H_

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tflite {
namespace eager {

// Converts a tensorflow:Status into a TfLiteStatus. If the original status
// represented an error, reports it using the given 'context'.
TfLiteStatus ConvertStatus(TfLiteContext* context,
                           const tensorflow::Status& status);

// Copies the given shape and type of the TensorFlow 'src' tensor into a TF Lite
// 'tensor'. Logs an error and returns kTfLiteError if the shape or type can't
// be converted.
TfLiteStatus CopyShapeAndType(TfLiteContext* context,
                              const tensorflow::Tensor& src,
                              TfLiteTensor* tensor);

// Returns the TF C API Data type that corresponds to the given TfLiteType.
TF_DataType GetTensorFlowDataType(TfLiteType type);

// Returns the TfLiteType that corresponds to the given TF C API Data type.
TfLiteType GetTensorFlowLiteType(TF_DataType);

}  // namespace eager
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_DELEGATES_EAGER_UTIL_H_
