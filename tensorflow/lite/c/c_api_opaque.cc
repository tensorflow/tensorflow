/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/c_api_opaque.h"

#include "tensorflow/lite/kernels/kernel_util.h"

TfLiteType TfLiteOpaqueTensorType(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorType(reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

int32_t TfLiteOpaqueTensorNumDims(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorNumDims(
      reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

int32_t TfLiteOpaqueTensorDim(const TfLiteOpaqueTensor* opaque_tensor,
                              int32_t dim_index) {
  return TfLiteTensorDim(reinterpret_cast<const TfLiteTensor*>(opaque_tensor),
                         dim_index);
}

size_t TfLiteOpaqueTensorByteSize(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorByteSize(
      reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

void* TfLiteOpaqueTensorData(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorData(reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

const char* TfLiteOpaqueTensorName(const TfLiteOpaqueTensor* opaque_tensor) {
  return TfLiteTensorName(reinterpret_cast<const TfLiteTensor*>(opaque_tensor));
}

TfLiteStatus TfLiteOpaqueTensorCopyFromBuffer(TfLiteOpaqueTensor* opaque_tensor,
                                              const void* input_data,
                                              size_t input_data_size) {
  return TfLiteTensorCopyFromBuffer(
      reinterpret_cast<TfLiteTensor*>(opaque_tensor), input_data,
      input_data_size);
}

TfLiteStatus TfLiteOpaqueTensorCopyToBuffer(
    const TfLiteOpaqueTensor* opaque_tensor, void* output_data,
    size_t output_data_size) {
  return TfLiteTensorCopyToBuffer(
      reinterpret_cast<const TfLiteTensor*>(opaque_tensor), output_data,
      output_data_size);
}

const TfLiteOpaqueTensor* TfLiteOpaqueNodeGetInput(
    TfLiteOpaqueContext* opaque_context, const TfLiteOpaqueNode* opaque_node,
    int index) {
  const TfLiteTensor* tensor =
      tflite::GetInput(reinterpret_cast<TfLiteContext*>(opaque_context),
                       reinterpret_cast<const TfLiteNode*>(opaque_node), index);
  return reinterpret_cast<const TfLiteOpaqueTensor*>(tensor);
}

TfLiteOpaqueTensor* TfLiteOpaqueNodeGetOutput(
    TfLiteOpaqueContext* opaque_context, const TfLiteOpaqueNode* opaque_node,
    int index) {
  TfLiteTensor* tensor = tflite::GetOutput(
      reinterpret_cast<TfLiteContext*>(opaque_context),
      reinterpret_cast<const TfLiteNode*>(opaque_node), index);
  return reinterpret_cast<TfLiteOpaqueTensor*>(tensor);
}
