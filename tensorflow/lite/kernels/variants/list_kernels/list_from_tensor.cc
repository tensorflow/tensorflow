/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <cstring>
#include <utility>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/list_ops_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

constexpr int kTensorInput = 0;
constexpr int kElementShapeInput = 1;
constexpr int kListOut = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);

  const TfLiteTensor* element_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kElementShapeInput, &element_shape));

  TF_LITE_ENSURE(context, element_shape->type == kTfLiteInt32);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kListOut, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteVariant);
  output->allocation_type = kTfLiteVariantObject;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* tensor_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kTensorInput, &tensor_input));
  const int rank = tensor_input->dims->size;

  // As in Tensorflow, input is not permitted be a scalar.
  TF_LITE_ENSURE(context, rank > 0);

  // Output list has `num_elements` equal to the first dim of `tensor_input`,
  // and element tensors with shape equal to `Shape(tensor_input)[1:]`.
  const int list_len = tensor_input->dims->data[0];
  IntArrayUniquePtr element_shape_for_tensors =
      BuildTfLiteArray(rank - 1, tensor_input->dims->data + 1);

  // `element_shape_tensor` is an auxillary input shape signature which
  // is to be used as the `ElementShape()` attribute of the resulting
  // `TensorArray`.
  const TfLiteTensor* element_shape_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kElementShapeInput,
                                          &element_shape_tensor));
  TF_LITE_ENSURE(context, (element_shape_tensor->dims->size == 1 &&
                           element_shape_tensor->dims->data[0] == rank - 1) ||
                              element_shape_tensor->dims->size == 0);

  IntArrayUniquePtr element_shape_for_list =
      TensorAsShape(*element_shape_tensor);
  // Check given element shape is compatible with the suffix of input tensor's
  // shape. TODO(b/257472333) consider wrapping this in `#ifndef NDEBUG`.
  if (element_shape_for_list->size > 0) {
    TF_LITE_ENSURE_EQ(context, element_shape_for_list->size,
                      element_shape_for_tensors->size);
    for (int i = 0; i < element_shape_for_tensors->size; ++i) {
      const int lhs = element_shape_for_list->data[i];
      const int rhs = element_shape_for_tensors->data[i];
      TF_LITE_ENSURE(context, lhs == -1 || rhs == -1 || lhs == rhs);
    }
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kListOut, &output));

  // Build and retrieve output list.
  TF_LITE_ENSURE_OK(context, TfLiteTensorVariantRealloc<TensorArray>(
                                 output, tensor_input->type,
                                 BuildTfLiteArray(*element_shape_for_list)));
  TensorArray* arr =
      static_cast<TensorArray*>(static_cast<VariantData*>(output->data.data));

  arr->Resize(list_len);

  // Copy each row of input into the elements of the new list.
  size_t data_offset = 0;
  for (int i = 0; i < list_len; ++i) {
    TensorUniquePtr tensor_to_set = BuildTfLiteTensor(
        tensor_input->type, BuildTfLiteArray(*element_shape_for_tensors),
        kTfLiteDynamic);

    memcpy(tensor_to_set->data.raw, tensor_input->data.raw + data_offset,
           tensor_to_set->bytes);
    data_offset += tensor_to_set->bytes;

    TF_LITE_ENSURE(context, arr->Set(i, std::move(tensor_to_set)));
  }

  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_LIST_FROM_TENSOR() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
