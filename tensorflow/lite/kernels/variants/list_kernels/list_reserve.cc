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
#include <utility>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/list_ops_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace variants {
namespace ops {
namespace list_reserve {
namespace {

using ::tflite::variants::TensorArray;
using ::tflite::variants::detail::ListReserveOptions;

TfLiteType ConvertTensorType(TensorType src) {
  switch (src) {
    case TensorType_INT32:
      return kTfLiteInt32;
    case TensorType_FLOAT32:
      return kTfLiteFloat32;
    case TensorType_BOOL:
      return kTfLiteBool;
    case TensorType_INT64:
      return kTfLiteInt64;
    default:
      return kTfLiteNoType;
  }
}

constexpr int kElementShapeInput = 0;
constexpr int kNumElementsInput = 1;
constexpr int kListOut = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);

  const TfLiteTensor* element_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kElementShapeInput, &element_shape));
  TF_LITE_ENSURE(context, element_shape->type == kTfLiteInt32);

  const TfLiteTensor* num_elements;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kNumElementsInput, &num_elements));
  TF_LITE_ENSURE_TYPES_EQ(context, num_elements->type, kTfLiteInt32);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kListOut, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteVariant);
  output->allocation_type = kTfLiteVariantObject;
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Parse element type from custom options.
  auto* options =
      reinterpret_cast<const ListReserveOptions*>(node->custom_initial_data);
  TfLiteType element_type = ConvertTensorType(options->element_type);
  TF_LITE_ENSURE(context, element_type != kTfLiteNoType);

  const TfLiteTensor* num_elements;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kNumElementsInput, &num_elements));
  TF_LITE_ENSURE_TYPES_EQ(context, num_elements->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, num_elements->dims->size, 0);
  const int num_elements_value = num_elements->data.i32[0];
  TF_LITE_ENSURE(context, num_elements_value >= 0);

  // Create int array representing constraint on list's constituent elements.
  const TfLiteTensor* element_shape_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kElementShapeInput,
                                          &element_shape_tensor));
  IntArrayUniquePtr element_shape = TensorAsShape(*element_shape_tensor);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kListOut, &output));

  // Construct new `TensorArray` underneath the output tensor.
  TfLiteStatus stat =
      TfLiteTensorVariantRealloc<TensorArray, TfLiteType, IntArrayUniquePtr>(
          output, std::move(element_type), std::move(element_shape));
  TF_LITE_ENSURE_OK(context, stat);

  // Set size of array.
  TensorArray* arr =
      static_cast<TensorArray*>(static_cast<VariantData*>(output->data.data));
  arr->Resize(num_elements_value);

  return kTfLiteOk;
}
}  // namespace
}  // namespace list_reserve

TfLiteRegistration* Register_LIST_RESERVE() {
  static TfLiteRegistration r = {nullptr, nullptr, list_reserve::Prepare,
                                 list_reserve::Eval};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
