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
#include "tensorflow/lite/kernels/variants/list_ops_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

constexpr int kListInput = 0;
constexpr int kShapeInput = 1;
constexpr int kTensorOutput = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kListInput, &list_input));
  TF_LITE_ENSURE_TYPES_EQ(context, list_input->type, kTfLiteVariant);

  const TfLiteTensor* shape_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kShapeInput, &shape_input));
  TF_LITE_ENSURE_TYPES_EQ(context, shape_input->type, kTfLiteInt32);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kTensorOutput, &output));

  // TODO(b/257472333) Consider leveraging arena when the shape is defined
  // at compile time.
  SetTensorToDynamic(output);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kListInput, &list_input));
  TF_LITE_ENSURE_EQ(context, list_input->allocation_type, kTfLiteVariantObject);
  TensorArray* arr = static_cast<TensorArray*>(
      static_cast<VariantData*>(list_input->data.data));

  const TfLiteTensor* shape_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kShapeInput, &shape_input));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kTensorOutput, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, arr->ElementType());

  IntArrayUniquePtr cur_shape_suffix;

  // If succeeds and result not nullptr, guaranteed to be fully defined.
  TF_LITE_ENSURE_OK(context, GetShapeIfAllEqual(*arr, cur_shape_suffix));

  // Confirm that input shape, shape of elements and list shape are all
  // compatible.
  cur_shape_suffix = MergeShapesOrNull(
      MergeShapesOrNull(TensorAsShape(*shape_input),
                        BuildTfLiteArray(*arr->ElementShape())),
      std::move(cur_shape_suffix));

  TF_LITE_ENSURE_MSG(
      context,
      cur_shape_suffix != nullptr && IsShapeFullyDefined(*cur_shape_suffix),
      "Shapes from input, list and elements are not compatible "
      "or do not resolve to fully defined shape.");

  // Now compute the first dimension and concat with computed suffix.
  IntArrayUniquePtr final_output_shape;

  const bool suffix_is_not_scalar =
      !(cur_shape_suffix->size == 0 ||
        (cur_shape_suffix->size == 1 && cur_shape_suffix->data[0] == 1));

  if (suffix_is_not_scalar) {
    final_output_shape = BuildTfLiteArray(cur_shape_suffix->size + 1);

    memcpy(final_output_shape->data + 1, cur_shape_suffix->data,
           cur_shape_suffix->size * sizeof(int));
    final_output_shape->data[0] = arr->NumElements();

  } else {
    final_output_shape = BuildTfLiteArray({arr->NumElements()});
  }

  context->ResizeTensor(context, output, final_output_shape.release());

  const auto num_elements = static_cast<int>(NumElements(output));
  if (num_elements == 0) {
    TfLiteTensorDataFree(output);
    return kTfLiteOk;
  }

  // This has to be an int and we would have returned already if divisor == 0.
  const int element_num_elements = num_elements / output->dims->data[0];
  const size_t bytes_per_element =
      element_num_elements * TfLiteTypeGetSize(output->type);

  // Copy buffer of constituent element tensors to output if they are present.
  // Otherwise, zero that chunk of memory.
  char* raw_data_offset = output->data.raw;
  for (int i = 0; i < arr->NumElements(); ++i) {
    if (arr->At(i) == nullptr) {
      memset(raw_data_offset, 0, bytes_per_element);
    } else {
      memcpy(raw_data_offset, arr->At(i)->data.data, bytes_per_element);
    }
    raw_data_offset = raw_data_offset + bytes_per_element;
  }

  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_LIST_STACK() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}
}  // namespace ops
}  // namespace variants
}  // namespace tflite
