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
#include <limits>
#include <utility>

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

  IntArrayUniquePtr shape_from_list = BuildTfLiteArray(*arr->ElementShape());

  const TfLiteTensor* shape_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kShapeInput, &shape_input));

  IntArrayUniquePtr shape_from_input = TensorAsShape(*shape_input);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kTensorOutput, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, arr->ElementType());

  // Attempt to compute all but the first dim of the output shape from
  // the shape signatures provided in second input tensor and input list.
  IntArrayUniquePtr cur_shape_suffix = MergeShapesOrNull(
      std::move(shape_from_input), std::move(shape_from_list));

  if (cur_shape_suffix == nullptr) {
    TF_LITE_KERNEL_LOG(
        context,
        "Given input shape is not compatible with input list's element shape.");
    return kTfLiteError;
  }

  // Get the shape of all present consituent elements if they are equal.
  IntArrayUniquePtr first_shape = nullptr;
  for (int i = 0; i < arr->NumElements(); ++i) {
    if (arr->At(i) == nullptr) {
      continue;
    }
    if (first_shape == nullptr) {
      first_shape = BuildTfLiteArray(*arr->At(i)->dims);
      continue;
    }
    TF_LITE_ENSURE(context,
                   TfLiteIntArrayEqual(first_shape.get(), arr->At(i)->dims));
  }
  if (first_shape != nullptr) {
    // List is non-empty, so compute updated shape suffix.
    cur_shape_suffix =
        MergeShapesOrNull(std::move(first_shape), std::move(cur_shape_suffix));
    TF_LITE_ENSURE(context, cur_shape_suffix != nullptr);
  }

  // At this point the shape suffix needs to be fully known.
  TF_LITE_ENSURE(context, IsShapeFullyDefined(*cur_shape_suffix));

  // Now compute the first dimension and concat with computed suffix.
  IntArrayUniquePtr final_output_shape = nullptr;

  const bool suffix_is_not_scalar =
      !(cur_shape_suffix->size == 0 ||
        (cur_shape_suffix->size == 1 && cur_shape_suffix->data[0] == 1));

  if (suffix_is_not_scalar) {
    final_output_shape = BuildTfLiteArray(cur_shape_suffix->size + 1);

    memcpy(final_output_shape->data + 1, cur_shape_suffix->data,
           cur_shape_suffix->size * sizeof(int));
    final_output_shape->data[0] = arr->NumElements();

    // Length zero will result in a tensor with empty allocation, so clear
    // data just in case and short circuit.
    if (arr->NumElements() == 0) {
      TfLiteTensorDataFree(output);
      if (output->dims) {
        TfLiteIntArrayFree(output->dims);
      }
      output->dims = final_output_shape.release();
      output->bytes = 0;
      return kTfLiteOk;
    }

  } else {
    final_output_shape = BuildTfLiteArray({arr->NumElements()});
  }

  context->ResizeTensor(context, output, final_output_shape.release());

  int num_elements = 1;
  for (int i = 0; i < output->dims->size; ++i) {
    const int d = output->dims->data[i];
    if (d > 0) {
      TF_LITE_ENSURE(context,
                     num_elements < std::numeric_limits<int>().max() / d);
      num_elements *= d;
    }
  }

  TF_LITE_ENSURE_EQ(context, output->bytes,
                    num_elements * TfLiteTypeGetSize(output->type));

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
