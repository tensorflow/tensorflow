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
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"

namespace tflite {
namespace variants {
namespace ops {
namespace list_element_shape {
namespace {

// This kernel returns a `TfLiteTensor` which represents the shape signature
// of the tensorlist elements stored in the `TensorArray::ElementShape` field as
// a `TfLiteIntArray`. Encoding shape signatures in tensors works as follows:
//
// An unranked shape signature:
//    `TfLiteIntArray` : []
//    `TfLiteTensor` : shape = [], data = [-1]
//
// A scalar shape signature:
//    `TfLiteIntArray` : [0]
//    `TfLiteTensor` : shape = [0], data = []
//
// A ranked tensor shape signature (with possibly dynamic dimensions):
//    `TfLiteIntArray` : [Dim1, Dim2 ... DimRank] (DimI >= -1)
//    `TfLiteTensor` : shape = [Rank], data = [Dim1, Dim2 ... DimRank]

using ::tflite::variants::TensorArray;

constexpr int kListInput = 0;
constexpr int kShapeOut = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kListInput, &list_input));
  TF_LITE_ENSURE(context, list_input->type == kTfLiteVariant);

  TfLiteTensor* shape_out;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kShapeOut, &shape_out));
  TF_LITE_ENSURE_TYPES_EQ(context, shape_out->type, kTfLiteInt32);

  SetTensorToDynamic(shape_out);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kListInput, &list_input));

  const TensorArray* const list =
      reinterpret_cast<const TensorArray*>(list_input->data.data);
  const TfLiteIntArray& element_shape = *list->ElementShape();

  TfLiteTensor* shape_out;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kShapeOut, &shape_out));

  if (element_shape.size == 0) {
    // Unranked
    context->ResizeTensor(context, shape_out, BuildTfLiteArray(0).release());
    GetTensorData<int32_t>(shape_out)[0] = -1;
  } else if (element_shape.data[0] == 0) {
    // Scalar
    context->ResizeTensor(context, shape_out, BuildTfLiteArray({0}).release());
  } else {
    // Ranked
    context->ResizeTensor(context, shape_out,
                          BuildTfLiteArray({element_shape.size}).release());
    memcpy(GetTensorData<int32_t>(shape_out), element_shape.data,
           element_shape.size * sizeof(int32_t));
  }
  return kTfLiteOk;
}
}  // namespace
}  // namespace list_element_shape

TfLiteRegistration* Register_LIST_ELEMENT_SHAPE() {
  static TfLiteRegistration r = {nullptr, nullptr, list_element_shape::Prepare,
                                 list_element_shape::Eval};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
