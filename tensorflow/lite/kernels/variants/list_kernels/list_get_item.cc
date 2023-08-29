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
#include <cstddef>
#include <cstring>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/list_ops_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

constexpr int kListInput = 0;
constexpr int kIndexInput = 1;
constexpr int kElementShapeInput = 2;
constexpr int kOutput = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 3);

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kListInput, &list_input));
  TF_LITE_ENSURE_TYPES_EQ(context, list_input->type, kTfLiteVariant);

  const TfLiteTensor* index_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndexInput, &index_input));
  TF_LITE_ENSURE_TYPES_EQ(context, index_input->type, kTfLiteInt32);

  const TfLiteTensor* element_shape_input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kElementShapeInput,
                                          &element_shape_input));
  TF_LITE_ENSURE_TYPES_EQ(context, index_input->type, kTfLiteInt32);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutput, &output));

  const TfLiteIntArray* const out_dims_sig = output->dims_signature;

  // Tensorflow python allows the passing of a `tf.TensorShape` at list
  // initialization and as an argument to this function as a hint. The compiler
  // will also give this information to `output`. Currently tflite has no
  // way to distinguish between `output->dims_signature` encoding a true scalar
  // shape, or an "unranked" shape signature which is meant to be compatible
  // with any shape (`tf.TensorShape(None)`). Because of this we need to fall
  // back to dynamic if `output->dims_signature` looks like a scalar.
  // Update this function after TODO(b/290271484) has been addressed.
  if (out_dims_sig == nullptr || out_dims_sig->size <= 0 ||
      !IsShapeFullyDefined(*out_dims_sig)) {
    SetTensorToDynamic(output);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* index_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndexInput, &index_input));
  TF_LITE_ENSURE_EQ(context, index_input->bytes, sizeof(int));

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kListInput, &list_input));
  TF_LITE_ENSURE_EQ(context, list_input->allocation_type, kTfLiteVariantObject);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutput, &output));

  const auto* arr = static_cast<const TensorArray*>(
      static_cast<VariantData*>(list_input->data.data));
  TF_LITE_ENSURE_TYPES_EQ(context, arr->ElementType(), output->type);

  const int idx = *GetTensorData<int>(index_input);
  TF_LITE_ENSURE(context, idx >= 0 && idx < arr->NumElements());

  const TfLiteTensor* element = arr->At(idx);

  if (element != nullptr) {
    if (IsDynamicTensor(output)) {
      size_t bytes;
      TF_LITE_ENSURE_OK(context,
                        BytesRequired(output->type, element->dims->data,
                                      element->dims->size, &bytes, context));
      TF_LITE_ENSURE_OK(context,
                        TfLiteTensorResizeMaybeCopy(bytes, output, false));
    }
    TF_LITE_ENSURE_OK(context, TfLiteTensorCopy(element, output));
    return kTfLiteOk;
  }

  // As in tensorflow, it is possible to "get" an empty element in the list.
  // In the dynamic case, try to infer the output shape first through input and
  // list shape, and then through the elements in the list. Otherwise,
  // the shape will be known at compile time so we can just use that.
  if (!IsDynamicTensor(output)) {
    memset(output->data.data, 0, output->bytes);
    return kTfLiteOk;
  }

  const TfLiteTensor* element_shape_input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kElementShapeInput,
                                          &element_shape_input));

  IntArrayUniquePtr output_shape =
      MergeShapesOrNull(BuildTfLiteArray(*arr->ElementShape()),
                        TensorAsShape(*element_shape_input));
  TF_LITE_ENSURE(context, output_shape != nullptr);

  const bool can_infer_shape = (element_shape_input->dims->size != 0 ||
                                arr->ElementShape()->size != 0) &&
                               IsShapeFullyDefined(*output_shape);

  if (!can_infer_shape) {
    TF_LITE_ENSURE_MSG(
        context,
        GetShapeIfAllEqual(*arr, output_shape) == kTfLiteOk &&
            output_shape != nullptr,
        "Failed to infer the output shape for an item which has not been set.");
  }

  context->ResizeTensor(context, output, output_shape.release());
  memset(output->data.data, 0, output->bytes);
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration* Register_LIST_GET_ITEM() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
