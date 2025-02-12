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

class GetItemSemantic {
 public:
  GetItemSemantic(TfLiteContext* ctx, TfLiteNode* node)
      : ctx_(ctx), node_(node) {}

  static constexpr int kElementShapeInputIdx = 2;
  static constexpr int kTensorOutputIdx = 0;
  static constexpr int kIndexInputIdx = 1;

  [[nodiscard]] TfLiteStatus CheckAndHandleTensors() const {
    TF_LITE_ENSURE(ctx_, NumInputs(node_) == 3 && NumOutputs(node_) == 1);
    const TfLiteTensor* index_input;
    TF_LITE_ENSURE_OK(ctx_,
                      GetInputSafe(ctx_, node_, kIndexInputIdx, &index_input));
    TF_LITE_ENSURE_TYPES_EQ(ctx_, index_input->type, kTfLiteInt32);
    return kTfLiteOk;
  }

  [[nodiscard]] TfLiteStatus GetIndexVal(const TensorArray* const arr,
                                         int& result) const {
    const TfLiteTensor* index_input;
    TF_LITE_ENSURE_OK(ctx_,
                      GetInputSafe(ctx_, node_, kIndexInputIdx, &index_input));
    TF_LITE_ENSURE_EQ(ctx_, index_input->bytes, sizeof(int));
    result = *GetTensorData<int>(index_input);
    return kTfLiteOk;
  }

  [[nodiscard]] TfLiteStatus HandleOutput(const TensorArray* const arr) const {
    return kTfLiteOk;
  }

 private:
  TfLiteContext* const ctx_;
  TfLiteNode* const node_;
};

class PopBackSemantic {
 public:
  PopBackSemantic(TfLiteContext* ctx, TfLiteNode* node)
      : ctx_(ctx), node_(node) {}

  static constexpr int kElementShapeInputIdx = 1;
  static constexpr int kTensorOutputIdx = 1;
  static constexpr int kListOutputIdx = 0;

  [[nodiscard]] TfLiteStatus CheckAndHandleTensors() const {
    TF_LITE_ENSURE(ctx_, NumInputs(node_) == 2 && NumOutputs(node_) == 2);
    TfLiteTensor* list_output;
    TF_LITE_ENSURE_OK(ctx_,
                      GetOutputSafe(ctx_, node_, kListOutputIdx, &list_output));
    TF_LITE_ENSURE_TYPES_EQ(ctx_, list_output->type, kTfLiteVariant);
    list_output->allocation_type = kTfLiteVariantObject;
    return kTfLiteOk;
  }

  [[nodiscard]] TfLiteStatus GetIndexVal(const TensorArray* const arr,
                                         int& result) const {
    result = arr->NumElements() - 1;
    return kTfLiteOk;
  }

  [[nodiscard]] TfLiteStatus HandleOutput(const TensorArray* const arr) const {
    TfLiteTensor* list_output;
    TF_LITE_ENSURE_OK(ctx_,
                      GetOutputSafe(ctx_, node_, kListOutputIdx, &list_output));
    TensorArray* output_arr = static_cast<TensorArray*>(
        arr->CloneTo(static_cast<VariantData*>(list_output->data.data)));
    output_arr->Resize(output_arr->NumElements() - 1);
    list_output->data.data = output_arr;
    return kTfLiteOk;
  }

 private:
  TfLiteContext* const ctx_;
  TfLiteNode* const node_;
};

template <class Semantic>
TfLiteStatus Prepare(TfLiteContext* ctx, TfLiteNode* node) {
  const auto semantic = Semantic(ctx, node);
  TF_LITE_ENSURE_OK(ctx, semantic.CheckAndHandleTensors());

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(ctx, GetInputSafe(ctx, node, kListInput, &list_input));
  TF_LITE_ENSURE_TYPES_EQ(ctx, list_input->type, kTfLiteVariant);

  const TfLiteTensor* element_shape_input;
  TF_LITE_ENSURE_OK(ctx, GetInputSafe(ctx, node, semantic.kElementShapeInputIdx,
                                      &element_shape_input));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(
      ctx, GetOutputSafe(ctx, node, semantic.kTensorOutputIdx, &output));

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

template <class Semantic>
TfLiteStatus Eval(TfLiteContext* ctx, TfLiteNode* node) {
  const auto semantic = Semantic(ctx, node);

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(ctx, GetInputSafe(ctx, node, kListInput, &list_input));
  TF_LITE_ENSURE_EQ(ctx, list_input->allocation_type, kTfLiteVariantObject);
  const auto* arr = static_cast<const TensorArray*>(
      static_cast<VariantData*>(list_input->data.data));

  int idx;
  TF_LITE_ENSURE_OK(ctx, semantic.GetIndexVal(arr, idx));
  TF_LITE_ENSURE(ctx, idx >= 0 && idx < arr->NumElements());

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(
      ctx, GetOutputSafe(ctx, node, semantic.kTensorOutputIdx, &output));
  TF_LITE_ENSURE_TYPES_EQ(ctx, arr->ElementType(), output->type);

  const TfLiteTensor* const element = arr->At(idx);

  if (element != nullptr) {
    if (IsDynamicTensor(output)) {
      size_t bytes;
      TF_LITE_ENSURE_OK(ctx, BytesRequired(output->type, element->dims->data,
                                           element->dims->size, &bytes, ctx));
      TF_LITE_ENSURE_OK(ctx, TfLiteTensorResizeMaybeCopy(bytes, output, false));
    }
    TF_LITE_ENSURE_OK(ctx, TfLiteTensorCopy(element, output));
    return semantic.HandleOutput(arr);
  }

  // As in tensorflow, it is possible to "get" an empty element in the list.
  // In the dynamic case, try to infer the output shape first through input and
  // list shape, and then through the elements in the list. Otherwise,
  // the shape will be known at compile time so we can just use that.
  if (!IsDynamicTensor(output)) {
    memset(output->data.data, 0, output->bytes);
    return semantic.HandleOutput(arr);
  }

  const TfLiteTensor* element_shape_input;
  TF_LITE_ENSURE_OK(ctx, GetInputSafe(ctx, node, semantic.kElementShapeInputIdx,
                                      &element_shape_input));

  IntArrayUniquePtr output_shape =
      MergeShapesOrNull(BuildTfLiteArray(*arr->ElementShape()),
                        TensorAsShape(*element_shape_input));
  TF_LITE_ENSURE(ctx, output_shape != nullptr);

  const bool can_infer_shape = (element_shape_input->dims->size != 0 ||
                                arr->ElementShape()->size != 0) &&
                               IsShapeFullyDefined(*output_shape);

  if (!can_infer_shape) {
    TF_LITE_ENSURE_MSG(
        ctx,
        GetShapeIfAllEqual(*arr, output_shape) == kTfLiteOk &&
            output_shape != nullptr,
        "Failed to infer the output shape for an item which has not been set.");
  }

  ctx->ResizeTensor(ctx, output, output_shape.release());
  memset(output->data.data, 0, output->bytes);

  return semantic.HandleOutput(arr);
}

}  // namespace

TfLiteRegistration* Register_LIST_GET_ITEM() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare<GetItemSemantic>,
                                 Eval<GetItemSemantic>};
  return &r;
}

TfLiteRegistration* Register_LIST_POP_BACK() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare<PopBackSemantic>,
                                 Eval<PopBackSemantic>};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
