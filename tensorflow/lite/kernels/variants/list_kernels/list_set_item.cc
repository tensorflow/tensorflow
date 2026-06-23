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

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

constexpr int kListInputIdx = 0;
constexpr int kIndexInputIdx = 1;
constexpr int kListOutputIdx = 0;

class SetItemSemantic {
 public:
  SetItemSemantic(TfLiteContext* ctx, TfLiteNode* node)
      : ctx_(ctx), node_(node) {}

  static constexpr int kItemInputIdx = 2;

  TfLiteStatus CheckIndexInput() const {
    const TfLiteTensor* index_input;
    TF_LITE_ENSURE_OK(ctx_,
                      GetInputSafe(ctx_, node_, kIndexInputIdx, &index_input));
    TF_LITE_ENSURE_TYPES_EQ(ctx_, index_input->type, kTfLiteInt32);
    return kTfLiteOk;
  }

  TfLiteStatus GetIndexVal(const TensorArray& /*arr*/, int& result) const {
    const TfLiteTensor* index_input;
    TF_LITE_ENSURE_OK(ctx_,
                      GetInputSafe(ctx_, node_, kIndexInputIdx, &index_input));
    TF_LITE_ENSURE_EQ(ctx_, index_input->bytes, sizeof(int));
    const int* index_data = GetTensorData<int>(index_input);
    TF_LITE_ENSURE(ctx_, index_data != nullptr);
    const int index = *index_data;
    TF_LITE_ENSURE(ctx_, index >= 0);
    result = index;
    return kTfLiteOk;
  }

 private:
  TfLiteContext* const ctx_;
  TfLiteNode* const node_;
};

class PushBackSemantic {
 public:
  PushBackSemantic(TfLiteContext* /*ctx*/, TfLiteNode* /*node*/) {}

  static constexpr int kItemInputIdx = 1;

  TfLiteStatus CheckIndexInput() const { return kTfLiteOk; }

  TfLiteStatus GetIndexVal(const TensorArray& arr, int& result) const {
    result = arr.NumElements();
    return kTfLiteOk;
  }
};

template <class Semantic>
TfLiteStatus Prepare(TfLiteContext* ctx, TfLiteNode* node) {
  const auto semantic = Semantic(ctx, node);
  TF_LITE_ENSURE(ctx, NumInputs(node) >= semantic.kItemInputIdx + 1);

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(ctx, GetInputSafe(ctx, node, kListInputIdx, &list_input));
  TF_LITE_ENSURE_TYPES_EQ(ctx, list_input->type, kTfLiteVariant);

  TF_LITE_ENSURE_OK(ctx, semantic.CheckIndexInput());

  const TfLiteTensor* item_input;
  TF_LITE_ENSURE_OK(
      ctx, GetInputSafe(ctx, node, semantic.kItemInputIdx, &item_input));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(ctx, GetOutputSafe(ctx, node, kListOutputIdx, &output));
  TF_LITE_ENSURE_TYPES_EQ(ctx, output->type, kTfLiteVariant);
  output->allocation_type = kTfLiteVariantObject;

  return kTfLiteOk;
}

template <class Semantic>
TfLiteStatus Eval(TfLiteContext* ctx, TfLiteNode* node) {
  const auto semantic = Semantic(ctx, node);

  const TfLiteTensor* list_input;
  TF_LITE_ENSURE_OK(ctx, GetInputSafe(ctx, node, kListInputIdx, &list_input));
  TF_LITE_ENSURE_EQ(ctx, list_input->allocation_type, kTfLiteVariantObject);
  TF_LITE_ENSURE(ctx, list_input->data.data != nullptr);

  TensorArray* input_arr = static_cast<TensorArray*>(
      static_cast<VariantData*>(list_input->data.data));
  TF_LITE_ENSURE(ctx, input_arr != nullptr);

  int index;
  TF_LITE_ENSURE_OK(ctx, semantic.GetIndexVal(*input_arr, index));

  const TfLiteTensor* item_input;
  TF_LITE_ENSURE_OK(
      ctx, GetInputSafe(ctx, node, semantic.kItemInputIdx, &item_input));
  TF_LITE_ENSURE_TYPES_EQ(ctx, input_arr->ElementType(), item_input->type);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(ctx, GetOutputSafe(ctx, node, kListOutputIdx, &output));

  TensorArray* output_arr = static_cast<TensorArray*>(
      input_arr->CloneTo(static_cast<VariantData*>(output->data.data)));
  output->data.data = static_cast<VariantData*>(output_arr);

  IntArrayUniquePtr item_dims = BuildTfLiteArray(*item_input->dims);
  TF_LITE_ENSURE(ctx, item_dims != nullptr);
  // TODO(b/288302706) Skip copy when tensor is used only once.
  TensorUniquePtr item_copy;
  if (item_input->type == kTfLiteVariant) {
    item_copy = BuildTfLiteTensor();
    TF_LITE_ENSURE(ctx, item_copy != nullptr);
    item_copy->type = kTfLiteVariant;
    item_copy->bytes = item_input->bytes;
    item_copy->allocation_type = kTfLiteVariantObject;
  } else {
    item_copy = BuildTfLiteTensor(item_input->type, std::move(item_dims),
                                  kTfLiteDynamic);
    TF_LITE_ENSURE(ctx, item_copy != nullptr);
    if (item_copy->bytes > 0) {
      TF_LITE_ENSURE(ctx, item_copy->data.data != nullptr);
    }
  }
  TF_LITE_ENSURE_OK(ctx, TfLiteTensorCopy(item_input, item_copy.get()));

  if (index >= output_arr->NumElements()) {
    TF_LITE_ENSURE(ctx, output_arr->Resize(index + 1));
  }
  TF_LITE_ENSURE(ctx, output_arr->Set(index, std::move(item_copy)));
  return kTfLiteOk;
}

}  // namespace
TfLiteRegistration* Register_LIST_SET_ITEM() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare<SetItemSemantic>,
                                 Eval<SetItemSemantic>};
  return &r;
}

TfLiteRegistration* Register_LIST_PUSH_BACK() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare<PushBackSemantic>,
                                 Eval<PushBackSemantic>};
  return &r;
}
}  // namespace ops
}  // namespace variants
}  // namespace tflite
