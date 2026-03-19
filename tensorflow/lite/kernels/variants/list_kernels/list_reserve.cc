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
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

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

constexpr int kListOut = 0;

struct SemanticOutType {
  TfLiteType element_type;
  IntArrayUniquePtr element_shape;
  int num_elements;
};

class ReserveSemantic {
 public:
  ReserveSemantic(TfLiteContext* context, TfLiteNode* node)
      : context_(context), node_(node) {}

  constexpr static int kElementShapeInput = 0;
  constexpr static int kNumElementsInput = 1;

  TfLiteStatus CheckInputs() const {
    TF_LITE_ENSURE_EQ(context_, NumInputs(node_), 2);
    const TfLiteTensor* element_shape;
    TF_LITE_ENSURE_OK(
        context_,
        GetInputSafe(context_, node_, kElementShapeInput, &element_shape));
    TF_LITE_ENSURE(context_, element_shape->type == kTfLiteInt32);
    const TfLiteTensor* num_elements;
    TF_LITE_ENSURE_OK(context_, GetInputSafe(context_, node_, kNumElementsInput,
                                             &num_elements));
    TF_LITE_ENSURE_TYPES_EQ(context_, num_elements->type, kTfLiteInt32);
    return kTfLiteOk;
  }

  TfLiteStatus Compute(SemanticOutType& result) const {
    // Parse element type from custom options.
    auto* options =
        reinterpret_cast<const ListReserveOptions*>(node_->custom_initial_data);
    TfLiteType element_type = ConvertTensorType(options->element_type);
    TF_LITE_ENSURE(context_, element_type != kTfLiteNoType);

    const TfLiteTensor* num_elements;
    TF_LITE_ENSURE_OK(context_, GetInputSafe(context_, node_, kNumElementsInput,
                                             &num_elements));
    TF_LITE_ENSURE_TYPES_EQ(context_, num_elements->type, kTfLiteInt32);
    TF_LITE_ENSURE_EQ(context_, num_elements->dims->size, 0);
    const int num_elements_value = num_elements->data.i32[0];
    TF_LITE_ENSURE(context_, num_elements_value >= 0);

    // Create int array representing constraint on list's constituent elements.
    const TfLiteTensor* element_shape_tensor;
    TF_LITE_ENSURE_OK(context_,
                      GetInputSafe(context_, node_, kElementShapeInput,
                                   &element_shape_tensor));
    IntArrayUniquePtr element_shape = TensorAsShape(*element_shape_tensor);

    result = SemanticOutType{element_type, std::move(element_shape),
                             num_elements_value};
    return kTfLiteOk;
  }

  TfLiteStatus PopulateOutput(TensorArray* const output) const {
    return kTfLiteOk;
  }

 private:
  TfLiteContext* const context_;
  TfLiteNode* const node_;
};

class ZerosLikeSemantic {
 public:
  ZerosLikeSemantic(TfLiteContext* context, TfLiteNode* node)
      : context_(context), node_(node) {}

  constexpr static int kListInput = 0;

  TfLiteStatus CheckInputs() const {
    TF_LITE_ENSURE_EQ(context_, NumInputs(node_), 1);
    const TfLiteTensor* list_input;
    TF_LITE_ENSURE_OK(context_,
                      GetInputSafe(context_, node_, kListInput, &list_input));
    TF_LITE_ENSURE(context_, list_input->type == kTfLiteVariant);
    return kTfLiteOk;
  }

  TfLiteStatus Compute(SemanticOutType& result) const {
    const TfLiteTensor* list_input;
    TF_LITE_ENSURE_OK(context_,
                      GetInputSafe(context_, node_, kListInput, &list_input));
    const TensorArray* const input =
        reinterpret_cast<const TensorArray*>(list_input->data.data);

    result = SemanticOutType{input->ElementType(),
                             BuildTfLiteArray(*input->ElementShape()),
                             input->NumElements()};
    return kTfLiteOk;
  }

  TfLiteStatus PopulateOutput(TensorArray* const output) const {
    const TfLiteTensor* list_input;
    TF_LITE_ENSURE_OK(context_,
                      GetInputSafe(context_, node_, kListInput, &list_input));
    const TensorArray* const input =
        reinterpret_cast<const TensorArray*>(list_input->data.data);
    for (int i = 0; i < input->NumElements(); ++i) {
      const TfLiteTensor* const at = input->At(i);
      if (at == nullptr) continue;
      // Tensorflow supports lazy allocation in this case which is not possible
      // with tflite tensors. If this proves to be a performance bottleneck we
      // can investigate storing more info in TensorArray putting off allocation
      // for later.
      TensorUniquePtr output_at = BuildTfLiteTensor(
          at->type, BuildTfLiteArray(*at->dims), kTfLiteDynamic);
      memset(output_at->data.data, 0, output_at->bytes);
      TF_LITE_ENSURE(context_, output->Set(i, std::move(output_at)));
    }
    return kTfLiteOk;
  }

 private:
  TfLiteContext* const context_;
  TfLiteNode* const node_;
};

template <class Semantic>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const Semantic sem(context, node);
  TF_LITE_ENSURE_OK(context, sem.CheckInputs());
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kListOut, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteVariant);
  output->allocation_type = kTfLiteVariantObject;
  return kTfLiteOk;
}

template <class Semantic>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const Semantic sem(context, node);
  SemanticOutType data;
  TF_LITE_ENSURE_OK(context, sem.Compute(data));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kListOut, &output));

  // Construct new `TensorArray` underneath the output tensor.
  TfLiteStatus stat = TfLiteTensorVariantRealloc<TensorArray>(
      output, data.element_type, std::move(data.element_shape));
  TF_LITE_ENSURE_OK(context, stat);

  // Set size of array.
  TensorArray* const arr =
      static_cast<TensorArray*>(static_cast<VariantData*>(output->data.data));
  arr->Resize(data.num_elements);
  TF_LITE_ENSURE_OK(context, sem.PopulateOutput(arr));

  return kTfLiteOk;
}
}  // namespace
}  // namespace list_reserve

TfLiteRegistration* Register_LIST_RESERVE() {
  static TfLiteRegistration r = {
      nullptr, nullptr, list_reserve::Prepare<list_reserve::ReserveSemantic>,
      list_reserve::Eval<list_reserve::ReserveSemantic>};
  return &r;
}

TfLiteRegistration* Register_VARIANT_ZEROS_LIKE() {
  static TfLiteRegistration r = {
      nullptr, nullptr, list_reserve::Prepare<list_reserve::ZerosLikeSemantic>,
      list_reserve::Eval<list_reserve::ZerosLikeSemantic>};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
