/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/kernel_util.h"

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace micro {

bool HaveSameShapes(const TfLiteEvalTensor* input1,
                    const TfLiteEvalTensor* input2) {
  TFLITE_DCHECK(input1 != nullptr);
  TFLITE_DCHECK(input2 != nullptr);
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

const RuntimeShape GetTensorShape(const TfLiteEvalTensor* tensor) {
  if (tensor == nullptr || tensor->dims == nullptr) {
    return RuntimeShape();
  }
  TfLiteIntArray* dims = tensor->dims;
  const int dims_size = dims->size;
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims->data);
  return RuntimeShape(dims_size, dims_data);
}

PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

// Relocate tensor dims from FlatBuffer to the persistent storage arena.
// The old dims data is copied to the new storage area.
// The tensor and eval_tensor must be the same tensor.
// Only use during Prepare phase.
TfLiteStatus CreateWritableTensorDimsWithCopy(TfLiteContext* context,
                                              TfLiteTensor* tensor,
                                              TfLiteEvalTensor* eval_tensor) {
  TF_LITE_ENSURE(context, tensor != nullptr);
  TF_LITE_ENSURE(context, eval_tensor != nullptr);
  int ranks = tensor->dims->size;
  size_t alloc_size = TfLiteIntArrayGetSizeInBytes(ranks);
  TfLiteIntArray* new_dims = static_cast<TfLiteIntArray*>(
      context->AllocatePersistentBuffer(context, alloc_size));
  TfLiteIntArray* old_dims = tensor->dims;
  new_dims->size = ranks;
  tensor->dims = new_dims;
  eval_tensor->dims = new_dims;
  for (int i = 0; i < ranks; i++) {
    new_dims->data[i] = old_dims->data[i];
  }

  return kTfLiteOk;
}

}  // namespace micro
}  // namespace tflite
