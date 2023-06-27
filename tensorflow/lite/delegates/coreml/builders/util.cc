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
#include "tensorflow/lite/delegates/coreml/builders/util.h"

#include <vector>

#include "fp16.h"  // from @FP16
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/coreml/builders/op_validator.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace coreml {
namespace {
void Get4DShape(const TfLiteTensor* tensor, std::vector<int>* shape) {
  const int rank = tensor->dims->size;
  shape->resize(4);
  for (int i = 0; i < 4 - rank; i++) {
    (*shape)[i] = 1;
  }
  for (int i = 4 - rank; i < 4; ++i) {
    (*shape)[i] = tensor->dims->data[i - (4 - rank)];
  }
}
}  // namespace

// Determines if two tensor shapes are broadcastable. See comment of
// IsBinaryOpSupported for more info.
bool IsBroadcastable(const TfLiteTensor* input_0, const TfLiteTensor* input_1) {
  std::vector<int> shape_0;
  std::vector<int> shape_1;
  Get4DShape(input_0, &shape_0);
  Get4DShape(input_1, &shape_1);
  const int B_0 = shape_0[0];
  const int B_1 = shape_1[0];
  const int H_0 = shape_0[1];
  const int H_1 = shape_1[1];
  const int W_0 = shape_0[2];
  const int W_1 = shape_1[2];
  const int C_0 = shape_0[3];
  const int C_1 = shape_1[3];

  // TFL tensor has [B, H, W, C] format.
  // comparing B: shape[0], (H, W): (shape[1], shape[2]), C: shape[3].

  // When B is different, it's not supported unless
  // one of the tensor is size 1 constant tensor.
  if (B_0 != B_1) {
    if (!((IsConstantTensor(input_0) && NumElements(input_0) == 1) ||
          (IsConstantTensor(input_1) && NumElements(input_1) == 1)))
      return false;
  }

  // When (H, W) are different, one of the (H, W) should be (1, 1).
  if (H_0 != H_1 || W_0 != W_1) {
    if (!((H_0 == 1 && W_0 == 1) || (H_1 == 1 && W_1 == 1))) {
      return false;
    }
  }

  // When C is different, one of the C should be 1.
  if (C_0 != C_1) {
    if (C_0 != 1 && C_1 != 1) return false;
  }
  return true;
}

bool IsBinaryOpSupportedType(const TfLiteTensor* tensor) {
  return tensor->type == kTfLiteFloat32 ||
         (tensor->type == kTfLiteFloat16 && IsConstantTensor(tensor));
}

bool IsBinaryOpSupported(const TfLiteRegistration* registration,
                         const TfLiteNode* node, TfLiteContext* context) {
  const auto* input_0 = GetInput(context, node, 0);
  const auto* input_1 = GetInput(context, node, 1);
  if (IsBinaryOpSupportedType(input_0) && IsBinaryOpSupportedType(input_1)) {
    return IsBroadcastable(input_0, input_1);
  }
  return false;
}

float GetScalarFloatFromTensor(const TfLiteTensor* tensor) {
  if (tensor->type == kTfLiteFloat16) {
    return fp16_ieee_to_fp32_value(GetTensorData<uint16_t>(tensor)[0]);
  }
  return GetTensorData<float>(tensor)[0];
}

}  // namespace coreml
}  // namespace delegates
}  // namespace tflite
