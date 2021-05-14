// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace sign {

// Performs common preparation for pointwise, unary ops, i.e., type checks and
// output tensor resizing.
TfLiteStatus PointwiseUnaryOpPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);

  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  // Validate size and type constraints
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

// Applies the operator Op pointwise to data of type T.
template <typename Op, typename T>
TfLiteStatus PointwiseUnaryOpDoEval(
    TfLiteContext* context,
    const TfLiteTensor* input,
    TfLiteTensor* output) {
  const T* data = tflite::GetTensorData<T>(input);
  T* data_output = tflite::GetTensorData<T>(output);

  const int64_t num_elements = NumElements(input);
  for (int64_t i = 0; i < num_elements; ++i) {
    data_output[i] = Op::template Eval<T>(data[i]);
  }

  return TfLiteStatus::kTfLiteOk;
}

// A generic evaluation function where the actual data processing is handled
// by the Op::Eval<T> function.
template <typename Op>
TfLiteStatus PointwiseUnaryOpEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  switch (output->type) {
    case kTfLiteFloat32:
      TF_LITE_ENSURE_OK(
          context,
          (PointwiseUnaryOpDoEval<Op, float>(context, input, output)));
      break;
    case kTfLiteFloat64:
      TF_LITE_ENSURE_OK(
          context,
          (PointwiseUnaryOpDoEval<Op, double>(context, input, output)));
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Unsupported datatype for atan2 output: %s",
          TfLiteTypeGetName(output->type));
  }

  return TfLiteStatus::kTfLiteOk;
}

// Operator that computes the sign function.
struct Sign {
  template <typename T>
  static T Eval(T x) {
    if (x > 0) {
      return 1;
    }
    if (x < 0) {
      return -1;
    }
    return 0;
  }
};

}  // namespace sign

TfLiteRegistration* Register_SIGN() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 sign::PointwiseUnaryOpPrepare,
                                 sign::PointwiseUnaryOpEval<sign::Sign>};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
