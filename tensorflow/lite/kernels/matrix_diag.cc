/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace matrix_diag {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteIntArray* input_dims = input->dims;
  int input_dims_size = input_dims->size;
  TF_LITE_ENSURE(context, input_dims_size >= 1);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  // Resize the output tensor.
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(input_dims_size + 1);
  for (int i = 0; i < input_dims_size; i++) {
    output_shape->data[i] = input_dims->data[i];
  }
  // Last dimension in the output is the same as the last dimension in the
  // input.
  output_shape->data[input_dims_size] = input_dims->data[input_dims_size - 1];
  output->type = input->type;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_shape));

  return kTfLiteOk;
}

// Fill the tensor to make a diagonal matrix in each batch, i.e., when
// row index and column index are the same, fill with the next input value.
// All other entries get zero.
// TODO(b/128636574) Move to reference_ops.
template <typename T>
void FillDiagImpl(const T* in, T* out, const int batch_size, const int row_size,
                  const int col_size) {
  int idx = 0;
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < row_size; i++) {
      for (int j = 0; j < col_size; ++j) {
        // input values go on the diagonal, 0 elsewhere
        if (i == j) {
          out[i * col_size + j] = in[idx];
          idx++;
        } else {
          out[i * col_size + j] = 0;
        }
      }
    }
    out += row_size * col_size;
  }
}

template <typename T>
void FillDiag(const TfLiteTensor* input, TfLiteTensor* output,
              const int batch_size, const int row_size, const int col_size) {
  FillDiagImpl<T>(GetTensorData<T>(input), GetTensorData<T>(output), batch_size,
                  row_size, col_size);
}

// Fill a tensor with given input on the diagonal, zero elsewhere
void FillDiagHelper(const TfLiteTensor* input, TfLiteTensor* output) {
  const int num_output_dims = output->dims->size;
  int batch_size = 1;
  for (int i = 0; i < num_output_dims - 2; ++i) {
    batch_size *= output->dims->data[i];
  }

  const int row_size = output->dims->data[num_output_dims - 2];
  const int col_size = output->dims->data[num_output_dims - 1];
  switch (output->type) {
    case kTfLiteInt64: {
      return FillDiag<int64_t>(input, output, batch_size, row_size, col_size);
    }
    case kTfLiteInt32: {
      return FillDiag<int32_t>(input, output, batch_size, row_size, col_size);
    }
    case kTfLiteInt16: {
      return FillDiag<int16_t>(input, output, batch_size, row_size, col_size);
    }
    case kTfLiteInt8: {
      return FillDiag<int8_t>(input, output, batch_size, row_size, col_size);
    }
    case kTfLiteUInt8: {
      return FillDiag<uint8_t>(input, output, batch_size, row_size, col_size);
    }
    default:
      return FillDiag<float>(input, output, batch_size, row_size, col_size);
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  FillDiagHelper(input, output);
  return kTfLiteOk;
}

}  // namespace matrix_diag

TfLiteRegistration* Register_MATRIX_DIAG() {
  static TfLiteRegistration r = {nullptr, nullptr, matrix_diag::Prepare,
                                 matrix_diag::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
