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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace squeeze {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// TODO(See TfLiteSqueezeParams): We can't have dynamic data, at least not yet.
// For now we will fix the maximum possible number of dimensions.
constexpr int max_num_dims = 8;

struct SqueezeContext {
  SqueezeContext(TfLiteContext* context, TfLiteNode* node)
      : params(reinterpret_cast<TfLiteSqueezeParams*>(node->builtin_data)),
        input(GetInput(context, node, kInputTensor)),
        output(GetOutput(context, node, kOutputTensor)) {}
  TfLiteSqueezeParams* params;
  const TfLiteTensor* const input;
  TfLiteTensor* output;
};

TfLiteStatus SqueezeOutput(TfLiteContext* context, TfLiteNode* node) {
  SqueezeContext op_context(context, node);
  // Determines number of dimensions of output tensor after squeeze.
  int input_num_dims = NumDimensions(op_context.input);
  int num_squeeze_dims = op_context.params->num_squeeze_dims;
  const int* squeeze_dims = op_context.params->squeeze_dims;
  const TfLiteIntArray* input_dims = op_context.input->dims;
  TF_LITE_ENSURE(context, input_num_dims <= max_num_dims);
  bool should_squeeze[max_num_dims] = {false};
  int num_squeezed_dims = 0;
  if (num_squeeze_dims == 0) {
    for (int idx = 0; idx < input_num_dims; ++idx) {
      if (input_dims->data[idx] == 1) {
        should_squeeze[idx] = true;
        ++num_squeezed_dims;
      }
    }
  } else {
    for (int idx = 0; idx < num_squeeze_dims; ++idx) {
      int current = squeeze_dims[idx] < 0 ? squeeze_dims[idx] + input_num_dims
                                          : squeeze_dims[idx];
      TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims &&
                                  input_dims->data[current] == 1);
      if (!should_squeeze[current]) ++num_squeezed_dims;
      should_squeeze[current] = true;
    }
  }
  // Sets output dimensions.
  // Allocate new output_dims from node's temporaries buffer
  TfLiteIntArray* output_dims = node->temporaries;
  output_dims->size = input_num_dims - num_squeezed_dims;
  for (int in_idx = 0, out_idx = 0; in_idx < input_num_dims; ++in_idx) {
    if (!should_squeeze[in_idx]) {
      output_dims->data[out_idx++] = input_dims->data[in_idx];
    }
  }
  op_context.output->dims = output_dims;

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  if (SqueezeOutput(context, node) != kTfLiteOk) {
    return kTfLiteError;
  }
  SqueezeContext op_context(context, node);
  // Copy input data to output data
  for (int i = 0; i < op_context.input->bytes; ++i) {
    op_context.output->data.raw[i] = op_context.input->data.raw[i];
  }
  return kTfLiteOk;
}

}  // namespace squeeze

TfLiteRegistration* Register_SQUEEZE() {
  static TfLiteRegistration r = {nullptr, nullptr, squeeze::Prepare,
                                 squeeze::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
