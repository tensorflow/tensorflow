/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <string.h>
#include <vector>
#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace squeeze {

struct SqueezeContext {
  SqueezeContext(TfLiteContext* context, TfLiteNode* node)
      : params(reinterpret_cast<TfLiteSqueezeParams*>(node->builtin_data)),
        input(GetInput(context, node, 0)),
        output(GetOutput(context, node, 0)) {}
  TfLiteSqueezeParams* params;
  const TfLiteTensor* const input;
  TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  SqueezeContext op_context(context, node);
  int input_num_dims = NumDimensions(op_context.input);
  int num_squeeze_dims = op_context.params->num_squeeze_dims;

  // Determines number of dimensions of output tensor after squeeze.
  const TfLiteIntArray* input_dims = op_context.input->dims;
  const int* squeeze_dims = op_context.params->squeeze_dims;
  TF_LITE_ENSURE(context, input_num_dims <= 8);
  bool should_squeeze[8] = {false};
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
  TfLiteIntArray* output_dims =
      TfLiteIntArrayCreate(input_num_dims - num_squeezed_dims);
  for (int in_idx = 0, out_idx = 0; in_idx < input_num_dims; ++in_idx) {
    if (!should_squeeze[in_idx]) {
      output_dims->data[out_idx++] = input_dims->data[in_idx];
    }
  }
  return context->ResizeTensor(context, op_context.output, output_dims);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  SqueezeContext op_context(context, node);
  TF_LITE_ENSURE_EQ(context, op_context.input->bytes, op_context.output->bytes);
  memcpy(op_context.output->data.raw, op_context.input->data.raw,
         op_context.input->bytes);
  return kTfLiteOk;
}

}  // namespace squeeze

TfLiteRegistration* Register_SQUEEZE() {
  static TfLiteRegistration r = {nullptr, nullptr, squeeze::Prepare,
                                 squeeze::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
