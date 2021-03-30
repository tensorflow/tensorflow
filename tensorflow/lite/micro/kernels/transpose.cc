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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kPermTensor = 1;
constexpr int kOutputTensor = 0;

struct TransposeContext {
    TransposeContext(TfLiteContext* context, TfLiteNode* node) {
        input = GetInput(context, node, kInputTensor);
        perm = GetInput(context, node, kPermTensor);
        output = GetOutput(context, node, kOutputTensor);
    }
    const TfLiteTensor* input;
    const TfLiteTensor* perm;
    TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    TransposeContext op_context(context, node);

    // Ensure validity of input tensor.
    TF_LITE_ENSURE_MSG(context, NumDimensions(op_context.input) <= 5,
                        "Transpose op only supports 1D-5D input arrays.");
    TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                            op_context.output->type);

    return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TransposeContext op_context(context, node);

  // Retrieve the perm permutation array
  const int32_t* perm_data = GetTensorData<int32_t>(op_context.perm);

  // Determine the number of dimensions in the perm array
  const int size = op_context.perm->dims->data[0];

  // Prepare an params object to store the perm data whilst implementing
  // the conversion 
  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }

  // Helper operation to acquire and convert data types
#define TF_LITE_TRANSPOSE(scalar)                     \
  reference_ops::Transpose(params, GetTensorShape(op_context.input), \
                  GetTensorData<scalar>(op_context.input),  \
                  GetTensorShape(op_context.output),        \
                  GetTensorData<scalar>(op_context.output))

  // Transpose really operates at the byte level,
  // and therefore we only really need to get the 
  // size of the scalar datatype in bytes.
  // Using this we can simplify the calls
  // to only use a small number of data types
  switch (op_context.input->type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
      TF_LITE_TRANSPOSE(int32_t);
      break;
    case kTfLiteInt8:
    case kTfLiteUInt8:
      TF_LITE_TRANSPOSE(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_TRANSPOSE(int16_t);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %s is currently not supported by Transpose.",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }

#undef TF_LITE_TRANSPOSE

  return kTfLiteOk;
}

} // namespace transpose

TfLiteRegistration Register_TRANSPOSE() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/2};
}

} // namespace tflite
