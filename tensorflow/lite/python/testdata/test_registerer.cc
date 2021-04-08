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
#include "tensorflow/lite/python/testdata/test_registerer.h"

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {

namespace {

static int num_test_registerer_calls = 0;

TfLiteRegistration* GetFakeRegistration() {
  static TfLiteRegistration fake_op;
  return &fake_op;
}

namespace double_op {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);

  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);

  const size_t size = GetTensorShape(input).FlatSize();

  if (input->type == kTfLiteFloat32) {
    const float* input_ptr = input->data.f;
    float* output_ptr = output->data.f;
    for (int i = 0; i < size; ++i) {
      output_ptr[i] = input_ptr[i] + input_ptr[i];
    }
  } else if (input->type == kTfLiteInt32) {
    const int32_t* input_ptr = input->data.i32;
    int32_t* output_ptr = output->data.i32;
    for (int i = 0; i < size; ++i) {
      output_ptr[i] = input_ptr[i] + input_ptr[i];
    }
  } else {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace double_op

TfLiteRegistration* GetDoubleRegistration() {
  static TfLiteRegistration double_op = {nullptr, nullptr, double_op::Prepare,
                                         double_op::Eval};
  return &double_op;
}
}  // namespace

// Dummy registerer function with the correct signature. Registers a fake custom
// op needed by test models. Increments the num_test_registerer_calls counter by
// one. The TF_ prefix is needed to get past the version script in the OSS
// build.
extern "C" void TF_TestRegisterer(tflite::MutableOpResolver *resolver) {
  resolver->AddCustom("FakeOp", GetFakeRegistration());
  resolver->AddCustom("Double", GetDoubleRegistration());
  num_test_registerer_calls++;
}

// Returns the num_test_registerer_calls counter and re-sets it.
int get_num_test_registerer_calls() {
  const int result = num_test_registerer_calls;
  num_test_registerer_calls = 0;
  return result;
}

}  // namespace tflite
