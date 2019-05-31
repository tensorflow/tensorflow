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

#include <jni.h>
#include <algorithm>
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_InterpreterTest_getNativeHandleForDelegate(
    JNIEnv* env, jclass clazz) {
  // A simple op which outputs a tensor with values of 7.
  static TfLiteRegistration registration = {
      .init = nullptr,
      .free = nullptr,
      .prepare =
          [](TfLiteContext* context, TfLiteNode* node) {
            TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
            TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
            TfLiteIntArray* output_dims = TfLiteIntArrayCopy(input->dims);
            output->type = kTfLiteFloat32;
            return context->ResizeTensor(context, output, output_dims);
          },
      .invoke =
          [](TfLiteContext* context, TfLiteNode* node) {
            TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
            std::fill(output->data.f,
                      output->data.f + tflite::NumElements(output), 7.0f);
            return kTfLiteOk;
          },
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "",
      .version = 1,
  };
  static TfLiteDelegate delegate = {
      .data_ = nullptr,
      .Prepare = [](TfLiteContext* context,
                    TfLiteDelegate* delegate) -> TfLiteStatus {
        TfLiteIntArray* execution_plan;
        TF_LITE_ENSURE_STATUS(
            context->GetExecutionPlan(context, &execution_plan));
        context->ReplaceNodeSubsetsWithDelegateKernels(
            context, registration, execution_plan, delegate);
        // Now bind delegate buffer handles for all tensors.
        for (size_t i = 0; i < context->tensors_size; ++i) {
          context->tensors[i].delegate = delegate;
          context->tensors[i].buffer_handle = static_cast<int>(i);
        }
        return kTfLiteOk;
      },
      .CopyFromBufferHandle = nullptr,
      .CopyToBufferHandle = nullptr,
      .FreeBufferHandle = nullptr,
      .flags = kTfLiteDelegateFlagsAllowDynamicTensors,
  };
  return reinterpret_cast<jlong>(&delegate);
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_InterpreterTest_getNativeHandleForInvalidDelegate(
    JNIEnv* env, jclass clazz) {
  // A simple delegate that fails during preparation.
  static TfLiteDelegate delegate = {
      .data_ = nullptr,
      .Prepare = [](TfLiteContext* context, TfLiteDelegate* delegate)
          -> TfLiteStatus { return kTfLiteError; },
      .CopyFromBufferHandle = nullptr,
      .CopyToBufferHandle = nullptr,
      .FreeBufferHandle = nullptr,
      .flags = kTfLiteDelegateFlagsNone,
  };
  return reinterpret_cast<jlong>(&delegate);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
