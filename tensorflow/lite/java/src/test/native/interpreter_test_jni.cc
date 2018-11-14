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
#include "tensorflow/lite/c/c_api_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_InterpreterTest_getNativeHandleForDelegate(
    JNIEnv* env, jclass clazz) {
  // A simple op which outputs a vector of length 1 with the value [7].
  static TfLiteRegistration registration = {
      .init = nullptr,
      .free = nullptr,
      .prepare =
          [](TfLiteContext* context, TfLiteNode* node) {
            TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
            TfLiteIntArray* scalar_size = TfLiteIntArrayCreate(1);
            scalar_size->data[0] = 1;
            output->type = kTfLiteFloat32;
            return context->ResizeTensor(context, output, scalar_size);
          },
      .invoke =
          [](TfLiteContext* context, TfLiteNode* node) {
            TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
            output->data.f[0] = 7.0f;
            return kTfLiteOk;
          },
      .profiling_string = nullptr,
      .builtin_code = 0,
      .custom_name = "",
      .version = 1,
  };
  // A simple delegate which replaces all ops with a single op that outputs a
  // vector of length 1 with the value [7].
  static TfLiteDelegate delegate = {
      .data_ = nullptr,
      .Prepare = [](TfLiteContext* context,
                    TfLiteDelegate* delegate) -> TfLiteStatus {
        TfLiteIntArray* execution_plan;
        TF_LITE_ENSURE_STATUS(
            context->GetExecutionPlan(context, &execution_plan));
        context->ReplaceNodeSubsetsWithDelegateKernels(
            context, registration, execution_plan, delegate);
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
