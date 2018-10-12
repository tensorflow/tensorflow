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

#include "tensorflow/contrib/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.h"
#include "tensorflow/contrib/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/contrib/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/contrib/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/contrib/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"
#include "tensorflow/contrib/lite/version.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = ::tflite::GetModel(g_tiny_conv_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }
  tflite::ops::micro::AllOpsResolver resolver;

  const int tensor_arena_size = 10 * 1024;
  uint8_t tensor_arena[tensor_arena_size];
  tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                                 tensor_arena_size);

  tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                       error_reporter);
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  error_reporter->Report("Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
