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

#include "tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_model.h"
#include "tensorflow/lite/experimental/micro/examples/mnist_demo/model/mnist_test_data.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(mnist_dense_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // This pulls in all the operation implementations we need.
  tflite::ops::micro::AllOpsResolver resolver;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, currently
  // determined by experimentation.
  const int tensor_arena_size = 5 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();

  // Make sure the input has the properties we expect.
  TfLiteTensor* model_input = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, model_input);
  TF_LITE_MICRO_EXPECT_EQ(3, model_input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, model_input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(28, model_input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(28, model_input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, model_input->type);

  // perform inference on each test sample and evalute accuracy of model
  int accurateCount = 0;
  for (int s = 0; s < mnistSampleCount; ++s) {
    // Set value of input tensor
    for (int d = 0; d < 784; ++d) model_input->data.f[d] = mnistInput[s][d];

    // perform inference
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed.\n");
      return 1;
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

    // Make sure the output has the properties we expect.
    TfLiteTensor* model_output = interpreter.output(0);
    TF_LITE_MICRO_EXPECT_NE(nullptr, model_output);
    TF_LITE_MICRO_EXPECT_EQ(1, model_output->dims->size);
    TF_LITE_MICRO_EXPECT_EQ(1, model_output->dims->data[0]);
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, model_output->type);

    if (model_output->data.i32[0] == mnistOutput[s]) ++accurateCount;
  }

  // test passes with an accuracy of 90% or more
  TF_LITE_MICRO_EXPECT_GE(accurateCount, mnistSampleCount * 0.9);

  error_reporter->Report("Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
