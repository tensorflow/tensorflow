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

#include "tensorflow/lite/experimental/micro/examples/network_tester/expected_output_data.h"
#include "tensorflow/lite/experimental/micro/examples/network_tester/input_data.h"
#include "tensorflow/lite/experimental/micro/examples/network_tester/network_model.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE (1024)
#endif

uint8_t tensor_arena[TENSOR_ARENA_SIZE];

#ifdef NUM_BYTES_TO_PRINT
inline void print_output_data(TfLiteTensor* output) {
  int num_bytes_to_print =
      (output->bytes < NUM_BYTES_TO_PRINT) ? output->bytes : NUM_BYTES_TO_PRINT;

  int dims_size = output->dims->size;
  printf("dims: {%d,", dims_size);
  for (int i = 0; i < output->dims->size - 1; ++i) {
    printf("%d,", output->dims->data[i]);
  }
  printf("%d}\n", output->dims->data[dims_size - 1]);

  printf("data_address: %p\n", output->data.raw);
  printf("data:\n{");
  for (int i = 0; i < num_bytes_to_print - 1; ++i) {
    if (i % 16 == 0) {
      printf("\n");
    }
    printf("0x%02x,", output->data.uint8[i]);
  }
  printf("0x%02x\n}\n", output->data.uint8[num_bytes_to_print - 1]);
}
#endif

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = ::tflite::GetModel(network_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  tflite::ops::micro::AllOpsResolver resolver;

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       TENSOR_ARENA_SIZE, error_reporter);
  interpreter.AllocateTensors();

  TfLiteTensor* input = interpreter.input(0);
  memcpy(input->data.uint8, input_data, input->bytes);

  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  TfLiteTensor* output = interpreter.output(0);

#ifdef NUM_BYTES_TO_PRINT
  print_output_data(output);
#endif

#ifndef NO_COMPARE_OUTPUT_DATA
  for (int i = 0; i < output->bytes; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(output->data.uint8[i], expected_output_data[i]);
  }
#endif
  error_reporter->Report("Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
