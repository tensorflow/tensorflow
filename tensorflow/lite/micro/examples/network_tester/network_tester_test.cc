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

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/network_tester/expected_output_data.h"
#include "tensorflow/lite/micro/examples/network_tester/input_data.h"
#include "tensorflow/lite/micro/examples/network_tester/network_model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#ifndef TENSOR_ARENA_SIZE
#define TENSOR_ARENA_SIZE (1024)
#endif

#ifndef NUM_INFERENCES
#define NUM_INFERENCES 1
#endif

uint8_t tensor_arena[TENSOR_ARENA_SIZE];

#ifdef NUM_BYTES_TO_PRINT
inline void print_output_data(TfLiteTensor* output) {
  int num_bytes_to_print =
      ((output->bytes < NUM_BYTES_TO_PRINT) || NUM_BYTES_TO_PRINT == 0)
          ? output->bytes
          : NUM_BYTES_TO_PRINT;

  int dims_size = output->dims->size;
  printf("{\n");
  printf("\"dims\": [%d,", dims_size);
  for (int i = 0; i < output->dims->size - 1; ++i) {
    printf("%d,", output->dims->data[i]);
  }
  printf("%d],\n", output->dims->data[dims_size - 1]);

  printf("\"data_address\": \"%p\",\n", output->data.raw);
  printf("\"data\":\"");
  for (int i = 0; i < num_bytes_to_print - 1; ++i) {
    if (i % 16 == 0 && i != 0) {
      printf("\n");
    }
    printf("0x%02x,", output->data.uint8[i]);
  }
  printf("0x%02x\"\n", output->data.uint8[num_bytes_to_print - 1]);
  printf("}");
}
#endif

template <typename T>
void check_output_elem(TfLiteTensor* output, const T* expected_output_data,
                       const int index) {
  TF_LITE_MICRO_EXPECT_EQ(tflite::GetTensorData<T>(output)[index],
                          expected_output_data[index]);
}

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = ::tflite::GetModel(network_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  tflite::AllOpsResolver resolver;

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       TENSOR_ARENA_SIZE, error_reporter);

  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Tensor allocation failed\n");
    return kTfLiteError;
  }

  for (int n = 0; n < NUM_INFERENCES; n++) {
    for (int i = 0; i < interpreter.inputs_size(); ++i) {
      TfLiteTensor* input = interpreter.input(i);
      memcpy(input->data.data, input_data[i], input->bytes);
    }
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
      return kTfLiteError;
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

#ifdef NUM_BYTES_TO_PRINT
    // Print all of the output data, or the first NUM_BYTES_TO_PRINT bytes,
    // whichever comes first as well as the output shape.
    printf("num_of_outputs: %d\n", interpreter.outputs_size());
    printf("output_begin\n");
    printf("[\n");
    for (int i = 0; i < interpreter.outputs_size(); i++) {
      TfLiteTensor* output = interpreter.output(i);
      print_output_data(output);
      if (i != interpreter.outputs_size() - 1) {
        printf(",\n");
      }
    }
    printf("]\n");
    printf("output_end\n");
#endif

#ifndef NO_COMPARE_OUTPUT_DATA
    for (int i = 0; i < interpreter.outputs_size(); i++) {
      TfLiteTensor* output = interpreter.output(i);
      for (int j = 0; j < tflite::ElementCount(*(output->dims)); ++j) {
        check_output_elem(output, expected_output_data[i], j);
      }
    }
#endif
  }
  TF_LITE_REPORT_ERROR(error_reporter, "Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
