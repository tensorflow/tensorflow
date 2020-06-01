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

#include "tensorflow/lite/micro/examples/image_recognition_experimental/first_10_cifar_images.h"
#include "tensorflow/lite/micro/examples/image_recognition_experimental/image_recognition_model.h"
#include "tensorflow/lite/micro/examples/image_recognition_experimental/util.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define IMAGE_BYTES 3072
#define LABEL_BYTES 1
#define ENTRY_BYTES (IMAGE_BYTES + LABEL_BYTES)

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestImageRecognitionInvoke) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = ::tflite::GetModel(image_recognition_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  tflite::MicroMutableOpResolver<4> micro_op_resolver;

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());

  const int tensor_arena_size = 50 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();

  TfLiteTensor* input = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(32, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(32, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(3, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  int num_correct = 0;
  int num_images = 10;
  for (int image_num = 0; image_num < num_images; image_num++) {
    memset(input->data.uint8, 0, input->bytes);

    uint8_t correct_label = 0;

    correct_label =
        tensorflow_lite_micro_tools_make_downloads_cifar10_test_batch_bin
            [image_num * ENTRY_BYTES];
    memcpy(input->data.uint8,
           &tensorflow_lite_micro_tools_make_downloads_cifar10_test_batch_bin
               [image_num * ENTRY_BYTES + LABEL_BYTES],
           IMAGE_BYTES);
    reshape_cifar_image(input->data.uint8, IMAGE_BYTES);

    TfLiteStatus invoke_status = interpreter.Invoke();
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
    }

    TfLiteTensor* output = interpreter.output(0);
    int guess = get_top_prediction(output->data.uint8, 10);

    if (correct_label == guess) {
      num_correct++;
    }
  }

  TF_LITE_MICRO_EXPECT_EQ(6, num_correct);
}

TF_LITE_MICRO_TESTS_END
