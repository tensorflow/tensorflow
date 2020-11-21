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

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"

// This file exists just to verify that the above header files above can build,
// link, and run as "C" code.

#ifdef __cplusplus
#error "This file should be compiled as C code, not as C++."
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void CheckFailed(const char *expression, const char *filename,
                        int line_number) {
  fprintf(stderr, "ERROR: CHECK failed: %s:%d: %s\n", filename, line_number,
          expression);
  fflush(stderr);
  abort();
}

// We use an extra level of macro indirection here to ensure that the
// macro arguments get evaluated, so that in a call to CHECK(foo),
// the call to STRINGIZE(condition) in the definition of the CHECK
// macro results in the string "foo" rather than the string "condition".
#define STRINGIZE(expression) STRINGIZE2(expression)
#define STRINGIZE2(expression) #expression

// Like assert(), but not dependent on NDEBUG.
#define CHECK(condition) \
  ((condition) ? (void)0 \
               : CheckFailed(STRINGIZE(condition), __FILE__, __LINE__))
#define ASSERT_EQ(expected, actual) CHECK((expected) == (actual))
#define ASSERT_NE(expected, actual) CHECK((expected) != (actual))
#define ASSERT_STREQ(expected, actual) \
    ASSERT_EQ(0, strcmp((expected), (actual)))

// Test the TfLiteVersion function.
static void TestVersion(void) {
  const char *version = TfLiteVersion();
  printf("Version = %s\n", version);
  CHECK(version[0] != '\0');
}

static void TestSmokeTest(void) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, NULL);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, NULL);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, NULL);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  int input_dims[1] = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims, 1),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, NULL);
  ASSERT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  ASSERT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  ASSERT_EQ(TfLiteTensorDim(input_tensor, 0), 2);
  ASSERT_EQ(TfLiteTensorByteSize(input_tensor), sizeof(float) * 2);
  ASSERT_NE(TfLiteTensorData(input_tensor), NULL);
  ASSERT_STREQ(TfLiteTensorName(input_tensor), "input");

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  ASSERT_EQ(input_params.scale, 0.f);
  ASSERT_EQ(input_params.zero_point, 0);

  float input[2] = {1.f, 3.f};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input,
                                       2 * sizeof(float)),
            kTfLiteOk);

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, NULL);
  ASSERT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  ASSERT_EQ(TfLiteTensorNumDims(output_tensor), 1);
  ASSERT_EQ(TfLiteTensorDim(output_tensor, 0), 2);
  ASSERT_EQ(TfLiteTensorByteSize(output_tensor), sizeof(float) * 2);
  ASSERT_NE(TfLiteTensorData(output_tensor), NULL);
  ASSERT_STREQ(TfLiteTensorName(output_tensor), "output");

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  ASSERT_EQ(output_params.scale, 0.f);
  ASSERT_EQ(output_params.zero_point, 0);

  float output[2];
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output,
                                     2 * sizeof(float)),
            kTfLiteOk);
  ASSERT_EQ(output[0], 3.f);
  ASSERT_EQ(output[1], 9.f);

  TfLiteInterpreterDelete(interpreter);
}

static void RunTests(void) {
  TestVersion();
  TestSmokeTest();
}

int main(void) {
  RunTests();
  return 0;
}
