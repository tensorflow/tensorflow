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

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_experimental.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

// This file exists just to verify that the above header files above can build,
// link, and run as "C" code.

#ifdef __cplusplus
#error "This file should be compiled as C code, not as C++."
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdbool.h>

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

static void TestInferenceUsingSignature(void) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/multi_signatures.bin");
  ASSERT_NE(model, NULL);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, NULL);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, NULL);

  // The options can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);

  // (optional) Validate signatures
  ASSERT_EQ(TfLiteInterpreterGetSignatureCount(interpreter), 2);
  ASSERT_STREQ(TfLiteInterpreterGetSignatureKey(interpreter, 0), "add");
  ASSERT_STREQ(TfLiteInterpreterGetSignatureKey(interpreter, 1), "sub");

  // Validate signature "add"
  TfLiteSignatureRunner* add_runner =
      TfLiteInterpreterGetSignatureRunner(interpreter, "add");
  ASSERT_NE(add_runner, NULL);
  ASSERT_EQ(TfLiteSignatureRunnerGetInputCount(add_runner), 1);
  ASSERT_STREQ(TfLiteSignatureRunnerGetInputName(add_runner, 0), "x");
  ASSERT_EQ(TfLiteSignatureRunnerGetOutputCount(add_runner), 1);
  ASSERT_STREQ(TfLiteSignatureRunnerGetOutputName(add_runner, 0), "output_0");

  // Resize signature "add" input tensor "x"
  int input_dims[1] = {2};
  ASSERT_EQ(
      TfLiteSignatureRunnerResizeInputTensor(add_runner, "x", input_dims, 1),
      kTfLiteOk);

  // Allocate tensors for signature "add"
  ASSERT_EQ(TfLiteSignatureRunnerAllocateTensors(add_runner), kTfLiteOk);

  // Validate signature "add" input tensor "x"
  TfLiteTensor* input_tensor =
      TfLiteSignatureRunnerGetInputTensor(add_runner, "x");
  ASSERT_NE(input_tensor, NULL);
  ASSERT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  ASSERT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  ASSERT_EQ(TfLiteTensorDim(input_tensor, 0), 2);
  ASSERT_EQ(TfLiteTensorByteSize(input_tensor), sizeof(float) * 2);
  ASSERT_NE(TfLiteTensorData(input_tensor), NULL);

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  ASSERT_EQ(input_params.scale, 0.f);
  ASSERT_EQ(input_params.zero_point, 0);

  float input[2] = {2.f, 4.f};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input, 2 * sizeof(float)),
            kTfLiteOk);
  ASSERT_EQ(TfLiteSignatureRunnerInvoke(add_runner), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteSignatureRunnerGetOutputTensor(add_runner, "output_0");
  ASSERT_NE(output_tensor, NULL);
  ASSERT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  ASSERT_EQ(TfLiteTensorNumDims(output_tensor), 1);
  ASSERT_EQ(TfLiteTensorDim(output_tensor, 0), 2);
  ASSERT_EQ(TfLiteTensorByteSize(output_tensor), sizeof(float) * 2);
  ASSERT_NE(TfLiteTensorData(output_tensor), NULL);

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  ASSERT_EQ(output_params.scale, 0.f);
  ASSERT_EQ(output_params.zero_point, 0);

  float output[2];
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output, 2 * sizeof(float)),
            kTfLiteOk);
  // Verify the result
  ASSERT_EQ(output[0], input[0] + 2.f);
  ASSERT_EQ(output[1], input[1] + 2.f);

  // The signature runner should be deleted before interpreter deletion.
  TfLiteSignatureRunnerDelete(add_runner);
  TfLiteInterpreterDelete(interpreter);
  // The model should only be deleted after destroying the interpreter.
  TfLiteModelDelete(model);
}

// This test checks if resizing the input (decreasing or increasing it's size)
// would invalidate input/output tensors.
static void TestRepeatResizeInputTensor(void) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/multi_signatures.bin");
  ASSERT_NE(model, NULL);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, NULL);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, NULL);

  TfLiteInterpreterOptionsDelete(options);

  ASSERT_EQ(TfLiteInterpreterGetSignatureCount(interpreter), 2);
  ASSERT_STREQ(TfLiteInterpreterGetSignatureKey(interpreter, 0), "add");
  ASSERT_STREQ(TfLiteInterpreterGetSignatureKey(interpreter, 1), "sub");

  TfLiteSignatureRunner* add_runner =
      TfLiteInterpreterGetSignatureRunner(interpreter, "add");
  ASSERT_NE(add_runner, NULL);
  ASSERT_EQ(TfLiteSignatureRunnerGetInputCount(add_runner), 1);
  ASSERT_STREQ(TfLiteSignatureRunnerGetInputName(add_runner, 0), "x");
  ASSERT_EQ(TfLiteSignatureRunnerGetOutputCount(add_runner), 1);
  ASSERT_STREQ(TfLiteSignatureRunnerGetOutputName(add_runner, 0), "output_0");

  TfLiteTensor* input_tensor =
      TfLiteSignatureRunnerGetInputTensor(add_runner, "x");
  const TfLiteTensor* output_tensor =
      TfLiteSignatureRunnerGetOutputTensor(add_runner, "output_0");

  // For different input sizes, resize the input/output tensors and check if
  // inferences runs as expected.

  int sizes[] = {3, 1, 5};
  float inputs_1[] = {3.f, 6.f, 11.f};
  float inputs_2[] = {4.f};
  float inputs_3[] = {5.f, 8.f, 11.f, 12.f, 20.f};
  float* all_inputs[] = {inputs_1, inputs_2, inputs_3};
  float actual_outputs1[] = {0.f, 0.f, 0.f};
  float actual_outputs2[] = {0.f};
  float actual_outputs3[] = {0.f, 0.f, 0.f, 0.f, 0.f};
  float* all_actual_outputs[] = {actual_outputs1, actual_outputs2,
                                 actual_outputs3};

  for (int i = 0; i < 3; i++) {
    int input_dims[] = {sizes[i]};
    float* inputs = all_inputs[i];
    ASSERT_EQ(
        TfLiteSignatureRunnerResizeInputTensor(add_runner, "x", input_dims, 1),
        kTfLiteOk);
    ASSERT_EQ(TfLiteSignatureRunnerAllocateTensors(add_runner), kTfLiteOk);
    ASSERT_NE(input_tensor, NULL);
    ASSERT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
    ASSERT_EQ(TfLiteTensorNumDims(input_tensor), 1);
    ASSERT_EQ(TfLiteTensorDim(input_tensor, 0), sizes[i]);
    ASSERT_EQ(TfLiteTensorByteSize(input_tensor), sizes[i] * sizeof(float));
    ASSERT_NE(TfLiteTensorData(input_tensor), NULL);
    ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, inputs,
                                         sizes[i] * sizeof(float)),
              kTfLiteOk);
    ASSERT_EQ(TfLiteSignatureRunnerInvoke(add_runner), kTfLiteOk);
    ASSERT_NE(output_tensor, NULL);
    ASSERT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
    ASSERT_EQ(TfLiteTensorNumDims(output_tensor), 1);
    ASSERT_EQ(TfLiteTensorDim(output_tensor, 0), sizes[i]);
    ASSERT_EQ(TfLiteTensorByteSize(output_tensor), sizes[i] * sizeof(float));
    ASSERT_NE(TfLiteTensorData(output_tensor), NULL);
    float* actual_outputs = all_actual_outputs[i];
    ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, actual_outputs,
                                       sizes[i] * sizeof(float)),
              kTfLiteOk);
    for (int j = 0; j < sizes[i]; j++) {
      ASSERT_EQ(actual_outputs[j], inputs[j] + 2);
    }
  }

  TfLiteSignatureRunnerDelete(add_runner);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

static void TestInferenceUsingInterpreter(void) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, NULL);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, NULL);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, NULL);

  // The options can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  int input_dims[1] = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, 1),
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
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input, 2 * sizeof(float)),
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
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output, 2 * sizeof(float)),
            kTfLiteOk);
  ASSERT_EQ(output[0], 3.f);
  ASSERT_EQ(output[1], 9.f);

  TfLiteInterpreterDelete(interpreter);

  // The model should only be deleted after destroying the interpreter.
  TfLiteModelDelete(model);
}

TfLiteStatus PrepareThatChecksExecutionPlanSizeEqualsTwo(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueDelegate* opaque_delegate, void* data) {
  bool* delegate_prepared = (bool*)data;
  *delegate_prepared = true;

  TfLiteIntArray* execution_plan;
  ASSERT_EQ(kTfLiteOk,
            TfLiteOpaqueContextGetExecutionPlan(context, &execution_plan));
  ASSERT_EQ(2, execution_plan->size);

  return kTfLiteOk;
}

static void TestTfLiteOpaqueContextGetExecutionPlan(void) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  bool delegate_prepared = false;
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder = { NULL };
  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = PrepareThatChecksExecutionPlanSizeEqualsTwo;
  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The delegate should have been applied.
  CHECK(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

static void TestTfLiteOpaqueContextReportErrorMacros(
    TfLiteStatus (*Prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueDelegate* delegate, void* data)) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  bool delegate_prepared_called = false;
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder = { NULL };
  opaque_delegate_builder.data = &delegate_prepared_called;
  opaque_delegate_builder.Prepare = Prepare;
  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The delegate's prepare function should have been called, even though it
  // returned an error code.
  CHECK(delegate_prepared_called);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TfLiteStatus TfLiteOpaqueContextReportErrorMacros_EnsureMsg_Prepare(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueDelegate* opaque_delegate, void* data) {
    bool* delegate_prepared = (bool*) data;
    *delegate_prepared = true;
    TF_LITE_OPAQUE_ENSURE_MSG(context, false, "false was not true!!!");
    return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextReportErrorMacros_Ensure_Prepare(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueDelegate* opaque_delegate, void* data) {
    bool* delegate_prepared = (bool*) data;
    *delegate_prepared = true;
    TF_LITE_OPAQUE_ENSURE(context, false);
    return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextReportErrorMacros_EnsureEq_Prepare(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueDelegate* opaque_delegate, void* data) {
    bool* delegate_prepared = (bool*) data;
    *delegate_prepared = true;
    TF_LITE_OPAQUE_ENSURE_EQ(context, 1, 2);
    return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextReportErrorMacros_EnsureTypesEq_Prepare(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueDelegate* opaque_delegate, void* data) {
    bool* delegate_prepared = (bool*) data;
    *delegate_prepared = true;
    TF_LITE_OPAQUE_ENSURE_TYPES_EQ(context, '1', 2);
    return kTfLiteOk;
}

TfLiteStatus TfLiteOpaqueContextReportErrorMacros_EnsureNear_Prepare(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueDelegate* opaque_delegate, void* data) {
    bool* delegate_prepared = (bool*) data;
    *delegate_prepared = true;
    TF_LITE_OPAQUE_ENSURE_NEAR(context, 3, 10, 1);
    return kTfLiteOk;
}

static void RunTests(void) {
  TestVersion();
  TestInferenceUsingSignature();
  TestRepeatResizeInputTensor();
  TestInferenceUsingInterpreter();
  TestTfLiteOpaqueContextGetExecutionPlan();
  TestTfLiteOpaqueContextReportErrorMacros(
      TfLiteOpaqueContextReportErrorMacros_Ensure_Prepare);
  TestTfLiteOpaqueContextReportErrorMacros(
      TfLiteOpaqueContextReportErrorMacros_EnsureMsg_Prepare);
  TestTfLiteOpaqueContextReportErrorMacros(
      TfLiteOpaqueContextReportErrorMacros_EnsureEq_Prepare);
  TestTfLiteOpaqueContextReportErrorMacros(
      TfLiteOpaqueContextReportErrorMacros_EnsureTypesEq_Prepare);
  TestTfLiteOpaqueContextReportErrorMacros(
      TfLiteOpaqueContextReportErrorMacros_EnsureNear_Prepare);
}

int main(void) {
  RunTests();
  return 0;
}
