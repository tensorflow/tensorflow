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

#include "tensorflow/lite/core/c/c_api_experimental.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/delegate_test_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"

using testing::HasSubstr;
using tflite::delegates::test_utils::SimpleDelegate;
using tflite::delegates::test_utils::TestDelegate;

namespace {

const TfLiteRegistration* GetNoOpRegistration() {
  static const TfLiteRegistration registration = {
      /*init=*/nullptr,
      /*free=*/nullptr,
      /*prepare=*/nullptr,
      /*invoke=*/[](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }};
  return &registration;
}

const TfLiteOperator* GetNoOpOperator() {
  static TfLiteOperator* registration = TfLiteOperatorCreate(
      kTfLiteBuiltinCustom, "NoOp", 1, /* user_data */ nullptr);
  TfLiteOperatorSetInvoke(
      registration,
      /*invoke=*/[](TfLiteOpaqueContext*, TfLiteOpaqueNode*) {
        return kTfLiteOk;
      });
  return registration;
}

TEST(CApiExperimentalTest, Smoke) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddBuiltinOp(options, kTfLiteBuiltinAdd,
                                       GetNoOpRegistration(), 1, 1);
  TfLiteInterpreterOptionsSetUseNNAPI(options, true);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// Test using TfLiteInterpreterCreateWithSelectedOps.
TEST(CApiExperimentalTest, SelectedBuiltins) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddBuiltinOp(options, kTfLiteBuiltinAdd,
                                       GetNoOpRegistration(), 1, 1);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// Test that when using TfLiteInterpreterCreateWithSelectedOps,
// we do NOT get the standard builtin operators by default.
TEST(CApiExperimentalTest, MissingBuiltin) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  // Install a custom error reporter into the interpreter by way of options.
  tflite::TestErrorReporter reporter;
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetErrorReporter(
      options,
      [](void* user_data, const char* format, va_list args) {
        reinterpret_cast<tflite::TestErrorReporter*>(user_data)->Report(format,
                                                                        args);
      },
      &reporter);

  // Create an interpreter with no builtins at all.
  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);

  // Check that interpreter creation failed, because the model contain a buitin
  // op that wasn't supported, and that we got the expected error messages.
  ASSERT_EQ(interpreter, nullptr);
  EXPECT_THAT(
      reporter.error_messages(),
      HasSubstr("Didn't find op for builtin opcode 'ADD' version '1'."));
  EXPECT_EQ(reporter.num_calls(), 2);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

struct OpResolverData {
  bool called_for_add = false;
};

const TfLiteRegistration* MyFindBuiltinOp(void* user_data,
                                          TfLiteBuiltinOperator op,
                                          int version) {
  OpResolverData* my_data = static_cast<OpResolverData*>(user_data);
  if (op == kTfLiteBuiltinAdd && version == 1) {
    my_data->called_for_add = true;
    return GetNoOpRegistration();
  }
  return nullptr;
}

const TfLiteRegistration* MyFindCustomOp(void*, const char* custom_op,
                                         int version) {
  if (absl::string_view(custom_op) == "foo" && version == 1) {
    return GetNoOpRegistration();
  }
  return nullptr;
}

// Test using TfLiteInterpreterCreateWithSelectedOps.
TEST(CApiExperimentalTest, SetOpResolver) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  OpResolverData my_data;
  TfLiteInterpreterOptionsSetOpResolver(options, MyFindBuiltinOp,
                                        MyFindCustomOp, &my_data);
  EXPECT_FALSE(my_data.called_for_add);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  EXPECT_TRUE(my_data.called_for_add);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

const TfLiteOperator* MyFindBuiltinOpExternal(void* user_data, int op,
                                              int version) {
  OpResolverData* my_data = static_cast<OpResolverData*>(user_data);
  if (op == kTfLiteBuiltinAdd && version == 1) {
    my_data->called_for_add = true;
    return GetNoOpOperator();
  }
  return nullptr;
}

const TfLiteOperator* MyFindCustomOpExternal(void*, const char* custom_op,
                                             int version) {
  if (absl::string_view(custom_op) == "foo" && version == 1) {
    return GetNoOpOperator();
  }
  return nullptr;
}

TfLiteStatus SinhPrepareOpaque(TfLiteOpaqueContext*, TfLiteOpaqueNode*) {
  return kTfLiteOk;
}

TfLiteStatus SinhEvalOpaque(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node) {
  EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfInputs(node));
  const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
  size_t input_bytes = TfLiteOpaqueTensorByteSize(input);
  const void* data_ptr = TfLiteOpaqueTensorData(input);
  float input_value;
  std::memcpy(&input_value, data_ptr, input_bytes);

  EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfOutputs(node));
  TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);
  float output_value = std::sinh(input_value);
  TfLiteOpaqueTensorCopyFromBuffer(output, &output_value, sizeof(output_value));
  return kTfLiteOk;
}

TfLiteStatus SinhPrepare(TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }

TfLiteStatus SinhEval(TfLiteContext* context, TfLiteNode* node) {
  EXPECT_EQ(1, node->inputs->size);
  const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  size_t input_bytes = TfLiteTensorByteSize(input);
  const void* data_ptr = TfLiteTensorData(input);
  float input_value;
  std::memcpy(&input_value, data_ptr, input_bytes);

  EXPECT_EQ(1, node->outputs->size);
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  float output_value = std::sinh(input_value);
  TfLiteTensorCopyFromBuffer(output, &output_value, sizeof(output_value));
  return kTfLiteOk;
}

const TfLiteOperator* SinhFindCustomOpExternal(void*, const char* custom_op,
                                               int version) {
  if (absl::string_view(custom_op) == "Sinh" && version == 1) {
    static TfLiteOperator* registration = []() {
      TfLiteOperator* reg = TfLiteOperatorCreate(kTfLiteBuiltinCustom, "Sinh",
                                                 1, /* user_data */ nullptr);
      TfLiteOperatorSetPrepare(reg, &SinhPrepareOpaque);
      TfLiteOperatorSetInvoke(reg, &SinhEvalOpaque);
      return reg;
    }();
    return registration;
  }
  return nullptr;
}

const TfLiteRegistration* SinhFindCustomOp(void*, const char* custom_op,
                                           int version) {
  if (absl::string_view(custom_op) == "Sinh" && version == 1) {
    static const TfLiteRegistration registration{/*init=*/nullptr,
                                                 /*free=*/nullptr,
                                                 /*prepare=*/SinhPrepare,
                                                 /*invoke=*/SinhEval};
    return &registration;
  }
  return nullptr;
}

// Test using TfLiteInterpreterOptionsSetOpResolverExternal and
// TfLiteInterpreterCreateWithSelectedOps.
TEST(CApiExperimentalTest, SetOpResolverExternal) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  OpResolverData my_data;
  TfLiteInterpreterOptionsSetOpResolverExternal(
      options, MyFindBuiltinOpExternal, MyFindCustomOpExternal, &my_data);
  EXPECT_FALSE(my_data.called_for_add);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  EXPECT_TRUE(my_data.called_for_add);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// Test using TfLiteInterpreterOptionsSetOpResolverExternalWithFallback and
// TfLiteInterpreterCreateWithSelectedOps, for a builtin op, for the normal
// case where the op is found with the primary op resolver callback that returns
// a TfLiteOperator pointer.
TEST(CApiExperimentalTest,
     SetOpResolverExternalWithFallback_BuiltinOp_NormalCase) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  OpResolverData my_data;
  TfLiteInterpreterOptionsSetOpResolverExternalWithFallback(
      options, MyFindBuiltinOpExternal, MyFindCustomOpExternal,
      [](void* user_data, TfLiteBuiltinOperator op,
         int version) -> const TfLiteRegistration* { return nullptr; },
      [](void* user_data, const char* custom_op,
         int version) -> const TfLiteRegistration* { return nullptr; },
      &my_data);
  EXPECT_FALSE(my_data.called_for_add);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  EXPECT_TRUE(my_data.called_for_add);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// Test using TfLiteInterpreterOptionsSetOpResolverExternalWithFallback and
// TfLiteInterpreterCreateWithSelectedOps, for a builtin op, for the fallback
// case where the op is found with the secondary op resolver callback that
// returns a TfLiteRegistration pointer.
TEST(CApiExperimentalTest,
     SetOpResolverExternalWithFallback_BuiltinOp_FallbackCase) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  OpResolverData my_data;
  TfLiteInterpreterOptionsSetOpResolverExternalWithFallback(
      options,
      [](void* user_data, int op, int version) -> const TfLiteOperator* {
        return nullptr;
      },
      [](void* user_data, const char* custom_op,
         int version) -> const TfLiteOperator* { return nullptr; },
      MyFindBuiltinOp, MyFindCustomOp, &my_data);
  EXPECT_FALSE(my_data.called_for_add);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  EXPECT_TRUE(my_data.called_for_add);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// Test using TfLiteInterpreterOptionsSetOpResolverExternalWithFallback and
// TfLiteInterpreterCreateWithSelectedOps, for a custom op, for the normal
// case where the op is found with the primary op resolver callback that returns
// a TfLiteOperator pointer.
TEST(CApiExperimentalTest,
     SetOpResolverExternalWithFallback_CustomOp_NormalCase) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/custom_sinh.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  OpResolverData my_data;
  TfLiteInterpreterOptionsSetOpResolverExternalWithFallback(
      options, MyFindBuiltinOpExternal, SinhFindCustomOpExternal,
      [](void* user_data, TfLiteBuiltinOperator op,
         int version) -> const TfLiteRegistration* { return nullptr; },
      [](void* user_data, const char* custom_op,
         int version) -> const TfLiteRegistration* { return nullptr; },
      &my_data);
  EXPECT_FALSE(my_data.called_for_add);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  const float input_value = 1.0f;
  TfLiteTensorCopyFromBuffer(input_tensor, &input_value, sizeof(float));

  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  float output_value;
  TfLiteTensorCopyToBuffer(output_tensor, &output_value, sizeof(float));
  EXPECT_EQ(output_value, std::sinh(input_value));

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// Test using TfLiteInterpreterOptionsSetOpResolverExternalWithFallback and
// TfLiteInterpreterCreateWithSelectedOps, for a custom op, for the fallback
// case where the op is found with the secondary op resolver callback that
// returns a TfLiteRegistration pointer.
TEST(CApiExperimentalTest,
     SetOpResolverExternalWithFallback_CustomOp_FallbackCase) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/custom_sinh.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  OpResolverData my_data;
  TfLiteInterpreterOptionsSetOpResolverExternalWithFallback(
      options,
      [](void* user_data, int op, int version) -> const TfLiteOperator* {
        return nullptr;
      },
      [](void* user_data, const char* custom_op,
         int version) -> const TfLiteOperator* { return nullptr; },
      MyFindBuiltinOp, SinhFindCustomOp, &my_data);
  EXPECT_FALSE(my_data.called_for_add);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  const float input_value = 1.0f;
  TfLiteTensorCopyFromBuffer(input_tensor, &input_value, sizeof(float));

  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  EXPECT_FALSE(my_data.called_for_add);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  float output_value;
  TfLiteTensorCopyToBuffer(output_tensor, &output_value, sizeof(float));
  EXPECT_EQ(output_value, std::sinh(input_value));

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// The following helper functions for custom allocation related tests are
// adapted from //tensorflow/lite/interpreter_test.cc.

// Returns the size of the alignment gap between `offset` and the address that
// is aligned to a multiple of 'alignment'. The value returned will be less than
// `alignment`.
size_t GetAlignGap(size_t alignment, uintptr_t offset) {
  return offset % alignment == 0 ? 0 : alignment - offset % alignment;
}

// Creates a new custom allocation. The allocation is aligned to the specified
// `required_alignment`. Actual initialized allocation is more than num_bytes,
// to account for required_alignment.
// Note that `new_alloc` will be pointed to the newly allocated memory for later
// destruction.
TfLiteCustomAllocation NewCustomAlloc(size_t num_bytes, int required_alignment,
                                      char** new_alloc) {
  *new_alloc = new char[num_bytes + required_alignment - 1];
  // Extra memory to ensure alignment.
  char* new_underlying_buffer_aligned_ptr =
      *new_alloc +
      GetAlignGap(required_alignment, reinterpret_cast<uintptr_t>(*new_alloc));

  return TfLiteCustomAllocation({new_underlying_buffer_aligned_ptr, num_bytes});
}

// Test using TfLiteInterpreterSetCustomAllocationForTensor.
TEST(CApiExperimentalTest, SetCustomAllocationForInputTensorSuccess) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  int tensor_idx = 0;
  // Checks null allocation.
  ASSERT_EQ(
      TfLiteInterpreterSetCustomAllocationForTensor(
          interpreter, tensor_idx, nullptr, kTfLiteCustomAllocationFlagsNone),
      kTfLiteError);

  int required_alignment = tflite::kDefaultTensorAlignment;
  const TfLiteTensor* input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, tensor_idx);
  char* new_alloc;
  auto input_tensor_alloc =
      NewCustomAlloc(input_tensor->bytes, required_alignment, &new_alloc);
  ASSERT_EQ(TfLiteInterpreterSetCustomAllocationForTensor(
                interpreter, tensor_idx, &input_tensor_alloc,
                kTfLiteCustomAllocationFlagsNone),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  delete[] new_alloc;
  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

TEST(CApiExperimentalTest, SetCustomAllocationForOutputTensorSuccess) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);
  int tensor_idx = 0;
  std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, tensor_idx, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  // Sets custom allocation for output tensor.
  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, tensor_idx);
  char* new_alloc;
  int required_alignment = tflite::kDefaultTensorAlignment;
  auto output_tensor_alloc =
      NewCustomAlloc(output_tensor->bytes, required_alignment, &new_alloc);
  ASSERT_EQ(TfLiteInterpreterSetCustomAllocationForTensor(
                interpreter, tensor_idx, &output_tensor_alloc,
                kTfLiteCustomAllocationFlagsNone),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  // Verifies output are expected.
  std::array<float, 2> input = {1.f, 3.f};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(
                TfLiteInterpreterGetInputTensor(interpreter, tensor_idx),
                input.data(), input.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  std::array<float, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(
                TfLiteInterpreterGetOutputTensor(interpreter, tensor_idx),
                output.data(), output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3.f);
  EXPECT_EQ(output[1], 9.f);

  delete[] new_alloc;
  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

TEST(CApiExperimentalTest, SetAndGetBufferHandleSuccess) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  auto simple_delegate = std::make_unique<SimpleDelegate>(
      // The delegate will handle the first (index 0) and the second (index 1)
      // op nodes in the TfLiteModel.
      /*nodes=*/std::vector<int>({0, 1}),
      /*delegate_flags=*/kTfLiteDelegateFlagsNone,
      /*fail_node_prepare=*/false, /*min_ops_per_subset=*/0,
      /*fail_node_invoke=*/false,
      /* automatic_shape_propagation=*/false, /*custom_op=*/false,
      /* set_output_tensor_dynamic =*/false);
  TfLiteDelegate* delegate = simple_delegate->get_tf_lite_delegate();

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);
  EXPECT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  // Tensor index is set to the input tensor (index 1) of the TfLiteModel.
  int tensor_index = 1;
  TfLiteTensor* tensor = TfLiteInterpreterGetTensor(interpreter, tensor_index);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);
  ASSERT_EQ(tensor->delegate, nullptr);

  // Use of an arbitrary non-negative int value for the buffer handle.
  TfLiteBufferHandle buffer_handle = 1234;

  TfLiteDelegate* expected_delegate = delegate;
  TfLiteBufferHandle expected_buffer_handle = buffer_handle;
  ASSERT_EQ(TfLiteInterpreterSetBufferHandle(interpreter, tensor, buffer_handle,
                                             delegate),
            kTfLiteOk);
  ASSERT_EQ(tensor->delegate, expected_delegate);
  ASSERT_EQ(tensor->buffer_handle, expected_buffer_handle);

  TfLiteOpaqueDelegate* fetched_delegate;
  TfLiteBufferHandle fetched_buffer_handle;
  ASSERT_EQ(
      TfLiteInterpreterGetBufferHandle(
          interpreter, tensor_index, &fetched_buffer_handle, &fetched_delegate),
      kTfLiteOk);
  ASSERT_EQ(fetched_delegate, expected_delegate);
  ASSERT_EQ(fetched_buffer_handle, expected_buffer_handle);

  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// A utility struct, intended to be used to record the interaction between a
// test delegate and the runtime.
struct DelegateState {
  bool delegate_prepared;
  bool copy_from_buffer_handle_called;
  bool free_buffer_handle_called;
  int buffer_handle;

  void Reset() {
    delegate_prepared = false;
    copy_from_buffer_handle_called = false;
    free_buffer_handle_called = false;
    buffer_handle = -1;
  }
};

struct OpaqueTestDelegate {
  static constexpr int kTestDelegateOutput = 42;

  static inline TfLiteStatus Prepare(TfLiteOpaqueContext* opaque_context,
                                     TfLiteOpaqueDelegate* opaque_delegate,
                                     void* data) {
    DelegateState* delegate_state = reinterpret_cast<DelegateState*>(data);
    delegate_state->delegate_prepared = true;

    // The buffer handle is set to one greater than the last allocated buffer
    // handle.
    delegate_state->buffer_handle++;

    TfLiteRegistration registration{};
    registration.registration_external = TfLiteOperatorCreate(
        kTfLiteBuiltinDelegate, "OpaqueTestDelegate delegate kernel",
        /* version = */ 1, /* user_data = */ nullptr);

    TfLiteOperatorSetPrepare(
        registration.registration_external,
        [](TfLiteOpaqueContext* context,
           TfLiteOpaqueNode* node) -> TfLiteStatus { return kTfLiteOk; });

    TfLiteOperatorSetInvoke(
        registration.registration_external,
        [](TfLiteOpaqueContext*, TfLiteOpaqueNode*) -> TfLiteStatus {
          return kTfLiteOk;
        });

    TfLiteIntArray* execution_plan;
    TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan);

    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, registration.registration_external, execution_plan,
        opaque_delegate);
    return kTfLiteOk;
  }

  static TfLiteStatus CopyFromBufferHandle(TfLiteOpaqueContext* context,
                                           TfLiteOpaqueDelegate* delegate,
                                           void* data,
                                           TfLiteBufferHandle buffer_handle,
                                           TfLiteOpaqueTensor* opaque_tensor) {
    DelegateState* delegate_state = reinterpret_cast<DelegateState*>(data);
    delegate_state->copy_from_buffer_handle_called = true;
    delegate_state->buffer_handle = buffer_handle;

    auto* output =
        reinterpret_cast<float*>(TfLiteOpaqueTensorData(opaque_tensor));
    int total_num_elements = 1;
    for (int i = 0; i < TfLiteOpaqueTensorNumDims(opaque_tensor); ++i) {
      total_num_elements *= TfLiteOpaqueTensorDim(opaque_tensor, i);
    }
    std::vector<float> meaning_of_life(total_num_elements, kTestDelegateOutput);
    memcpy(output, meaning_of_life.data(),
           meaning_of_life.size() * sizeof(float));
    return kTfLiteOk;
  }

  static inline void FreeBufferHandle(TfLiteOpaqueContext* context,
                                      TfLiteOpaqueDelegate* delegate,
                                      void* data,
                                      TfLiteBufferHandle* buffer_handle) {
    DelegateState* delegate_state = reinterpret_cast<DelegateState*>(data);
    delegate_state->free_buffer_handle_called = true;
  }
};

TEST(CApiExperimentalTest, SetAllowBufferHandleOutputFalse) {
  DelegateState delegate_state;
  delegate_state.Reset();

  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  int kNumTensorElements = 1 * 8 * 8 * 3;

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.CopyFromBufferHandle =
      OpaqueTestDelegate::CopyFromBufferHandle;
  opaque_delegate_builder.FreeBufferHandle =
      OpaqueTestDelegate::FreeBufferHandle;
  opaque_delegate_builder.Prepare = OpaqueTestDelegate::Prepare;

  TfLiteOpaqueDelegate* tflite_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, tflite_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // Allocate tensor buffers.
  EXPECT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  // Fill input buffers
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  float* input = reinterpret_cast<float*>(input_tensor->data.raw);
  std::fill(input, input + kNumTensorElements, 1);

  // We set the buffer handle of the output tensor and mark its data as stale.
  // This will make the interpreter call 'CopyFromBufferHandle' to refresh the
  // output tensor's data.
  int first_buffer_handle = 0;

  // Tensor index is set to the output tensor (index 2) of the TfLite model.
  int tensor_index = 2;

  TfLiteTensor* output_tensor =
      TfLiteInterpreterGetTensor(interpreter, tensor_index);

  ASSERT_EQ(
      TfLiteInterpreterSetBufferHandle(interpreter, output_tensor,
                                       first_buffer_handle, tflite_delegate),
      kTfLiteOk);

  output_tensor->data_is_stale = true;

  TfLiteSetAllowBufferHandleOutput(interpreter,
                                   /*allow_buffer_handle_output=*/false);

  // Run inference
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  EXPECT_TRUE(delegate_state.delegate_prepared);
  EXPECT_TRUE(delegate_state.copy_from_buffer_handle_called);
  EXPECT_EQ(delegate_state.buffer_handle, first_buffer_handle);
  EXPECT_FALSE(delegate_state.free_buffer_handle_called);
  float* outputs = reinterpret_cast<float*>(output_tensor->data.raw);
  for (int i = 0; i < kNumTensorElements; ++i) {
    EXPECT_EQ(outputs[i], OpaqueTestDelegate::kTestDelegateOutput);
  }
  ASSERT_EQ(output_tensor->buffer_handle, first_buffer_handle);
  ASSERT_EQ(output_tensor->delegate, tflite_delegate);

  // Destroying the interpreter will release any buffer handles that are
  // associated with the tensors owner by the interpreter.
  delegate_state.Reset();
  TfLiteInterpreterDelete(interpreter);
  TfLiteOpaqueDelegateDelete(tflite_delegate);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
  EXPECT_FALSE(delegate_state.copy_from_buffer_handle_called);
  EXPECT_TRUE(delegate_state.free_buffer_handle_called);
}

TEST(CApiExperimentalTest, SetAllowBufferHandleOutputTrue) {
  DelegateState delegate_state;
  delegate_state.Reset();

  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  int kNumTensorElements = 1 * 8 * 8 * 3;

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.CopyFromBufferHandle =
      OpaqueTestDelegate::CopyFromBufferHandle;
  opaque_delegate_builder.FreeBufferHandle =
      OpaqueTestDelegate::FreeBufferHandle;
  opaque_delegate_builder.Prepare = OpaqueTestDelegate::Prepare;

  TfLiteOpaqueDelegate* tflite_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, tflite_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // Allocate tensor buffers.
  EXPECT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  // Fill input buffers
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  float* input = reinterpret_cast<float*>(input_tensor->data.raw);
  std::fill(input, input + kNumTensorElements, 1);

  // We set the buffer handle of the output tensor and mark its data as stale.
  // This will make the interpreter call 'CopyFromBufferHandle' to refresh the
  // output tensor's data.
  EXPECT_FALSE(delegate_state.free_buffer_handle_called);
  int first_buffer_handle = 0;

  // Tensor index is set to the output tensor (index 2) of the TfLite model.
  int tensor_index = 2;

  TfLiteTensor* output_tensor =
      TfLiteInterpreterGetTensor(interpreter, tensor_index);

  ASSERT_EQ(
      TfLiteInterpreterSetBufferHandle(interpreter, output_tensor,
                                       first_buffer_handle, tflite_delegate),
      kTfLiteOk);

  output_tensor->data_is_stale = true;

  TfLiteSetAllowBufferHandleOutput(interpreter,
                                   /*allow_buffer_handle_output=*/true);

  // Run inference
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  EXPECT_TRUE(delegate_state.delegate_prepared);
  EXPECT_FALSE(delegate_state.copy_from_buffer_handle_called);
  EXPECT_EQ(delegate_state.buffer_handle, first_buffer_handle);
  EXPECT_FALSE(delegate_state.free_buffer_handle_called);
  ASSERT_EQ(output_tensor->buffer_handle, first_buffer_handle);
  ASSERT_EQ(output_tensor->delegate, tflite_delegate);

  // Destroying the interpreter will release any buffer handles that are
  // associated with the tensors owner by the interpreter.
  delegate_state.Reset();
  TfLiteInterpreterDelete(interpreter);
  TfLiteOpaqueDelegateDelete(tflite_delegate);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
  EXPECT_FALSE(delegate_state.copy_from_buffer_handle_called);
  EXPECT_TRUE(delegate_state.free_buffer_handle_called);
}

TEST(CApiExperimentalTest, SetInvalidHandleToTensor) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  auto simple_delegate = std::make_unique<SimpleDelegate>(
      // The delegate will handle the first (index 0) and the second (index 1)
      // op nodes in the TfLiteModel.
      /*nodes=*/std::vector<int>({0, 1}),
      /*delegate_flags=*/kTfLiteDelegateFlagsNone,
      /*fail_node_prepare=*/false, /*min_ops_per_subset=*/0,
      /*fail_node_invoke=*/false,
      /* automatic_shape_propagation=*/false, /*custom_op=*/false,
      /* set_output_tensor_dynamic =*/false);
  TfLiteDelegate* delegate = simple_delegate->get_tf_lite_delegate();

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  EXPECT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  auto another_simple_delegate = std::make_unique<SimpleDelegate>(
      // The delegate will handle the 0th, 1st and the 2nd indexed nodes in
      // the TfLiteModel.
      /*nodes=*/std::vector<int>({0, 1, 2}),
      /*delegate_flags=*/kTfLiteDelegateFlagsNone,
      /*fail_node_prepare=*/false, /*min_ops_per_subset=*/0,
      /*fail_node_invoke=*/false, /* automatic_shape_propagation=*/false,
      /*custom_op=*/false, /*set_output_tensor_dynamic=*/false);

  // Tensor index is set to the output tensor (index 2) of the TfLite model.
  int tensor_index = 2;
  TfLiteTensor* tensor = TfLiteInterpreterGetTensor(interpreter, tensor_index);

  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  // Buffer handle is set to one greater than the last allocated buffer handle.
  TfLiteBufferHandle buffer_handle = kTfLiteNullBufferHandle + 1;

  // Setting a buffer handle to a tensor with another delegate will fail.
  ASSERT_EQ(TfLiteInterpreterSetBufferHandle(
                interpreter, tensor, buffer_handle,
                another_simple_delegate->get_tf_lite_delegate()),
            kTfLiteError);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

void AllocateAndSetInputs(TfLiteInterpreter* interpreter) {
  std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  std::array<float, 2> input = {1.f, 3.f};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);
}

void VerifyOutputs(TfLiteInterpreter* interpreter) {
  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  std::array<float, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3.f);
  EXPECT_EQ(output[1], 9.f);
}

void CheckExecution(TfLiteInterpreterOptions* options,
                    TfLiteStatus expected_first_result,
                    TfLiteStatus expected_subsequent_results) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  AllocateAndSetInputs(interpreter);
  for (int i = 0; i < 4; i++) {
    bool result = TfLiteInterpreterInvoke(interpreter);
    bool expected_result =
        ((i == 0) ? expected_first_result : expected_subsequent_results);
    EXPECT_EQ(result, expected_result);
    if (result != kTfLiteError) {
      VerifyOutputs(interpreter);
    }
  }

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

TEST_F(TestDelegate, NoDelegate) {
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  // Execution without any delegate should succeed.
  CheckExecution(options, kTfLiteOk, kTfLiteOk);
  TfLiteInterpreterOptionsDelete(options);
}

TEST_F(TestDelegate, DelegateNodeInvokeFailure) {
  // Initialize a delegate that will fail when invoked.
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/,
      false /**automatic_shape_propagation**/, false /**custom_op**/));
  // Create another interpreter with the delegate, without fallback.
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options,
                                      delegate_->get_tf_lite_delegate());
  // Execution with the delegate should fail.
  CheckExecution(options, kTfLiteError, kTfLiteError);
  TfLiteInterpreterOptionsDelete(options);
}

TEST_F(TestDelegate, DelegateNodeInvokeFailureFallback) {
  // Initialize a delegate that will fail when invoked.
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0, 1}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/,
      false /**automatic_shape_propagation**/, false /**custom_op**/));
  // Create another interpreter with the delegate, with fallback enabled.
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options,
                                      delegate_->get_tf_lite_delegate());
  TfLiteInterpreterOptionsSetEnableDelegateFallback(options, true);
  CheckExecution(options,
                 // First execution will report DelegateError which indicates
                 // that the delegate failed but fallback succeeded.
                 kTfLiteDelegateError,
                 // Subsequent executions will not use the delegate and
                 // should therefore succeed.
                 kTfLiteOk);
  TfLiteInterpreterOptionsDelete(options);
}

TEST_F(TestDelegate, TestFallbackWithMultipleDelegates) {
  // First delegate only supports node 0.
  // This delegate should support dynamic tensors, otherwise the second won't be
  // applied.
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0}, kTfLiteDelegateFlagsAllowDynamicTensors,
      false /**fail_node_prepare**/, 0 /**min_ops_per_subset**/,
      true /**fail_node_invoke**/, false /**automatic_shape_propagation**/,
      false /**custom_op**/));
  // Second delegate supports node 1, and makes the graph immutable.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {1}, kTfLiteDelegateFlagsNone, false /**fail_node_prepare**/,
      0 /**min_ops_per_subset**/, true /**fail_node_invoke**/,
      false /**automatic_shape_propagation**/, false /**custom_op**/));
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options,
                                      delegate_->get_tf_lite_delegate());
  TfLiteInterpreterOptionsAddDelegate(options,
                                      delegate2_->get_tf_lite_delegate());
  TfLiteInterpreterOptionsSetEnableDelegateFallback(options, true);
  CheckExecution(options,
                 // First execution will report DelegateError which indicates
                 // that the delegate failed but fallback succeeded.
                 kTfLiteDelegateError,
                 // Subsequent executions will not use the delegate and
                 // should therefore succeed.
                 kTfLiteOk);
  TfLiteInterpreterOptionsDelete(options);
}

}  // namespace
