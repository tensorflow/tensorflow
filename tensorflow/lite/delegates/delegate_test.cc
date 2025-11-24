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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/experimental/remat/metadata_util.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/delegate_test_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace delegates {

using test_utils::SimpleDelegate;
using test_utils::TestDelegate;
using test_utils::TestDelegateWithControlEdges;
using test_utils::TestFP16Delegation;
using test_utils::TestTwoDelegates;

namespace {

TEST_F(TestDelegate, NullDelegate) {
  TfLiteOpaqueDelegate* delegate = nullptr;
  EXPECT_EQ(interpreter_->ModifyGraphWithDelegate(delegate),
            kTfLiteDelegateError);
}

TEST_F(TestDelegate, BasicDelegate) {
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  int node = interpreter_->execution_plan()[0];
  const auto* node_and_reg = interpreter_->node_and_registration(node);
  EXPECT_EQ(node_and_reg->second.custom_name,
            delegate_->FakeFusedRegistration().custom_name);

  const TfLiteDelegateParams* params = static_cast<const TfLiteDelegateParams*>(
      node_and_reg->first.builtin_data);
  ASSERT_EQ(params->nodes_to_replace->size, 3);
  EXPECT_EQ(params->nodes_to_replace->data[0], 0);
  EXPECT_EQ(params->nodes_to_replace->data[1], 1);
  EXPECT_EQ(params->nodes_to_replace->data[2], 2);

  ASSERT_EQ(params->input_tensors->size, 2);
  EXPECT_EQ(params->input_tensors->data[0], 0);
  EXPECT_EQ(params->input_tensors->data[1], 1);

  ASSERT_EQ(params->output_tensors->size, 2);
  EXPECT_EQ(params->output_tensors->data[0], 3);
  EXPECT_EQ(params->output_tensors->data[1], 4);
}

TEST_F(TestDelegate, DelegateNodeInitFailure) {
  delegate_ = std::make_unique<SimpleDelegate>(
      std::vector<int>{0, 1, 2}, kTfLiteDelegateFlagsNone,
      SimpleDelegate::Options::kFailOnInit);
  // ModifyGraphWithDelegate fails, since the Init() method in the node's
  // TfLiteRegistration returns an error status.
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteDelegateError);
}

TEST_F(TestDelegate, DelegateNodePrepareFailure) {
  delegate_ = std::make_unique<SimpleDelegate>(
      std::vector<int>{0, 1, 2}, kTfLiteDelegateFlagsNone,
      SimpleDelegate::Options::kFailOnPrepare);
  // ModifyGraphWithDelegate fails, since the Prepare() method in the node's
  // TfLiteRegistration returns an error status.
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteDelegateError);
  // Execution plan should remain unchanged.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, DelegateNodeInvokeFailure) {
  delegate_ = std::make_unique<SimpleDelegate>(
      std::vector<int>{0, 1, 2}, kTfLiteDelegateFlagsNone,
      SimpleDelegate::Options::kFailOnInvoke);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Delegation modified execution plan.
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;

  // Verify Invoke() behavior: fails first, succeeds after RemoveAllDelegates().
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  ASSERT_EQ(RemoveAllDelegates(), kTfLiteOk);
  // Delegation removed, returning to original execution plan.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, StaticDelegateMakesGraphImmutable) {
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  // Deliberately try to set tensor params with quantization while immutable,
  // ensuring quantization is properly freed.
  TfLiteQuantization quant = {};
  quant.type = kTfLiteAffineQuantization;
  auto quant_params = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  quant_params->scale = nullptr;
  quant_params->zero_point = nullptr;
  quant_params->quantized_dimension = 0;
  quant.params = quant_params;
  ASSERT_NE(interpreter_->SetTensorParametersReadWrite(0, kTfLiteInt8, "", {3},
                                                       quant),
            kTfLiteOk);
}

TEST_F(TestDelegate, ComplexDelegate) {
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{1, 2});
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  ASSERT_EQ(interpreter_->execution_plan().size(), 2);
  // 0th should be a non-delegated original op
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
  // 1st should be a new macro op (3) which didn't exist)
  ASSERT_EQ(interpreter_->execution_plan()[1], 3);
  const auto* node_and_reg = interpreter_->node_and_registration(3);
  ASSERT_EQ(node_and_reg->second.custom_name,
            delegate_->FakeFusedRegistration().custom_name);
}

TEST_F(TestDelegate, SetBufferHandleToInput) {
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kInputTensorIndex = 0;
  TfLiteTensor* tensor = interpreter_->tensor(kInputTensorIndex);
  ASSERT_EQ(tensor->delegate, nullptr);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kInputTensorIndex, handle, delegate);
  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
}

TEST_F(TestDelegate, SetBufferHandleToOutput) {
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kOutputTensorIndex, handle, delegate);
  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
}

TEST_F(TestDelegate, SetInvalidHandleToTensor) {
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  SimpleDelegate another_simple_delegate({0, 1, 2});

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status = interpreter_->SetBufferHandle(
      kOutputTensorIndex, handle,
      another_simple_delegate.get_tf_lite_delegate());
  // Setting a buffer handle to a tensor with another delegate will fail.
  ASSERT_EQ(status, kTfLiteError);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);
}

TEST_F(TestDelegate, TestResizeInputWithNonDynamicDelegate) {
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  // Try resizing input to same shape as before (which should be a No-op).
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  // ModifyGraphWithDelegate shouldn't fail, but graph won't change.
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Ensure graph has been restored to its valid delegated state.
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }

  // Resize again, but call AllocateTensors as usual afterwards.
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

// If a delegate sets kTfLiteDelegateFlagsRequirePropagatedShapes but not
// kTfLiteDelegateFlagsAllowDynamicTensors, the former is redundant.
TEST_F(TestDelegate, TestRequirePropagatedShapes_NonDynamicDelegate) {
  delegate_ = std::make_unique<SimpleDelegate>(
      std::vector<int>{0, 1, 2}, kTfLiteDelegateFlagsRequirePropagatedShapes);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  // Resizing should revert execution plan to original state.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestDelegate, TestRequirePropagatedShapes_DynamicDelegateWithFlag) {
  // Delegate sets both flags and in its Prepare, ensures that shapes have been
  // propagated by runtime.
  int delegate_flags = kTfLiteDelegateFlagsAllowDynamicTensors |
                       kTfLiteDelegateFlagsRequirePropagatedShapes;
  delegate_ = SimpleDelegate::DelegateWithRuntimeShapePropagation(
      {0, 1, 2}, delegate_flags, 3);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

// If the delegate implementation expects shapes to be automatically propagated
// but does not set the required flag, its Prepare should fail.
TEST_F(TestDelegate, TestRequirePropagatedShapes_DynamicDelegateWithoutFlag) {
  // Delegate sets both flags and in its Prepare, ensures that shapes have been
  // propagated by runtime.
  int delegate_flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  delegate_ = SimpleDelegate::DelegateWithRuntimeShapePropagation(
      {0, 1, 2}, delegate_flags, 3);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);
}

TEST_F(TestDelegate, TestCopyFromBufferInvoke) {
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  std::vector<float> floats = {1.0f, 2.0f, 3.0f};
  memcpy(interpreter_->typed_tensor<float>(0), floats.data(),
         floats.size() * sizeof(float));

  memcpy(interpreter_->typed_tensor<float>(1), floats.data(),
         floats.size() * sizeof(float));

  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  // Called Invoke without setting the buffer will not call the CopyFromBuffer
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  std::vector<float> res = {2.0f, 4.0f, 6.0f};
  for (int i = 0; i < tensor->dims->data[0]; ++i) {
    ASSERT_EQ(tensor->data.f[i], res[i]);
  }
}

TEST_F(TestDelegate, TestCopyFromBuffer) {
  interpreter_->Invoke();
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  std::vector<float> floats = {1.0f, 2.0f, 3.0f};
  memcpy(interpreter_->typed_tensor<float>(0), floats.data(),
         floats.size() * sizeof(float));

  memcpy(interpreter_->typed_tensor<float>(1), floats.data(),
         floats.size() * sizeof(float));

  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kOutputTensorIndex, handle, delegate);
  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
  for (int i = 0; i < tensor->dims->data[0]; ++i) {
    ASSERT_EQ(tensor->data.f[i], 6.0f);
  }
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

    TfLiteRegistration registration{};
    registration.registration_external = TfLiteOperatorCreate(
        kTfLiteBuiltinDelegate, "OpaqueTestDelegate delegate kernel", 1,
        /*user_data=*/nullptr);

    registration.prepare = [](TfLiteContext* context,
                              TfLiteNode* node) -> TfLiteStatus {
      return kTfLiteOk;
    };
    registration.invoke = [](TfLiteContext* context,
                             TfLiteNode* node) -> TfLiteStatus {
      return kTfLiteOk;
    };

    TfLiteContext* context = reinterpret_cast<TfLiteContext*>(opaque_context);
    TfLiteIntArray* execution_plan;
    context->GetExecutionPlan(context, &execution_plan);
    context->ReplaceNodeSubsetsWithDelegateKernels(
        context, registration, execution_plan,
        reinterpret_cast<TfLiteDelegate*>(opaque_delegate));
    return kTfLiteOk;
  }

  static inline TfLiteStatus CopyFromBufferHandle(
      TfLiteOpaqueContext* context, TfLiteOpaqueDelegate* delegate, void* data,
      TfLiteBufferHandle buffer_handle, TfLiteOpaqueTensor* opaque_tensor) {
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
    delegate_state->buffer_handle = *buffer_handle;
  }
};

// Ensure that the runtime correctly interacts with a delegate that uses the
// 'TfLiteOpaqueDelegateBuilder'.  This test:
// 1. Defines a delegate that will replace the full graph will a delegate
//    kernel.
// 2. Associates the model's output tensor with the delegate and marks the
//    output tensor's data as stale, to prompt the runtime to use the delegate's
//    'CopyFromBufferHandle' callback.
// 3. The test driver will overwrite the output tensor's buffer handle, to
//    prompt the runtime to use the delegate's 'FreeBufferHandle' callback.
// 4. Eventually the test driver destroys the interpreter, and checks that
//    also the second buffer handle gets deallocated via the delegate callback.
TEST(TestOpaqueDelegate, PrepareCopyFromFree) {
  DelegateState delegate_state;
  delegate_state.Reset();

  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          "tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  constexpr int kNumTensorElements = 1 * 8 * 8 * 3;

  TfLiteOpaqueDelegateBuilder opaque_delegate{};
  opaque_delegate.data = &delegate_state;
  opaque_delegate.CopyFromBufferHandle =
      OpaqueTestDelegate::CopyFromBufferHandle;
  opaque_delegate.FreeBufferHandle = OpaqueTestDelegate::FreeBufferHandle;
  opaque_delegate.Prepare = OpaqueTestDelegate::Prepare;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  TfLiteDelegate tflite_delegate{};
  tflite_delegate.opaque_delegate_builder = &opaque_delegate;
  builder.AddDelegate(&tflite_delegate);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  ASSERT_NE(interpreter, nullptr);

  // Allocate tensor buffers.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  // Fill input buffers
  float* input = interpreter->typed_input_tensor<float>(0);
  std::fill(input, input + kNumTensorElements, 1);

  // We set the buffer handle of the output tensor and mark its data as stale.
  // This will make the interpreter call 'CopyFromBufferHandle' to refresh the
  // output tensor's data.
  EXPECT_FALSE(delegate_state.free_buffer_handle_called);
  int first_buffer_handle = 1;
  const int kOutputTensorIndex = 2;
  interpreter->SetBufferHandle(kOutputTensorIndex, first_buffer_handle,
                               &tflite_delegate);
  TfLiteTensor* output_t = interpreter->output_tensor(0);
  output_t->data_is_stale = true;

  // Run inference
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  EXPECT_TRUE(delegate_state.delegate_prepared);
  EXPECT_TRUE(delegate_state.copy_from_buffer_handle_called);
  EXPECT_EQ(delegate_state.buffer_handle, first_buffer_handle);
  EXPECT_FALSE(delegate_state.free_buffer_handle_called);
  float* outputs = interpreter->typed_output_tensor<float>(0);
  for (int i = 0; i < kNumTensorElements; ++i) {
    EXPECT_EQ(outputs[i], OpaqueTestDelegate::kTestDelegateOutput);
  }

  delegate_state.Reset();
  // Setting a buffer handle on a tensor that already has a buffer handle
  // associated with it will free the previously installed buffer handle.
  int second_buffer_handle = first_buffer_handle + 1;
  interpreter->SetBufferHandle(kOutputTensorIndex, second_buffer_handle,
                               &tflite_delegate);
  EXPECT_FALSE(delegate_state.copy_from_buffer_handle_called);
  EXPECT_EQ(delegate_state.buffer_handle, first_buffer_handle);
  EXPECT_TRUE(delegate_state.free_buffer_handle_called);

  // Destroying the interpreter will release any buffer handles that are
  // associated with the tensors owner by the interpreter.
  delegate_state.Reset();
  interpreter.reset();
  EXPECT_FALSE(delegate_state.copy_from_buffer_handle_called);
  EXPECT_EQ(delegate_state.buffer_handle, second_buffer_handle);
  EXPECT_TRUE(delegate_state.free_buffer_handle_called);
}

TEST(TestDelegateKernel, WithoutName) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          "tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  TfLiteDelegate tflite_delegate{};
  tflite_delegate.Prepare =
      [](TfLiteContext* context,
         struct TfLiteDelegate* delegate) -> TfLiteStatus {
    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
    TfLiteRegistration registration{};
    registration.init = [](TfLiteContext* context, const char* buffer,
                           size_t length) -> void* { return nullptr; };
    context->ReplaceNodeSubsetsWithDelegateKernels(context, registration,
                                                   execution_plan, delegate);
    return kTfLiteOk;
  };
  builder.AddDelegate(&tflite_delegate);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  ASSERT_NE(interpreter, nullptr);
}

TEST_F(TestDelegate, DelegateCustomOpResolution) {
  // Build a flatbuffer model that contains the "my_add" custom op which gets
  // resolved only after SimpleDelegate is applied.
  flatbuffers::FlatBufferBuilder builder;
  // Tensors.
  const int32_t shape[1] = {3};
  flatbuffers::Offset<Tensor> tensors[3] = {
      CreateTensor(builder, builder.CreateVector<int32_t>(shape, 1),
                   TensorType_FLOAT32, /*buffer=*/0, builder.CreateString("X")),
      CreateTensor(builder, builder.CreateVector<int32_t>(shape, 1),
                   TensorType_FLOAT32, /*buffer=*/0, builder.CreateString("Y")),
      CreateTensor(builder, builder.CreateVector<int32_t>(shape, 1),
                   TensorType_FLOAT32, /*buffer=*/0, builder.CreateString("Z")),
  };
  // Custom op definition.
  flatbuffers::Offset<OperatorCode> op_code =
      CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM, "my_add");
  const int32_t inputs[2] = {0, 1};
  const int32_t outputs[1] = {2};
  flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector<int32_t>(inputs, 2),
      builder.CreateVector<int32_t>(outputs, 1), BuiltinOptions_NONE,
      /*builtin_options=*/0,
      /*custom_options=*/0, tflite::CustomOptionsFormat_FLEXBUFFERS);
  // Subgraph & Model.
  flatbuffers::Offset<SubGraph> subgraph =
      CreateSubGraph(builder, builder.CreateVector(tensors, 3),
                     builder.CreateVector<int32_t>(inputs, 2),
                     builder.CreateVector<int32_t>(outputs, 1),
                     builder.CreateVector(&op, 1), /*name=*/0);
  flatbuffers::Offset<Buffer> buffers[1] = {
      CreateBuffer(builder, builder.CreateVector({})),
  };
  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&op_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("test_model"),
      builder.CreateVector(buffers, 1));
  builder.Finish(model_buffer);
  std::vector<char> buffer =
      std::vector<char>(builder.GetBufferPointer(),
                        builder.GetBufferPointer() + builder.GetSize());
  const Model* model = GetModel(buffer.data());

  // Build an interpreter with the model. Initialization should work fine.
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model, ::tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
      kTfLiteOk);
  // AllocateTensors should fail, since my_add hasn't been resolved.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteUnresolvedOps);

  // Applying static delegate won't work, since the interpreter will first try
  // to Prepare all original nodes.
  std::unique_ptr<SimpleDelegate> static_delegate(new SimpleDelegate({0}));
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(
                static_delegate->get_tf_lite_delegate()),
            kTfLiteUnresolvedOps);

  // Applying delegate that supports dynamic tensors should work.
  std::unique_ptr<SimpleDelegate> dynamic_delegate(
      new SimpleDelegate({0}, kTfLiteDelegateFlagsAllowDynamicTensors));
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(
                dynamic_delegate->get_tf_lite_delegate()),
            kTfLiteOk);
  // AllocateTensors will now work.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
}

TEST_F(TestDelegate, AllSubgraphsAreDelegatedByDefault) {
  AddSubgraphs(1);
  SetUpSubgraph(interpreter_->subgraph(1));
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  for (int subgraph_index = 0; subgraph_index < 2; subgraph_index++) {
    ASSERT_EQ(interpreter_->subgraph(subgraph_index)->execution_plan().size(),
              1);
    int node = interpreter_->subgraph(subgraph_index)->execution_plan()[0];
    const auto* node_and_reg =
        interpreter_->subgraph(subgraph_index)->node_and_registration(node);
    EXPECT_EQ(node_and_reg->second.custom_name,
              delegate_->FakeFusedRegistration().custom_name);
  }
}

TEST_F(TestDelegate, ValidationSubgraphsAreNotDelegated) {
  AddSubgraphs(1);
  SetUpSubgraph(interpreter_->subgraph(1));
  interpreter_->subgraph(1)->SetName("VALIDATION:foo");
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1, 2});
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->subgraph(1)->execution_plan().size(), 3);
  int node = interpreter_->subgraph(1)->execution_plan()[0];
  const auto* node_and_reg =
      interpreter_->subgraph(1)->node_and_registration(node);
  EXPECT_NE(node_and_reg->second.custom_name,
            delegate_->FakeFusedRegistration().custom_name);
}

TEST_P(TestTwoDelegates, SecondDelegationPrepareFailure) {
  auto delegate_flag_pair = GetParam();
  // First delegate only supports nodes 1, 2. Gets applied successfully.
  delegate_ = std::unique_ptr<SimpleDelegate>(
      new SimpleDelegate({1, 2}, delegate_flag_pair.first));
  // Second delegate supports node 0, but fails during the delegate-node's
  // Prepare.
  delegate2_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate(
      {0}, delegate_flag_pair.second, SimpleDelegate::Options::kFailOnPrepare));

  // Initially, execution plan has 3 nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  if (delegate_flag_pair.first == kTfLiteDelegateFlagsAllowDynamicTensors &&
      delegate_flag_pair.second == kTfLiteDelegateFlagsAllowDynamicTensors) {
    // If both delegates support dynamic tensors, the execution plan isn't
    // prepared by ModifyGraphWithDelegate unless the graph was previously
    // invokable. This is mainly because dynamic tensors anyway need
    // allocations during Invoke.
    // But for this test, we call AllocateTensors() to trigger allocations.
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  }
  // First delegate should be applied successfully, yielding a plan with 2
  // nodes.
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  // Second delegate won't get applied.
  // As a result, previous delegate should also get undone, restoring the
  // execution plan to its original state.
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteDelegateError);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_P(TestTwoDelegates, SecondDelegationInvokeFailure) {
  auto delegate_flag_pair = GetParam();
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{1, 2},
                                               delegate_flag_pair.first);
  delegate2_ = std::make_unique<SimpleDelegate>(
      std::vector<int>{0}, delegate_flag_pair.second,
      SimpleDelegate::Options::kFailOnInvoke);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);
  if (delegate_flag_pair.first == kTfLiteDelegateFlagsAllowDynamicTensors &&
      delegate_flag_pair.second == kTfLiteDelegateFlagsAllowDynamicTensors) {
    // If both delegates support dynamic tensors, the execution plan isn't
    // prepared by ModifyGraphWithDelegate unless the graph was previously
    // invokable. This is mainly because dynamic tensors anyway need
    // allocations during Invoke.
    // Call AllocateTensors() to trigger allocations.
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  }

  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  // Outputs match the AddOp path, rather than delegate path.
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};
  constexpr int kOutputTensorIndex = 3;

  // Verify Invoke() behavior to ensure Interpreter isn't broken.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  EXPECT_EQ(interpreter_->Invoke(), kTfLiteError);
  EXPECT_EQ(RemoveAllDelegates(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

// This test ensures that node indices in multi-delegate application are handled
// correctly by the TFLite partitioning algorithm.
TEST_P(TestTwoDelegates, NodeIndicesCorrectlyHandledAfterDelegation) {
  auto delegate_flag_pair = GetParam();
  // First delegate supports nodes 0, 1.
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0, 1},
                                               delegate_flag_pair.first);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  // Second delegate supports (original) node index 2.
  // The execution plan has 2 nodes, so this verifies that the partitioning
  // algorithm correctly refers to (original) node indices instead of execution
  // plan indices.
  delegate2_ = std::make_unique<SimpleDelegate>(std::vector<int>{2},
                                                delegate_flag_pair.second);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);
}

TEST_P(TestTwoDelegates, TestResizeInputTensors) {
  auto delegate_flag_pair = GetParam();
  // First delegate only supports node 0.
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0},
                                               delegate_flag_pair.first);
  // Second delegate supports nodes 1 & 2.
  delegate2_ = std::make_unique<SimpleDelegate>(std::vector<int>{1, 2},
                                                delegate_flag_pair.second);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should be two delegated nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  // Try resizing input to same shape as before (which should be a No-op).
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {3}), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  // Resize inputs to new shape.
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  if (delegate_flag_pair.first == kTfLiteDelegateFlagsAllowDynamicTensors &&
      delegate_flag_pair.second == kTfLiteDelegateFlagsAllowDynamicTensors) {
    // If both delegates support dynamic tensors, execution plan won't be reset.
    ASSERT_EQ(interpreter_->execution_plan().size(), 2);
  } else {
    // In the presence of a static delegate, the runtime will reset execution
    // plan to its original state until AllocateTensors or
    // ModifyGraphWithDelegate
    ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  }

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  // Irrespective of whether one or more delegates support dynamic shapes,
  // execution plan should have 2 (delegated) nodes now.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 2;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

// We utilize delegation in such a way as to allow node subsets with a minimum
// number of ops only.
TEST_P(TestTwoDelegates, TestDelegationWithPartitionPreview) {
  auto delegate_flag_pair = GetParam();
  // Ops 0 and 2 are delegated but end up in the same partition (based on
  // dependency analysis). However, since min_ops_per_subset = 3, no delegation
  // takes place.
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>({0, 2}),
                                               delegate_flag_pair.first,
                                               SimpleDelegate::Options::kNone,
                                               /*min_ops_per_subset=*/3);
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  // Original execution plan remains.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
  ASSERT_EQ(interpreter_->execution_plan()[1], 1);
  ASSERT_EQ(interpreter_->execution_plan()[2], 2);

  // Same ops supported, but min_ops_per_subset = 2.
  delegate2_ = std::make_unique<SimpleDelegate>(std::vector<int>({0, 2}),
                                                delegate_flag_pair.second,
                                                SimpleDelegate::Options::kNone,
                                                /*min_ops_per_subset=*/2);
  interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate());

  ASSERT_EQ(interpreter_->execution_plan().size(), 2);
  ASSERT_EQ(interpreter_->execution_plan()[0], 3);
  const auto* node_and_reg = interpreter_->node_and_registration(3);
  ASSERT_EQ(node_and_reg->second.custom_name,
            delegate2_->FakeFusedRegistration().custom_name);
  ASSERT_EQ(interpreter_->execution_plan()[1], 1);
}

TEST_P(TestTwoDelegates, TestRequirePropagatedShapes) {
  // We do not use kTfLiteDelegateFlagsNone in this test, since shape
  // propagation always requires the delegate to support dynamic tensors. This
  // delegate does not require automatic propagation.
  delegate_ = std::make_unique<SimpleDelegate>(
      std::vector<int>{0}, kTfLiteDelegateFlagsAllowDynamicTensors);
  // Second delegate supports nodes 1 & 2, and requires automatic shape
  // propagation.
  int delegate_flags = kTfLiteDelegateFlagsAllowDynamicTensors |
                       kTfLiteDelegateFlagsRequirePropagatedShapes;
  delegate2_ = SimpleDelegate::DelegateWithRuntimeShapePropagation(
      {1, 2}, delegate_flags, 1);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should be two delegate nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 2;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 4 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 4 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_P(TestTwoDelegates, ReleaseNonPersistentMemoryWithDelegates) {
  auto delegate_flag_pair = GetParam();
  // First delegate only supports node 0.
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0},
                                               delegate_flag_pair.first);
  // Second delegate supports nodes 1 & 2, and makes the graph immutable.
  delegate2_ = std::make_unique<SimpleDelegate>(std::vector<int>{1, 2},
                                                delegate_flag_pair.second);

  // No-op.
  ASSERT_EQ(interpreter_->ReleaseNonPersistentMemory(), kTfLiteOk);

  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should be two delegates nodes.
  ASSERT_EQ(interpreter_->execution_plan().size(), 2);

  ASSERT_EQ(interpreter_->ReleaseNonPersistentMemory(), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f};
  constexpr int kOutputTensorIndex = 2;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);

  // Verify Invoke() behavior.
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }

  ASSERT_EQ(interpreter_->ReleaseNonPersistentMemory(), kTfLiteOk);
}

// This test ensures that after a static delegate is applied, a future delegate
// that accepts previous nodes doesn't make them dynamic.
TEST_F(TestTwoDelegates, DynamicTensorBeforeStaticDelegate) {
  // First delegate only supports node {1, 2}.
  // This makes the graph immutable.
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{1, 2});
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Second delegate supports node 0, & tries to mark its output as
  // dynamic. This should result in kTfLiteApplicationError.
  delegate2_ = SimpleDelegate::DelegateWithDynamicOutput({0});
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteApplicationError);
  // Execution plan reset to original.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

// Same as bove, except a tensor later in the graph is marked static.
// Even in this case, to be safe, we return an error.
TEST_F(TestTwoDelegates, DynamicTensorAfterStaticDelegate) {
  // First delegate only supports node 0.
  // This makes the graph immutable.
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>{0});
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Second delegate supports node 1, 2 & tries to mark its output as
  // dynamic. This should result in kTfLiteApplicationError.
  delegate2_ = SimpleDelegate::DelegateWithDynamicOutput({1, 2});
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate2_->get_tf_lite_delegate()),
      kTfLiteApplicationError);
  // Execution plan reset to original.
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(
    TestTwoDelegatesInstance, TestTwoDelegates,
    ::testing::Values(std::make_pair(kTfLiteDelegateFlagsNone,
                                     kTfLiteDelegateFlagsNone),
                      std::make_pair(kTfLiteDelegateFlagsAllowDynamicTensors,
                                     kTfLiteDelegateFlagsNone),
                      std::make_pair(kTfLiteDelegateFlagsNone,
                                     kTfLiteDelegateFlagsAllowDynamicTensors),
                      std::make_pair(kTfLiteDelegateFlagsAllowDynamicTensors,
                                     kTfLiteDelegateFlagsAllowDynamicTensors)));

class TestDelegateWithDynamicTensors : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_ =
        test_utils::TestDelegation::NewInterpreterWithDefaultDelegates();

    interpreter_->AddTensors(3);
    interpreter_->SetInputs({0});
    interpreter_->SetOutputs({1, 2});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3},
                                               quant);
    TfLiteRegistration reg = DynamicCopyOpRegistration();
    interpreter_->AddNodeWithParameters({0}, {1, 2}, nullptr, 0, nullptr, &reg);

    delegate_ = TfLiteDelegateCreate();
    delegate_.Prepare = [](TfLiteContext* context,
                           TfLiteDelegate* delegate) -> TfLiteStatus {
      // In this test, the delegate replaces all the nodes if this function is
      // called.
      TfLiteIntArray* execution_plan;
      TF_LITE_ENSURE_STATUS(
          context->GetExecutionPlan(context, &execution_plan));
      TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
          context, DelegateRegistration(), execution_plan, delegate);
      return status;
    };
    delegate_.flags = kTfLiteDelegateFlagsNone;
  }

  static TfLiteRegistration DynamicCopyOpRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Output 0 is dynamic
      TfLiteTensor* output0;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output0));
      SetTensorToDynamic(output0);
      // Output 1 has the same shape as input.
      const TfLiteTensor* input;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
      TfLiteTensor* output1;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &output1));
      TF_LITE_ENSURE_STATUS(context->ResizeTensor(
          context, output1, TfLiteIntArrayCopy(input->dims)));
      return kTfLiteOk;
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      // Not implemented since this isn't required in testing.
      return kTfLiteOk;
    };
    return reg;
  }

  static TfLiteRegistration DelegateRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // If tensors are resized, the runtime should propagate shapes
      // automatically if correct flag is set. Ensure values are correct.
      // Output 0 should be dynamic.
      TfLiteTensor* output0;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output0));
      TF_LITE_ENSURE(context, IsDynamicTensor(output0));
      // Output 1 has the same shape as input.
      const TfLiteTensor* input;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
      TfLiteTensor* output1;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &output1));
      TF_LITE_ENSURE(context, input->dims->size == output1->dims->size);
      TF_LITE_ENSURE(context, input->dims->data[0] == output1->dims->data[0]);
      return kTfLiteOk;
    };

    return reg;
  }

  std::unique_ptr<Interpreter> interpreter_;
  TfLiteDelegate delegate_;
};

TfLiteOperator* CreateTfLiteOperator() {
  auto* registration = TfLiteOperatorCreate(
      kTfLiteBuiltinDelegate, "OpaqueDelegateKernel", 1, /*user_data=*/nullptr);
  TfLiteOperatorSetPrepareWithData(
      registration,
      [](void* user_data, TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        // If tensors are resized, the runtime should propagate shapes
        // automatically if 'kTfLiteDelegateFlagsRequirePropagatedShapes' flag
        // is set.

        // Output 0 should be dynamic.
        TfLiteOpaqueTensor* output0 =
            TfLiteOpaqueNodeGetOutput(context, opaque_node, 0);
        EXPECT_EQ(kTfLiteDynamic, TfLiteOpaqueTensorGetAllocationType(output0));

        // Output 1 has the same shape as input.
        const TfLiteOpaqueTensor* input =
            TfLiteOpaqueNodeGetInput(context, opaque_node, 0);
        const TfLiteOpaqueTensor* output1 =
            TfLiteOpaqueNodeGetOutput(context, opaque_node, 1);

        if (TfLiteOpaqueTensorNumDims(input) !=
            TfLiteOpaqueTensorNumDims(output1)) {
          return kTfLiteError;
        }
        // When 'kTfLiteDelegateFlagsRequirePropagatedShapes' is *not* set then
        // changes to the dimensions of the 'input' tensor won't automatically
        // propagate to the 'output1' tensor dimensions.
        if (TfLiteOpaqueTensorDim(input, 0) !=
            TfLiteOpaqueTensorDim(output1, 0)) {
          return kTfLiteError;
        }

        return kTfLiteOk;
      });

  return registration;
}

class TestOpaqueDelegateBuilderWithDynamicTensors
    : public TestDelegateWithDynamicTensors {
 public:
  void SetUp() override {
    // Re-use the base class setup in terms of nodes, tensors and registrations.
    TestDelegateWithDynamicTensors::SetUp();
    // But with the difference that we'll apply a delegate to the graph that
    // uses its opaque_delegate_builder field.
    delegate_.Prepare = nullptr;
    delegate_.opaque_delegate_builder = &delegate_external_;
    delegate_external_.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                    TfLiteOpaqueDelegate* opaque_delegate,
                                    void* data) -> TfLiteStatus {
      TfLiteIntArray* execution_plan;
      TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan);
      return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
          opaque_context, CreateTfLiteOperator(), execution_plan,
          opaque_delegate);
    };
    delegate_external_.flags = kTfLiteDelegateFlagsNone;
  }

 private:
  TfLiteOpaqueDelegateBuilder delegate_external_{};
};

TEST_F(TestDelegateWithDynamicTensors, DisallowDynamicTensors) {
  interpreter_->ModifyGraphWithDelegate(&delegate_);

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  // The interpreter should not call delegate's `Prepare` when dynamic tensors
  // exist. So the node ID isn't changed.
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
}

TEST_F(TestOpaqueDelegateBuilderWithDynamicTensors, DisallowDynamicTensors) {
  interpreter_->ModifyGraphWithDelegate(&delegate_);

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  // The interpreter should not call delegate's `Prepare` when dynamic tensors
  // exist. So the node ID isn't changed.
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
}

TEST_F(TestDelegateWithDynamicTensors, AllowDynamicTensors) {
  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  interpreter_->ModifyGraphWithDelegate(&delegate_);

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  // The node should be replaced because dynamic tensors are allowed. Therefore
  // only node ID in the execution plan is changed from 0 to 1.
  ASSERT_EQ(interpreter_->execution_plan()[0], 1);
}

TEST_F(TestOpaqueDelegateBuilderWithDynamicTensors, AllowDynamicTensors) {
  delegate_.opaque_delegate_builder->flags =
      kTfLiteDelegateFlagsAllowDynamicTensors;
  interpreter_->ModifyGraphWithDelegate(&delegate_);

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  // The node should be replaced because dynamic tensors are allowed. Therefore
  // only node ID in the execution plan is changed from 0 to 1.
  ASSERT_EQ(interpreter_->execution_plan()[0], 1);
}

TEST_F(TestDelegateWithDynamicTensors, ModifyGraphAfterAllocate) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  ASSERT_EQ(interpreter_->execution_plan()[0], 1);

  // Allocation should still succeed.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

TEST_F(TestOpaqueDelegateBuilderWithDynamicTensors, ModifyGraphAfterAllocate) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.opaque_delegate_builder->flags =
      kTfLiteDelegateFlagsAllowDynamicTensors;
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  ASSERT_EQ(interpreter_->execution_plan()[0], 1);

  // Allocation should still succeed.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

TEST_F(TestDelegateWithDynamicTensors, ShapePropagation_FlagSet) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors |
                    kTfLiteDelegateFlagsRequirePropagatedShapes;
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);

  // Allocation before & after resizing tensors should work.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

TEST_F(TestOpaqueDelegateBuilderWithDynamicTensors, ShapePropagation_FlagSet) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.opaque_delegate_builder->flags =
      kTfLiteDelegateFlagsAllowDynamicTensors |
      kTfLiteDelegateFlagsRequirePropagatedShapes;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);

  // Allocation before & after resizing tensors should work.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

TEST_F(TestDelegateWithDynamicTensors, ShapePropagation_FlagNotSet) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);

  // Allocation after resizing tensors should NOT work, since runtime won't
  // propagate shape - causing delegate kernel to fail.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);
}

TEST_F(TestOpaqueDelegateBuilderWithDynamicTensors,
       ShapePropagation_FlagNotSet) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.opaque_delegate_builder->flags =
      kTfLiteDelegateFlagsAllowDynamicTensors;

  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);

  // Allocation after resizing tensors should NOT work, since runtime won't
  // propagate shape - causing delegate kernel to fail.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {4}), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);
}

class TestReleaseDynamicTensorWithDelegate : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_ =
        test_utils::TestDelegation::NewInterpreterWithDefaultDelegates();

    interpreter_->AddTensors(3);
    interpreter_->SetInputs({0});
    interpreter_->SetOutputs({2});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3},
                                               quant);
    TfLiteRegistration reg = DynamicCopyOpRegistration();
    interpreter_->AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr, &reg);

    delegate_.Prepare = [](TfLiteContext* context,
                           TfLiteDelegate* delegate) -> TfLiteStatus {
      TfLiteIntArray* execution_plan;
      TF_LITE_ENSURE_STATUS(
          context->GetExecutionPlan(context, &execution_plan));
      // Only replace the second execution node with delegate.
      TfLiteIntArray* nodes_to_replace = TfLiteIntArrayCreate(1);
      nodes_to_replace->data[0] = execution_plan->data[1];
      TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
          context, DelegateRegistration(), nodes_to_replace, delegate);
      TfLiteIntArrayFree(nodes_to_replace);
      return status;
    };
    delegate_.flags = kTfLiteDelegateFlagsNone;
  }

  static TfLiteRegistration DynamicCopyOpRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Output is dynamic and has the same size as input.
      TfLiteTensor* output;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
      SetTensorToDynamic(output);
      const TfLiteTensor* input;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
      TfLiteTensorRealloc(input->bytes, output);
      return kTfLiteOk;
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      // Not implemented since this isn't required in testing.
      return kTfLiteOk;
    };
    return reg;
  }

  static TfLiteRegistration DelegateRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Check that input is dynamic.
      const TfLiteTensor* input;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
      TF_LITE_ENSURE(context, IsDynamicTensor(input));
      return kTfLiteOk;
    };
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      // Not implemented since this isn't required in testing.
      return kTfLiteOk;
    };
    return reg;
  }

  std::unique_ptr<Interpreter> interpreter_;
  TfLiteDelegate delegate_;
};

TEST_F(TestReleaseDynamicTensorWithDelegate, ShapePropagation_FlagNotSet) {
  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_NE(interpreter_->tensor(1)->data.raw, nullptr);

  InterpreterOptions options;
  options.SetEnsureDynamicTensorsAreReleased();
  interpreter_->ApplyOptions(&options);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter_->tensor(1)->data.raw, nullptr);
}

// Tests for control edges passed in metadata
// ==========================================

TEST_F(TestDelegateWithControlEdges, NoControlEdges) {
  // Put {0,2} on a super-node, if possible
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>({0, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());
  ASSERT_EQ(interpreter_->execution_plan().size(), 3);     // [ {0, 2}, 1, 3]
  EXPECT_EQ(interpreter_->execution_plan().data()[0], 4);  // new super-node
  EXPECT_EQ(interpreter_->execution_plan().data()[1], 1);  // undelegated
  EXPECT_EQ(interpreter_->execution_plan().data()[2], 3);  // undelegated
}

TEST_F(TestDelegateWithControlEdges, OverrideControlEdges) {
  // Execute node 1 before node 2.
  SetMetadata({{kModelControlDependenciesMetadataKey,
                SerializeModelControlDependencies({{{1, 2}}})}});
  // Put {0,2} on a super-node, if possible
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>({0, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  // 1 has to be executed before 2, so original execution order is
  // preserved. Nodes 0 and 2 both get rewritten into new delegate nodes
  // 4 and 5.
  ASSERT_EQ(interpreter_->execution_plan().size(), 4);  // [ 0, 1, 2, 3]
  EXPECT_EQ(interpreter_->execution_plan().data()[0], 4);
  EXPECT_EQ(interpreter_->execution_plan().data()[1], 1);
  EXPECT_EQ(interpreter_->execution_plan().data()[2], 5);
  EXPECT_EQ(interpreter_->execution_plan().data()[3], 3);
}

// Test that empty control edge metadata for subgraph 0 don't change anything.
TEST_F(TestDelegateWithControlEdges, EmptyControlEdges) {
  SetMetadata({{kModelControlDependenciesMetadataKey,
                SerializeModelControlDependencies({{}})}});
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>({0, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());
  EXPECT_EQ(interpreter_->execution_plan().size(), 3);  // [ {0, 2}, 1, 3]
}

// Test that control edges that are compatible with execution order
// [0, 2, 1, 3] don't change anything (case 1).
TEST_F(TestDelegateWithControlEdges, CompatibleControlEdges1) {
  // Execute node 0 before node 2 and node 1 before node 3.
  SetMetadata({{kModelControlDependenciesMetadataKey,
                SerializeModelControlDependencies({{{0, 2}, {1, 3}}})}});
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>({0, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());
  EXPECT_EQ(interpreter_->execution_plan().size(), 3);  // [ {0, 2}, 1, 3]
}

// Test that control edges that are compatible with execution order
// [0, 2, 1, 3] don't change anything (case 2).
TEST_F(TestDelegateWithControlEdges, CompatibleControlEdges2) {
  // Execute node 0 before node 1 and node 1 before node 3.
  SetMetadata({{kModelControlDependenciesMetadataKey,
                SerializeModelControlDependencies({{{0, 1}, {1, 3}}})}});
  delegate_ = std::make_unique<SimpleDelegate>(std::vector<int>({0, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());
  EXPECT_EQ(interpreter_->execution_plan().size(), 3);  // [ {0, 2}, 1, 3]
}

// Tests for FP16 graphs
// =====================

TEST_P(TestFP16Delegation, NonDelegatedInterpreterWorks) {
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();
}

TEST_F(TestFP16Delegation, NullDelegate) {
  TfLiteOpaqueDelegate* delegate = nullptr;
  EXPECT_EQ(interpreter_->ModifyGraphWithDelegate(delegate),
            kTfLiteDelegateError);
  // Verify that resulting interpreter still works, despite null delegate.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();
}

TEST_P(TestFP16Delegation, DelegationWorks) {
  delegate_ = std::make_unique<FP16Delegate>(
      /**num_delegated_subsets**/ GetParam());
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  // Should have 7 nodes: delegate, mul, add2 & 4 dequantize ops.
  ASSERT_EQ(interpreter_->execution_plan().size(), 7);
  VerifyInvoke();
}

TEST_P(TestFP16Delegation, DelegatePrepareFails) {
  delegate_ = std::make_unique<FP16Delegate>(
      /**num_delegated_subsets**/ GetParam(), /**fail_node_prepare**/ true);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteDelegateError);
  // Delegation failed, but runtime should go back to correct previous state.
  ASSERT_EQ(interpreter_->execution_plan().size(), 8);
  VerifyInvoke();
}

INSTANTIATE_TEST_SUITE_P(TestFP16Delegation, TestFP16Delegation,
                         ::testing::Values(1, 2));

}  // anonymous namespace
}  // namespace delegates
}  // namespace tflite
