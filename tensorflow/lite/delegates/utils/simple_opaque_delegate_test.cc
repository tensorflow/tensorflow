/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/delegate_test_util.h"
#include "tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {

class TestDelegate : public ::testing::Test {};

TEST_F(TestDelegate, TestDataAddBin_SingleInputSingleOutput_FullyDelegated) {
  //
  // Create the opaque delegate
  //
  TfLiteOpaqueDelegateUniquePtr my_opaque_delegate =
      TfLiteOpaqueDelegateFactory::Create(
          std::make_unique<example::SampleStableDelegate>());

  //
  // Create the model and the interpreter
  //
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreterOptionsAddDelegate(options, my_opaque_delegate.get());
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);
  // The options can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);

  //
  // Allocate the tensors and fill the input tensor.
  //
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  TfLiteTensor* input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  EXPECT_NE(TfLiteTensorData(input_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(input_tensor), "input");

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.f);
  EXPECT_EQ(input_params.zero_point, 0);

  const float kTensorCellValue = 3.f;
  int64_t n = tflite::NumElements(input_tensor);
  std::vector<float> input(n, kTensorCellValue);
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);

  //
  // Run the interpreter
  //
  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  EXPECT_NE(TfLiteTensorData(output_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(output_tensor), "output");

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);

  // The 'add.bin' model does the following operation ('t_output' denotes the
  // single output tensor, and 't_input' denotes the single input tensor):
  //
  // t_output = t_input + t_input + t_input = t_input * 3
  std::vector<float> output(n, 0);
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], kTensorCellValue * 3);
  }

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

TEST(DelegateTest,
     TestDataAddBin_SingleInputSingleOutput_FullyDelegated_ResizeInputTensors) {
  TfLiteOpaqueDelegateUniquePtr my_opaque_delegate =
      TfLiteOpaqueDelegateFactory::Create(
          std::make_unique<example::SampleStableDelegate>());

  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreterOptionsAddDelegate(options, my_opaque_delegate.get());

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  TfLiteInterpreterOptionsDelete(options);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 2);
  EXPECT_EQ(TfLiteTensorByteSize(input_tensor), sizeof(float) * 2);
  EXPECT_NE(TfLiteTensorData(input_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(input_tensor), "input");

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.f);
  EXPECT_EQ(input_params.zero_point, 0);

  std::array<float, 2> input = {1.f, 3.f};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(output_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(output_tensor, 0), 2);
  EXPECT_EQ(TfLiteTensorByteSize(output_tensor), sizeof(float) * 2);
  EXPECT_NE(TfLiteTensorData(output_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(output_tensor), "output");

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<float, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3.f);
  EXPECT_EQ(output[1], 9.f);

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

TEST(DelegateTest, TestDataMultiAddBin_MultiInputMultiOutput_FullyDelegated) {
  TfLiteOpaqueDelegateUniquePtr my_opaque_delegate =
      TfLiteOpaqueDelegateFactory::Create(
          std::make_unique<example::SampleStableDelegate>());

  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/multi_add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreterOptionsAddDelegate(options, my_opaque_delegate.get());

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  TfLiteInterpreterOptionsDelete(options);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 4);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 2);

  TfLiteTensor* input_tensor0 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/0);
  TfLiteTensor* input_tensor1 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/1);
  TfLiteTensor* input_tensor2 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/2);
  TfLiteTensor* input_tensor3 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/3);

  std::vector<TfLiteTensor*> input_tensors{input_tensor0, input_tensor1,
                                           input_tensor2, input_tensor3};
  for (TfLiteTensor* input_tensor : input_tensors) {
    const float kTensorCellValue = 1.f;
    int64_t n = tflite::NumElements(input_tensor);
    std::vector<float> input(n, kTensorCellValue);
    ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                         input.size() * sizeof(float)),
              kTfLiteOk);
  }

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor0 =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  const TfLiteTensor* output_tensor1 =
      TfLiteInterpreterGetOutputTensor(interpreter, 1);
  std::vector<const TfLiteTensor*> output_tensors{output_tensor0,
                                                  output_tensor1};
  for (const TfLiteTensor* output_tensor : output_tensors) {
    int64_t n = tflite::NumElements(output_tensor);
    std::vector<float> output_tensor_values(n, 0);
    ASSERT_EQ(
        TfLiteTensorCopyToBuffer(output_tensor, output_tensor_values.data(),
                                 output_tensor_values.size() * sizeof(float)),
        kTfLiteOk);
    for (int i = 0; i < n; ++i) {
      // We know that the model is wired in a way so that every output tensor
      // holds the sum of three input tensors.  And because every input tensor
      // is filled with 1s we can assert on the output tensors storing 3s.
      EXPECT_EQ(output_tensor_values[i], 3.f);
    }
  }

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

TfLiteOperator* CreateDelegateKernelRegistrationImpl(
    SimpleOpaqueDelegateInterface* delegate) {
  TfLiteOperator* kernel_registration = TfLiteOperatorCreate(
      kTfLiteBuiltinDelegate, delegate->Name(), 1, /*user_data=*/nullptr);
  TfLiteOperatorSetFreeWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context, void* buffer) -> void {
        delete reinterpret_cast<SimpleOpaqueDelegateInterface*>(buffer);
      });

  TfLiteOperatorSetInitWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context, const char* buffer,
         size_t length) -> void* {
        auto* params =
            reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
        if (params == nullptr) {
          return nullptr;
        }
        auto* simple_delegate =
            reinterpret_cast<SimpleOpaqueDelegateInterface*>(
                params->delegate_data);
        std::unique_ptr<SimpleOpaqueDelegateKernelInterface> delegate_kernel(
            simple_delegate->CreateDelegateKernelInterface());
        if (delegate_kernel->Init(context, params) != kTfLiteOk) {
          return nullptr;
        }
        return delegate_kernel.release();
      });
  TfLiteOperatorSetPrepareWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        SimpleOpaqueDelegateKernelInterface* delegate_kernel =
            reinterpret_cast<SimpleOpaqueDelegateKernelInterface*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));
        return delegate_kernel->Prepare(context, opaque_node);
      });
  TfLiteOperatorSetInvokeWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        SimpleOpaqueDelegateKernelInterface* delegate_kernel =
            reinterpret_cast<SimpleOpaqueDelegateKernelInterface*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));
        TFLITE_DCHECK(delegate_kernel != nullptr);
        return delegate_kernel->Eval(context, opaque_node);
      });

  return kernel_registration;
}

using ::tflite::delegates::test_utils::TestFP16Delegation;

TEST_F(TestFP16Delegation, MultipleDelegateKernels) {
  auto my_simple_delegate = std::make_unique<example::SampleStableDelegate>();
  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(
          std::move(my_simple_delegate));
  // The following cast is safe only because this code is part of the
  // TF Lite tests.  Apps using TF Lite should not rely on
  // TfLiteOpaqueDelegate and TfLiteDelegate being equivalent.
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(
                reinterpret_cast<TfLiteDelegate*>(opaque_delegate)),
            kTfLiteOk);
  // Should have 7 nodes: delegate, mul, add2 & 4 dequantize ops.
  ASSERT_EQ(interpreter_->execution_plan().size(), 7);
  VerifyInvoke();
  TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(opaque_delegate);
}

// A test facility used in the 'SetBufferHandle' unit test.  See the tests
// comments for further context on the implementation of this class.
class MySimpleOpaqueDelegateWithBufferHandleSupport
    : public example::SampleStableDelegate {
 public:
  static constexpr int kDelegateOutputValue = 42;
  TfLiteStatus CopyFromBufferHandle(TfLiteOpaqueContext* context,
                                    TfLiteBufferHandle buffer_handle,
                                    TfLiteOpaqueTensor* tensor) override {
    auto* output = reinterpret_cast<float*>(TfLiteOpaqueTensorData(tensor));
    std::vector<float> test_output(
        example::helpers::CalculateNumElements(tensor), kDelegateOutputValue);
    memcpy(output, test_output.data(), test_output.size() * sizeof(float));

    return kTfLiteOk;
  }

  void FreeBufferHandle(TfLiteOpaqueContext* context,  // NOLINT
                        TfLiteBufferHandle* handle) override {
    recorded_buffer_handle_ = *handle;
    free_buffer_handle_called_ = true;
  }

  int recorded_buffer_handle_ = -1;
  bool free_buffer_handle_called_ = false;
};

TEST_F(TestDelegate, SetBufferHandle) {
  // Set up a simple delegate that defines a 'CopyFromBufferHandle' callback as
  // as well as a 'FreeBufferHandle' callback.  The purpose of these functions
  // is to check that the TFLite runtime interacts with the delegate as
  // expected. In this case we want to make sure that the 'CopyFromBufferHandle'
  // callback is used if the output tensor's data is marked as stale.  In
  // addition we want to verify that the runtime frees the delegate's buffer
  // handles when either a new buffer handle is set, or the buffer handle is no
  // longer needed.
  MySimpleOpaqueDelegateWithBufferHandleSupport my_simple_delegate;
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  // A 'Prepare' callback that blindly replaces the full execution plan.
  // We do this because all that we are interested is to verify the buffer
  // handle-related code.
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    auto* simple_opaque_delegate =
        reinterpret_cast<SimpleOpaqueDelegateInterface*>(data);
    TF_LITE_ENSURE_STATUS(simple_opaque_delegate->Initialize(opaque_context));
    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOperator* delegate_kernel_registration =
        CreateDelegateKernelRegistrationImpl(simple_opaque_delegate);

    return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, delegate_kernel_registration, execution_plan,
        opaque_delegate);
  };
  opaque_delegate_builder.flags = kTfLiteDelegateFlagsNone;
  opaque_delegate_builder.data = &my_simple_delegate;
  opaque_delegate_builder.CopyFromBufferHandle =
      [](TfLiteOpaqueContext* context, TfLiteOpaqueDelegate* delegate,
         void* data, TfLiteBufferHandle buffer_handle,
         TfLiteOpaqueTensor* tensor) -> TfLiteStatus {
    auto* simple_opaque_delegate =
        reinterpret_cast<MySimpleOpaqueDelegateWithBufferHandleSupport*>(data);
    simple_opaque_delegate->CopyFromBufferHandle(context, buffer_handle,
                                                 tensor);
    return kTfLiteOk;
  };
  opaque_delegate_builder.FreeBufferHandle = [](TfLiteOpaqueContext* context,
                                                TfLiteOpaqueDelegate* delegate,
                                                void* data,
                                                TfLiteBufferHandle* handle) {
    auto* simple_opaque_delegate =
        reinterpret_cast<MySimpleOpaqueDelegateWithBufferHandleSupport*>(data);
    simple_opaque_delegate->FreeBufferHandle(context, handle);
  };
  TfLiteDelegate tflite_delegate{};
  tflite_delegate.opaque_delegate_builder = &opaque_delegate_builder;

  // Load a model and build an interpreter.
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          "tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(&tflite_delegate);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  ASSERT_NE(interpreter, nullptr);

  // Allocate tensor buffers.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  // Fill input buffers
  constexpr int kTensorDimensions = 1 * 8 * 8 * 3;
  std::vector<float> floats(kTensorDimensions, 1);
  memcpy(interpreter->typed_input_tensor<float>(0), floats.data(),
         floats.size() * sizeof(float));

  // We set the buffer handle of the output tensor and mark its data as stale.
  // This will make the interpreter call 'CopyFromBufferHandle' to refresh the
  // output tensor's data.  We simply hardcode the values that will be copied to
  // the output tensor to
  // MySimpleOpaqueDelegateWithBufferHandleSupport::kDelegateOutputValue.
  EXPECT_FALSE(my_simple_delegate.free_buffer_handle_called_);
  int first_buffer_handle = 1;
  const int kOutputTensorIndex = 2;
  interpreter->SetBufferHandle(
      kOutputTensorIndex, first_buffer_handle,
      reinterpret_cast<TfLiteDelegate*>(&tflite_delegate));
  TfLiteTensor* output_t = interpreter->output_tensor(0);
  output_t->data_is_stale = true;

  EXPECT_FALSE(my_simple_delegate.free_buffer_handle_called_);
  EXPECT_NE(my_simple_delegate.recorded_buffer_handle_, first_buffer_handle);

  // Run inference
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  std::vector<float> outputs(kTensorDimensions, 0);
  memcpy(outputs.data(), interpreter->typed_output_tensor<float>(0),
         outputs.size() * sizeof(float));
  for (int i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(
        outputs[i],
        MySimpleOpaqueDelegateWithBufferHandleSupport::kDelegateOutputValue);
  }

  // Call 'SetBufferHandle' on a tensor that already has a buffer handle will
  // lead to a call of 'FreeBufferHandle' for the currently set
  int next_buffer_handle = first_buffer_handle + 1;
  interpreter->SetBufferHandle(kOutputTensorIndex, next_buffer_handle,
                               &tflite_delegate);
  EXPECT_TRUE(my_simple_delegate.free_buffer_handle_called_);
  EXPECT_EQ(my_simple_delegate.recorded_buffer_handle_, first_buffer_handle);

  // Destroying the interpreter will free the currently installed buffer
  // handle.
  my_simple_delegate.free_buffer_handle_called_ = false;
  my_simple_delegate.recorded_buffer_handle_ = first_buffer_handle = -1;
  interpreter.reset();
  EXPECT_TRUE(my_simple_delegate.free_buffer_handle_called_);
  EXPECT_EQ(my_simple_delegate.recorded_buffer_handle_, next_buffer_handle);
}

TEST(DelegateTest,
     TestDataConvHugeIm2ColBin_MultiInputSingleOutput_PartiallyDelegated) {
  TfLiteOpaqueDelegateUniquePtr my_opaque_delegate =
      TfLiteOpaqueDelegateFactory::Create(
          std::make_unique<example::SampleStableDelegate>());
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/conv_huge_im2col.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreterOptionsAddDelegate(options, my_opaque_delegate.get());

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // The options can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 4);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  TfLiteTensor* input_tensor0 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/0);
  TfLiteTensor* input_tensor1 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/1);
  TfLiteTensor* input_tensor2 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/2);
  TfLiteTensor* input_tensor3 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/3);
  std::vector<TfLiteTensor*> input_tensors{input_tensor0, input_tensor1,
                                           input_tensor2, input_tensor3};
  for (TfLiteTensor* input_tensor : input_tensors) {
    const float kTensorCellValue = 4.f;
    int64_t n = tflite::NumElements(input_tensor);
    std::vector<float> input(n, kTensorCellValue);
    ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                         input.size() * sizeof(float)),
              kTfLiteOk);
  }

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  EXPECT_NE(TfLiteTensorData(output_tensor), nullptr);

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);
  int64_t n = tflite::NumElements(output_tensor);
  std::vector<float> output(n, 0);
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);
  for (int i = 0; i < n; ++i) {
    // We know that we can expect '4' because that is the model's output when
    // no delegate gets applied.  The purpose of this expectation is that we
    // arrive at the same result when the delegate is applied.
    EXPECT_EQ(output[i], 4);
  }

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

}  // namespace tflite
