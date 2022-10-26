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

#include <array>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/delegate_test_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {

static const char kDelegateName[] = "My opaque delegate";

int CalculateNumElements(const TfLiteOpaqueTensor* opaque_tensor) {
  int total_num_elements = 1;
  for (int i = 0; i < TfLiteOpaqueTensorNumDims(opaque_tensor); ++i) {
    total_num_elements *= TfLiteOpaqueTensorDim(opaque_tensor, i);
  }
  return total_num_elements;
}

class MySimpleOpaqueDelegateKernel
    : public SimpleOpaqueDelegateKernelInterface {
  bool IsExternalTensor(const TfLiteOpaqueTensor* opaque_tensor) const {
    return external_tensors_.count(opaque_tensor) != 0;
  }

  void DeriveExternalTensors() {
    for (const TfLiteOpaqueTensor* tensor : node_input_tensors_set_) {
      if (node_output_tensors_set_.count(tensor) == 0) {
        external_tensors_.insert(tensor);
      }
    }

    for (const TfLiteOpaqueTensor* tensor : node_output_tensors_set_) {
      if (node_input_tensors_set_.count(tensor) == 0) {
        external_tensors_.insert(tensor);
      }
    }
  }

 public:
  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override {
    if (params->delegate == nullptr) return kTfLiteDelegateError;

    context_ = context;
    builtin_code_.resize(params->nodes_to_replace->size);

    node_input_tensors_.resize(params->nodes_to_replace->size);
    node_output_tensors_.resize(params->nodes_to_replace->size);

    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteOpaqueNode* delegated_node = nullptr;
      TfLiteRegistrationExternal* delegated_node_registration = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(
          context, node_index, &delegated_node, &delegated_node_registration);

      auto input_tensor1 = TfLiteOpaqueNodeGetInput(context, delegated_node, 0);
      node_input_tensors_[i].push_back(input_tensor1);
      node_input_tensors_set_.insert(input_tensor1);

      auto input_tensor2 = TfLiteOpaqueNodeGetInput(context, delegated_node, 1);
      node_input_tensors_[i].push_back(input_tensor2);
      node_input_tensors_set_.insert(input_tensor2);

      auto output_tensor =
          TfLiteOpaqueNodeGetOutput(context, delegated_node, 0);
      node_output_tensors_[i] = output_tensor;
      node_output_tensors_set_.insert(output_tensor);

      builtin_code_[i] =
          TfLiteRegistrationExternalGetBuiltInCode(delegated_node_registration);
    }

    // Determine which tensors are external (the TFLite runtime takes care
    // of them) so that we know which tensors are 'internal' to this delegate.
    // For the internal tensors we need to ensure they have memory allocated to
    // store their data, and take care of re-sizing etc.
    DeriveExternalTensors();

    return kTfLiteOk;
  }

  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* delegated_node) override {
    if (external_tensors_.empty()) return kTfLiteOk;

    const int kTheInputTensorSize =
        CalculateNumElements((*external_tensors_.begin()));
    for (std::vector<const TfLiteOpaqueTensor*>& vecs : node_input_tensors_) {
      for (const TfLiteOpaqueTensor* tensor : vecs) {
        if (IsExternalTensor(tensor)) continue;

        std::vector<float>& vec_memory = internal_tensors_memory_[tensor];
        vec_memory.resize(kTheInputTensorSize);
      }
    }
    for (const TfLiteOpaqueTensor* tensor : node_output_tensors_) {
      if (IsExternalTensor(tensor)) continue;

      std::vector<float>& vec_memory = internal_tensors_memory_[tensor];
      vec_memory.resize(kTheInputTensorSize);
    }

    return kTfLiteOk;
  }

  void ComputeImpl(float* input_1, float* input_2, float* output,
                   int builtin_code, int number_of_elements) {
    for (int i = 0; i < number_of_elements; ++i) {
      if (builtin_code == kTfLiteBuiltinAdd) {
        output[i] = input_1[i] + input_2[i];
      } else {
        output[i] = input_1[i] - input_2[i];
      }
    }
  }

  float* GetRawDataSource(TfLiteOpaqueContext* context,
                          const TfLiteOpaqueTensor* tensor) {
    if (IsExternalTensor(tensor)) {
      return reinterpret_cast<float*>(TfLiteOpaqueTensorData(tensor));
    } else {
      return internal_tensors_memory_[tensor].data();
    }
  }

  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* delegated_node) override {
    for (int i = 0; i < node_input_tensors_.size(); ++i) {
      float* input1 = GetRawDataSource(context, node_input_tensors_[i][0]);
      float* input2 = GetRawDataSource(context, node_input_tensors_[i][1]);
      float* output = GetRawDataSource(context, node_output_tensors_[i]);
      // We assume that all input, output and intermediate tensors of the
      // delegated subgraph have the same size.
      ComputeImpl(input1, input2, output, builtin_code_[i],
                  CalculateNumElements(node_output_tensors_[i]));
    }
    return kTfLiteOk;
  }

 private:
  std::vector<std::vector<const TfLiteOpaqueTensor*>> node_input_tensors_;
  absl::flat_hash_set<const TfLiteOpaqueTensor*> node_input_tensors_set_;
  std::vector<const TfLiteOpaqueTensor*> node_output_tensors_;
  absl::flat_hash_set<const TfLiteOpaqueTensor*> node_output_tensors_set_;
  absl::flat_hash_set<const TfLiteOpaqueTensor*> external_tensors_;
  absl::flat_hash_map<const TfLiteOpaqueTensor*, std::vector<float>>
      internal_tensors_memory_;
  TfLiteOpaqueContext* context_;
  // Holds the builtin code of the ops.
  // builtin_code_[i] is the type of node at index 'i'
  std::vector<int> builtin_code_;
};

class MySimpleOpaqueDelegate : public SimpleOpaqueDelegateInterface {
 public:
  bool IsNodeSupportedByDelegate(
      const TfLiteRegistrationExternal* registration_external,
      const TfLiteOpaqueNode* node,
      TfLiteOpaqueContext* context) const override {
    if (kTfLiteBuiltinAdd !=
            TfLiteRegistrationExternalGetBuiltInCode(registration_external) &&
        kTfLiteBuiltinSub !=
            TfLiteRegistrationExternalGetBuiltInCode(registration_external))
      return false;

    // This delegate only supports float32 types.
    for (int i = 0; i < TfLiteOpaqueNodeNumberOfInputs(node); ++i) {
      const TfLiteOpaqueTensor* tensor =
          TfLiteOpaqueNodeGetInput(context, node, i);
      if (TfLiteOpaqueTensorType(tensor) != kTfLiteFloat32) return false;
    }

    return true;
  }

  TfLiteStatus Initialize(TfLiteOpaqueContext* context) override {
    return kTfLiteOk;
  }

  const char* Name() const override { return kDelegateName; }

  std::unique_ptr<SimpleOpaqueDelegateKernelInterface>
  CreateDelegateKernelInterface() override {
    return std::make_unique<MySimpleOpaqueDelegateKernel>();
  }
};

class TestDelegate : public ::testing::Test {};

TEST_F(TestDelegate, TestDataAddBin_SingleInputSingleOutput_FullyDelegated) {
  //
  // Create the opaque delegate
  //
  TfLiteOpaqueDelegateUniquePtr my_opaque_delegate =
      TfLiteOpaqueDelegateFactory::Create(
          std::make_unique<MySimpleOpaqueDelegate>());

  //
  // Create the model and the interpreter
  //
  TfLiteModel* model =
      TfLiteModelCreateFromFile("third_party/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreterOptionsAddOpaqueDelegate(options, my_opaque_delegate.get());
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
          std::make_unique<MySimpleOpaqueDelegate>());

  TfLiteModel* model =
      TfLiteModelCreateFromFile("third_party/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreterOptionsAddOpaqueDelegate(options, my_opaque_delegate.get());

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
          std::make_unique<MySimpleOpaqueDelegate>());

  TfLiteModel* model = TfLiteModelCreateFromFile(
      "third_party/tensorflow/lite/testdata/multi_add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreterOptionsAddOpaqueDelegate(options, my_opaque_delegate.get());

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

TfLiteRegistrationExternal* GetDelegateKernelRegistrationImpl(
    SimpleOpaqueDelegateInterface* delegate) {
  TfLiteRegistrationExternal* kernel_registration =
      TfLiteRegistrationExternalCreate(kTfLiteBuiltinDelegate, delegate->Name(),
                                       1);
  TfLiteRegistrationExternalSetFree(
      kernel_registration,
      [](TfLiteOpaqueContext* context, void* buffer) -> void {
        delete reinterpret_cast<SimpleOpaqueDelegateInterface*>(buffer);
      });

  TfLiteRegistrationExternalSetInit(
      kernel_registration,
      [](TfLiteOpaqueContext* context, const char* buffer,
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
  TfLiteRegistrationExternalSetPrepare(
      kernel_registration,
      [](TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        SimpleOpaqueDelegateKernelInterface* delegate_kernel =
            reinterpret_cast<SimpleOpaqueDelegateKernelInterface*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));
        return delegate_kernel->Prepare(context, opaque_node);
      });
  TfLiteRegistrationExternalSetInvoke(
      kernel_registration,
      [](TfLiteOpaqueContext* context,
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
  auto my_simple_delegate = std::make_unique<MySimpleOpaqueDelegate>();
  TfLiteOpaqueDelegateStruct* opaque_delegate =
      TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(
          std::move(my_simple_delegate));
  // The following cast is safe only because this code is part of the
  // TF Lite tests.  Apps using TF Lite should not rely on
  // TfLiteOpaqueDelegateStruct and TfLiteDelegate being equivalent.
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(
                reinterpret_cast<TfLiteDelegate*>(opaque_delegate)),
            kTfLiteOk);
  // Should have 7 nodes: delegate, mul, add2 & 4 dequantize ops.
  ASSERT_EQ(interpreter_->execution_plan().size(), 7);
  VerifyInvoke();
  TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(opaque_delegate);
}

// A test facilty used in the 'SetBufferHandle' unit test.  See the tests
// comments for further context on the implementation of this class.
class MySimpleOpaqueDelegateWithBufferHandleSupport
    : public MySimpleOpaqueDelegate {
 public:
  static constexpr int kDelegateOutputValue = 42;
  TfLiteStatus CopyFromBufferHandle(TfLiteOpaqueContext* context,
                                    TfLiteBufferHandle buffer_handle,
                                    TfLiteOpaqueTensor* tensor) {
    auto* output = reinterpret_cast<float*>(TfLiteOpaqueTensorData(tensor));
    std::vector<float> test_output(CalculateNumElements(tensor),
                                   kDelegateOutputValue);
    memcpy(output, test_output.data(), test_output.size() * sizeof(float));

    return kTfLiteOk;
  }

  void FreeBufferHandle(TfLiteOpaqueContext* context,  // NOLINT
                        struct TfLiteOpaqueDelegateStruct* delegate,
                        TfLiteBufferHandle* handle) {
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
  opaque_delegate_builder.Prepare =
      [](TfLiteOpaqueContext* opaque_context,
         struct TfLiteOpaqueDelegateStruct* opaque_delegate, void* data) {
        auto* simple_opaque_delegate =
            reinterpret_cast<SimpleOpaqueDelegateInterface*>(data);
        TF_LITE_ENSURE_STATUS(
            simple_opaque_delegate->Initialize(opaque_context));
        TfLiteIntArray* execution_plan;
        TF_LITE_ENSURE_STATUS(TfLiteOpaqueContextGetExecutionPlan(
            opaque_context, &execution_plan));
        TfLiteRegistrationExternal* delegate_kernel_registration =
            GetDelegateKernelRegistrationImpl(simple_opaque_delegate);

        return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
            opaque_context, delegate_kernel_registration, execution_plan,
            opaque_delegate);
      };
  opaque_delegate_builder.flags = kTfLiteDelegateFlagsNone;
  opaque_delegate_builder.data = &my_simple_delegate;
  opaque_delegate_builder.CopyFromBufferHandle =
      [](TfLiteOpaqueContext* context,
         struct TfLiteOpaqueDelegateStruct* delegate, void* data,
         TfLiteBufferHandle buffer_handle,
         TfLiteOpaqueTensor* tensor) -> TfLiteStatus {
    auto* simple_opaque_delegate =
        reinterpret_cast<MySimpleOpaqueDelegateWithBufferHandleSupport*>(data);
    simple_opaque_delegate->CopyFromBufferHandle(context, buffer_handle,
                                                 tensor);
    return kTfLiteOk;
  };
  opaque_delegate_builder.FreeBufferHandle =
      [](TfLiteOpaqueContext* context,
         struct TfLiteOpaqueDelegateStruct* delegate, void* data,
         TfLiteBufferHandle* handle) {
        auto* simple_opaque_delegate =
            reinterpret_cast<MySimpleOpaqueDelegateWithBufferHandleSupport*>(
                data);
        simple_opaque_delegate->FreeBufferHandle(context, delegate, handle);
      };
  TfLiteDelegate tflite_delegate{};
  tflite_delegate.opaque_delegate_builder = &opaque_delegate_builder;

  // Load a model and build an interpreter.
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          "third_party/tensorflow/lite/testdata/add.bin");
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
          std::make_unique<MySimpleOpaqueDelegate>());
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "third_party/tensorflow/lite/testdata/conv_huge_im2col.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreterOptionsAddOpaqueDelegate(options, my_opaque_delegate.get());

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
