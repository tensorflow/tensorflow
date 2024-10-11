// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <cstring>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"
#include "tensorflow/lite/experimental/lrt/test/testdata/simple_model_test_vectors.h"

TEST(DispatchApi, Pixel) {
#if !defined(__ANDROID__)
  GTEST_SKIP() << "This test is specific to Android devices with a Pixel eTPU";
#endif

  EXPECT_EQ(LrtDispatchInitialize(/*shared_lib_path=*/nullptr), kLrtStatusOk);

  const char* vendor_id;
  EXPECT_EQ(LrtDispatchGetVendorId(&vendor_id), kLrtStatusOk);
  ABSL_LOG(INFO) << "vendor_id: " << vendor_id;

  const char* build_id;
  EXPECT_EQ(LrtDispatchGetBuildId(&build_id), kLrtStatusOk);
  ABSL_LOG(INFO) << "build_id: " << build_id;

  LrtDispatchApiVersion api_version;
  EXPECT_EQ(LrtDispatchGetApiVersion(&api_version), kLrtStatusOk);
  ABSL_LOG(INFO) << "api_version: " << api_version.major << "."
                 << api_version.minor << "." << api_version.patch;

  int capabilities;
  EXPECT_EQ(LrtDispatchGetCapabilities(&capabilities), kLrtStatusOk);
  ABSL_LOG(INFO) << "capabilities: " << capabilities;

  LrtDispatchDeviceContext device_context = nullptr;
  EXPECT_EQ(LrtDispatchDeviceContextCreate(&device_context), kLrtStatusOk);
  ABSL_LOG(INFO) << "device_context: " << device_context;

  auto model_file_name = kPixelModelFileName;
  auto model = lrt::testing::LoadBinaryFile(model_file_name);
  EXPECT_TRUE(model.ok());
  ABSL_LOG(INFO) << "Loaded model " << model_file_name << ", " << model->size()
                 << " bytes";

  // ///////////////////////////////////////////////////////////////////////////
  // Set up an invocation context for a given model.
  // ///////////////////////////////////////////////////////////////////////////

  LrtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_EQ(LrtDispatchInvocationContextCreate(
                device_context, kLrtDispatchExecutableTypeMlModel,
                model->data(), model->size(), /*function_name=*/nullptr,
                /*num_inputs=*/2, /*num_outputs=*/1, &invocation_context),
            kLrtStatusOk);
  ABSL_LOG(INFO) << "Invocation context: " << invocation_context;

  // ///////////////////////////////////////////////////////////////////////////
  // Determine tensor buffer requirements.
  // ///////////////////////////////////////////////////////////////////////////

  int num_tensor_buffer_types;
  LrtTensorBufferRequirements input_0_tensor_buffer_requirements;
  EXPECT_EQ(LrtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/0, &kInput0TensorType,
                &input_0_tensor_buffer_requirements),
            kLrtStatusOk);
  EXPECT_EQ(LrtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
                input_0_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLrtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LrtTensorBufferType input_0_tensor_buffer_type;
  EXPECT_EQ(LrtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_0_tensor_buffer_requirements, /*type_index=*/0,
                &input_0_tensor_buffer_type),
            kLrtStatusOk);
  EXPECT_EQ(input_0_tensor_buffer_type, kLrtTensorBufferTypeAhwb);
  size_t input_0_tensor_buffer_size;
  EXPECT_EQ(
      LrtGetTensorBufferRequirementsBufferSize(
          input_0_tensor_buffer_requirements, &input_0_tensor_buffer_size),
      kLrtStatusOk);
  EXPECT_GE(input_0_tensor_buffer_size, sizeof(kTestInput0Tensor));

  LrtTensorBufferRequirements input_1_tensor_buffer_requirements;
  EXPECT_EQ(LrtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/1, &kInput1TensorType,
                &input_1_tensor_buffer_requirements),
            kLrtStatusOk);
  EXPECT_EQ(LrtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
                input_1_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLrtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LrtTensorBufferType input_1_tensor_buffer_type;
  EXPECT_EQ(LrtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_1_tensor_buffer_requirements, /*type_index=*/0,
                &input_1_tensor_buffer_type),
            kLrtStatusOk);
  EXPECT_EQ(input_1_tensor_buffer_type, kLrtTensorBufferTypeAhwb);
  size_t input_1_tensor_buffer_size;
  EXPECT_EQ(
      LrtGetTensorBufferRequirementsBufferSize(
          input_1_tensor_buffer_requirements, &input_1_tensor_buffer_size),
      kLrtStatusOk);
  EXPECT_GE(input_1_tensor_buffer_size, sizeof(kTestInput1Tensor));

  LrtTensorBufferRequirements output_tensor_buffer_requirements;
  EXPECT_EQ(LrtDispatchGetOutputRequirements(
                invocation_context, /*output_index=*/0, &kOutputTensorType,
                &output_tensor_buffer_requirements),
            kLrtStatusOk);
  EXPECT_EQ(LrtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
                output_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLrtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 1);
  LrtTensorBufferType output_tensor_buffer_type;
  EXPECT_EQ(LrtGetTensorBufferRequirementsSupportedTensorBufferType(
                output_tensor_buffer_requirements, /*type_index=*/0,
                &output_tensor_buffer_type),
            kLrtStatusOk);
  EXPECT_EQ(output_tensor_buffer_type, kLrtTensorBufferTypeAhwb);
  size_t output_tensor_buffer_size;
  EXPECT_EQ(LrtGetTensorBufferRequirementsBufferSize(
                output_tensor_buffer_requirements, &output_tensor_buffer_size),
            kLrtStatusOk);
  EXPECT_GE(output_tensor_buffer_size, sizeof(kTestOutputTensor));

  // ///////////////////////////////////////////////////////////////////////////
  // Allocate tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

  LrtTensorBuffer input_0_tensor_buffer;
  EXPECT_EQ(LrtCreateManagedTensorBuffer(
                input_0_tensor_buffer_type, &kInput0TensorType,
                input_0_tensor_buffer_size, &input_0_tensor_buffer),
            kLrtStatusOk);

  LrtTensorBuffer input_1_tensor_buffer;
  EXPECT_EQ(LrtCreateManagedTensorBuffer(
                input_1_tensor_buffer_type, &kInput1TensorType,
                input_1_tensor_buffer_size, &input_1_tensor_buffer),
            kLrtStatusOk);

  LrtTensorBuffer output_tensor_buffer;
  EXPECT_EQ(LrtCreateManagedTensorBuffer(
                output_tensor_buffer_type, &kOutputTensorType,
                output_tensor_buffer_size, &output_tensor_buffer),
            kLrtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Register tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

  LrtTensorBufferHandle input_1_handle;
  EXPECT_EQ(LrtDispatchRegisterTensorBuffer(
                device_context, input_1_tensor_buffer, &input_1_handle),
            kLrtStatusOk);

  LrtTensorBufferHandle input_0_handle;
  EXPECT_EQ(LrtDispatchRegisterTensorBuffer(
                device_context, input_0_tensor_buffer, &input_0_handle),
            kLrtStatusOk);

  LrtTensorBufferHandle output_handle;
  EXPECT_EQ(LrtDispatchRegisterTensorBuffer(
                device_context, output_tensor_buffer, &output_handle),
            kLrtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Attach tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

  EXPECT_EQ(LrtDispatchAttachInput(invocation_context, /*graph_input_index=*/0,
                                   input_0_handle),
            kLrtStatusOk);
  EXPECT_EQ(LrtDispatchAttachInput(invocation_context, /*graph_input_index=*/1,
                                   input_1_handle),
            kLrtStatusOk);
  EXPECT_EQ(LrtDispatchAttachOutput(invocation_context,
                                    /*graph_output_index=*/0, output_handle),
            kLrtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Fill the input buffers with data.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Filling inputs with data";
    void* host_mem_addr;

    ASSERT_EQ(LrtLockTensorBuffer(input_0_tensor_buffer, &host_mem_addr,
                                  /*event=*/nullptr),
              kLrtStatusOk);
    std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
    ASSERT_EQ(LrtUnlockTensorBuffer(input_0_tensor_buffer), kLrtStatusOk);

    ASSERT_EQ(LrtLockTensorBuffer(input_1_tensor_buffer, &host_mem_addr,
                                  /*event=*/nullptr),
              kLrtStatusOk);
    std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
    ASSERT_EQ(LrtUnlockTensorBuffer(input_1_tensor_buffer), kLrtStatusOk);
  }

  // ///////////////////////////////////////////////////////////////////////////
  // Execute model.
  // ///////////////////////////////////////////////////////////////////////////

  ABSL_LOG(INFO) << "Invoking execution...";
  EXPECT_EQ(LrtDispatchInvoke(invocation_context), kLrtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Check output for correctness.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    ASSERT_EQ(LrtLockTensorBuffer(output_tensor_buffer, &host_mem_addr,
                                  /*event=*/nullptr),
              kLrtStatusOk);
    auto* output = static_cast<float*>(host_mem_addr);
    constexpr auto output_size =
        sizeof(kTestOutputTensor) / sizeof(kTestOutputTensor[0]);
    for (auto i = 0; i < output_size; ++i) {
      EXPECT_NEAR(output[i], kTestOutputTensor[i], 1e-3);
    }
    ASSERT_EQ(LrtUnlockTensorBuffer(output_tensor_buffer), kLrtStatusOk);
  }

  // ///////////////////////////////////////////////////////////////////////////
  // Clean up resources.
  // ///////////////////////////////////////////////////////////////////////////

  EXPECT_EQ(LrtDispatchDetachInput(invocation_context, /*graph_input_index=*/0,
                                   input_0_handle),
            kLrtStatusOk);
  EXPECT_EQ(LrtDispatchDetachInput(invocation_context, /*graph_input_index=*/1,
                                   input_1_handle),
            kLrtStatusOk);
  EXPECT_EQ(LrtDispatchDetachOutput(invocation_context,
                                    /*graph_output_index=*/0, output_handle),
            kLrtStatusOk);
  EXPECT_EQ(LrtDispatchUnregisterTensorBuffer(device_context, output_handle),
            kLrtStatusOk);
  EXPECT_EQ(LrtDispatchUnregisterTensorBuffer(device_context, input_1_handle),
            kLrtStatusOk);
  EXPECT_EQ(LrtDispatchUnregisterTensorBuffer(device_context, input_0_handle),
            kLrtStatusOk);
  LrtDestroyTensorBuffer(output_tensor_buffer);
  LrtDestroyTensorBuffer(input_1_tensor_buffer);
  LrtDestroyTensorBuffer(input_0_tensor_buffer);
  EXPECT_EQ(LrtDispatchInvocationContextDestroy(invocation_context),
            kLrtStatusOk);
  EXPECT_EQ(LrtDispatchDeviceContextDestroy(device_context), kLrtStatusOk);
}
