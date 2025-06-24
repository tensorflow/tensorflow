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

#include "tensorflow/lite/core/c/c_api.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/delegate_test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"

namespace {

TEST(CApiSimple, Version) {
  const char* version = TfLiteVersion();
  ASSERT_NE(version, nullptr);
  EXPECT_STRNE(version, "");
  int major = -1, minor = -1, patch = -1;
  int ret = sscanf(version, "%d.%d.%d", &major, &minor, &patch);
  // The version number should contain all three components.
  EXPECT_GE(ret, 3);
  // The following checks should work for all TF Lite 2.* versions,
  // but will need updating for TF Lite version 3.0.0.
  EXPECT_EQ(major, 2);
  EXPECT_GE(minor, 12);
  EXPECT_GE(patch, 0);
  // Calling the function again should give the same result.
  EXPECT_STREQ(TfLiteVersion(), version);
}

TEST(CApiSimple, ExtensionApisVersion) {
  const char* version = TfLiteExtensionApisVersion();
  ASSERT_NE(version, nullptr);
  EXPECT_STRNE(version, "");
  int major = -1, minor = -1, patch = -1;
  int ret = sscanf(version, "%d.%d.%d", &major, &minor, &patch);
  // The version number should contain all three components.
  EXPECT_GE(ret, 3);
  EXPECT_GE(major, 0);
  EXPECT_GE(minor, 0);
  EXPECT_GE(patch, 0);
  // Calling the function again should give the same result.
  EXPECT_STREQ(TfLiteExtensionApisVersion(), version);
}

TEST(CApiSimple, SchemaVersion) {
  // The following checks will need updating if we change the schema version.
  EXPECT_EQ(TfLiteSchemaVersion(), 3);
  // Calling the function again should give the same result.
  EXPECT_EQ(TfLiteSchemaVersion(), 3);
}

TEST(CApiSimple, Smoke) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);
  EXPECT_NE(TfLiteInterpreterInputTensorIndices(interpreter), nullptr);
  EXPECT_EQ(TfLiteInterpreterInputTensorIndices(interpreter)[0], 1);
  EXPECT_NE(TfLiteInterpreterOutputTensorIndices(interpreter), nullptr);
  EXPECT_EQ(TfLiteInterpreterOutputTensorIndices(interpreter)[0], 2);

  std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
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
}

TEST(CApiSimple, QuantizationParams) {
  TfLiteModel* model = TfLiteModelCreateFromFile(

      "tensorflow/lite/testdata/add_quantized.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, nullptr);
  ASSERT_NE(interpreter, nullptr);

  TfLiteModelDelete(model);

  const std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteUInt8);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 2);

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.003922f);
  EXPECT_EQ(input_params.zero_point, 0);

  const std::array<uint8_t, 2> input = {1, 3};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(uint8_t)),
            kTfLiteOk);

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.003922f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<uint8_t, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(uint8_t)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3);
  EXPECT_EQ(output[1], 9);

  const float dequantizedOutput0 =
      output_params.scale * (output[0] - output_params.zero_point);
  const float dequantizedOutput1 =
      output_params.scale * (output[1] - output_params.zero_point);
  EXPECT_EQ(dequantizedOutput0, 0.011766f);
  EXPECT_EQ(dequantizedOutput1, 0.035298f);

  TfLiteInterpreterDelete(interpreter);
}

TEST(CApiSimple, TfLiteInterpreterGetTensor) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  // The 'tensorflow/lite/testdata/add.bin' model uses model tensor
  // at index 1 as the input tensor.
  TfLiteTensor* input_tensor = TfLiteInterpreterGetTensor(interpreter, 1);
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

  // The 'third_party/tensorflow/testdata/add.bin' model uses model tensor
  // at index 2 as the output tensor.
  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetTensor(interpreter, 2);
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
}

#if !TFLITE_USE_OPAQUE_DELEGATE
TEST(CApiSimple, Delegate) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  bool delegate_prepared = false;
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  delegate.data_ = &delegate_prepared;
  delegate.Prepare = [](TfLiteContext* context, TfLiteDelegate* delegate) {
    *static_cast<bool*>(delegate->data_) = true;
    return kTfLiteOk;
  };
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, &delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_prepared);

  // Subsequent execution should behave properly (the delegate is a no-op).
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  TfLiteInterpreterDelete(interpreter);
}
#endif

TEST(CApiSimple, DelegateExternal_GetExecutionPlan) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  bool delegate_prepared = false;
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* context,  // NOLINT
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    *static_cast<bool*>(data) = true;

    TfLiteIntArray* execution_plan;
    EXPECT_EQ(kTfLiteOk,
              TfLiteOpaqueContextGetExecutionPlan(context, &execution_plan));
    EXPECT_EQ(2, execution_plan->size);

    return kTfLiteOk;
  };

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

// NOTE: This function does not illustrate intended usage by applications, and
// clients should not mimic such a scenario in their code.
// This is a helper function that retrieves whether the subgraph pointed by the
// given subgraph_index is marked as "delegation-skippable", a check that is
// expected to happen in the TFLite runtime (in the
// Interpreter::ModifyGraphWithDelegate function call).
// The following cast is safe only because this code is part of the API testing.
bool SubgraphIsDelegationSkippable(TfLiteOpaqueContext* context,
                                   int subgraph_index) {
  TfLiteOpaqueContext* skipped_subgraph_context;
  EXPECT_EQ(kTfLiteOk, TfLiteOpaqueContextAcquireSubgraphContext(
                           context, subgraph_index, &skipped_subgraph_context));
  tflite::Subgraph* subgraph = reinterpret_cast<::tflite::Subgraph*>(
      reinterpret_cast<TfLiteContext*>(skipped_subgraph_context)->impl_);
  EXPECT_EQ(kTfLiteOk,
            TfLiteOpaqueContextReleaseSubgraphContext(context, subgraph_index));
  return subgraph->IsDelegationSkippable();
}

TEST(CApiSimple, DelegateExternal_MarkSubgraphAsDelegationSkippable) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/2_subgraphs.bin");

  // Create and install a delegate instance.
  bool delegate_prepared = false;
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* context,  // NOLINT
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    *static_cast<bool*>(data) = true;

    EXPECT_EQ(kTfLiteOk,
              TfLiteOpaqueContextMarkSubgraphAsDelegationSkippable(context, 1));
    EXPECT_TRUE(SubgraphIsDelegationSkippable(context, 1));

    return kTfLiteOk;
  };

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_prepared);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

#if !TFLITE_USE_OPAQUE_DELEGATE
TEST(CApiSimple, DelegateFails) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  TfLiteDelegate delegate = TfLiteDelegateCreate();
  delegate.Prepare = [](TfLiteContext* context, TfLiteDelegate* delegate) {
    return kTfLiteError;
  };
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, &delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // Interpreter creation should fail as delegate preparation failed.
  EXPECT_EQ(nullptr, interpreter);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}
#endif

struct DelegateState {
  bool delegate_prepared;
};

struct OpState {
  bool op_init_called;
};

std::vector<int>* g_nodes_to_replace;
TfLiteOpaqueDelegate* g_opaque_delegate_struct;

TfLiteOperator* CreateDelegateKernelExternalRegistration() {
  TfLiteOperator* delegate_kernel_registration_external = TfLiteOperatorCreate(
      kTfLiteBuiltinDelegate, "TEST DELEGATE KERNEL", /*version=*/1,
      /*user_data=*/nullptr);
  TfLiteOperatorSetInitWithData(
      delegate_kernel_registration_external,
      [](void* user_data, TfLiteOpaqueContext* context, const char* buffer,
         size_t length) -> void* {
        const TfLiteOpaqueDelegateParams* params =
            reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
        // Ensure that the TFLite runtime passes the opaque delegate's address
        // into the delegate kernel.
        EXPECT_EQ(g_opaque_delegate_struct, params->delegate);
        // Ensure that the TFLite runtime passes the opaque delegate's data
        // field into the delegate kernel.  Note that our delegate's 'Prepare'
        // callback marks the delegate as 'prepared' before it calls
        // 'TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels'.
        DelegateState* delegate_state =
            static_cast<DelegateState*>(params->delegate_data);
        EXPECT_TRUE(delegate_state->delegate_prepared);

        for (int i = 0; i < params->nodes_to_replace->size; ++i) {
          g_nodes_to_replace->push_back(params->nodes_to_replace->data[i]);
        }
        return new OpState{true};
      });
  TfLiteOperatorSetFreeWithData(
      delegate_kernel_registration_external,
      [](void* user_data, TfLiteOpaqueContext* context, void* buffer) {
        delete (reinterpret_cast<OpState*>(buffer));
      });
  return delegate_kernel_registration_external;
}

TEST(CApiSimple, OpaqueDelegate_ReplaceNodeSubsetsWithDelegateKernels) {
  g_nodes_to_replace = new std::vector<int>();

  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  DelegateState delegate_state{false};
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    DelegateState* delegate_state = static_cast<DelegateState*>(data);
    delegate_state->delegate_prepared = true;

    TfLiteIntArray* execution_plan;
    TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan);
    EXPECT_EQ(execution_plan->size, 2);

    TfLiteOpaqueNode* node = nullptr;
    TfLiteOperator* registration_external = nullptr;
    TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, 0, &node,
                                              &registration_external);
    EXPECT_NE(node, nullptr);
    EXPECT_NE(registration_external, nullptr);

    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, CreateDelegateKernelExternalRegistration(),
        execution_plan, opaque_delegate);

    return kTfLiteOk;
  };

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  g_opaque_delegate_struct = opaque_delegate;

  EXPECT_EQ(g_nodes_to_replace->size(), 0);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  TfLiteModelDelete(model);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_state.delegate_prepared);
  std::vector<int>& nodes_to_replace = *g_nodes_to_replace;
  // We know that "tensorflow/lite/testdata/add.bin" contains two
  // nodes, 0 and 1, and that 0 comes before 1 in the execution plan.
  EXPECT_EQ(nodes_to_replace.size(), 2);
  EXPECT_EQ(nodes_to_replace[0], 0);
  EXPECT_EQ(nodes_to_replace[1], 1);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
  delete g_nodes_to_replace;
  g_opaque_delegate_struct = nullptr;
}

TEST(CApiSimple, OpaqueDelegate_TransferOperatorOwnershipWithoutNodeToReplace) {
  g_nodes_to_replace = new std::vector<int>();

  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  // Create and install a delegate instance.
  DelegateState delegate_state{false};
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    DelegateState* delegate_state = static_cast<DelegateState*>(data);
    delegate_state->delegate_prepared = true;

    TfLiteOpaqueNode* node = nullptr;
    TfLiteOperator* registration_external = nullptr;
    TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, 0, &node,
                                              &registration_external);
    EXPECT_NE(node, nullptr);
    EXPECT_NE(registration_external, nullptr);

    // Create a fake execution plan to avoid replacing nodes.
    TfLiteIntArray* fake_execution_plan = TfLiteIntArrayCreate(0);
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, CreateDelegateKernelExternalRegistration(),
        fake_execution_plan, opaque_delegate);
    TfLiteIntArrayFree(fake_execution_plan);

    return kTfLiteOk;
  };

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  g_opaque_delegate_struct = opaque_delegate;

  EXPECT_EQ(g_nodes_to_replace->size(), 0);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  TfLiteModelDelete(model);

  // The delegate should have been applied with 0 node to replace.
  EXPECT_TRUE(delegate_state.delegate_prepared);
  std::vector<int>& nodes_to_replace = *g_nodes_to_replace;
  EXPECT_EQ(nodes_to_replace.size(), 0);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
  delete g_nodes_to_replace;
  g_opaque_delegate_struct = nullptr;
}

using ::tflite::delegates::test_utils::TestFP16Delegation;

TEST_F(TestFP16Delegation,
       ReplaceNodeSubsetsWithDelegateKernels_MultipleDelegateKernels) {
  g_nodes_to_replace = new std::vector<int>();

  // Create and install a delegate instance.
  DelegateState delegate_state{false};
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    DelegateState* delegate_state = static_cast<DelegateState*>(data);
    delegate_state->delegate_prepared = true;
    TfLiteIntArray* execution_plan;
    TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan);

    std::vector<int> nodes_to_replace;
    for (int i = 0; i < execution_plan->size; i++) {
      TfLiteOpaqueNode* node = nullptr;
      TfLiteOperator* registration_external = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(opaque_context,
                                                execution_plan->data[i], &node,
                                                &registration_external);
      EXPECT_NE(node, nullptr);
      EXPECT_NE(registration_external, nullptr);
      if (TfLiteOperatorGetBuiltInCode(registration_external) ==
          kTfLiteBuiltinAdd) {
        nodes_to_replace.push_back(execution_plan->data[i]);
      }
    }

    TfLiteIntArray* subset_to_replace =
        TfLiteIntArrayCreate(nodes_to_replace.size());
    for (int i = 0; i < nodes_to_replace.size(); i++) {
      subset_to_replace->data[i] = nodes_to_replace[i];
    }

    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, CreateDelegateKernelExternalRegistration(),
        subset_to_replace, opaque_delegate);

    TfLiteIntArrayFree(subset_to_replace);
    return kTfLiteOk;
  };
  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  g_opaque_delegate_struct = opaque_delegate;
  EXPECT_EQ(g_nodes_to_replace->size(), 0);
  EXPECT_EQ(interpreter_->execution_plan().size(), 8);
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(opaque_delegate), kTfLiteOk);
  EXPECT_EQ(interpreter_->execution_plan().size(), 7);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_state.delegate_prepared);
  std::vector<int>& nodes_to_replace = *g_nodes_to_replace;
  EXPECT_EQ(nodes_to_replace.size(), 3);
  // We know based on the graph structure that nodes 1, 3 and 7 are nodes with
  // an ADD operation.
  EXPECT_EQ(nodes_to_replace[0], 1);
  EXPECT_EQ(nodes_to_replace[1], 3);
  EXPECT_EQ(nodes_to_replace[2], 7);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
  delete g_nodes_to_replace;
  g_opaque_delegate_struct = nullptr;
}

static void error_reporter(void* user_data, const char* format, va_list args) {
  reinterpret_cast<tflite::TestErrorReporter*>(user_data)->Report(format, args);
}

TEST(CApiSimple, InterpreterOptionsCopy) {
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptions* copy = TfLiteInterpreterOptionsCopy(options);
  ASSERT_NE(copy, nullptr);
  ASSERT_NE(copy, options);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterOptionsDelete(copy);
}

TEST(CApiSimple, ErrorReporter) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  // Install a custom error reporter into the interpreter by way of options.
  tflite::TestErrorReporter reporter;
  TfLiteInterpreterOptionsSetErrorReporter(options, error_reporter, &reporter);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  // Invoke the interpreter before tensor allocation.
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteError);

  // The error should propagate to the custom error reporter.
  EXPECT_EQ(reporter.error_messages(),
            "Invoke called on model that is not ready.");
  EXPECT_EQ(reporter.num_calls(), 1);

  TfLiteInterpreterDelete(interpreter);
}

TEST(CApiSimple, ModelCreateWithErrorReporter) {
  TfLiteModel* model = nullptr;
  tflite::TestErrorReporter reporter;

  // valid model with error reporter
  std::ifstream model_file("tensorflow/lite/testdata/add.bin");
  model_file.seekg(0, std::ios_base::end);
  std::vector<char> model_buffer(model_file.tellg());
  model_file.seekg(0, std::ios_base::beg);
  model_file.read(model_buffer.data(), model_buffer.size());
  model = TfLiteModelCreateWithErrorReporter(
      model_buffer.data(), model_buffer.size(), error_reporter, &reporter);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(reporter.error_messages(), "");
  TfLiteModelDelete(model);

  // invalid model with error reporter
  std::vector<char> invalid_model(20, 'c');
  model = TfLiteModelCreateWithErrorReporter(
      invalid_model.data(), invalid_model.size(), error_reporter, &reporter);
  ASSERT_EQ(model, nullptr);
  EXPECT_EQ(reporter.error_messages(),
            "The model is not a valid Flatbuffer buffer");
  TfLiteModelDelete(model);
}

TEST(CApiSimple, ModelCreateFromFileWithErrorReporter) {
  TfLiteModel* model = nullptr;
  tflite::TestErrorReporter reporter;

  // valid model file with error reporter
  model = TfLiteModelCreateFromFileWithErrorReporter(
      "tensorflow/lite/testdata/add.bin", error_reporter,
      &reporter);
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(reporter.error_messages(), "");
  TfLiteModelDelete(model);

  // invalid model file with error reporter
  std::vector<char> invalid_model(20, 'c');
  model = TfLiteModelCreateFromFileWithErrorReporter("invalid/path/foo.tflite",
                                                     error_reporter, &reporter);
  ASSERT_EQ(model, nullptr);
  ASSERT_THAT(
      reporter.error_messages(),
      testing::ContainsRegex("Could not open 'invalid/path/foo.tflite'."));
  TfLiteModelDelete(model);
}

struct DelegateKernelState {
  TfLiteOpaqueTensor* input_tensor = nullptr;
  TfLiteOpaqueTensor* output_tensor = nullptr;
};

TfLiteOperator* CreateReg() {
  auto reg_ex = TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                                     "Test driver delegate", /*version=*/1,
                                     /*user_data=*/nullptr);
  TfLiteOperatorSetInitWithData(
      reg_ex,
      [](void* user_data, TfLiteOpaqueContext* context, const char* buffer,
         size_t length) -> void* {
        const TfLiteOpaqueDelegateParams* params =
            reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);

        for (int i = 0; i < params->nodes_to_replace->size; ++i) {
          TfLiteOpaqueNode* node = nullptr;
          TfLiteOperator* registration_external = nullptr;
          TfLiteOpaqueContextGetNodeAndRegistration(
              context, params->nodes_to_replace->data[i], &node,
              &registration_external);
          EXPECT_NE(nullptr, node);
          EXPECT_NE(nullptr, registration_external);
          EXPECT_EQ(2, TfLiteOpaqueNodeNumberOfInputs(node));
          EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfOutputs(node));
          EXPECT_EQ(kTfLiteBuiltinAdd,
                    TfLiteOperatorGetBuiltInCode(registration_external));
        }

        TfLiteIntArray* input_tensors = params->input_tensors;
        EXPECT_EQ(1, input_tensors->size);
        TfLiteIntArray* output_tensors = params->output_tensors;
        EXPECT_EQ(1, output_tensors->size);

        TfLiteOpaqueTensor* input_tensor =
            TfLiteOpaqueContextGetOpaqueTensor(context, input_tensors->data[0]);
        TfLiteOpaqueTensor* output_tensor = TfLiteOpaqueContextGetOpaqueTensor(
            context, output_tensors->data[0]);
        return new DelegateKernelState{input_tensor, output_tensor};
      });

  TfLiteOperatorSetInvokeWithData(
      reg_ex,
      [](void* user_data, TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        DelegateKernelState* delegate_kernel =
            reinterpret_cast<DelegateKernelState*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));

        float* input_data = reinterpret_cast<float*>(
            TfLiteOpaqueTensorData(delegate_kernel->input_tensor));
        float* output_data = reinterpret_cast<float*>(
            TfLiteOpaqueTensorData(delegate_kernel->output_tensor));
        for (int i = 0; i < (1 * 8 * 8 * 3); ++i) {
          output_data[i] = input_data[i] * 3;
        }
        return kTfLiteOk;
      });

  TfLiteOperatorSetFreeWithData(
      reg_ex, [](void* user_data, TfLiteOpaqueContext* context, void* data) {
        DelegateKernelState* state =
            reinterpret_cast<DelegateKernelState*>(data);
        delete state;
      });
  return reg_ex;
}

TEST(CApiSimple, OpaqueDelegate_TfLiteOpaqueTensorGet) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  struct DelegateState {
    bool delegate_prepared = false;
  };
  DelegateState delegate_state{false};

  // Create and install a delegate instance.
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    auto delegate_state = static_cast<DelegateState*>(data);
    delegate_state->delegate_prepared = true;

    TfLiteIntArray* execution_plan = nullptr;
    TfLiteOpaqueContextGetExecutionPlan(context, &execution_plan);

    EXPECT_EQ(2, execution_plan->size);
    std::vector<int> node_ids_to_replace;
    for (int i = 0; i < execution_plan->size; ++i) {
      TfLiteOpaqueNode* node = nullptr;
      TfLiteOperator* registration_external = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(
          context, execution_plan->data[i], &node, &registration_external);
      EXPECT_NE(nullptr, node);
      EXPECT_NE(nullptr, registration_external);
      EXPECT_EQ(2, TfLiteOpaqueNodeNumberOfInputs(node));
      EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfOutputs(node));
      EXPECT_EQ(kTfLiteBuiltinAdd,
                TfLiteOperatorGetBuiltInCode(registration_external));
      node_ids_to_replace.push_back(execution_plan->data[i]);
    }

    TfLiteIntArray* nodes_to_replace =
        TfLiteIntArrayCreate(node_ids_to_replace.size());
    for (int i = 0; i < node_ids_to_replace.size(); ++i) {
      nodes_to_replace->data[i] = node_ids_to_replace[i];
    }

    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        context, CreateReg(), nodes_to_replace, opaque_delegate);

    TfLiteIntArrayFree(nodes_to_replace);
    return kTfLiteOk;
  };

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_state.delegate_prepared);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  EXPECT_NE(TfLiteTensorData(input_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(input_tensor), "input");

  const float kTensorCellValue = 3.f;
  int64_t n = 1 * 8 * 8 * 3;
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

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST(CApiSimple, OpaqueContextGetNodeAndRegistration) {
  struct DelegatePrepareStatus {
    bool prepared;
  };
  DelegatePrepareStatus delegate_state{false};

  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    DelegatePrepareStatus* delegate_state =
        static_cast<DelegatePrepareStatus*>(data);
    delegate_state->prepared = true;

    TfLiteIntArray* execution_plan;
    TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan);
    EXPECT_EQ(execution_plan->size, 2);

    for (int i = 0; i < execution_plan->size; i++) {
      TfLiteOpaqueNode* node = nullptr;
      TfLiteOperator* registration_external = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, 0, &node,
                                                &registration_external);
      EXPECT_NE(node, nullptr);
      EXPECT_NE(registration_external, nullptr);
      EXPECT_EQ(kTfLiteBuiltinAdd,
                TfLiteOperatorGetBuiltInCode(registration_external));
      EXPECT_EQ(1, TfLiteOperatorGetVersion(registration_external));
      EXPECT_EQ(2, TfLiteOpaqueNodeNumberOfInputs(node));
      EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfOutputs(node));
    }

    {
      TfLiteOpaqueNode* node = nullptr;
      TfLiteOperator* registration_external = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, 0, &node,
                                                &registration_external);
      EXPECT_EQ(1, TfLiteOpaqueNodeGetInputTensorIndex(node, 0));
      EXPECT_EQ(1, TfLiteOpaqueNodeGetInputTensorIndex(node, 1));
      EXPECT_EQ(-1, TfLiteOpaqueNodeGetInputTensorIndex(node, 2));
      EXPECT_EQ(0, TfLiteOpaqueNodeGetOutputTensorIndex(node, 0));
      EXPECT_EQ(-1, TfLiteOpaqueNodeGetOutputTensorIndex(node, 123));
      EXPECT_EQ(-1, TfLiteOpaqueNodeGetOutputTensorIndex(node, -1));

      const TfLiteOpaqueTensor* opaque_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(opaque_context, 0);
      EXPECT_NE(opaque_tensor, nullptr);
      EXPECT_EQ(kTfLiteFloat32, TfLiteOpaqueTensorType(opaque_tensor));
      size_t bytes_float_32 = 0;
      EXPECT_EQ(kTfLiteOk,
                TfLiteOpaqueContextGetSizeOfType(opaque_context, kTfLiteFloat32,
                                                 &bytes_float_32));
      EXPECT_EQ(bytes_float_32, sizeof(float));
    }
    {
      TfLiteOpaqueNode* node = nullptr;
      TfLiteOperator* registration_external = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, 1, &node,
                                                &registration_external);
      EXPECT_EQ(0, TfLiteOpaqueNodeGetInputTensorIndex(node, 0));
      EXPECT_EQ(1, TfLiteOpaqueNodeGetInputTensorIndex(node, 1));
      EXPECT_EQ(2, TfLiteOpaqueNodeGetOutputTensorIndex(node, 0));
    }
    return kTfLiteOk;
  };

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  TfLiteModelDelete(model);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_state.prepared);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST(CApiSimple, TfLiteOperatorGetVersionNullptr) {
  EXPECT_EQ(TfLiteOperatorGetVersion(nullptr), -1);
}

TEST(CApiSimple, TfLiteOpaqueContextResizeTensor) {
  struct DelegatePrepareStatus {
    bool prepared;
  };
  DelegatePrepareStatus delegate_state{false};

  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) {
    DelegatePrepareStatus* delegate_state =
        static_cast<DelegatePrepareStatus*>(data);
    delegate_state->prepared = true;

    TfLiteOpaqueTensor* tensor =
        TfLiteOpaqueContextGetOpaqueTensor(opaque_context, 0);
    EXPECT_EQ(4, TfLiteOpaqueTensorNumDims(tensor));
    EXPECT_EQ(1, TfLiteOpaqueTensorDim(tensor, 0));
    EXPECT_EQ(8, TfLiteOpaqueTensorDim(tensor, 1));
    EXPECT_EQ(8, TfLiteOpaqueTensorDim(tensor, 2));
    EXPECT_EQ(3, TfLiteOpaqueTensorDim(tensor, 3));

    TfLiteIntArray* new_dims = TfLiteIntArrayCreate(3);
    new_dims->data[0] = 2;
    new_dims->data[1] = 3;
    new_dims->data[2] = 4;
    EXPECT_EQ(kTfLiteOk, TfLiteOpaqueContextResizeTensor(opaque_context, tensor,
                                                         new_dims));

    EXPECT_EQ(new_dims->size, TfLiteOpaqueTensorNumDims(tensor));
    EXPECT_EQ(new_dims->data[0], TfLiteOpaqueTensorDim(tensor, 0));
    EXPECT_EQ(new_dims->data[1], TfLiteOpaqueTensorDim(tensor, 1));
    EXPECT_EQ(new_dims->data[2], TfLiteOpaqueTensorDim(tensor, 2));
    return kTfLiteOk;
  };

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  TfLiteModelDelete(model);

  // The delegate should have been applied.
  EXPECT_TRUE(delegate_state.prepared);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST(CApiSimple, ValidModel) {
  std::ifstream model_file("tensorflow/lite/testdata/add.bin");

  model_file.seekg(0, std::ios_base::end);
  std::vector<char> model_buffer(model_file.tellg());

  model_file.seekg(0, std::ios_base::beg);
  model_file.read(model_buffer.data(), model_buffer.size());

  TfLiteModel* model =
      TfLiteModelCreate(model_buffer.data(), model_buffer.size());
  ASSERT_NE(model, nullptr);
  TfLiteModelDelete(model);
}

TEST(CApiSimple, ValidModelFromFile) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  TfLiteModelDelete(model);
}

TEST(CApiSimple, InvalidModel) {
  std::vector<char> invalid_model(20, 'c');
  TfLiteModel* model =
      TfLiteModelCreate(invalid_model.data(), invalid_model.size());
  ASSERT_EQ(model, nullptr);
}

TEST(CApiSimple, InvalidModelFromFile) {
  TfLiteModel* model = TfLiteModelCreateFromFile("invalid/path/foo.tflite");
  ASSERT_EQ(model, nullptr);
}

struct SinhParams {
  bool use_cosh_instead = false;
};

void* FlexSinhInit(void* user_data, TfLiteOpaqueContext* context,
                   const char* buffer, size_t length) {
  auto sinh_params = new SinhParams;
  // The buffer that is passed into here is the custom_options
  // field from the flatbuffer
  // (third_party/tensorflow/compiler/mlir/lite/schema/schema.fbs) `Operator`
  // for this node. Typically it should be stored as a FlexBuffer, but for this
  // test we assume that it is just a string.
  if (std::string(buffer, length) == "use_cosh") {
    sinh_params->use_cosh_instead = true;
  }
  return sinh_params;
}

void FlexSinhFree(void* user_data, TfLiteOpaqueContext* context, void* data) {
  delete static_cast<SinhParams*>(data);
}

TfLiteStatus FlexSinhPrepare(void* user_data, TfLiteOpaqueContext* context,
                             TfLiteOpaqueNode* node) {
  return kTfLiteOk;
}

TfLiteStatus FlexSinhEval(void* user_data, TfLiteOpaqueContext* context,
                          TfLiteOpaqueNode* node) {
  auto sinh_params =
      static_cast<SinhParams*>(TfLiteOpaqueNodeGetUserData(node));
  EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfInputs(node));
  const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
  size_t input_bytes = TfLiteOpaqueTensorByteSize(input);
  void* data_ptr = TfLiteOpaqueTensorData(input);
  float input_value;
  memcpy(&input_value, data_ptr, input_bytes);

  EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfOutputs(node));
  TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);
  float output_value = sinh_params->use_cosh_instead ? std::cosh(input_value)
                                                     : std::sinh(input_value);
  TfLiteOpaqueTensorCopyFromBuffer(output, &output_value, sizeof(output_value));
  return kTfLiteOk;
}

TEST(CApiSimple, CustomOpSupport) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/custom_sinh.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOperator* reg =
      TfLiteOperatorCreate(kTfLiteBuiltinCustom, "Sinh", /*version=*/1,
                           /*user_data=*/nullptr);
  TfLiteOperatorSetPrepareWithData(reg, &FlexSinhPrepare);
  TfLiteOperatorSetInitWithData(reg, &FlexSinhInit);
  TfLiteOperatorSetFreeWithData(reg, &FlexSinhFree);
  TfLiteOperatorSetInvokeWithData(reg, &FlexSinhEval);

  const char* kCustomName = TfLiteOperatorGetCustomName(reg);
  EXPECT_EQ("Sinh", kCustomName);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddOperator(options, reg);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  TfLiteInterpreterOptionsDelete(options);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  float input_value = 1.0f;
  TfLiteTensorCopyFromBuffer(input_tensor, &input_value, sizeof(float));

  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  float output_value;
  TfLiteTensorCopyToBuffer(output_tensor, &output_value, sizeof(float));
  EXPECT_EQ(output_value, std::sinh(1.0f));

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOperatorDelete(reg);
}

const TfLiteRegistration* find_builtin_op_add(void* user_data,
                                              TfLiteBuiltinOperator op,
                                              int version) {
  static TfLiteRegistration registration{/*init=*/nullptr,
                                         /*free=*/nullptr,
                                         /*prepare=*/nullptr,
                                         /*invoke=*/nullptr,
                                         /*profiling_string=*/nullptr,
                                         /*builtin_code=*/kTfLiteBuiltinAdd,
                                         /*custom_name=*/nullptr,
                                         /*version=*/1};
  if (op == kTfLiteBuiltinAdd && version == 1) {
    return &registration;
  }
  return nullptr;
}

const TfLiteRegistration* find_custom_op_sinh(void* user_data, const char* op,
                                              int version) {
  static TfLiteRegistration registration{/*init=*/nullptr,
                                         /*free=*/nullptr,
                                         /*prepare=*/nullptr,
                                         /*invoke=*/nullptr,
                                         /*profiling_string=*/nullptr,
                                         /*builtin_code=*/kTfLiteBuiltinCustom,
                                         /*custom_name=*/"Sinh",
                                         /*version=*/1};
  if (strcmp(op, "Sinh") == 0 && version == 1) {
    return &registration;
  }
  return nullptr;
}

TEST(CApiSimple, CallbackOpResolver) {
  tflite::internal::CallbackOpResolver resolver;
  struct TfLiteOpResolverCallbacks callbacks {};
  callbacks.find_builtin_op = find_builtin_op_add;
  callbacks.find_custom_op = find_custom_op_sinh;

  resolver.SetCallbacks(callbacks);
  auto reg_add = resolver.FindOp(
      static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
  ASSERT_NE(reg_add, nullptr);
  EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
  EXPECT_EQ(reg_add->version, 1);
  EXPECT_EQ(reg_add->registration_external, nullptr);

  EXPECT_EQ(
      resolver.FindOp(
          static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinConv2d), 1),
      nullptr);

  auto reg_sinh = resolver.FindOp("Sinh", 1);
  ASSERT_NE(reg_sinh, nullptr);
  EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
  EXPECT_EQ(reg_sinh->custom_name, "Sinh");
  EXPECT_EQ(reg_sinh->version, 1);
  EXPECT_EQ(reg_sinh->registration_external, nullptr);

  EXPECT_EQ(resolver.FindOp("Cosh", 1), nullptr);
}

const TfLiteRegistration_V1* dummy_find_builtin_op_v1(void* user_data,
                                                      TfLiteBuiltinOperator op,
                                                      int version) {
  static TfLiteRegistration_V1 registration_v1{
      nullptr, nullptr,           nullptr, nullptr,
      nullptr, kTfLiteBuiltinAdd, nullptr, 1};
  if (op == kTfLiteBuiltinAdd) {
    return &registration_v1;
  }
  return nullptr;
}

const TfLiteRegistration_V1* dummy_find_custom_op_v1(void* user_data,
                                                     const char* op,
                                                     int version) {
  static TfLiteRegistration_V1 registration_v1{
      nullptr, nullptr, nullptr, nullptr, nullptr, kTfLiteBuiltinCustom,
      "Sinh",  1};
  if (strcmp(op, "Sinh") == 0) {
    return &registration_v1;
  }
  return nullptr;
}

TEST(CApiSimple, CallbackOpResolver_V1) {
  tflite::internal::CallbackOpResolver resolver;
  struct TfLiteOpResolverCallbacks callbacks {};
  callbacks.find_builtin_op_v1 = dummy_find_builtin_op_v1;
  callbacks.find_custom_op_v1 = dummy_find_custom_op_v1;

  resolver.SetCallbacks(callbacks);
  auto reg_add = resolver.FindOp(
      static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
  ASSERT_NE(reg_add, nullptr);
  EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
  EXPECT_EQ(reg_add->version, 1);
  EXPECT_EQ(reg_add->registration_external, nullptr);

  EXPECT_EQ(
      resolver.FindOp(
          static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinConv2d), 1),
      nullptr);

  // Query kTfLiteBuiltinAdd multiple times to check if caching logic works.
  for (int i = 0; i < 10; ++i) {
    auto reg_add = resolver.FindOp(
        static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
    ASSERT_NE(reg_add, nullptr);
    EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
    EXPECT_EQ(reg_add->version, 1);
    EXPECT_EQ(reg_add->registration_external, nullptr);
  }

  auto reg_sinh = resolver.FindOp("Sinh", 1);
  ASSERT_NE(reg_sinh, nullptr);
  EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
  EXPECT_EQ(reg_sinh->custom_name, "Sinh");
  EXPECT_EQ(reg_sinh->version, 1);
  EXPECT_EQ(reg_sinh->registration_external, nullptr);

  EXPECT_EQ(resolver.FindOp("Cosh", 1), nullptr);

  // Query "Sinh" multiple times to check if caching logic works.
  for (int i = 0; i < 10; ++i) {
    auto reg_sinh = resolver.FindOp("Sinh", 1);
    ASSERT_NE(reg_sinh, nullptr);
    EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
    EXPECT_EQ(reg_sinh->custom_name, "Sinh");
    EXPECT_EQ(reg_sinh->version, 1);
    EXPECT_EQ(reg_sinh->registration_external, nullptr);
  }
}

const TfLiteRegistration_V3* dummy_find_builtin_op_v3(void* user_data,
                                                      TfLiteBuiltinOperator op,
                                                      int version) {
  static TfLiteRegistration_V3 registration_v3{
      nullptr,           nullptr, nullptr, nullptr, nullptr,
      kTfLiteBuiltinAdd, nullptr, 1,       nullptr};
  if (op == kTfLiteBuiltinAdd) {
    return &registration_v3;
  }
  return nullptr;
}

const TfLiteRegistration_V3* dummy_find_custom_op_v3(void* user_data,
                                                     const char* op,
                                                     int version) {
  static TfLiteRegistration_V3 registration_v3{
      nullptr, nullptr, nullptr, nullptr, nullptr, kTfLiteBuiltinCustom,
      "Sinh",  1,       nullptr};
  if (strcmp(op, "Sinh") == 0) {
    return &registration_v3;
  }
  return nullptr;
}

TEST(CApiSimple, CallbackOpResolver_V3) {
  tflite::internal::CallbackOpResolver resolver;
  struct TfLiteOpResolverCallbacks callbacks {};
  callbacks.find_builtin_op_v3 = dummy_find_builtin_op_v3;
  callbacks.find_custom_op_v3 = dummy_find_custom_op_v3;

  resolver.SetCallbacks(callbacks);
  auto reg_add = resolver.FindOp(
      static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
  ASSERT_NE(reg_add, nullptr);
  EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
  EXPECT_EQ(reg_add->version, 1);
  EXPECT_EQ(reg_add->registration_external, nullptr);

  EXPECT_EQ(
      resolver.FindOp(
          static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinConv2d), 1),
      nullptr);

  // Query kTfLiteBuiltinAdd multiple times to check if caching logic works.
  for (int i = 0; i < 10; ++i) {
    auto reg_add = resolver.FindOp(
        static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
    ASSERT_NE(reg_add, nullptr);
    EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
    EXPECT_EQ(reg_add->version, 1);
    EXPECT_EQ(reg_add->registration_external, nullptr);
  }

  auto reg_sinh = resolver.FindOp("Sinh", 1);
  ASSERT_NE(reg_sinh, nullptr);
  EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
  EXPECT_EQ(reg_sinh->custom_name, "Sinh");
  EXPECT_EQ(reg_sinh->version, 1);
  EXPECT_EQ(reg_sinh->registration_external, nullptr);

  EXPECT_EQ(resolver.FindOp("Cosh", 1), nullptr);

  // Query "Sinh" multiple times to check if caching logic works.
  for (int i = 0; i < 10; ++i) {
    auto reg_sinh = resolver.FindOp("Sinh", 1);
    ASSERT_NE(reg_sinh, nullptr);
    EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
    EXPECT_EQ(reg_sinh->custom_name, "Sinh");
    EXPECT_EQ(reg_sinh->version, 1);
    EXPECT_EQ(reg_sinh->registration_external, nullptr);
  }
}

const TfLiteRegistration_V2* dummy_find_builtin_op_v2(void* user_data,
                                                      TfLiteBuiltinOperator op,
                                                      int version) {
  static TfLiteRegistration_V2 registration_v2{
      nullptr,           nullptr, nullptr, nullptr, nullptr,
      kTfLiteBuiltinAdd, nullptr, 1,       nullptr};
  if (op == kTfLiteBuiltinAdd) {
    return &registration_v2;
  }
  return nullptr;
}

const TfLiteRegistration_V2* dummy_find_custom_op_v2(void* user_data,
                                                     const char* op,
                                                     int version) {
  static TfLiteRegistration_V2 registration_v2{
      nullptr, nullptr, nullptr, nullptr, nullptr, kTfLiteBuiltinCustom,
      "Sinh",  1,       nullptr};
  if (strcmp(op, "Sinh") == 0) {
    return &registration_v2;
  }
  return nullptr;
}

TEST(CApiSimple, CallbackOpResolver_V2) {
  tflite::internal::CallbackOpResolver resolver;
  struct TfLiteOpResolverCallbacks callbacks {};
  callbacks.find_builtin_op_v2 = dummy_find_builtin_op_v2;
  callbacks.find_custom_op_v2 = dummy_find_custom_op_v2;

  resolver.SetCallbacks(callbacks);
  auto reg_add = resolver.FindOp(
      static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
  ASSERT_NE(reg_add, nullptr);
  EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
  EXPECT_EQ(reg_add->version, 1);
  EXPECT_EQ(reg_add->registration_external, nullptr);

  EXPECT_EQ(
      resolver.FindOp(
          static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinConv2d), 1),
      nullptr);

  // Query kTfLiteBuiltinAdd multiple times to check if caching logic works.
  for (int i = 0; i < 10; ++i) {
    auto reg_add = resolver.FindOp(
        static_cast<::tflite::BuiltinOperator>(kTfLiteBuiltinAdd), 1);
    ASSERT_NE(reg_add, nullptr);
    EXPECT_EQ(reg_add->builtin_code, kTfLiteBuiltinAdd);
    EXPECT_EQ(reg_add->version, 1);
    EXPECT_EQ(reg_add->registration_external, nullptr);
  }

  auto reg_sinh = resolver.FindOp("Sinh", 1);
  ASSERT_NE(reg_sinh, nullptr);
  EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
  EXPECT_EQ(reg_sinh->custom_name, "Sinh");
  EXPECT_EQ(reg_sinh->version, 1);
  EXPECT_EQ(reg_sinh->registration_external, nullptr);

  EXPECT_EQ(resolver.FindOp("Cosh", 1), nullptr);

  // Query "Sinh" multiple times to check if caching logic works.
  for (int i = 0; i < 10; ++i) {
    auto reg_sinh = resolver.FindOp("Sinh", 1);
    ASSERT_NE(reg_sinh, nullptr);
    EXPECT_EQ(reg_sinh->builtin_code, kTfLiteBuiltinCustom);
    EXPECT_EQ(reg_sinh->custom_name, "Sinh");
    EXPECT_EQ(reg_sinh->version, 1);
    EXPECT_EQ(reg_sinh->registration_external, nullptr);
  }
}

const char* kSubgraphName = "TheName";

TEST(CApiSimple, OpaqueApiAccessors) {
  //
  // Construct a model in-memory with various node and tensor properties that
  // we are going to query with the API functions that work with opaque types.
  //
  ::tflite::Interpreter interpreter;
  interpreter.primary_subgraph().SetName(kSubgraphName);
  interpreter.AddTensors(3);
  std::vector<int> dims = {1, 3};
  std::vector<int> dims_signature = {-1, 3};
  interpreter.SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "a", dims, TfLiteQuantizationParams{1.0, 0},
      /*is_variable=*/false, &dims_signature);
  interpreter.SetTensorParametersReadWrite(
      1, kTfLiteFloat32, "b", dims, TfLiteQuantizationParams{1.0, 0},
      /*is_variable=*/true, &dims_signature);
  interpreter.SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "c", dims, TfLiteQuantizationParams{1.0, 0},
      /*is_variable=*/false, &dims_signature);
  // Add an additional "blank" tensor that doesn't have its properties set via
  // an API like 'SetTensorParametersReadWrite' to simulate the case where one
  // or multiple blank tensors are added after the model has been loaded.
  interpreter.AddTensors(1);

  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({2});
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  const TfLiteRegistration* reg =
      resolver.FindOp(::tflite::BuiltinOperator_ADD, 1);
  interpreter.AddNodeWithParameters({0, 1}, {2}, initial_data, 0, builtin_data,
                                    reg);
  interpreter.primary_subgraph().variables().push_back(1);

  //
  // We delegate all nodes to a kernel, so that the TFLite runtime provides us
  // with an opaque context that represents the model that we constructed
  // in-memory.
  //
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_kernel_invoked = false;
  opaque_delegate_builder.data = &delegate_kernel_invoked;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* context,
                                       TfLiteOpaqueDelegate* delegate,
                                       void* data) -> TfLiteStatus {
    //
    // Define a delegate kernel that checks that the properties of the model
    // are accessible via the opaque API function.
    //
    TfLiteOperator* reg = TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                                               "my delegate", /*version=*/123,
                                               /*user_data=*/nullptr);
    EXPECT_EQ(123, TfLiteOperatorGetVersion(reg));
    TfLiteOperatorSetInitWithData(
        reg,
        [](void* user_data, TfLiteOpaqueContext* opaque_context,
           const char* buffer, size_t length) -> void* {
          const TfLiteOpaqueDelegateParams* params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
          EXPECT_EQ(2, params->input_tensors->size);
          TfLiteOpaqueTensor* opaque_input_tensor =
              TfLiteOpaqueContextGetOpaqueTensor(
                  opaque_context, params->input_tensors->data[0]);
          EXPECT_EQ(2, TfLiteOpaqueTensorNumDims(opaque_input_tensor));
          EXPECT_EQ(1, TfLiteOpaqueTensorDim(opaque_input_tensor, 0));
          EXPECT_EQ(3, TfLiteOpaqueTensorDim(opaque_input_tensor, 1));

          int32_t num_dims = 0;
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueTensorGetNumDimsSignature(
                                   opaque_input_tensor, &num_dims));
          EXPECT_EQ(2, num_dims);

          int32_t dim_length = 0;
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueTensorGetDimSignature(
                                   opaque_input_tensor, 0, &dim_length));
          EXPECT_EQ(-1, dim_length);
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueTensorGetDimSignature(
                                   opaque_input_tensor, 1, &dim_length));
          EXPECT_EQ(3, dim_length);

          EXPECT_FALSE(TfLiteOpaqueTensorIsVariable(opaque_input_tensor));
          EXPECT_TRUE(
              TfLiteOpaqueTensorIsVariable(TfLiteOpaqueContextGetOpaqueTensor(
                  opaque_context, params->input_tensors->data[1])));
          EXPECT_EQ(kTfLiteArenaRw,
                    TfLiteOpaqueTensorGetAllocationType(opaque_input_tensor));

          TfLiteQuantizationParams quantization_params =
              TfLiteOpaqueTensorGetQuantizationParams(opaque_input_tensor);
          EXPECT_EQ(1.0, quantization_params.scale);
          EXPECT_EQ(0, quantization_params.zero_point);

          TfLiteQuantization quantization =
              TfLiteOpaqueTensorGetQuantization(opaque_input_tensor);
          EXPECT_EQ(kTfLiteAffineQuantization, quantization.type);
          EXPECT_STREQ(kSubgraphName,
                       TfLiteOpaqueContextGetName(opaque_context));
          EXPECT_EQ(4, TfLiteOpaqueContextGetNumTensors(opaque_context));

          EXPECT_EQ(-1,
                    TfLiteOpaqueTensorNumDims(
                        TfLiteOpaqueContextGetOpaqueTensor(opaque_context, 3)));

          EXPECT_EQ(kTfLiteOk,
                    TfLiteOpaqueTensorGetNumDimsSignature(
                        TfLiteOpaqueContextGetOpaqueTensor(opaque_context, 3),
                        &num_dims));
          EXPECT_EQ(-1, num_dims);

          // 1 node for ADD and 1 node for the delegate kernel.
          EXPECT_EQ(2, TfLiteOpaqueContextGetNumNodes(opaque_context));

          TfLiteOpaqueContext* acquired_opaque_context;
          EXPECT_EQ(kTfLiteOk,
                    TfLiteOpaqueContextAcquireSubgraphContext(
                        opaque_context, 0, &acquired_opaque_context));
          EXPECT_EQ(opaque_context, acquired_opaque_context);
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueContextReleaseSubgraphContext(
                                   opaque_context, 0));

          //
          // Create and configure a dynamic tensor.
          //
          {
            TfLiteOpaqueTensorBuilder* builder =
                TfLiteOpaqueTensorBuilderCreate();
            TfLiteOpaqueTensorBuilderSetType(builder, kTfLiteUInt8);
            void* dummy_data = malloc(sizeof(uint8_t));
            TfLiteOpaqueTensorBuilderSetData(builder, dummy_data);
            TfLiteQuantizationParams new_quantization_params{};
            new_quantization_params.scale = 16.0 / 256;
            new_quantization_params.zero_point = 255;

            TfLiteQuantization new_quantization{};
            new_quantization.type = kTfLiteAffineQuantization;
            TfLiteAffineQuantization* affine_quant =
                (TfLiteAffineQuantization*)malloc(
                    sizeof(TfLiteAffineQuantization));
            affine_quant->scale = TfLiteFloatArrayCreate(1);
            affine_quant->zero_point = TfLiteIntArrayCreate(1);
            new_quantization.params = affine_quant;
            TfLiteOpaqueTensorBuilderSetQuantization(
                TfLiteOpaqueTensorBuilderSetQuantizationParams(
                    TfLiteOpaqueTensorBuilderSetAllocationType(builder,
                                                               kTfLiteDynamic),
                    new_quantization_params),
                new_quantization);

            int new_tensor_index = -1;
            EXPECT_EQ(kTfLiteOk,
                      TfLiteOpaqueContextAddTensor(acquired_opaque_context,
                                                   builder, &new_tensor_index));
            TfLiteOpaqueTensorBuilderDelete(builder);

            TfLiteOpaqueTensor* new_tensor = TfLiteOpaqueContextGetOpaqueTensor(
                opaque_context, new_tensor_index);
            EXPECT_NE(new_tensor, nullptr);
            EXPECT_EQ(dummy_data, TfLiteOpaqueTensorData(new_tensor));
            EXPECT_EQ(kTfLiteUInt8, TfLiteOpaqueTensorType(new_tensor));
            EXPECT_EQ(new_quantization.type,
                      TfLiteOpaqueTensorGetQuantization(new_tensor).type);
            EXPECT_NEAR(
                new_quantization_params.scale,
                TfLiteOpaqueTensorGetQuantizationParams(new_tensor).scale,
                0.0001);
            EXPECT_EQ(
                new_quantization_params.zero_point,
                TfLiteOpaqueTensorGetQuantizationParams(new_tensor).zero_point);
          }
          //
          // Create and configure a 'kTfLiteArenaRw' tensor
          //
          {
            TfLiteOpaqueTensorBuilder* builder =
                TfLiteOpaqueTensorBuilderCreate();
            TfLiteOpaqueTensorBuilderSetType(builder, kTfLiteUInt8);
            TfLiteQuantizationParams new_quantization_params{};
            new_quantization_params.scale = 16.0 / 256;
            new_quantization_params.zero_point = 255;

            TfLiteQuantization new_quantization{};
            new_quantization.type = kTfLiteNoQuantization;
            TfLiteOpaqueTensorBuilderSetQuantization(
                TfLiteOpaqueTensorBuilderSetQuantizationParams(
                    TfLiteOpaqueTensorBuilderSetAllocationType(builder,
                                                               kTfLiteArenaRw),
                    new_quantization_params),
                new_quantization);

            int new_tensor_index = -1;
            EXPECT_EQ(kTfLiteOk,
                      TfLiteOpaqueContextAddTensor(acquired_opaque_context,
                                                   builder, &new_tensor_index));
            TfLiteOpaqueTensorBuilderDelete(builder);

            TfLiteOpaqueTensor* new_tensor = TfLiteOpaqueContextGetOpaqueTensor(
                opaque_context, new_tensor_index);
            EXPECT_NE(new_tensor, nullptr);
            EXPECT_EQ(kTfLiteUInt8, TfLiteOpaqueTensorType(new_tensor));
            EXPECT_EQ(new_quantization.type,
                      TfLiteOpaqueTensorGetQuantization(new_tensor).type);
            EXPECT_NEAR(
                new_quantization_params.scale,
                TfLiteOpaqueTensorGetQuantizationParams(new_tensor).scale,
                0.0001);
            EXPECT_EQ(
                new_quantization_params.zero_point,
                TfLiteOpaqueTensorGetQuantizationParams(new_tensor).zero_point);
            EXPECT_EQ(kTfLiteArenaRw,
                      TfLiteOpaqueTensorGetAllocationType(new_tensor));

            // Now switch the tensor's allocation type to dynamic after it has
            // been created.
            TfLiteOpaqueTensorSetAllocationTypeToDynamic(new_tensor);
            EXPECT_EQ(kTfLiteDynamic,
                      TfLiteOpaqueTensorGetAllocationType(new_tensor));
          }
          //
          // Create and configure a 'kTfLiteVariantObject' tensor, which will
          // result in an error.
          //
          {
            TfLiteOpaqueTensorBuilder* builder =
                TfLiteOpaqueTensorBuilderCreate();
            TfLiteOpaqueTensorBuilderSetAllocationType(builder,
                                                       kTfLiteVariantObject);
            int new_tensor_index = -1;
            EXPECT_EQ(kTfLiteError,
                      TfLiteOpaqueContextAddTensor(acquired_opaque_context,
                                                   builder, &new_tensor_index));
            TfLiteOpaqueTensorBuilderDelete(builder);
          }
          //
          // Create and configure a 'kTfLiteArenaRw' tensor and attempt to
          // provide 'data', which will result in an error because the TF Lite
          // runtime will provide the memory in this case.
          //
          {
            TfLiteOpaqueTensorBuilder* builder =
                TfLiteOpaqueTensorBuilderCreate();
            TfLiteOpaqueTensorBuilderSetType(builder, kTfLiteUInt8);
            static int dummy_data = 0;
            TfLiteOpaqueTensorBuilderSetData(builder, &dummy_data);
            TfLiteOpaqueTensorBuilderSetAllocationType(builder, kTfLiteArenaRw);
            int new_tensor_index = -1;
            EXPECT_EQ(kTfLiteError,
                      TfLiteOpaqueContextAddTensor(acquired_opaque_context,
                                                   builder, &new_tensor_index));
            TfLiteOpaqueTensorBuilderDelete(builder);
          }
          TfLiteOpaqueNode* node = nullptr;
          TfLiteOperator* registration_external = nullptr;
          TfLiteOpaqueContextGetNodeAndRegistration(
              opaque_context, params->nodes_to_replace->data[0], &node,
              &registration_external);
          // ADD is a builtin OP, not a custom OP.
          const char* kCustomName =
              TfLiteOperatorGetCustomName(registration_external);
          EXPECT_EQ(nullptr, kCustomName);

          const void* node_custom_init_data = nullptr;
          int node_custom_init_data_size = 0;
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeGetCustomInitialData(
                                   node, &node_custom_init_data,
                                   &node_custom_init_data_size));
          EXPECT_EQ(nullptr, node_custom_init_data);
          EXPECT_EQ(0, node_custom_init_data_size);

          void* builtin = TfLiteOpaqueNodeGetBuiltinData(node);
          TfLiteAddParams* add_params =
              reinterpret_cast<TfLiteAddParams*>(builtin);
          EXPECT_EQ(add_params->activation, kTfLiteActNone);

          const int* node_inputs = nullptr;
          int node_inputs_size = 0;
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeInputs(node, &node_inputs,
                                                      &node_inputs_size));
          EXPECT_EQ(2, node_inputs_size);
          EXPECT_EQ(0, node_inputs[0]);
          EXPECT_EQ(1, node_inputs[1]);

          const int* node_outputs = nullptr;
          int node_outputs_size = 0;
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeOutputs(node, &node_outputs,
                                                       &node_outputs_size));
          EXPECT_EQ(1, node_outputs_size);
          EXPECT_EQ(2, node_outputs[0]);

          const int* node_temporaries = nullptr;
          int node_temporaries_size = 0;
          EXPECT_EQ(kTfLiteOk,
                    TfLiteOpaqueNodeTemporaries(node, &node_temporaries,
                                                &node_temporaries_size));
          EXPECT_EQ(0, node_temporaries_size);

          const int* context_inputs = nullptr;
          int context_inputs_size = 0;
          EXPECT_EQ(kTfLiteOk,
                    TfLiteOpaqueContextGetInputs(
                        opaque_context, &context_inputs, &context_inputs_size));
          EXPECT_EQ(2, context_inputs_size);
          EXPECT_EQ(0, context_inputs[0]);
          EXPECT_EQ(1, context_inputs[1]);

          const int* context_outputs = nullptr;
          int context_outputs_size = 0;
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueContextGetOutputs(
                                   opaque_context, &context_outputs,
                                   &context_outputs_size));
          EXPECT_EQ(1, context_outputs_size);
          EXPECT_EQ(2, context_outputs[0]);

          const int* context_variables = nullptr;
          int context_variables_size = 0;
          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueContextGetVariables(
                                   opaque_context, &context_variables,
                                   &context_variables_size));
          EXPECT_EQ(1, context_variables_size);
          EXPECT_EQ(1, context_variables[0]);

          bool* delegate_kernel_invoked =
              static_cast<bool*>(params->delegate_data);
          *delegate_kernel_invoked = true;
          return nullptr;
        });

    TfLiteIntArray* execution_plan{};
    TfLiteOpaqueContextGetExecutionPlan(context, &execution_plan);
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        context, reg, execution_plan, delegate);
    EXPECT_EQ(kTfLiteOk, TfLiteOpaqueContextReleaseSubgraphContext(context, 0));
    return kTfLiteOk;
  };
  TfLiteDelegate my_delegate{};
  my_delegate.opaque_delegate_builder = &opaque_delegate_builder;

  EXPECT_EQ(kTfLiteOk, interpreter.ModifyGraphWithDelegate(&my_delegate));
  EXPECT_TRUE(delegate_kernel_invoked);
}

TEST(CApiSimple, OpaqueApiAccessorsStrings) {
  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(3);
  std::vector<int> dims = {1};
  TfLiteQuantizationParams quant{};
  interpreter.SetTensorParametersReadWrite(0, kTfLiteString, "a", dims, quant,
                                           /*is_variable=*/false);
  interpreter.SetTensorParametersReadWrite(1, kTfLiteString, "b", dims, quant,
                                           /*is_variable=*/false);
  interpreter.SetTensorParametersReadWrite(2, kTfLiteString, "c", dims, quant,
                                           /*is_variable=*/false);

  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({2});
  const char* initial_data = "";
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  const TfLiteRegistration* registration =
      resolver.FindOp(::tflite::BuiltinOperator_ADD, 1);
  interpreter.AddNodeWithParameters({0, 1}, {2}, initial_data, 0, builtin_data,
                                    registration);

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  bool delegate_kernel_invoked = false;
  opaque_delegate_builder.data = &delegate_kernel_invoked;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* context,
                                       TfLiteOpaqueDelegate* delegate,
                                       void* data) -> TfLiteStatus {
    TfLiteOperator* registration = TfLiteOperatorCreate(
        kTfLiteBuiltinDelegate, "my delegate", /*version=*/123,
        /*user_data=*/nullptr);
    TfLiteOperatorSetInitWithData(
        registration,
        [](void* user_data, TfLiteOpaqueContext* opaque_context,
           const char* buffer, size_t length) -> void* {
          const TfLiteOpaqueDelegateParams* params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
          EXPECT_EQ(2, params->input_tensors->size);
          TfLiteOpaqueTensor* opaque_input_tensor =
              TfLiteOpaqueContextGetOpaqueTensor(
                  opaque_context, params->input_tensors->data[0]);
          EXPECT_EQ(1, TfLiteOpaqueTensorNumDims(opaque_input_tensor));
          EXPECT_EQ(1, TfLiteOpaqueTensorDim(opaque_input_tensor, 0));
          EXPECT_EQ(kTfLiteDynamic,
                    TfLiteOpaqueTensorGetAllocationType(opaque_input_tensor));

          bool* delegate_kernel_invoked =
              static_cast<bool*>(params->delegate_data);
          *delegate_kernel_invoked = true;
          return nullptr;
        });

    TfLiteOperatorSetPrepareWithData(
        registration,
        [](void* user_data, TfLiteOpaqueContext* context,
           TfLiteOpaqueNode* node) -> TfLiteStatus { return kTfLiteOk; });

    TfLiteOperatorSetInvokeWithData(
        registration,
        [](void* user_data, TfLiteOpaqueContext* context,
           TfLiteOpaqueNode* node) -> TfLiteStatus {
          const TfLiteOpaqueTensor* input0 =
              TfLiteOpaqueNodeGetInput(context, node, 0);

          EXPECT_EQ(TfLiteOpaqueTensorGetStringCount(input0), 4);
          const char* input0_string2 = nullptr;
          int input0_string2_len = -1;
          EXPECT_EQ(kTfLiteOk,
                    TfLiteOpaqueTensorGetString(input0, 2, &input0_string2,
                                                &input0_string2_len));
          EXPECT_EQ(std::string(input0_string2, input0_string2_len), "F");
          EXPECT_EQ(1, input0_string2_len);

          const TfLiteOpaqueTensor* input1 =
              TfLiteOpaqueNodeGetInput(context, node, 1);
          EXPECT_EQ(TfLiteOpaqueTensorGetStringCount(input1), 1);
          const char* input1_string0 = nullptr;
          int input1_string0_len = -1;
          EXPECT_EQ(kTfLiteOk,
                    TfLiteOpaqueTensorGetString(input1, 0, &input1_string0,
                                                &input1_string0_len));
          EXPECT_EQ(std::string(input1_string0, input1_string0_len), "XYZ");
          EXPECT_EQ(3, input1_string0_len);

          TfLiteOpaqueTensor* opaque_output0 =
              TfLiteOpaqueNodeGetOutput(context, node, 0);

          //
          // First use 'TfLiteOpaqueTensorWriteString' to check that we can copy
          // a string from an input tensor to an output tensor.
          //
          EXPECT_EQ(kTfLiteOk,
                    TfLiteOpaqueTensorWriteString(
                        opaque_output0, input0_string2, input0_string2_len));
          const char* output_str_from_opaque_tensor = nullptr;
          int output_str_from_opaque_tensor_len = -1;
          EXPECT_EQ(kTfLiteOk,
                    TfLiteOpaqueTensorGetString(
                        opaque_output0, 0, &output_str_from_opaque_tensor,
                        &output_str_from_opaque_tensor_len));
          EXPECT_EQ(std::string(output_str_from_opaque_tensor,
                                output_str_from_opaque_tensor_len),
                    "F");
          EXPECT_EQ(1, output_str_from_opaque_tensor_len);

          //
          // Then perform the 'actual' ADD operation of adding the input tensor
          // string to the output tensor.
          //
          std::vector<const char*> str_array;
          std::vector<int> str_array_len;
          for (int i = 0; i < TfLiteOpaqueTensorGetStringCount(input0); ++i) {
            const char* input_string = nullptr;
            int input_string_len = -1;
            EXPECT_EQ(kTfLiteOk,
                      TfLiteOpaqueTensorGetString(input0, i, &input_string,
                                                  &input_string_len));
            str_array.push_back(input_string);
            str_array_len.push_back(input_string_len);
          }
          str_array.push_back(input1_string0);
          str_array_len.push_back(input1_string0_len);

          EXPECT_EQ(kTfLiteOk, TfLiteOpaqueTensorWriteStrings(
                                   opaque_output0, str_array.data(),
                                   str_array.size(), str_array_len.data()));
          return kTfLiteOk;
        });

    TfLiteIntArray* execution_plan{};
    TfLiteOpaqueContextGetExecutionPlan(context, &execution_plan);
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        context, registration, execution_plan, delegate);
    return kTfLiteOk;
  };

  TfLiteDelegate my_delegate{};
  my_delegate.opaque_delegate_builder = &opaque_delegate_builder;
  EXPECT_EQ(kTfLiteOk, interpreter.ModifyGraphWithDelegate(&my_delegate));
  EXPECT_TRUE(delegate_kernel_invoked);
  EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());

  //
  // Load input tensors with string data.
  //
  TfLiteTensor* t0 = interpreter.tensor(0);
  tflite::DynamicBuffer buf0;
  const char* raw_buf_with_embedded_null = "DDD\0EEE";
  const char* raw_buf_without_embedded_null = "12345678";
  std::vector<std::string> t0_strings{
      "ABC",
      std::string(raw_buf_with_embedded_null, raw_buf_with_embedded_null + 6),
      "F",
      std::string(raw_buf_without_embedded_null,
                  raw_buf_without_embedded_null + 4)};
  for (const std::string& s : t0_strings) {
    ASSERT_EQ(buf0.AddString(s.data(), s.size()), kTfLiteOk);
  }
  buf0.WriteToTensorAsVector(t0);

  TfLiteTensor* t1 = interpreter.tensor(1);
  char s1[] = "XYZ";
  tflite::DynamicBuffer buf1;
  ASSERT_EQ(buf1.AddString(s1, 3), kTfLiteOk);
  buf1.WriteToTensorAsVector(t1);

  //
  // Invoke the interpreter, so that the input tensor strings get copied to the
  // output tensor.
  //
  EXPECT_EQ(kTfLiteOk, interpreter.Invoke());

  //
  // Check that the output tensor stores the combination of the input strings.
  //
  const std::vector<std::string> expected_strings{
      "ABC",
      std::string(raw_buf_with_embedded_null, raw_buf_with_embedded_null + 6),
      "F", "1234", "XYZ"};
  TfLiteTensor* t2 = interpreter.tensor(2);
  EXPECT_EQ(tflite::GetStringCount(t2), expected_strings.size());
  for (int i = 0; i < tflite::GetStringCount(t2); ++i) {
    tflite::StringRef str_ref = tflite::GetString(t2, i);
    EXPECT_EQ(std::string(str_ref.str, str_ref.len), expected_strings[i]);
  }
}

void AddNode(
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates* resolver,
    ::tflite::Interpreter* interpreter) {
  const char* initial_data = "";
  TfLiteAddParams* builtin_data =
      reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
  builtin_data->activation = kTfLiteActNone;
  builtin_data->pot_scale_int16 = false;
  const TfLiteRegistration* reg =
      resolver->FindOp(::tflite::BuiltinOperator_ADD, 1);
  interpreter->AddNodeWithParameters({0, 1}, {2}, initial_data, 0, builtin_data,
                                     reg);
}

TEST(CApiSimple, AddNodesAfterApplyingDelegate) {
  // NOTE: This test does not illustrate intended usage by applications, and
  // clients should not mimic such a scenario in their code.  This is a
  // regression test for issue b/268405910.
  //
  // The test adds 'several' nodes after a delegate kernel has been applied
  // that visits the existing nodes and registrations.  The motivation behind
  // this scenario is that nodes (and registration) objects are stored within a
  // dynamic data structure and adding a significant amount of nodes has the
  // potential to exercise the scenario wherein internal re-allocations are
  // happening within that data structure.  For test coverage purposes this test
  // is valuable, though users should not write such code.

  ::tflite::Interpreter interpreter;
  interpreter.AddTensors(3);
  std::vector<int> dims = {1, 3};
  std::vector<int> dims_signature = {-1, 3};
  interpreter.SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "a", dims, TfLiteQuantizationParams{1.0, 0},
      /*is_variable=*/false, &dims_signature);
  interpreter.SetTensorParametersReadWrite(
      1, kTfLiteFloat32, "b", dims, TfLiteQuantizationParams{1.0, 0},
      /*is_variable=*/false, &dims_signature);
  interpreter.SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "c", dims, TfLiteQuantizationParams{1.0, 0},
      /*is_variable=*/false, &dims_signature);

  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({2});
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  AddNode(&resolver, &interpreter);

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  bool delegate_prepare_invoked = false;
  opaque_delegate_builder.data = &delegate_prepare_invoked;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* context,
                                       TfLiteOpaqueDelegate* delegate,
                                       void* data) -> TfLiteStatus {
    bool* delegate_prepare_invoked_pointer = static_cast<bool*>(data);
    *delegate_prepare_invoked_pointer = true;

    TfLiteIntArray* execution_plan;
    TfLiteOpaqueContextGetExecutionPlan(context, &execution_plan);
    EXPECT_EQ(execution_plan->size, 1);

    TfLiteOpaqueNode* node = nullptr;
    TfLiteOperator* registration_external = nullptr;
    TfLiteOpaqueContextGetNodeAndRegistration(context, execution_plan->data[0],
                                              &node, &registration_external);
    EXPECT_NE(node, nullptr);
    EXPECT_NE(registration_external, nullptr);
    return kTfLiteOk;
  };
  TfLiteDelegate my_delegate{};
  my_delegate.opaque_delegate_builder = &opaque_delegate_builder;

  EXPECT_EQ(kTfLiteOk, interpreter.ModifyGraphWithDelegate(&my_delegate));
  EXPECT_TRUE(delegate_prepare_invoked);

  for (int i = 0; i < 500; ++i) {
    AddNode(&resolver, &interpreter);
  }
}

}  // namespace
