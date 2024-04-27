/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/c/c_api_opaque.h"

#include <stddef.h>

#include <cstring>
#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api.h"

namespace tflite {
namespace {

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithMemNoneBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithMmapRoBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithArenaRwBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithDynamicBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithPersistentRoBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithCustomBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetAllocationStrategy,
     WithVariantObjectBehavesAsTfLiteTensorGetAllocationStrategy) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetAllocationStrategy(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetAllocationStrategy(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithMemNoneBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithMmapRoBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithArenaRwBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithDynamicBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithPersistentRoBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithCustomBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetBufferAddressStability,
     WithVariantObjectBehavesAsTfLiteTensorGetBufferAddressStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetBufferAddressStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetBufferAddressStability(&t));
}

TEST(TestTfLiteOpaqueTensorData, ValidInput) {
  TfLiteTensor t;
  char data[] = "data";
  t.data.raw = data;
  EXPECT_EQ(TfLiteOpaqueTensorData(reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            data);
}

TEST(TestTfLiteOpaqueTensorData, NullInput) {
  EXPECT_EQ(TfLiteOpaqueTensorData(nullptr), nullptr);
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithMemNoneBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithMmapRoBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithArenaRwBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithDynamicBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithPersistentRoBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithCustomBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataStability,
     WithVariantObjectBehavesAsTfLiteTensorGetDataStability) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataStability(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataStability(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithMemNoneBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithMmapRoBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithArenaRwBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithDynamicBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithPersistentRoBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithCustomBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetDataKnownStep,
     WithVariantObjectBehavesAsTfLiteTensorGetDataKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetDataKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetDataKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithMemNoneBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMemNone;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithMmapRoBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteMmapRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithArenaRwBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRw;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithArenaRwPersistentBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteArenaRwPersistent;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithDynamicBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteDynamic;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithPersistentRoBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLitePersistentRo;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithCustomBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteCustom;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueTensorGetShapeKnownStep,
     WithVariantObjectBehavesAsTfLiteTensorGetShapeKnownStep) {
  TfLiteTensor t;
  t.allocation_type = kTfLiteVariantObject;
  EXPECT_EQ(TfLiteOpaqueTensorGetShapeKnownStep(
                reinterpret_cast<TfLiteOpaqueTensor*>(&t)),
            TfLiteTensorGetShapeKnownStep(&t));
}

TEST(TestTfLiteOpaqueDelegate, CreateAndDelete) {
  std::unique_ptr<TfLiteOpaqueDelegateBuilder> opaque_delegate_builder(
      new TfLiteOpaqueDelegateBuilder{});

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(opaque_delegate_builder.get());

  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST(TestTfLiteOpaqueDelegate, Create_WithNull) {
  EXPECT_EQ(nullptr, TfLiteOpaqueDelegateCreate(nullptr));
}

TEST(TestTfLiteOpaqueDelegate, Delete_WithNull) {
  TfLiteOpaqueDelegateDelete(nullptr);
}

TEST(TestTfLiteOpaqueDelegate, GetData_WellFormedOpaqueDelegate) {
  int delegate_data = 42;
  TfLiteOpaqueDelegateBuilder builder{};
  builder.data = &delegate_data;

  TfLiteOpaqueDelegate* opaque_delegate = TfLiteOpaqueDelegateCreate(&builder);

  EXPECT_EQ(&delegate_data, TfLiteOpaqueDelegateGetData(opaque_delegate));

  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST(TestTfLiteOpaqueDelegate,
     GetData_NotConstructedWithTfLiteOpaqueDelegateCreate) {
  // Given a non-opaque delegate, that was created with 'TfLiteDelegateCreate'
  // and has its 'data_' field set manually.
  int delegate_data = 42;
  TfLiteDelegate non_opaque_delegate = TfLiteDelegateCreate();
  non_opaque_delegate.data_ = &delegate_data;
  // The following cast is safe only because this code is part of the
  // TF Lite test suite.  Apps using TF Lite should not rely on
  // 'TfLiteOpaqueDelegate' and 'TfLiteDelegate' being equivalent.
  auto* opaque_delegate =
      reinterpret_cast<TfLiteOpaqueDelegate*>(&non_opaque_delegate);

  // The accessor returns '&delegate_data', because the
  // 'opaque_delegate_builder' field inside the delegate was not set so it falls
  // back to returning the data_ field of TfLiteDelegate.
  EXPECT_EQ(&delegate_data, TfLiteOpaqueDelegateGetData(opaque_delegate));
}

TEST(TestTfLiteOpaqueDelegate, GetData_NoDataSetViaOpaqueDelegateBuilder) {
  TfLiteOpaqueDelegateBuilder builder{};
  TfLiteOpaqueDelegate* opaque_delegate = TfLiteOpaqueDelegateCreate(&builder);
  // The accessor returns 'nullptr', because the 'data' field inside the opaque
  // delegate builder was not set.
  EXPECT_EQ(nullptr, TfLiteOpaqueDelegateGetData(opaque_delegate));
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

// For testing TfLiteOpaqueNodeSetTemporaries, we define a custom op which uses
// TfLiteOpaqueNodeSetTemporaries in its Prepare method.
namespace my_custom_op {

struct MyOpData {
  int temp_tensor_index;
};

// Allocates MyOpData.
void* Init(TfLiteOpaqueContext* context, const char* buffer, size_t length) {
  auto* op_data = new MyOpData{};
  return op_data;
}

// Deallocates MyOpData.
void Free(TfLiteOpaqueContext* context, void* buffer) {
  delete reinterpret_cast<MyOpData*>(buffer);
}

// Allocates a temp tensor and stores it using TfLiteOpaqueTensorSetTemporaries.
// Also does some tests of TfLiteOpaqueTensorTemporaries
// and TfLiteOpaqueTensorSetTemporaries.
TfLiteStatus Prepare(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  auto* op_data =
      reinterpret_cast<MyOpData*>(TfLiteOpaqueNodeGetUserData(node));

  // Test #1: calling TfLiteOpaqueNodeSetTemporaries with a negative number
  // should result in a kTfLiteError status.
  const int num_temporaries = 1;
  int temporary_tensor_indices[num_temporaries];
  TfLiteStatus status =
      TfLiteOpaqueNodeSetTemporaries(node, temporary_tensor_indices,
                                     /*num_temporaries=*/-1);
  TF_LITE_OPAQUE_ENSURE(context, status == kTfLiteError);

  // Test #2: calling TfLiteOpaqueNodeSetTemporaries with zero
  // should result in a kTfLiteOk status.
  status = TfLiteOpaqueNodeSetTemporaries(node, temporary_tensor_indices,
                                          /*num_temporaries=*/0);
  TF_LITE_OPAQUE_ENSURE(context, status == kTfLiteOk);

  // Test #3: calling TfLiteOpaqueNodeSetTemporaries with a positive number.
  // For this test, the setup occurs here in Prepare, but the test assertion
  // is done in Invoke below.

  // Allocate a temp tensor index.
  TfLiteOpaqueTensorBuilder* builder = TfLiteOpaqueTensorBuilderCreate();
  TfLiteOpaqueTensorBuilderSetType(builder, kTfLiteFloat32);
  TfLiteOpaqueTensorBuilderSetAllocationType(builder, kTfLiteArenaRw);
  TfLiteOpaqueContextAddTensor(context, builder, &temporary_tensor_indices[0]);
  TfLiteOpaqueTensorBuilderDelete(builder);

  // Store the temp tensor index in the node temporaries and also in MyOpData
  // (so that we can verify the node's temporaries' contents later).
  status = TfLiteOpaqueNodeSetTemporaries(node, temporary_tensor_indices,
                                          num_temporaries);
  TF_LITE_OPAQUE_ENSURE(context, status == kTfLiteOk);
  op_data->temp_tensor_index = temporary_tensor_indices[0];

  // Allocate the temp tensor data.
  TfLiteOpaqueTensor* temp_tensor =
      TfLiteOpaqueContextGetOpaqueTensor(context, op_data->temp_tensor_index);
  TfLiteIntArray* temp_size = TfLiteIntArrayCreate(1);
  temp_size->data[0] = 1;
  return TfLiteOpaqueContextResizeTensor(context, temp_tensor, temp_size);
}

// Copies input tensor to output tensor via a previously allocated temp tensor.
// Also does some tests of TfLiteOpaqueTensorTemporaries /
// TfLiteOpaqueTensorSetTemporaries.
TfLiteStatus Invoke(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  auto* op_data =
      reinterpret_cast<MyOpData*>(TfLiteOpaqueNodeGetUserData(node));
  // Test that TfLiteOpaqueNodeTemporaries() returns the expected result.
  const int* temporary_tensor_indices;
  int num_temporaries;
  TfLiteOpaqueNodeTemporaries(node, &temporary_tensor_indices,
                              &num_temporaries);
  TF_LITE_OPAQUE_ENSURE(context, num_temporaries == 1);
  TF_LITE_OPAQUE_ENSURE(
      context, temporary_tensor_indices[0] == op_data->temp_tensor_index);

  // Get the temp tensor previously allocated by Prepare().
  TfLiteOpaqueTensor* temp_tensor =
      TfLiteOpaqueContextGetOpaqueTensor(context, op_data->temp_tensor_index);
  TF_LITE_OPAQUE_ENSURE(context,
                        TfLiteOpaqueTensorType(temp_tensor) == kTfLiteFloat32);
  TF_LITE_OPAQUE_ENSURE(context, TfLiteOpaqueTensorGetAllocationType(
                                     temp_tensor) == kTfLiteArenaRw);
  size_t temp_bytes = TfLiteOpaqueTensorByteSize(temp_tensor);
  void* temp_data = TfLiteOpaqueTensorData(temp_tensor);
  TF_LITE_OPAQUE_ENSURE(context, temp_bytes != 0);
  TF_LITE_OPAQUE_ENSURE(context, temp_data != nullptr);

  // Copy input tensor to temp tensor.
  EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfInputs(node));
  const TfLiteOpaqueTensor* input = TfLiteOpaqueNodeGetInput(context, node, 0);
  size_t input_bytes = TfLiteOpaqueTensorByteSize(input);
  void* input_data = TfLiteOpaqueTensorData(input);
  EXPECT_EQ(input_bytes, temp_bytes);
  std::memcpy(temp_data, input_data, input_bytes);

  // Copy temp tensor to output.
  EXPECT_EQ(1, TfLiteOpaqueNodeNumberOfOutputs(node));
  TfLiteOpaqueTensor* output = TfLiteOpaqueNodeGetOutput(context, node, 0);
  size_t output_bytes = TfLiteOpaqueTensorByteSize(output);
  void* output_data = TfLiteOpaqueTensorData(output);
  EXPECT_EQ(output_bytes, temp_bytes);
  std::memcpy(output_data, temp_data, output_bytes);

  return kTfLiteOk;
}

}  // namespace my_custom_op

TEST(TestTfLiteOpaqueNode, CustomOpWithSetAndGetTemporaries) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/custom_sinh.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOperator* reg = TfLiteOperatorCreate(kTfLiteBuiltinCustom, "Sinh", 1);
  TfLiteOperatorSetPrepare(reg, my_custom_op::Prepare);
  TfLiteOperatorSetInit(reg, my_custom_op::Init);
  TfLiteOperatorSetFree(reg, my_custom_op::Free);
  TfLiteOperatorSetInvoke(reg, my_custom_op::Invoke);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddOperator(options, reg);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  TfLiteInterpreterOptionsDelete(options);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  const float input_value = 42.0f;
  TfLiteTensorCopyFromBuffer(input_tensor, &input_value, sizeof(float));

  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  float output_value;
  TfLiteTensorCopyToBuffer(output_tensor, &output_value, sizeof(float));
  EXPECT_EQ(output_value, input_value);

  TfLiteInterpreterDelete(interpreter);
  TfLiteOperatorDelete(reg);
  TfLiteModelDelete(model);
}

}  // namespace
}  // namespace tflite
