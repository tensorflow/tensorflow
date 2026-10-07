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
#include "tensorflow/lite/experimental/resource/resource_variable.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/resource/mock_resource.h"
#include "tensorflow/lite/experimental/resource/resource_base.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace resource {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;

// Helper util that initialize 'tensor'.
void InitTensor(const std::vector<int>& shape, TfLiteAllocationType alloc_type,
                float default_value, TfLiteTensor* tensor) {
  memset(tensor, 0, sizeof(TfLiteTensor));
  int num_elements = 1;
  for (auto dim : shape) num_elements *= dim;
  if (shape.empty()) num_elements = 0;
  float* buf = static_cast<float*>(malloc(sizeof(float) * num_elements));
  for (int i = 0; i < num_elements; ++i) buf[i] = default_value;
  const int bytes = num_elements * sizeof(buf[0]);
  auto* dims = ConvertArrayToTfLiteIntArray(shape.size(), shape.data());
  TfLiteTensorReset(TfLiteType::kTfLiteFloat32, nullptr, dims, {},
                    reinterpret_cast<char*>(buf), bytes, alloc_type, nullptr,
                    false, tensor);
}

TEST(ResourceTest, NonDynamicTensorAssign) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  TfLiteTensor tensor;
  std::vector<int> shape = {1};
  InitTensor(shape, kTfLiteArenaRw, 1.0f, &tensor);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();

  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  ASSERT_THAT(value, DimsAre({1}));
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Cleanup
  // For non dynamic tensors we need to delete the buffers manually.
  free(tensor.data.raw);
  TfLiteTensorFree(&tensor);
}

TEST(ResourceTest, DynamicTensorAssign) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  TfLiteTensor tensor;
  std::vector<int> shape = {1};
  InitTensor(shape, kTfLiteDynamic, 1.0f, &tensor);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();

  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  ASSERT_THAT(value, DimsAre({1}));
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Cleanup
  TfLiteTensorFree(&tensor);
}

TEST(ResourceTest, AssignSameSizeTensor) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  // We create 2 tensors and make 2 calls for Assign.
  // The second Assign call should trigger the case of assign with same size.
  TfLiteTensor tensor_a, tensor_b;
  std::vector<int> shape_a = {1};
  std::vector<int> shape_b = {1};
  InitTensor(shape_a, kTfLiteDynamic, 1.0, &tensor_a);
  InitTensor(shape_b, kTfLiteDynamic, 4.0, &tensor_b);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_a));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  ASSERT_THAT(value, DimsAre({1}));
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Second AssignFrom but now tensor_b has same size as the variable.
  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_b));
  EXPECT_TRUE(var.IsInitialized());
  value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  ASSERT_THAT(value, DimsAre({1}));
  EXPECT_EQ(4.0f, value->data.f[0]);

  // Cleanup
  TfLiteTensorFree(&tensor_a);
  TfLiteTensorFree(&tensor_b);
}

TEST(ResourceTest, AssignDifferentSizeTensor) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  // We create 2 tensors and make 2 calls for Assign.
  // The second Assign call should trigger the case of assign with different
  // size.
  TfLiteTensor tensor_a, tensor_b;
  std::vector<int> shape_a = {1};
  std::vector<int> shape_b = {2};
  InitTensor(shape_a, kTfLiteDynamic, 1.0, &tensor_a);
  InitTensor(shape_b, kTfLiteDynamic, 4.0, &tensor_b);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_a));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Second AssignFrom but now tensor_b has different size from the variable.
  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_b));
  EXPECT_TRUE(var.IsInitialized());
  value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float) * 2, value->bytes);
  ASSERT_THAT(value, DimsAre({2}));
  EXPECT_EQ(4.0f, value->data.f[0]);

  // Cleanup
  TfLiteTensorFree(&tensor_a);
  TfLiteTensorFree(&tensor_b);
}

TEST(IsBuiltinResource, IsBuiltinResourceTest) {
  TfLiteTensor tensor;
  tensor.type = kTfLiteResource;
  tensor.delegate = nullptr;
  // Resource type and not delegate output.
  EXPECT_TRUE(IsBuiltinResource(&tensor));

  // Not valid tensor.
  EXPECT_FALSE(IsBuiltinResource(nullptr));

  // Not a resource type.
  tensor.type = kTfLiteFloat32;
  EXPECT_FALSE(IsBuiltinResource(&tensor));

  // Resource but coming from a delegate.
  tensor.type = kTfLiteResource;
  TfLiteDelegate delegate;
  tensor.delegate = &delegate;
  EXPECT_FALSE(IsBuiltinResource(&tensor));
}

TEST(ResourceTest, GetMemoryUsage) {
  ResourceVariable var;
  EXPECT_FALSE(var.IsInitialized());

  TfLiteTensor tensor;
  std::vector<int> shape = {100};
  InitTensor(shape, kTfLiteArenaRw, 1.0f, &tensor);

  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor));
  EXPECT_TRUE(var.IsInitialized());
  auto* value = var.GetTensor();

  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(100 * sizeof(float), value->bytes);
  ASSERT_THAT(value, DimsAre({100}));
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Check memory usage
  EXPECT_EQ(100 * sizeof(float), var.GetMemoryUsage());

  // Cleanup
  // For non dynamic tensors we need to delete the buffers manually.
  free(tensor.data.raw);
  TfLiteTensorFree(&tensor);
}

TEST(ResourceTest, CreateResourceVariableWhenNotAvailableCreatesNew) {
  ResourceMap resources;
  EXPECT_EQ(CreateResourceVariableIfNotAvailable(&resources, /*resource_id=*/1),
            kTfLiteOk);
  EXPECT_THAT(resources,
              ElementsAre(Pair(
                  1, Pointee(Property(
                         &ResourceBase::GetResourceType,
                         ResourceBase::ResourceType::kResourceVariable)))));
}

TEST(ResourceTest, CreateResourceVariableWhenMatchingTypeExistsSucceeds) {
  ResourceMap resources;
  EXPECT_EQ(CreateResourceVariableIfNotAvailable(&resources, /*resource_id=*/1),
            kTfLiteOk);
  EXPECT_EQ(CreateResourceVariableIfNotAvailable(&resources, /*resource_id=*/1),
            kTfLiteOk);
  ASSERT_EQ(resources.size(), 1);
}

TEST(ResourceTest, CreateResourceVariableWhenTypeMismatchesReturnsError) {
  ResourceMap resources;
  resources.emplace(1, std::make_unique<MockHashTableResource>());
  EXPECT_EQ(CreateResourceVariableIfNotAvailable(&resources, /*resource_id=*/1),
            kTfLiteError);
}

TEST(ResourceTest, CreateResourceVariableWhenEntryIsNullReturnsError) {
  ResourceMap resources;
  resources.emplace(1, nullptr);
  EXPECT_EQ(CreateResourceVariableIfNotAvailable(&resources, /*resource_id=*/1),
            kTfLiteError);
  EXPECT_EQ(GetResourceVariable(&resources, /*resource_id=*/1), nullptr);
}

TEST(ResourceTest, CreateResourceVariableWhenResourcesMapIsNullReturnsError) {
  EXPECT_EQ(CreateResourceVariableIfNotAvailable(nullptr, /*resource_id=*/1),
            kTfLiteError);
}

TEST(ResourceTest, GetResourceVariableWhenResourcesMapIsNullReturnsNull) {
  EXPECT_EQ(GetResourceVariable(nullptr, /*resource_id=*/1), nullptr);
}

TEST(ResourceTest, GetResourceVariableWhenNotFoundReturnsNull) {
  ResourceMap resources;
  EXPECT_EQ(GetResourceVariable(&resources, /*resource_id=*/1), nullptr);
}

TEST(ResourceTest, GetResourceVariableWhenTypeMismatchesReturnsNull) {
  ResourceMap resources;
  resources.emplace(1, std::make_unique<MockHashTableResource>());
  EXPECT_EQ(GetResourceVariable(&resources, /*resource_id=*/1), nullptr);
}

}  // namespace
}  // namespace resource
}  // namespace tflite
