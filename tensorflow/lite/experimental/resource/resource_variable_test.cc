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

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace resource {
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
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
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
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
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
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
  EXPECT_EQ(1.0f, value->data.f[0]);

  // Second AssignFrom but now tensor_b has same size as the variable.
  EXPECT_EQ(kTfLiteOk, var.AssignFrom(&tensor_b));
  EXPECT_TRUE(var.IsInitialized());
  value = var.GetTensor();
  // Variables are always dynamic type.
  EXPECT_EQ(kTfLiteDynamic, value->allocation_type);
  EXPECT_EQ(kTfLiteFloat32, value->type);
  EXPECT_EQ(sizeof(float), value->bytes);
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(1, value->dims->data[0]);
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
  EXPECT_EQ(1, value->dims->size);
  EXPECT_EQ(2, value->dims->data[0]);
  EXPECT_EQ(4.0f, value->data.f[0]);

  // Cleanup
  TfLiteTensorFree(&tensor_a);
  TfLiteTensorFree(&tensor_b);
}

}  // namespace resource
}  // namespace tflite
