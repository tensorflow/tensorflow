/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/tensor.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/cl_test.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

Status TensorGenericTest(const BHWC& shape, const TensorDescriptor& descriptor,
                         Environment* env) {
  TensorFloat32 tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    tensor_cpu.data[i] = half(0.3f * i);
  }
  TensorFloat32 tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0.0f;
  }

  Tensor tensor;
  RETURN_IF_ERROR(
      CreateTensor(env->context(), env->device(), shape, descriptor, &tensor));
  RETURN_IF_ERROR(tensor.WriteData(env->queue(), tensor_cpu));
  RETURN_IF_ERROR(tensor.ReadData(env->queue(), &tensor_gpu));

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return InternalError("Wrong value.");
    }
  }
  return OkStatus();
}

Status TensorTests(const TensorDescriptor& descriptor, Environment* env) {
  RETURN_IF_ERROR(TensorGenericTest(BHWC(1, 6, 7, 3), descriptor, env));
  RETURN_IF_ERROR(TensorGenericTest(BHWC(1, 1, 4, 12), descriptor, env));
  RETURN_IF_ERROR(TensorGenericTest(BHWC(1, 6, 1, 7), descriptor, env));

  // Batch tests
  RETURN_IF_ERROR(TensorGenericTest(BHWC(2, 6, 7, 3), descriptor, env));
  RETURN_IF_ERROR(TensorGenericTest(BHWC(4, 1, 4, 12), descriptor, env));
  RETURN_IF_ERROR(TensorGenericTest(BHWC(7, 6, 1, 7), descriptor, env));
  RETURN_IF_ERROR(TensorGenericTest(BHWC(13, 7, 3, 3), descriptor, env));
  return OkStatus();
}

TEST_F(OpenCLTest, BufferF32) {
  ASSERT_OK(TensorTests({DataType::FLOAT32, TensorStorageType::BUFFER}, &env_));
}

TEST_F(OpenCLTest, BufferF16) {
  ASSERT_OK(TensorTests({DataType::FLOAT16, TensorStorageType::BUFFER}, &env_));
}

TEST_F(OpenCLTest, Texture2DF32) {
  ASSERT_OK(
      TensorTests({DataType::FLOAT32, TensorStorageType::TEXTURE_2D}, &env_));
}

TEST_F(OpenCLTest, Texture2DF16) {
  ASSERT_OK(
      TensorTests({DataType::FLOAT16, TensorStorageType::TEXTURE_2D}, &env_));
}

TEST_F(OpenCLTest, TextureArrayF32) {
  ASSERT_OK(TensorTests({DataType::FLOAT32, TensorStorageType::TEXTURE_ARRAY},
                        &env_));
}

TEST_F(OpenCLTest, TextureArrayF16) {
  ASSERT_OK(TensorTests({DataType::FLOAT16, TensorStorageType::TEXTURE_ARRAY},
                        &env_));
}

TEST_F(OpenCLTest, ImageBufferF32) {
  ASSERT_OK(
      TensorTests({DataType::FLOAT32, TensorStorageType::IMAGE_BUFFER}, &env_));
}

TEST_F(OpenCLTest, ImageBufferF16) {
  ASSERT_OK(
      TensorTests({DataType::FLOAT16, TensorStorageType::IMAGE_BUFFER}, &env_));
}

TEST_F(OpenCLTest, SingleTextureF32) {
  ASSERT_OK(TensorGenericTest(
      BHWC(1, 6, 14, 1),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D}, &env_));
  ASSERT_OK(TensorGenericTest(
      BHWC(1, 6, 14, 2),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D}, &env_));

  // Batch tests
  ASSERT_OK(TensorGenericTest(
      BHWC(7, 6, 14, 1),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D}, &env_));
  ASSERT_OK(TensorGenericTest(
      BHWC(3, 6, 14, 2),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D}, &env_));
}

TEST_F(OpenCLTest, SingleTextureF16) {
  ASSERT_OK(TensorGenericTest(
      BHWC(1, 6, 3, 1),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D}, &env_));
  ASSERT_OK(TensorGenericTest(
      BHWC(1, 6, 3, 2),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D}, &env_));

  // Batch tests
  ASSERT_OK(TensorGenericTest(
      BHWC(7, 6, 3, 1),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D}, &env_));
  ASSERT_OK(TensorGenericTest(
      BHWC(3, 6, 3, 2),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D}, &env_));
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
