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

#include <cmath>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

template <DataType T>
absl::Status TensorBHWCTest(const BHWC& shape,
                            const TensorDescriptor& descriptor,
                            Environment* env) {
  tflite::gpu::Tensor<BHWC, T> tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    // val = [0, 1];
    const double val = static_cast<double>(i) /
                       static_cast<double>(tensor_cpu.data.size() - 1);
    double transformed_val = sin(val * 2.0 * M_PI) * 256.0;
    if (descriptor.data_type == DataType::INT16 ||
        descriptor.data_type == DataType::UINT16) {
      transformed_val *= 256.0;
    }
    if (descriptor.data_type == DataType::INT32 ||
        descriptor.data_type == DataType::UINT32) {
      transformed_val *= 256.0 * 256.0 * 256.0 * 256.0;
    }
    if (descriptor.data_type == DataType::FLOAT16) {
      transformed_val = half(transformed_val);
    }
    tensor_cpu.data[i] = transformed_val;
  }
  tflite::gpu::Tensor<BHWC, T> tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0;
  }

  Tensor tensor;
  RETURN_IF_ERROR(CreateTensor(env->context(), shape, descriptor, &tensor));
  RETURN_IF_ERROR(tensor.WriteData(env->queue(), tensor_cpu));
  RETURN_IF_ERROR(tensor.ReadData(env->queue(), &tensor_gpu));

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return absl::InternalError("Wrong value.");
    }
  }
  return absl::OkStatus();
}

template absl::Status TensorBHWCTest<DataType::FLOAT32>(
    const BHWC& shape, const TensorDescriptor& descriptor, Environment* env);
template absl::Status TensorBHWCTest<DataType::INT32>(
    const BHWC& shape, const TensorDescriptor& descriptor, Environment* env);

template absl::Status TensorBHWCTest<DataType::INT16>(
    const BHWC& shape, const TensorDescriptor& descriptor, Environment* env);

template absl::Status TensorBHWCTest<DataType::INT8>(
    const BHWC& shape, const TensorDescriptor& descriptor, Environment* env);
template absl::Status TensorBHWCTest<DataType::UINT32>(
    const BHWC& shape, const TensorDescriptor& descriptor, Environment* env);

template absl::Status TensorBHWCTest<DataType::UINT16>(
    const BHWC& shape, const TensorDescriptor& descriptor, Environment* env);

template absl::Status TensorBHWCTest<DataType::UINT8>(
    const BHWC& shape, const TensorDescriptor& descriptor, Environment* env);

template <DataType T>
absl::Status TensorBHWDCTest(const BHWDC& shape,
                             const TensorDescriptor& descriptor,
                             Environment* env) {
  tflite::gpu::Tensor<BHWDC, T> tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    // val = [0, 1];
    const double val = static_cast<double>(i) /
                       static_cast<double>(tensor_cpu.data.size() - 1);
    double transformed_val = sin(val * 2.0 * M_PI) * 256.0;
    if (descriptor.data_type == DataType::INT16 ||
        descriptor.data_type == DataType::UINT16) {
      transformed_val *= 256.0;
    }
    if (descriptor.data_type == DataType::INT32 ||
        descriptor.data_type == DataType::UINT32) {
      transformed_val *= 256.0 * 256.0 * 256.0 * 256.0;
    }
    if (descriptor.data_type == DataType::FLOAT16) {
      transformed_val = half(transformed_val);
    }
    tensor_cpu.data[i] = transformed_val;
  }
  tflite::gpu::Tensor<BHWDC, T> tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0;
  }

  Tensor tensor;
  RETURN_IF_ERROR(CreateTensor(env->context(), shape, descriptor, &tensor));
  RETURN_IF_ERROR(tensor.WriteData(env->queue(), tensor_cpu));
  RETURN_IF_ERROR(tensor.ReadData(env->queue(), &tensor_gpu));

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return absl::InternalError("Wrong value.");
    }
  }
  return absl::OkStatus();
}

template absl::Status TensorBHWDCTest<DataType::FLOAT32>(
    const BHWDC& shape, const TensorDescriptor& descriptor, Environment* env);
template absl::Status TensorBHWDCTest<DataType::INT32>(
    const BHWDC& shape, const TensorDescriptor& descriptor, Environment* env);

template absl::Status TensorBHWDCTest<DataType::INT16>(
    const BHWDC& shape, const TensorDescriptor& descriptor, Environment* env);

template absl::Status TensorBHWDCTest<DataType::INT8>(
    const BHWDC& shape, const TensorDescriptor& descriptor, Environment* env);
template absl::Status TensorBHWDCTest<DataType::UINT32>(
    const BHWDC& shape, const TensorDescriptor& descriptor, Environment* env);

template absl::Status TensorBHWDCTest<DataType::UINT16>(
    const BHWDC& shape, const TensorDescriptor& descriptor, Environment* env);

template absl::Status TensorBHWDCTest<DataType::UINT8>(
    const BHWDC& shape, const TensorDescriptor& descriptor, Environment* env);

template <DataType T>
absl::Status TensorTests(DataType data_type, TensorStorageType storage_type,
                         Environment* env) {
  RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(1, 6, 7, 3), {data_type, storage_type, Layout::HWC}, env));
  RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(1, 1, 4, 12), {data_type, storage_type, Layout::HWC}, env));
  RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(1, 6, 1, 7), {data_type, storage_type, Layout::HWC}, env));

  // Batch tests
  RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(2, 6, 7, 3), {data_type, storage_type, Layout::BHWC}, env));
  RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(4, 1, 4, 12), {data_type, storage_type, Layout::BHWC}, env));
  RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(7, 6, 1, 7), {data_type, storage_type, Layout::BHWC}, env));
  RETURN_IF_ERROR(TensorBHWCTest<T>(
      BHWC(13, 7, 3, 3), {data_type, storage_type, Layout::BHWC}, env));

  // 5D tests with batch = 1
  RETURN_IF_ERROR(TensorBHWDCTest<T>(
      BHWDC(1, 6, 7, 4, 3), {data_type, storage_type, Layout::HWDC}, env));
  RETURN_IF_ERROR(TensorBHWDCTest<T>(
      BHWDC(1, 1, 4, 3, 12), {data_type, storage_type, Layout::HWDC}, env));
  RETURN_IF_ERROR(TensorBHWDCTest<T>(
      BHWDC(1, 6, 1, 7, 7), {data_type, storage_type, Layout::HWDC}, env));

  // 5D tests
  RETURN_IF_ERROR(TensorBHWDCTest<T>(
      BHWDC(2, 6, 7, 1, 3), {data_type, storage_type, Layout::BHWDC}, env));
  RETURN_IF_ERROR(TensorBHWDCTest<T>(
      BHWDC(4, 1, 4, 2, 12), {data_type, storage_type, Layout::BHWDC}, env));
  RETURN_IF_ERROR(TensorBHWDCTest<T>(
      BHWDC(7, 6, 1, 3, 7), {data_type, storage_type, Layout::BHWDC}, env));
  RETURN_IF_ERROR(TensorBHWDCTest<T>(
      BHWDC(13, 7, 3, 4, 3), {data_type, storage_type, Layout::BHWDC}, env));
  return absl::OkStatus();
}

template absl::Status TensorTests<DataType::FLOAT32>(
    DataType data_type, TensorStorageType storage_type, Environment* env);
template absl::Status TensorTests<DataType::INT32>(
    DataType data_type, TensorStorageType storage_type, Environment* env);
template absl::Status TensorTests<DataType::INT16>(
    DataType data_type, TensorStorageType storage_type, Environment* env);
template absl::Status TensorTests<DataType::INT8>(
    DataType data_type, TensorStorageType storage_type, Environment* env);
template absl::Status TensorTests<DataType::UINT32>(
    DataType data_type, TensorStorageType storage_type, Environment* env);
template absl::Status TensorTests<DataType::UINT16>(
    DataType data_type, TensorStorageType storage_type, Environment* env);
template absl::Status TensorTests<DataType::UINT8>(
    DataType data_type, TensorStorageType storage_type, Environment* env);

TEST_F(OpenCLTest, BufferF32) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(DataType::FLOAT32,
                                           TensorStorageType::BUFFER, &env_));
}

TEST_F(OpenCLTest, BufferF16) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(DataType::FLOAT16,
                                           TensorStorageType::BUFFER, &env_));
}

TEST_F(OpenCLTest, BufferInt32) {
  ASSERT_OK(TensorTests<DataType::INT32>(DataType::INT32,
                                         TensorStorageType::BUFFER, &env_));
}

TEST_F(OpenCLTest, BufferInt16) {
  ASSERT_OK(TensorTests<DataType::INT16>(DataType::INT16,
                                         TensorStorageType::BUFFER, &env_));
}

TEST_F(OpenCLTest, BufferInt8) {
  ASSERT_OK(TensorTests<DataType::INT8>(DataType::INT8,
                                        TensorStorageType::BUFFER, &env_));
}

TEST_F(OpenCLTest, BufferUint32) {
  ASSERT_OK(TensorTests<DataType::UINT32>(DataType::UINT32,
                                          TensorStorageType::BUFFER, &env_));
}

TEST_F(OpenCLTest, BufferUint16) {
  ASSERT_OK(TensorTests<DataType::UINT16>(DataType::UINT16,
                                          TensorStorageType::BUFFER, &env_));
}

TEST_F(OpenCLTest, BufferUint8) {
  ASSERT_OK(TensorTests<DataType::UINT8>(DataType::UINT8,
                                         TensorStorageType::BUFFER, &env_));
}

TEST_F(OpenCLTest, Texture2DF32) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(
      DataType::FLOAT32, TensorStorageType::TEXTURE_2D, &env_));
}

TEST_F(OpenCLTest, Texture2DF16) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(
      DataType::FLOAT16, TensorStorageType::TEXTURE_2D, &env_));
}

TEST_F(OpenCLTest, Texture2DInt32) {
  ASSERT_OK(TensorTests<DataType::INT32>(DataType::INT32,
                                         TensorStorageType::TEXTURE_2D, &env_));
}

TEST_F(OpenCLTest, Texture2DInt16) {
  ASSERT_OK(TensorTests<DataType::INT16>(DataType::INT16,
                                         TensorStorageType::TEXTURE_2D, &env_));
}

TEST_F(OpenCLTest, Texture2DInt8) {
  ASSERT_OK(TensorTests<DataType::INT8>(DataType::INT8,
                                        TensorStorageType::TEXTURE_2D, &env_));
}

TEST_F(OpenCLTest, Texture2DUint32) {
  ASSERT_OK(TensorTests<DataType::UINT32>(
      DataType::UINT32, TensorStorageType::TEXTURE_2D, &env_));
}

TEST_F(OpenCLTest, Texture2DUint16) {
  ASSERT_OK(TensorTests<DataType::UINT16>(
      DataType::UINT16, TensorStorageType::TEXTURE_2D, &env_));
}

TEST_F(OpenCLTest, Texture2DUint8) {
  ASSERT_OK(TensorTests<DataType::UINT8>(DataType::UINT8,
                                         TensorStorageType::TEXTURE_2D, &env_));
}

TEST_F(OpenCLTest, Texture3DF32) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(
      DataType::FLOAT32, TensorStorageType::TEXTURE_3D, &env_));
}

TEST_F(OpenCLTest, Texture3DF16) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(
      DataType::FLOAT16, TensorStorageType::TEXTURE_3D, &env_));
}

TEST_F(OpenCLTest, Texture3DInt32) {
  ASSERT_OK(TensorTests<DataType::INT32>(DataType::INT32,
                                         TensorStorageType::TEXTURE_3D, &env_));
}

TEST_F(OpenCLTest, Texture3DInt16) {
  ASSERT_OK(TensorTests<DataType::INT16>(DataType::INT16,
                                         TensorStorageType::TEXTURE_3D, &env_));
}

TEST_F(OpenCLTest, Texture3DInt8) {
  ASSERT_OK(TensorTests<DataType::INT8>(DataType::INT8,
                                        TensorStorageType::TEXTURE_3D, &env_));
}

TEST_F(OpenCLTest, Texture3DUint32) {
  ASSERT_OK(TensorTests<DataType::UINT32>(
      DataType::UINT32, TensorStorageType::TEXTURE_3D, &env_));
}

TEST_F(OpenCLTest, Texture3DUint16) {
  ASSERT_OK(TensorTests<DataType::UINT16>(
      DataType::UINT16, TensorStorageType::TEXTURE_3D, &env_));
}

TEST_F(OpenCLTest, Texture3DUint8) {
  ASSERT_OK(TensorTests<DataType::UINT8>(DataType::UINT8,
                                         TensorStorageType::TEXTURE_3D, &env_));
}

TEST_F(OpenCLTest, TextureArrayF32) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(
      DataType::FLOAT32, TensorStorageType::TEXTURE_ARRAY, &env_));
}

TEST_F(OpenCLTest, TextureArrayF16) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(
      DataType::FLOAT16, TensorStorageType::TEXTURE_ARRAY, &env_));
}

TEST_F(OpenCLTest, TextureArrayInt32) {
  ASSERT_OK(TensorTests<DataType::INT32>(
      DataType::INT32, TensorStorageType::TEXTURE_ARRAY, &env_));
}

TEST_F(OpenCLTest, TextureArrayInt16) {
  ASSERT_OK(TensorTests<DataType::INT16>(
      DataType::INT16, TensorStorageType::TEXTURE_ARRAY, &env_));
}

TEST_F(OpenCLTest, TextureArrayInt8) {
  ASSERT_OK(TensorTests<DataType::INT8>(
      DataType::INT8, TensorStorageType::TEXTURE_ARRAY, &env_));
}

TEST_F(OpenCLTest, TextureArrayUint32) {
  ASSERT_OK(TensorTests<DataType::UINT32>(
      DataType::UINT32, TensorStorageType::TEXTURE_ARRAY, &env_));
}

TEST_F(OpenCLTest, TextureArrayUint16) {
  ASSERT_OK(TensorTests<DataType::UINT16>(
      DataType::UINT16, TensorStorageType::TEXTURE_ARRAY, &env_));
}

TEST_F(OpenCLTest, TextureArrayUint8) {
  ASSERT_OK(TensorTests<DataType::UINT8>(
      DataType::UINT8, TensorStorageType::TEXTURE_ARRAY, &env_));
}

TEST_F(OpenCLTest, ImageBufferF32) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(
      DataType::FLOAT32, TensorStorageType::IMAGE_BUFFER, &env_));
}

TEST_F(OpenCLTest, ImageBufferF16) {
  ASSERT_OK(TensorTests<DataType::FLOAT32>(
      DataType::FLOAT16, TensorStorageType::IMAGE_BUFFER, &env_));
}

TEST_F(OpenCLTest, ImageBufferInt32) {
  ASSERT_OK(TensorTests<DataType::INT32>(
      DataType::INT32, TensorStorageType::IMAGE_BUFFER, &env_));
}

TEST_F(OpenCLTest, ImageBufferInt16) {
  ASSERT_OK(TensorTests<DataType::INT16>(
      DataType::INT16, TensorStorageType::IMAGE_BUFFER, &env_));
}

TEST_F(OpenCLTest, ImageBufferInt8) {
  ASSERT_OK(TensorTests<DataType::INT8>(
      DataType::INT8, TensorStorageType::IMAGE_BUFFER, &env_));
}

TEST_F(OpenCLTest, ImageBufferUint32) {
  ASSERT_OK(TensorTests<DataType::UINT32>(
      DataType::UINT32, TensorStorageType::IMAGE_BUFFER, &env_));
}

TEST_F(OpenCLTest, ImageBufferUint16) {
  ASSERT_OK(TensorTests<DataType::UINT16>(
      DataType::UINT16, TensorStorageType::IMAGE_BUFFER, &env_));
}

TEST_F(OpenCLTest, ImageBufferUint8) {
  ASSERT_OK(TensorTests<DataType::UINT8>(
      DataType::UINT8, TensorStorageType::IMAGE_BUFFER, &env_));
}

TEST_F(OpenCLTest, SingleTextureF32) {
  ASSERT_OK(TensorBHWCTest<DataType::FLOAT32>(
      BHWC(1, 6, 14, 1),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC},
      &env_));
  ASSERT_OK(TensorBHWCTest<DataType::FLOAT32>(
      BHWC(1, 6, 14, 2),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC},
      &env_));

  // Batch tests
  ASSERT_OK(TensorBHWCTest<DataType::FLOAT32>(
      BHWC(7, 6, 14, 1),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D, Layout::BHWC},
      &env_));
  ASSERT_OK(TensorBHWCTest<DataType::FLOAT32>(
      BHWC(3, 6, 14, 2),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D, Layout::BHWC},
      &env_));

  // 5D tests with batch = 1
  ASSERT_OK(TensorBHWDCTest<DataType::FLOAT32>(
      BHWDC(1, 6, 14, 7, 1),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWDC},
      &env_));
  ASSERT_OK(TensorBHWDCTest<DataType::FLOAT32>(
      BHWDC(1, 6, 14, 4, 2),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWDC},
      &env_));

  // 5D tests
  ASSERT_OK(TensorBHWDCTest<DataType::FLOAT32>(
      BHWDC(7, 6, 14, 5, 1),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D, Layout::BHWDC},
      &env_));
  ASSERT_OK(TensorBHWDCTest<DataType::FLOAT32>(
      BHWDC(3, 6, 14, 3, 2),
      {DataType::FLOAT32, TensorStorageType::SINGLE_TEXTURE_2D, Layout::BHWDC},
      &env_));
}

TEST_F(OpenCLTest, SingleTextureF16) {
  ASSERT_OK(TensorBHWCTest<DataType::FLOAT32>(
      BHWC(1, 6, 3, 1),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC},
      &env_));
  ASSERT_OK(TensorBHWCTest<DataType::FLOAT32>(
      BHWC(1, 6, 3, 2),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWC},
      &env_));

  // Batch tests
  ASSERT_OK(TensorBHWCTest<DataType::FLOAT32>(
      BHWC(7, 6, 3, 1),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::BHWC},
      &env_));
  ASSERT_OK(TensorBHWCTest<DataType::FLOAT32>(
      BHWC(3, 6, 3, 2),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::BHWC},
      &env_));

  // 5D tests with batch = 1
  ASSERT_OK(TensorBHWDCTest<DataType::FLOAT32>(
      BHWDC(1, 6, 14, 7, 1),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWDC},
      &env_));
  ASSERT_OK(TensorBHWDCTest<DataType::FLOAT32>(
      BHWDC(1, 6, 14, 4, 2),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::HWDC},
      &env_));

  // 5D tests
  ASSERT_OK(TensorBHWDCTest<DataType::FLOAT32>(
      BHWDC(7, 6, 14, 5, 1),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::BHWDC},
      &env_));
  ASSERT_OK(TensorBHWDCTest<DataType::FLOAT32>(
      BHWDC(3, 6, 14, 3, 2),
      {DataType::FLOAT16, TensorStorageType::SINGLE_TEXTURE_2D, Layout::BHWDC},
      &env_));
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
