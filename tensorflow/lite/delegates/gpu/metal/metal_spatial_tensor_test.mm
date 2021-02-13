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

#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

#include "tensorflow/lite/delegates/gpu/common/types.h"

#import <XCTest/XCTest.h>

#import <Metal/Metal.h>

@interface MetalSpatialTensorTest : XCTestCase
@end

@implementation MetalSpatialTensorTest
- (void)setUp {
  [super setUp];
}

using tflite::gpu::half;
using tflite::gpu::TensorDescriptor;
using tflite::gpu::TensorStorageType;
using tflite::gpu::DataType;
using tflite::gpu::BHWC;
using tflite::gpu::BHWDC;
using tflite::gpu::Layout;

static absl::Status TensorGenericTest(const BHWC& shape,
                               const TensorDescriptor& descriptor,
                               id<MTLDevice> device) {
  tflite::gpu::TensorFloat32 tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    tensor_cpu.data[i] = half(0.3f * i);
  }
  tflite::gpu::TensorFloat32 tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0.0f;
  }

  tflite::gpu::metal::MetalSpatialTensor tensor;
  RETURN_IF_ERROR(CreateTensor(device, shape, descriptor, &tensor));
  RETURN_IF_ERROR(tensor.WriteData(device, tensor_cpu));
  RETURN_IF_ERROR(tensor.ReadData(device, &tensor_gpu));

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return absl::InternalError("Wrong value at index - " + std::to_string(i) + ". GPU - " +
                                 std::to_string(tensor_gpu.data[i]) + ", CPU - " +
                                 std::to_string(tensor_cpu.data[i]));
    }
  }
  return absl::OkStatus();
}

static absl::Status Tensor5DGenericTest(const BHWDC& shape,
                                 const TensorDescriptor& descriptor,
                                 id<MTLDevice> device) {
  tflite::gpu::Tensor5DFloat32 tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    tensor_cpu.data[i] = half(0.3f * i);
  }
  tflite::gpu::Tensor5DFloat32 tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0.0f;
  }

  tflite::gpu::metal::MetalSpatialTensor tensor;
  RETURN_IF_ERROR(CreateTensor(device, shape, descriptor, &tensor));
  RETURN_IF_ERROR(tensor.WriteData(device, tensor_cpu));
  RETURN_IF_ERROR(tensor.ReadData(device, &tensor_gpu));

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return absl::InternalError("Wrong value.");
    }
  }
  return absl::OkStatus();
}

static absl::Status TensorTests(DataType data_type, TensorStorageType storage_type,
                         id<MTLDevice> device) {
  RETURN_IF_ERROR(TensorGenericTest(
      BHWC(1, 6, 7, 3), {data_type, storage_type, Layout::HWC}, device));
  RETURN_IF_ERROR(TensorGenericTest(
      BHWC(1, 1, 4, 12), {data_type, storage_type, Layout::HWC}, device));
  RETURN_IF_ERROR(TensorGenericTest(
      BHWC(1, 6, 1, 7), {data_type, storage_type, Layout::HWC}, device));

  // Batch tests
  RETURN_IF_ERROR(TensorGenericTest(
      BHWC(2, 6, 7, 3), {data_type, storage_type, Layout::BHWC}, device));
  RETURN_IF_ERROR(TensorGenericTest(
      BHWC(4, 1, 4, 12), {data_type, storage_type, Layout::BHWC}, device));
  RETURN_IF_ERROR(TensorGenericTest(
      BHWC(7, 6, 1, 7), {data_type, storage_type, Layout::BHWC}, device));
  RETURN_IF_ERROR(TensorGenericTest(
      BHWC(13, 7, 3, 3), {data_type, storage_type, Layout::BHWC}, device));

  // 5D tests with batch = 1
  RETURN_IF_ERROR(Tensor5DGenericTest(
      BHWDC(1, 6, 7, 4, 3), {data_type, storage_type, Layout::HWDC}, device));
  RETURN_IF_ERROR(Tensor5DGenericTest(
      BHWDC(1, 1, 4, 3, 12), {data_type, storage_type, Layout::HWDC}, device));
  RETURN_IF_ERROR(Tensor5DGenericTest(
      BHWDC(1, 6, 1, 7, 7), {data_type, storage_type, Layout::HWDC}, device));

  // 5D tests
  RETURN_IF_ERROR(Tensor5DGenericTest(
      BHWDC(2, 6, 7, 1, 3), {data_type, storage_type, Layout::BHWDC}, device));
  RETURN_IF_ERROR(Tensor5DGenericTest(
      BHWDC(4, 1, 4, 2, 12), {data_type, storage_type, Layout::BHWDC}, device));
  RETURN_IF_ERROR(Tensor5DGenericTest(
      BHWDC(7, 6, 1, 3, 7), {data_type, storage_type, Layout::BHWDC}, device));
  RETURN_IF_ERROR(Tensor5DGenericTest(
      BHWDC(13, 7, 3, 4, 3), {data_type, storage_type, Layout::BHWDC}, device));
  return absl::OkStatus();
}

- (void)testBufferF32 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT32, TensorStorageType::BUFFER, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferF16 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT16, TensorStorageType::BUFFER, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DF32 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT32, TensorStorageType::TEXTURE_2D, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DF16 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT16, TensorStorageType::TEXTURE_2D, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DF32 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT32, TensorStorageType::TEXTURE_3D, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DF16 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT16, TensorStorageType::TEXTURE_3D, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayF32 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT32, TensorStorageType::TEXTURE_ARRAY, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayF16 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT16, TensorStorageType::TEXTURE_ARRAY, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferF32 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT32, TensorStorageType::IMAGE_BUFFER, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferF16 {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  auto status = TensorTests(DataType::FLOAT16, TensorStorageType::IMAGE_BUFFER, device);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
