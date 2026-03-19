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

#include <cmath>

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

namespace {
template <DataType T>
absl::Status TensorBHWCTest(const BHWC& shape, const TensorDescriptor& descriptor,
                            id<MTLDevice> device) {
  tflite::gpu::Tensor<BHWC, T> tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    // val = [0, 1];
    const double val = static_cast<double>(i) / static_cast<double>(tensor_cpu.data.size() - 1);
    double transformed_val = sin(val * 2.0 * M_PI) * 256.0;
    if (descriptor.GetDataType() == DataType::INT16 ||
        descriptor.GetDataType() == DataType::UINT16) {
      transformed_val *= 256.0;
    }
    if (descriptor.GetDataType() == DataType::INT32 ||
        descriptor.GetDataType() == DataType::UINT32) {
      transformed_val *= 256.0 * 256.0 * 256.0 * 256.0;
    }
    if (descriptor.GetDataType() == DataType::FLOAT16) {
      transformed_val = half(transformed_val);
    }
    if (descriptor.GetDataType() == DataType::BOOL) {
      transformed_val = i % 7;
    }
    tensor_cpu.data[i] = transformed_val;
  }
  tflite::gpu::Tensor<BHWC, T> tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0;
  }

  tflite::gpu::metal::MetalSpatialTensor tensor;
  tflite::gpu::TensorDescriptor descriptor_with_data = descriptor;
  descriptor_with_data.UploadData(tensor_cpu);
  RETURN_IF_ERROR(tensor.CreateFromDescriptor(descriptor_with_data, device));
  tflite::gpu::TensorDescriptor output_descriptor;
  RETURN_IF_ERROR(tensor.ToDescriptor(&output_descriptor, device));
  output_descriptor.DownloadData(&tensor_gpu);

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return absl::InternalError("Wrong value at index - " + std::to_string(i) + ". GPU - " +
                                 std::to_string(tensor_gpu.data[i]) + ", CPU - " +
                                 std::to_string(tensor_cpu.data[i]));
    }
  }
  return absl::OkStatus();
}

template absl::Status TensorBHWCTest<DataType::FLOAT32>(const BHWC& shape,
                                                        const TensorDescriptor& descriptor,
                                                        id<MTLDevice> device);
template absl::Status TensorBHWCTest<DataType::INT32>(const BHWC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::INT16>(const BHWC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::INT8>(const BHWC& shape,
                                                     const TensorDescriptor& descriptor,
                                                     id<MTLDevice> device);
template absl::Status TensorBHWCTest<DataType::UINT32>(const BHWC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::UINT16>(const BHWC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::UINT8>(const BHWC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      id<MTLDevice> device);

template absl::Status TensorBHWCTest<DataType::BOOL>(const BHWC& shape,
                                                     const TensorDescriptor& descriptor,
                                                     id<MTLDevice> device);

template <DataType T>
absl::Status TensorBHWDCTest(const BHWDC& shape, const TensorDescriptor& descriptor,
                             id<MTLDevice> device) {
  tflite::gpu::Tensor<BHWDC, T> tensor_cpu;
  tensor_cpu.shape = shape;
  tensor_cpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_cpu.data.size(); ++i) {
    // val = [0, 1];
    const double val = static_cast<double>(i) / static_cast<double>(tensor_cpu.data.size() - 1);
    double transformed_val = sin(val * 2.0 * M_PI) * 256.0;
    if (descriptor.GetDataType() == DataType::INT16 ||
        descriptor.GetDataType() == DataType::UINT16) {
      transformed_val *= 256.0;
    }
    if (descriptor.GetDataType() == DataType::INT32 ||
        descriptor.GetDataType() == DataType::UINT32) {
      transformed_val *= 256.0 * 256.0 * 256.0 * 256.0;
    }
    if (descriptor.GetDataType() == DataType::FLOAT16) {
      transformed_val = half(transformed_val);
    }
    if (descriptor.GetDataType() == DataType::BOOL) {
      transformed_val = i % 7;
    }
    tensor_cpu.data[i] = transformed_val;
  }
  tflite::gpu::Tensor<BHWDC, T> tensor_gpu;
  tensor_gpu.shape = shape;
  tensor_gpu.data.resize(shape.DimensionsProduct());
  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    tensor_gpu.data[i] = 0;
  }

  tflite::gpu::metal::MetalSpatialTensor tensor;
  tflite::gpu::TensorDescriptor descriptor_with_data = descriptor;
  descriptor_with_data.UploadData(tensor_cpu);
  RETURN_IF_ERROR(tensor.CreateFromDescriptor(descriptor_with_data, device));
  tflite::gpu::TensorDescriptor output_descriptor;
  RETURN_IF_ERROR(tensor.ToDescriptor(&output_descriptor, device));
  output_descriptor.DownloadData(&tensor_gpu);

  for (int i = 0; i < tensor_gpu.data.size(); ++i) {
    if (tensor_gpu.data[i] != tensor_cpu.data[i]) {
      return absl::InternalError("Wrong value.");
    }
  }
  return absl::OkStatus();
}

template absl::Status TensorBHWDCTest<DataType::FLOAT32>(const BHWDC& shape,
                                                         const TensorDescriptor& descriptor,
                                                         id<MTLDevice> device);
template absl::Status TensorBHWDCTest<DataType::INT32>(const BHWDC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::INT16>(const BHWDC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::INT8>(const BHWDC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      id<MTLDevice> device);
template absl::Status TensorBHWDCTest<DataType::UINT32>(const BHWDC& shape,
                                                        const TensorDescriptor& descriptor,
                                                        id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::UINT16>(const BHWDC& shape,
                                                        const TensorDescriptor& descriptor,
                                                        id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::UINT8>(const BHWDC& shape,
                                                       const TensorDescriptor& descriptor,
                                                       id<MTLDevice> device);

template absl::Status TensorBHWDCTest<DataType::BOOL>(const BHWDC& shape,
                                                      const TensorDescriptor& descriptor,
                                                      id<MTLDevice> device);

template <DataType T>
absl::Status TensorTests(DataType data_type, TensorStorageType storage_type) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(1, 6, 7, 3), {data_type, storage_type, Layout::HWC}, device));
  RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(1, 1, 4, 12), {data_type, storage_type, Layout::HWC}, device));
  RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(1, 6, 1, 7), {data_type, storage_type, Layout::HWC}, device));

  // Batch tests
  RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(2, 6, 7, 3), {data_type, storage_type, Layout::BHWC}, device));
  RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(4, 1, 4, 12), {data_type, storage_type, Layout::BHWC}, device));
  RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(7, 6, 1, 7), {data_type, storage_type, Layout::BHWC}, device));
  RETURN_IF_ERROR(
      TensorBHWCTest<T>(BHWC(13, 7, 3, 3), {data_type, storage_type, Layout::BHWC}, device));

  // 5D tests with batch = 1
  RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(1, 6, 7, 4, 3), {data_type, storage_type, Layout::HWDC}, device));
  RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(1, 1, 4, 3, 12), {data_type, storage_type, Layout::HWDC}, device));
  RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(1, 6, 1, 7, 7), {data_type, storage_type, Layout::HWDC}, device));

  // 5D tests
  RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(2, 6, 7, 1, 3), {data_type, storage_type, Layout::BHWDC}, device));
  RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(4, 1, 4, 2, 12), {data_type, storage_type, Layout::BHWDC}, device));
  RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(7, 6, 1, 3, 7), {data_type, storage_type, Layout::BHWDC}, device));
  RETURN_IF_ERROR(
      TensorBHWDCTest<T>(BHWDC(13, 7, 3, 4, 3), {data_type, storage_type, Layout::BHWDC}, device));
  return absl::OkStatus();
}

template absl::Status TensorTests<DataType::FLOAT32>(DataType data_type,
                                                     TensorStorageType storage_type);
template absl::Status TensorTests<DataType::INT32>(DataType data_type,
                                                   TensorStorageType storage_type);
template absl::Status TensorTests<DataType::INT16>(DataType data_type,
                                                   TensorStorageType storage_type);
template absl::Status TensorTests<DataType::INT8>(DataType data_type,
                                                  TensorStorageType storage_type);
template absl::Status TensorTests<DataType::UINT32>(DataType data_type,
                                                    TensorStorageType storage_type);
template absl::Status TensorTests<DataType::UINT16>(DataType data_type,
                                                    TensorStorageType storage_type);
template absl::Status TensorTests<DataType::UINT8>(DataType data_type,
                                                   TensorStorageType storage_type);
template absl::Status TensorTests<DataType::BOOL>(DataType data_type,
                                                  TensorStorageType storage_type);

}  // namespace

- (void)testBufferF32 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT32, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferF16 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT16, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferInt32 {
  auto status = TensorTests<DataType::INT32>(DataType::INT32, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferInt16 {
  auto status = TensorTests<DataType::INT16>(DataType::INT16, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferInt8 {
  auto status = TensorTests<DataType::INT8>(DataType::INT8, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferUint32 {
  auto status = TensorTests<DataType::UINT32>(DataType::UINT32, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferUint16 {
  auto status = TensorTests<DataType::UINT16>(DataType::UINT16, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferUint8 {
  auto status = TensorTests<DataType::UINT8>(DataType::UINT8, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testBufferBool {
  auto status = TensorTests<DataType::BOOL>(DataType::BOOL, TensorStorageType::BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DF32 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT32, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DF16 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT16, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DInt32 {
  auto status = TensorTests<DataType::INT32>(DataType::INT32, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DInt16 {
  auto status = TensorTests<DataType::INT16>(DataType::INT16, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DInt8 {
  auto status = TensorTests<DataType::INT8>(DataType::INT8, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DUint32 {
  auto status = TensorTests<DataType::UINT32>(DataType::UINT32, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DUint16 {
  auto status = TensorTests<DataType::UINT16>(DataType::UINT16, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DUint8 {
  auto status = TensorTests<DataType::UINT8>(DataType::UINT8, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DBool {
  auto status = TensorTests<DataType::BOOL>(DataType::BOOL, TensorStorageType::TEXTURE_2D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DF32 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT32, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DF16 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT16, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DInt32 {
  auto status = TensorTests<DataType::INT32>(DataType::INT32, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DInt16 {
  auto status = TensorTests<DataType::INT16>(DataType::INT16, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DInt8 {
  auto status = TensorTests<DataType::INT8>(DataType::INT8, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DUint32 {
  auto status = TensorTests<DataType::UINT32>(DataType::UINT32, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DUint16 {
  auto status = TensorTests<DataType::UINT16>(DataType::UINT16, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DUint8 {
  auto status = TensorTests<DataType::UINT8>(DataType::UINT8, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture3DBool {
  auto status = TensorTests<DataType::BOOL>(DataType::BOOL, TensorStorageType::TEXTURE_3D);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayF32 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT32, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayF16 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT16, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayInt32 {
  auto status = TensorTests<DataType::INT32>(DataType::INT32, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayInt16 {
  auto status = TensorTests<DataType::INT16>(DataType::INT16, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayInt8 {
  auto status = TensorTests<DataType::INT8>(DataType::INT8, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayUint32 {
  auto status = TensorTests<DataType::UINT32>(DataType::UINT32, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayUint16 {
  auto status = TensorTests<DataType::UINT16>(DataType::UINT16, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayUint8 {
  auto status = TensorTests<DataType::UINT8>(DataType::UINT8, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTexture2DArrayBool {
  auto status = TensorTests<DataType::BOOL>(DataType::BOOL, TensorStorageType::TEXTURE_ARRAY);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferF32 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT32, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferF16 {
  auto status = TensorTests<DataType::FLOAT32>(DataType::FLOAT16, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferInt32 {
  auto status = TensorTests<DataType::INT32>(DataType::INT32, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferInt16 {
  auto status = TensorTests<DataType::INT16>(DataType::INT16, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferInt8 {
  auto status = TensorTests<DataType::INT8>(DataType::INT8, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferUint32 {
  auto status = TensorTests<DataType::UINT32>(DataType::UINT32, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferUint16 {
  auto status = TensorTests<DataType::UINT16>(DataType::UINT16, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferUint8 {
  auto status = TensorTests<DataType::UINT8>(DataType::UINT8, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTextureBufferBool {
  auto status = TensorTests<DataType::BOOL>(DataType::BOOL, TensorStorageType::IMAGE_BUFFER);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
