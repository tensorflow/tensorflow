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

#include "tensorflow/lite/delegates/gpu/metal/kernels/winograd.h"

#import <XCTest/XCTest.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"
#include "tensorflow/lite/delegates/gpu/common/winograd_util.h"

using ::tflite::gpu::BHWC;
using ::tflite::gpu::ValueId;
using ::tflite::gpu::TensorFloat32;
using ::tflite::gpu::metal::CompareVectors;

@interface WinogradTest : XCTestCase
@end

@implementation WinogradTest
- (void)setUp {
  [super setUp];
}

- (void)testWinograd4x4To36 {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 4, 4, 1);
  src_tensor.data.resize(16);
  for (int i = 0; i < 16; ++i) {
    src_tensor.data[i] = sin(i);
  }

  TensorFloat32 dst_tensor;
  dst_tensor.shape = BHWC(1, 36, 1, 1);
  dst_tensor.data.resize(36, 0.0f);
  auto b_t = tflite::gpu::BtMatrixForWinograd4x4To6x6();

  // Bt * Src * B
  // 1: temp = Src * B
  std::vector<float> temp(36, 0.0f);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        if (y < 1 || y > 4 || i < 1 || i > 4) continue;
        const int index = src_tensor.shape.LinearIndex({0, y - 1, i - 1, 0});
        sum += src_tensor.data[index] * b_t[x * 6 + i];
      }
      temp[y * 6 + x] = sum;
    }
  }
  // 2: dst_tensor = Bt * temp
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        sum += b_t[y * 6 + i] * temp[i * 6 + x];
      }
      const int index = dst_tensor.shape.LinearIndex({0, y * 6 + x, 0, 0});
      dst_tensor.data[index] = sum;
    }
  }

  tflite::gpu::metal::Winograd4x4To36Attributes attr;
  attr.padding.prepended = tflite::gpu::HW(1, 1);
  attr.padding.appended = tflite::gpu::HW(1, 1);
  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;
  tflite::gpu::TensorDescriptor tensor_descriptor = tflite::gpu::TensorDescriptor{
      tflite::gpu::DataType::FLOAT32,
      tflite::gpu::TensorStorageType::BUFFER,
      tflite::gpu::Layout::HWC};
  op_def.src_tensors.push_back(tensor_descriptor);
  op_def.dst_tensors.push_back(tensor_descriptor);
  auto gpu_op = tflite::gpu::metal::Winograd4x4To36(op_def, attr);

  tflite::gpu::metal::MetalExecutionEnvironment env;
  auto op_ptr = absl::make_unique<tflite::gpu::metal::ComputeTaskDescriptor>(std::move(gpu_op));
  TensorFloat32 gpu_output;
  auto status = env.ExecuteGPUOperation(src_tensor, std::move(op_ptr),
                                        BHWC(1, 36, 1, 1), &gpu_output);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());

  status = CompareVectors(dst_tensor.data, gpu_output.data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testWinograd4x4To36TileX6 {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 4, 4, 1);
  src_tensor.data.resize(16);
  for (int i = 0; i < 16; ++i) {
    src_tensor.data[i] = sin(i);
  }

  TensorFloat32 dst_tensor;
  dst_tensor.shape = BHWC(1, 36, 1, 1);
  dst_tensor.data.resize(36, 0.0f);
  auto b_t = tflite::gpu::BtMatrixForWinograd4x4To6x6();

  // Bt * Src * B
  // 1: temp = Src * B
  std::vector<float> temp(36, 0.0f);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        if (y < 1 || y > 4 || i < 1 || i > 4) continue;
        const int index = src_tensor.shape.LinearIndex({0, y - 1, i - 1, 0});
        sum += src_tensor.data[index] * b_t[x * 6 + i];
      }
      temp[y * 6 + x] = sum;
    }
  }
  // 2: dst_tensor = Bt * temp
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 6; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        sum += b_t[y * 6 + i] * temp[i * 6 + x];
      }
      const int index = dst_tensor.shape.LinearIndex({0, y * 6 + x, 0, 0});
      dst_tensor.data[index] = sum;
    }
  }

  tflite::gpu::metal::Winograd4x4To36Attributes attr;
  attr.padding.prepended = tflite::gpu::HW(1, 1);
  attr.padding.appended = tflite::gpu::HW(1, 1);
  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;
  tflite::gpu::TensorDescriptor tensor_descriptor = tflite::gpu::TensorDescriptor{
      tflite::gpu::DataType::FLOAT32,
      tflite::gpu::TensorStorageType::BUFFER,
      tflite::gpu::Layout::HWC};
  op_def.src_tensors.push_back(tensor_descriptor);
  op_def.dst_tensors.push_back(tensor_descriptor);
  auto gpu_op = tflite::gpu::metal::Winograd4x4To36TileX6(op_def, attr);

  tflite::gpu::metal::MetalExecutionEnvironment env;
  auto op_ptr = absl::make_unique<tflite::gpu::metal::ComputeTaskDescriptor>(std::move(gpu_op));
  TensorFloat32 gpu_output;
  auto status = env.ExecuteGPUOperation(src_tensor, std::move(op_ptr),
                                        BHWC(1, 36, 1, 1), &gpu_output);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());

  status = CompareVectors(dst_tensor.data, gpu_output.data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testWinograd36To4x4 {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 36, 1, 1);
  src_tensor.data.resize(36);
  for (int i = 0; i < 36; ++i) {
    src_tensor.data[i] = sin(i);
  }

  TensorFloat32 dst_tensor;
  dst_tensor.shape = BHWC(1, 4, 4, 1);
  dst_tensor.data.resize(16, 0.0f);
  auto a_t = tflite::gpu::AtMatrixForWinograd4x4To6x6();

  // At * Src * A
  // 1: temp = Src * A
  std::vector<float> temp(24, 0.0f);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 4; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        const int index = src_tensor.shape.LinearIndex({0, y * 6 + i, 0, 0});
        sum += src_tensor.data[index] * a_t[x * 6 + i];
      }
      temp[y * 4 + x] = sum;
    }
  }
  // 2: dst_tensor = At * temp
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        sum += a_t[y * 6 + i] * temp[i * 4 + x];
      }
      const int index = dst_tensor.shape.LinearIndex({0, y, x, 0});
      dst_tensor.data[index] = sum;
    }
  }

  tflite::gpu::metal::Winograd36To4x4Attributes attr;
  attr.output_shape = BHWC(1, 4, 4, 1);
  attr.biases.shape = tflite::gpu::Linear(1);
  attr.biases.data.resize(1, 0.0f);

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;
  tflite::gpu::TensorDescriptor tensor_descriptor = tflite::gpu::TensorDescriptor{
      tflite::gpu::DataType::FLOAT32,
      tflite::gpu::TensorStorageType::BUFFER,
      tflite::gpu::Layout::HWC};
  op_def.src_tensors.push_back(tensor_descriptor);
  op_def.dst_tensors.push_back(tensor_descriptor);
  auto gpu_op = tflite::gpu::metal::Winograd36To4x4(op_def, attr);

  tflite::gpu::metal::MetalExecutionEnvironment env;
  auto op_ptr = absl::make_unique<tflite::gpu::metal::ComputeTaskDescriptor>(std::move(gpu_op));
  TensorFloat32 gpu_output;
  auto status = env.ExecuteGPUOperation(src_tensor, std::move(op_ptr),
                                        BHWC(1, 4, 4, 1), &gpu_output);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());

  status = CompareVectors(dst_tensor.data, gpu_output.data, 1e-5f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testWinograd36To4x4Tile4x1 {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 36, 1, 1);
  src_tensor.data.resize(36);
  for (int i = 0; i < 36; ++i) {
    src_tensor.data[i] = sin(i);
  }

  TensorFloat32 dst_tensor;
  dst_tensor.shape = BHWC(1, 4, 4, 1);
  dst_tensor.data.resize(16, 0.0f);
  auto a_t = tflite::gpu::AtMatrixForWinograd4x4To6x6();

  // At * Src * A
  // 1: temp = Src * A
  std::vector<float> temp(24, 0.0f);
  for (int y = 0; y < 6; ++y) {
    for (int x = 0; x < 4; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        const int index = src_tensor.shape.LinearIndex({0, y * 6 + i, 0, 0});
        sum += src_tensor.data[index] * a_t[x * 6 + i];
      }
      temp[y * 4 + x] = sum;
    }
  }
  // 2: dst_tensor = At * temp
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      float sum = 0.0f;
      for (int i = 0; i < 6; ++i) {
        sum += a_t[y * 6 + i] * temp[i * 4 + x];
      }
      const int index = dst_tensor.shape.LinearIndex({0, y, x, 0});
      dst_tensor.data[index] = sum;
    }
  }

  tflite::gpu::metal::Winograd36To4x4Attributes attr;
  attr.output_shape = BHWC(1, 4, 4, 1);
  attr.biases.shape = tflite::gpu::Linear(1);
  attr.biases.data.resize(1, 0.0f);

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;
  tflite::gpu::TensorDescriptor tensor_descriptor = tflite::gpu::TensorDescriptor{
      tflite::gpu::DataType::FLOAT32,
      tflite::gpu::TensorStorageType::BUFFER,
      tflite::gpu::Layout::HWC};
  op_def.src_tensors.push_back(tensor_descriptor);
  op_def.dst_tensors.push_back(tensor_descriptor);
  auto gpu_op = tflite::gpu::metal::Winograd36To4x4Tile4x1(op_def, attr);

  tflite::gpu::metal::MetalExecutionEnvironment env;
  auto op_ptr = absl::make_unique<tflite::gpu::metal::ComputeTaskDescriptor>(std::move(gpu_op));
  TensorFloat32 gpu_output;
  auto status = env.ExecuteGPUOperation(src_tensor, std::move(op_ptr),
                                        BHWC(1, 4, 4, 1), &gpu_output);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());

  status = CompareVectors(dst_tensor.data, gpu_output.data, 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

@end
