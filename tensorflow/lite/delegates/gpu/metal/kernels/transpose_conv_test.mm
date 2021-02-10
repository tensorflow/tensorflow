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

#include "tensorflow/lite/delegates/gpu/metal/kernels/transpose_conv.h"

#import <XCTest/XCTest.h>

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_4x4_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/convolution_transposed_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface TransposeConvTest : XCTestCase
@end

@implementation TransposeConvTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

namespace tflite {
namespace gpu {
namespace metal {

absl::Status TransposeConvO2H2W1I1Stride1x1DAdjacent1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1, 1, 1, 1};

  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 2, 1, 1);
  attr.weights.data = {1, 2, 3, 4};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1, 1};
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 0);
  attr.adjacent = HW(1, 1);
  attr.stride = HW(1, 1);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      ConvolutionTransposed operation =
          CreateConvolutionTransposed(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionTransposed>(std::move(operation)),
          BHWC(1, 3, 3, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2, 4, 2, 4, 1, 1, 4, 8, 4, 8, 1, 1, 3, 5, 3, 5, 1, 1},
                                    dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status TransposeConvO1H2W2I1Stride1x1Adjacent2x2Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 3, 1);
  src_tensor.data = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(1, 2, 2, 1);
  attr.weights.data = {1, 2, 3, 4};
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0};
  attr.adjacent = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(1, 1);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      ConvolutionTransposed operation =
          CreateConvolutionTransposed(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionTransposed>(std::move(operation)),
          BHWC(1, 6, 6, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({1, 3, 3, 2, 0, 0, 4, 10, 10, 6, 0, 0, 4, 10, 10, 6, 0, 0,
                                     3, 7, 7, 4, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0,  0,  0, 0, 0},
                                    dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status TransposeConvO1H3W3I1Stride1x1Adjacent1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1, 1, 1, 1};

  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(1, 3, 3, 1);
  attr.weights.data = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  attr.bias.shape = Linear(1);
  attr.bias.data = {1.0};
  attr.adjacent = HW(1, 1);
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(1, 1);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      ConvolutionTransposed operation =
          CreateConvolutionTransposed(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionTransposed>(std::move(operation)),
          BHWC(1, 4, 4, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({7, 11, 7, 1, 7, 11, 7, 1, 4, 6, 4, 1, 1, 1, 1, 1}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status TransposeConvO2H1W1I2Stride1x1Dilation1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1, 1, 1, 1};

  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {1, 2, 3, 4};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.0, 1.0};
  attr.adjacent = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(1, 1);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      ConvolutionTransposed operation =
          CreateConvolutionTransposed(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionTransposed>(std::move(operation)),
          BHWC(1, 3, 2, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({4, 8, 1, 1, 4, 8, 1, 1, 1, 1, 1, 1}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status TransposeConvO1H1W1I1Stride2x2Dilation1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 3, 1);
  src_tensor.data = {1, 0, 2, 0, 0, 0, 4, 0, 8};

  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(1, 1, 1, 1);
  attr.weights.data = {2.0};
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0};
  attr.adjacent = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      ConvolutionTransposed operation =
          CreateConvolutionTransposed(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionTransposed>(std::move(operation)),
          BHWC(1, 6, 6, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0},
                                    dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status TransposeConv4x4Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 4, 4, 1);
  attr.weights.data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                      2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                       2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0, 0.0};
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(1, 1);
  attr.stride = HW(2, 2);

  for (auto storage : {TensorStorageType::BUFFER, TensorStorageType::IMAGE_BUFFER}) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      ConvolutionTransposed4x4 operation =
          CreateConvolutionTransposed4x4(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionTransposed4x4>(std::move(operation)),
          BHWC(1, 4, 4, 2), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f,  0.0f, 1.0f,  2.0f, 1.0f, 2.0f,  1.0f, 2.0f,  2.0f,  4.0f, 6.0f,
                         12.0f, 6.0f, 12.0f, 4.0f, 8.0f, 2.0f,  4.0f, 6.0f,  12.0f, 6.0f, 12.0f,
                         4.0f,  8.0f, 2.0f,  4.0f, 5.0f, 10.0f, 5.0f, 10.0f, 3.0f,  6.0f},
                        dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

- (void)testTransposeConvO2H2W1I1Stride1x1DAdjacent1x1 {
  auto status = tflite::gpu::metal::TransposeConvO2H2W1I1Stride1x1DAdjacent1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTransposeConvO1H2W2I1Stride1x1Adjacent2x2 {
  auto status = tflite::gpu::metal::TransposeConvO1H2W2I1Stride1x1Adjacent2x2Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTransposeConvO1H3W3I1Stride1x1Adjacent1x1 {
  auto status = tflite::gpu::metal::TransposeConvO1H3W3I1Stride1x1Adjacent1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTransposeConvO2H1W1I2Stride1x1Dilation1x1 {
  auto status = tflite::gpu::metal::TransposeConvO2H1W1I2Stride1x1Dilation1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTransposeConvO1H1W1I1Stride2x2Dilation1x1 {
  auto status = tflite::gpu::metal::TransposeConvO1H1W1I1Stride2x2Dilation1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTransposeConv4x4 {
  auto status = tflite::gpu::metal::TransposeConv4x4Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposedSimpleWeights {
  auto status = ConvolutionTransposedSimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposed {
  auto status = ConvolutionTransposedTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvolutionTransposed4x4SimpleWeights {
  auto status = ConvolutionTransposed4x4SimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
