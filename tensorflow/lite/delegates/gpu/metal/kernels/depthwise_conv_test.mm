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

#import <XCTest/XCTest.h>

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_3x3_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface DepthwiseConvMetalTest : XCTestCase
@end

@implementation DepthwiseConvMetalTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

namespace tflite {
namespace gpu {
namespace metal {

absl::Status DepthWiseO4H1W1I2Strides1x1Dilation1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 2);
  src_tensor.data = {1, 3};

  DepthwiseConvolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {1, 3, 2, 4};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1, 2, 3, 4};
  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      DepthWiseConvolution operation = CreateDepthWiseConvolution(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<DepthWiseConvolution>(std::move(operation)),
          BHWC(1, 1, 1, 4), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2, 4, 12, 16}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status DepthWiseO2H1W1I1Strides2x2Dilation1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 3, 1);
  src_tensor.data = {1, 0, 1, 1, 0, 1, 1, 0, 1};

  DepthwiseConvolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 1);
  attr.weights.data = {1, 3};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};
  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      DepthWiseConvolution operation = CreateDepthWiseConvolution(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<DepthWiseConvolution>(std::move(operation)),
          BHWC(1, 2, 2, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({1, 3, 1, 3, 1, 3, 1, 3}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status DepthWiseO2H2W2I1Strides1x1Dilation2x2Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 3, 1);
  src_tensor.data = {1, 0, 1, 1, 0, 1, 1, 0, 1};

  DepthwiseConvolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 2, 2, 1);
  attr.weights.data = {1, 2, 3, 4, 5, 6, 7, 8};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};
  attr.dilations = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      DepthWiseConvolution operation = CreateDepthWiseConvolution(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<DepthWiseConvolution>(std::move(operation)),
          BHWC(1, 1, 1, 2), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({10, 26}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status DepthWiseShape2x2Kernel2x2Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1, 4, 9, 16};

  DepthwiseConvolution2DAttributes attr;
  attr.weights.shape = OHWI(1, 2, 2, 1);
  attr.weights.data = {1, 2, 3, 4};
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};
  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      DepthWiseConvolution operation = CreateDepthWiseConvolution(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<DepthWiseConvolution>(std::move(operation)),
          BHWC(1, 2, 2, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({100, 52, 41, 16}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

- (void)testO4H1W1I2Strides1x1Dilation1x1 {
  auto status = tflite::gpu::metal::DepthWiseO4H1W1I2Strides1x1Dilation1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO2H1W1I1Strides2x2Dilation1x1 {
  auto status = tflite::gpu::metal::DepthWiseO2H1W1I1Strides2x2Dilation1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO2H2W2I1Strides1x1Dilation2x2 {
  auto status = tflite::gpu::metal::DepthWiseO2H2W2I1Strides1x1Dilation2x2Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testShape2x2Kernel2x2 {
  auto status = tflite::gpu::metal::DepthWiseShape2x2Kernel2x2Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConvSimpleWeights {
  auto status = DepthwiseConvSimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConvNoMultiplier {
  auto status = DepthwiseConvNoMultiplierTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConvMultiplier2 {
  auto status = DepthwiseConvMultiplier2Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConv3x3SimpleWeights {
  auto status = DepthwiseConv3x3SimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDepthwiseConv3x3 {
  auto status = DepthwiseConv3x3Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
