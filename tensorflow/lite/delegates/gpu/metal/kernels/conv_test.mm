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

#include "tensorflow/lite/delegates/gpu/common/tasks/conv_metal.h"

#import <XCTest/XCTest.h>
#include "tensorflow/lite/delegates/gpu/common/tasks/winograd.h"

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_buffer_1x1_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_constants_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_powervr_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface ConvTest : XCTestCase
@end

@implementation ConvTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

namespace tflite {
namespace gpu {
namespace metal {

absl::Status ConvolutionO2H2W1I1Stride1x1Dilation1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1, 1, 1, 1};

  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 2, 1, 1);
  attr.weights.data = {1, 2, 3, 4};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1, 1};
  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 0);
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
      ConvolutionMetal operation =
          CreateConvolutionMetal(op_def, BHWC(1, 2, 2, 2), attr, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionMetal>(std::move(operation)), BHWC(1, 2, 2, 2),
          &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({4, 8, 4, 8, 2, 4, 2, 4}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status ConvolutionO1H2W2I1Stride1x1Dilation2x2Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 3, 1);
  src_tensor.data = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(1, 2, 2, 1);
  attr.weights.data = {1, 2, 3, 4};
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};
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
      ConvolutionMetal operation =
          CreateConvolutionMetal(op_def, BHWC(1, 1, 1, 1), attr, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionMetal>(std::move(operation)), BHWC(1, 1, 1, 1),
          &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({10}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status ConvolutionO1H3W3I1Stride1x1Dilation1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1, 1, 1, 1};

  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(1, 3, 3, 1);
  attr.weights.data = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  attr.bias.shape = Linear(1);
  attr.bias.data = {1.0f};
  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(1, 1);
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
      ConvolutionMetal operation =
          CreateConvolutionMetal(op_def, BHWC(1, 1, 1, 1), attr, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionMetal>(std::move(operation)), BHWC(1, 1, 1, 1),
          &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({11}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status ConvolutionO2H1W1I2Stride1x1Dilation1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1, 1, 1, 1};

  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {1, 2, 3, 4};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.0f, 1.0f};
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
      ConvolutionMetal operation =
          CreateConvolutionMetal(op_def, BHWC(1, 2, 1, 2), attr, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionMetal>(std::move(operation)), BHWC(1, 2, 1, 2),
          &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({4, 8, 4, 8}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status ConvolutionO1H1W1I1Stride2x2Dilation1x1Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 3, 1);
  src_tensor.data = {1, 0, 2, 0, 0, 0, 4, 0, 8};

  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(1, 1, 1, 1);
  attr.weights.data = {2.0f};
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};
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
      ConvolutionMetal operation =
          CreateConvolutionMetal(op_def, BHWC(1, 2, 2, 1), attr, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<ConvolutionMetal>(std::move(operation)), BHWC(1, 2, 2, 1),
          &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2, 4, 8, 16}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

absl::Status Winograd4x4To6x6Test(TestExecutionEnvironment* env) {
  const int src_channels = 7;
  const int dst_channels = 13;
  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(10, 10);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(dst_channels, 3, 3, src_channels);
  attr.weights.data.resize(attr.weights.shape.DimensionsProduct());
  for (int i = 0; i < attr.weights.data.size(); ++i) {
    attr.weights.data[i] = sin(i);
  }
  attr.bias.shape = Linear(dst_channels);
  attr.bias.data.resize(attr.bias.shape.DimensionsProduct());
  for (int i = 0; i < attr.bias.data.size(); ++i) {
    attr.bias.data[i] = sin(i);
  }

  auto src_shape = BHWC(1, 17, 13, src_channels);
  auto dst_shape = CalculateOutputShape(src_shape, attr);
  int new_width = src_shape.w + attr.padding.prepended.w + attr.padding.appended.w - 2;
  int new_height = src_shape.h + attr.padding.prepended.h + attr.padding.appended.h - 2;
  BHWC conv_shape;
  conv_shape.b = dst_shape.b;
  conv_shape.h = 36;
  conv_shape.w = DivideRoundUp(new_width, 4) * DivideRoundUp(new_height, 4);
  conv_shape.c = dst_shape.c;

  TensorFloat32 src_tensor;
  src_tensor.shape = src_shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = sin(i);
  }

  for (auto storage : {TensorStorageType::BUFFER, TensorStorageType::IMAGE_BUFFER}) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-4f : 0.4f;

      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});

      TensorFloat32 output0;
      auto gpu_op0 = CreateConvolutionMetal(op_def, dst_shape, attr, env->GetGpuInfo());
      auto op0_ptr = absl::make_unique<ConvolutionMetal>(std::move(gpu_op0));
      RETURN_IF_ERROR(
          env->ExecuteGPUOperation(src_tensor, std::move(op0_ptr), dst_shape, &output0));

      auto gpu_op1 = CreateWinograd4x4To36(op_def, attr.padding);
      std::unique_ptr<GPUOperation> op1_ptr =
          absl::make_unique<Winograd4x4To36>(std::move(gpu_op1));

      auto gpu_op2 =
          CreateConvolutionMetalWino4x4To6x6(op_def, conv_shape, attr, env->GetGpuInfo());
      auto op2_ptr = absl::make_unique<ConvolutionMetal>(std::move(gpu_op2));

      auto gpu_op3 = CreateWinograd36To4x4(op_def, attr.bias);
      std::unique_ptr<GPUOperation> op3_ptr =
          absl::make_unique<Winograd36To4x4>(std::move(gpu_op3));

      TensorFloat32 output1;
      BHWC output1_shape = conv_shape;
      output1_shape.c = src_shape.c;
      RETURN_IF_ERROR(
          env->ExecuteGPUOperation(src_tensor, std::move(op1_ptr), output1_shape, &output1));

      TensorFloat32 output2;
      BHWC output2_shape = conv_shape;
      RETURN_IF_ERROR(
          env->ExecuteGPUOperation(output1, std::move(op2_ptr), output2_shape, &output2));

      TensorFloat32 output3;
      BHWC output3_shape = dst_shape;
      RETURN_IF_ERROR(
          env->ExecuteGPUOperation(output2, std::move(op3_ptr), output3_shape, &output3));

      RETURN_IF_ERROR(PointWiseNear(output0.data, output3.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

- (void)testO2H2W1I1Stride1x1Dilation1x1 {
  auto status = tflite::gpu::metal::ConvolutionO2H2W1I1Stride1x1Dilation1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO1H2W2I1Stride1x1Dilation2x2 {
  auto status = tflite::gpu::metal::ConvolutionO1H2W2I1Stride1x1Dilation2x2Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO1H3W3I1Stride1x1Dilation1x1 {
  auto status = tflite::gpu::metal::ConvolutionO1H3W3I1Stride1x1Dilation1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO2H1W1I2Stride1x1Dilation1x1 {
  auto status = tflite::gpu::metal::ConvolutionO2H1W1I2Stride1x1Dilation1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO1H1W1I1Stride2x2Dilation1x1 {
  auto status = tflite::gpu::metal::ConvolutionO1H1W1I1Stride2x2Dilation1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testWinograd4x4To6x6 {
  auto status = tflite::gpu::metal::Winograd4x4To6x6Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvPowerVR1x1SimpleWeights {
  const auto status = ConvPowerVR1x1SimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvPowerVR1x1 {
  const auto status = ConvPowerVR1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvPowerVRSimpleWeights {
  const auto status = ConvPowerVRSimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvPowerVR {
  const auto status = ConvPowerVRTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvConstantsSimpleWeights {
  const auto status = ConvConstantsSimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvConstants {
  const auto status = ConvConstantsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvBuffer1x1SimpleWeights {
  const auto status = ConvBuffer1x1SimpleWeightsTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testConvBuffer1x1 {
  const auto status = ConvBuffer1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
