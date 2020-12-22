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

#include "tensorflow/lite/delegates/gpu/metal/kernels/conv.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/winograd.h"

#import <XCTest/XCTest.h>

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

using ::tflite::gpu::Axis;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::Convolution2DAttributes;
using ::tflite::gpu::DataType;
using ::tflite::gpu::DivideRoundUp;
using ::tflite::gpu::HW;
using ::tflite::gpu::Linear;
using ::tflite::gpu::OHWI;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::Tensor;
using ::tflite::gpu::TensorFloat32;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::ValueId;
using ::tflite::gpu::metal::ConvolutionGeneric;
using ::tflite::gpu::metal::ConvolutionWino4x4To6x6;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface ConvTest : XCTestCase
@end

@implementation ConvTest
- (void)setUp {
  [super setUp];
}

- (void)testO2H2W1I1Stride1x1Dilation1x1 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 2;
  bias.id = 1;
  bias.data = {1, 1};
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(2, 2, 1, 1);
  weights.id = 2;
  weights.data = {1, 2, 3, 4};
  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model({ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 1, 1, 1}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({4, 8, 4, 8, 2, 4, 2, 4}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO1H2W2I1Stride1x1Dilation2x2 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 3, 1);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 2;
  bias.id = 1;
  bias.data.push_back(0.0);
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 2, 2, 1);
  weights.id = 2;
  weights.data = {1, 2, 3, 4};
  attr.weights = std::move(weights);

  attr.dilations = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 1, 1, 1);

  SingleOpModel model({ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 1, 1, 1, 1, 1, 1, 1, 1}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({10}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO1H3W3I1Stride1x1Dilation1x1 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 1;
  bias.id = 1;
  bias.data.push_back(1.0);
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 3, 3, 1);
  weights.id = 2;
  weights.data = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 1, 1, 1);

  SingleOpModel model({ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 1, 1, 1}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({11}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO2H1W1I2Stride1x1Dilation1x1 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 1, 2);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 2;
  bias.id = 1;
  bias.data = {1, 1};
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(2, 1, 1, 2);
  weights.id = 2;
  weights.data = {1, 2, 3, 4};
  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 2, 1, 2);

  SingleOpModel model({ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 1, 1, 1}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({4, 8, 4, 8}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testO1H1W1I1Stride2x2Dilation1x1 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 3, 1);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 2;
  bias.id = 1;
  bias.data.push_back(0.0);
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 1, 1, 1);
  weights.id = 2;
  weights.data.push_back(2.0);

  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 0, 2, 0, 0, 0, 4, 0, 8}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({2, 4, 8, 16}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testWinograd4x4To6x6 {
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
  int new_width = src_shape.w + attr.padding.prepended.w +
                        attr.padding.appended.w - 2;
  int new_height = src_shape.h + attr.padding.prepended.h +
                         attr.padding.appended.h - 2;
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

  TensorFloat32 output0;

  tflite::gpu::OperationDef op_def;
  op_def.precision = tflite::gpu::CalculationsPrecision::F32;
  tflite::gpu::TensorDescriptor tensor_descriptor = tflite::gpu::TensorDescriptor{
    DataType::FLOAT32, tflite::gpu::TensorStorageType::BUFFER, tflite::gpu::Layout::HWC};
  op_def.src_tensors.push_back(tensor_descriptor);
  op_def.dst_tensors.push_back(tensor_descriptor);

  tflite::gpu::metal::MetalExecutionEnvironment env;
  auto gpu_op0 = ConvolutionGeneric(op_def, dst_shape, attr, env.GetGpuInfo());
  auto op0_ptr = absl::make_unique<tflite::gpu::metal::ComputeTaskDescriptor>(std::move(gpu_op0));
  auto status = env.ExecuteGPUOperation(src_tensor, std::move(op0_ptr), dst_shape, &output0);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());

  tflite::gpu::metal::Winograd4x4To36Attributes wino_up_attr;
  wino_up_attr.padding = attr.padding;
  auto gpu_op1 = tflite::gpu::metal::Winograd4x4To36(op_def, wino_up_attr);
  auto op1_ptr = absl::make_unique<tflite::gpu::metal::ComputeTaskDescriptor>(std::move(gpu_op1));

  auto gpu_op2 = ConvolutionWino4x4To6x6(op_def, conv_shape, attr, env.GetGpuInfo());
  auto op2_ptr = absl::make_unique<tflite::gpu::metal::ComputeTaskDescriptor>(std::move(gpu_op2));

  tflite::gpu::metal::Winograd36To4x4Attributes wino_down_attr;
  wino_down_attr.output_shape = dst_shape;
  wino_down_attr.biases = attr.bias;
  auto gpu_op3 = tflite::gpu::metal::Winograd36To4x4(op_def, wino_down_attr);
  auto op3_ptr = absl::make_unique<tflite::gpu::metal::ComputeTaskDescriptor>(std::move(gpu_op3));

  TensorFloat32 output1;
  BHWC output1_shape = conv_shape;
  output1_shape.c = src_shape.c;
  status = env.ExecuteGPUOperation(src_tensor, std::move(op1_ptr), output1_shape, &output1);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());

  TensorFloat32 output2;
  BHWC output2_shape = conv_shape;
  status = env.ExecuteGPUOperation(output1, std::move(op2_ptr), output2_shape, &output2);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());

  TensorFloat32 output3;
  BHWC output3_shape = dst_shape;
  status = env.ExecuteGPUOperation(output2, std::move(op3_ptr), output3_shape, &output3);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());

  status = CompareVectors(output0.data, output3.data, 1e-4f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

@end
