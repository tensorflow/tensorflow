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

#include "tensorflow/lite/delegates/gpu/metal/kernels/add.h"

#import <XCTest/XCTest.h>

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

using ::tflite::gpu::Axis;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::Convolution2DAttributes;
using ::tflite::gpu::DataType;
using ::tflite::gpu::HW;
using ::tflite::gpu::Linear;
using ::tflite::gpu::OHWI;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::Tensor;
using ::tflite::gpu::TensorRef;
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
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({4, 8, 4, 8, 2, 4, 2, 4}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
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
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({10}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
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
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({11}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
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
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({4, 8, 4, 8}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
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
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({2, 4, 8, 16}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

@end
