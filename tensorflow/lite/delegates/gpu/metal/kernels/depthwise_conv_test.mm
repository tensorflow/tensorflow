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
using ::tflite::gpu::DataType;
using ::tflite::gpu::DepthwiseConvolution2DAttributes;
using ::tflite::gpu::HW;
using ::tflite::gpu::Linear;
using ::tflite::gpu::OHWI;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::Tensor;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface DepthwiseConvTest : XCTestCase
@end

@implementation DepthwiseConvTest
- (void)setUp {
  [super setUp];
}

- (void)testO4H1W1I2Strides1x1Dilation1x1 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 2);

  DepthwiseConvolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 4;
  bias.id = 1;
  bias.data = {1, 2, 3, 4};
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(2, 1, 1, 2);
  weights.id = 2;
  weights.data = {1, 3, 2, 4};

  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 1, 1, 4);

  SingleOpModel model({ToString(OperationType::DEPTHWISE_CONVOLUTION), std::move(attr)}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 3}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({2, 4, 12, 16}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testO2H1W1I1Strides2x2Dilation1x1 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 3, 1);

  DepthwiseConvolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 4;
  bias.id = 1;
  bias.data = {0, 0};
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(2, 1, 1, 1);
  weights.id = 1;
  weights.data = {1, 3};

  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model({ToString(OperationType::DEPTHWISE_CONVOLUTION), std::move(attr)}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 0, 1, 1, 0, 1, 1, 0, 1}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({1, 3, 1, 3, 1, 3, 1, 3}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testO2H2W2I1Strides1x1Dilation2x2 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 3, 1);

  DepthwiseConvolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 4;
  bias.id = 1;
  bias.data = {0, 0};
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(2, 2, 2, 1);
  weights.id = 1;
  weights.data = {1, 2, 3, 4, 5, 6, 7, 8};

  attr.weights = std::move(weights);

  attr.dilations = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 1, 1, 2);

  SingleOpModel model({ToString(OperationType::DEPTHWISE_CONVOLUTION), std::move(attr)}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 0, 1, 1, 0, 1, 1, 0, 1}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({10, 26}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

@end
