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

#include "tensorflow/lite/delegates/gpu/metal/kernels/softmax.h"

#import <XCTest/XCTest.h>

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/softmax_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

using ::tflite::gpu::Axis;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::DataType;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::SoftmaxAttributes;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface SoftmaxMetalTest : XCTestCase
@end

@implementation SoftmaxMetalTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

- (void)setUp {
  [super setUp];
}

- (void)testSoftmaxOp {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.1, 0.2, 0.1, 0.2}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1, 1, 1, 1}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmaxDoesNotWorkForHeightAxis {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftmaxAttributes attr;
  attr.axis = Axis::HEIGHT;

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.1, 0.2, 0.3, 0.4}));
  auto status = model.Invoke();
  XCTAssertFalse(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmaxDoesNotWorkForWidthAxis {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftmaxAttributes attr;
  attr.axis = Axis::WIDTH;

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.1, 0.2, 0.3, 0.4}));
  auto status = model.Invoke();
  XCTAssertFalse(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmax1x1Op {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 1, 4);

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;

  const float sum = std::exp(0.1f) + std::exp(0.2f) + std::exp(0.3f) + std::exp(0.4f);

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.1f, 0.2f, 0.3f, 0.4f}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors(
      {std::exp(0.1f) / sum, std::exp(0.2f) / sum, std::exp(0.3f) / sum, std::exp(0.4f) / sum},
      model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmaxBigNumberOp {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 1, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 1, 2);

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;

  double doubles[4] = {1.0, 2.0, 3.0, 100.0};
  // exp(100) is inf in float (32 bit) but representable in double (64 bit)
  XCTAssertTrue(std::isinf(std::exp(static_cast<float>(doubles[3]))));
  XCTAssertFalse(std::isinf(std::exp(doubles[3])));
  double s0 = std::exp(doubles[0]) + std::exp(doubles[1]);
  double s1 = std::exp(doubles[2]) + std::exp(doubles[3]);

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {static_cast<float>(doubles[0]),
                                         static_cast<float>(doubles[1]),
                                         static_cast<float>(doubles[2]),
                                         static_cast<float>(doubles[3])}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({static_cast<float>(std::exp(doubles[0]) / s0),
                           static_cast<float>(std::exp(doubles[1]) / s0),
                           static_cast<float>(std::exp(doubles[2]) / s1),
                           static_cast<float>(std::exp(doubles[3]) / s1)},
                          model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmax1x1BigNumberOp {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 1, 4);

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;

  double doubles[4] = {1.0, 2.0, 3.0, 100.0};
  // exp(100) is inf in float (32 bit) but representable in double (64 bit)
  XCTAssertTrue(std::isinf(std::exp(static_cast<float>(doubles[3]))));
  XCTAssertFalse(std::isinf(std::exp(doubles[3])));
  double s0 = std::exp(doubles[0]) + std::exp(doubles[1]) +
      std::exp(doubles[2]) + std::exp(doubles[3]);

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {static_cast<float>(doubles[0]),
                                         static_cast<float>(doubles[1]),
                                         static_cast<float>(doubles[2]),
                                         static_cast<float>(doubles[3])}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({static_cast<float>(std::exp(doubles[0]) / s0),
                           static_cast<float>(std::exp(doubles[1]) / s0),
                           static_cast<float>(std::exp(doubles[2]) / s0),
                           static_cast<float>(std::exp(doubles[3]) / s0)},
                          model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmax {
  auto status = SoftmaxTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmaxBigNumber {
  auto status = SoftmaxBigNumberTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmax1x1 {
  auto status = Softmax1x1Test(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSoftmax1x1BigNumber {
  auto status = Softmax1x1BigNumberTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
