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

#include <string>
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
using ::tflite::gpu::ConcatAttributes;
using ::tflite::gpu::DataType;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface ConcatTest : XCTestCase
@end

@implementation ConcatTest
- (void)setUp {
  [super setUp];
}

- (void)testTwoInputTensorsByUnalignedChannel {
  TensorRef<BHWC> input1, input2, output;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 2, 2, 1);

  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 2, 2, 1);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 2);

  ConcatAttributes attr;
  attr.axis = Axis::CHANNELS;

  SingleOpModel model({ToString(OperationType::CONCAT), attr}, {input1, input2}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 3, 5, 7}));
  XCTAssertTrue(model.PopulateTensor(1, {2, 4, 6, 8}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1, 2, 3, 4, 5, 6, 7, 8}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTwoInputTensorsByAlignedChannel {
  TensorRef<BHWC> input1, input2, output;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 1, 1, 4);

  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 1, 1, 4);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 1, 8);

  ConcatAttributes attr;
  attr.axis = Axis::CHANNELS;

  SingleOpModel model({ToString(OperationType::CONCAT), attr}, {input1, input2}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 2, 3, 4}));
  XCTAssertTrue(model.PopulateTensor(1, {5, 6, 7, 8}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1, 2, 3, 4, 5, 6, 7, 8}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTwoInputTensorsByHeight {
  TensorRef<BHWC> input1, input2, output;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 1, 2, 1);

  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 2, 2, 1);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 3, 2, 1);

  ConcatAttributes attr;
  attr.axis = Axis::HEIGHT;

  SingleOpModel model({ToString(OperationType::CONCAT), attr}, {input1, input2}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 2}));
  XCTAssertTrue(model.PopulateTensor(1, {3, 4, 5, 6}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1, 2, 3, 4, 5, 6}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTwoInputTensorsByWidth {
  TensorRef<BHWC> input1, input2, output;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 2, 1, 1);

  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 2, 2, 1);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 3, 1);

  ConcatAttributes attr;
  attr.axis = Axis::WIDTH;

  SingleOpModel model({ToString(OperationType::CONCAT), attr}, {input1, input2}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 4}));
  XCTAssertTrue(model.PopulateTensor(1, {2, 3, 5, 6}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1, 2, 3, 4, 5, 6}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}
@end
