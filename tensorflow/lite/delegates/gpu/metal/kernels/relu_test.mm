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

#include "tensorflow/lite/delegates/gpu/metal/kernels/slice.h"

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

using ::tflite::gpu::BHWC;
using ::tflite::gpu::DataType;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::ReLUAttributes;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface SliceTest : XCTestCase
@end

@implementation SliceTest
- (void)setUp {
  [super setUp];
}

TensorRef<BHWC> GetTensorRef(int ref) {
  TensorRef<BHWC> tensor_ref;
  tensor_ref.type = DataType::FLOAT32;
  tensor_ref.ref = ref;
  tensor_ref.shape = BHWC(1, 2, 2, 1);
  return tensor_ref;
}

- (void)testReluSmoke {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.clip = 0;
  attr.alpha = 0;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)}, {GetTensorRef(1)});
  XCTAssertTrue(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, 0.0, 2.0, 8.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testReluClipOnly {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.clip = 6;
  attr.alpha = 0;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)}, {GetTensorRef(1)});
  XCTAssertTrue(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, 0.0, 2.0, 6.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testReluAlphaOnly {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.clip = 0;
  attr.alpha = 0.5;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)}, {GetTensorRef(1)});
  XCTAssertTrue(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({-3.0, 0.0, 2.0, 8.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testReluClipAndAlpha {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.clip = 6;
  attr.alpha = 0.5;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)}, {GetTensorRef(1)});
  XCTAssertTrue(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({-3.0, 0.0, 2.0, 6.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
