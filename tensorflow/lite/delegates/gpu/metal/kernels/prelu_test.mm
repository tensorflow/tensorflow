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
using ::tflite::gpu::HWC;
using ::tflite::gpu::Linear;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::PReLUAttributes;
using ::tflite::gpu::Tensor;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface SoftmaxTest : XCTestCase
@end

@implementation SoftmaxTest
- (void)setUp {
  [super setUp];
}

- (void)testPReluLinearAlphaNoClip {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  PReLUAttributes attr;
  attr.clip = 0;
  Tensor<Linear, DataType::FLOAT32> alpha;
  alpha.shape.v = 1;
  alpha.id = 1;
  alpha.data = {2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(OperationType::PRELU), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {-1.0, -2.0, 1.0, 2.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({-2, -4, 1, 2}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testPReluLinearAlphaWithClip {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  PReLUAttributes attr;
  attr.clip = 1.0;
  Tensor<Linear, DataType::FLOAT32> alpha;
  alpha.shape.v = 1;
  alpha.id = 1;
  alpha.data = {2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(OperationType::PRELU), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {-1.0, -2.0, 1.0, 2.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({-2, -4, 1, 1}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testPRelu3DAlphaNoClip {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  OperationType op_type = OperationType::PRELU;
  PReLUAttributes attr;
  attr.clip = 0;
  Tensor<HWC, DataType::FLOAT32> alpha;
  alpha.shape = HWC(2, 2, 1);
  alpha.id = 1;
  alpha.data = {1, 2, 2, 2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(op_type), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -1.0, 2.0, -3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({0, -2, 2, -6}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testPRelu3DAlphaWithClip {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  OperationType op_type = OperationType::PRELU;
  PReLUAttributes attr;
  attr.clip = 1.0;
  Tensor<HWC, DataType::FLOAT32> alpha;
  alpha.shape = HWC(2, 2, 1);
  alpha.id = 1;
  alpha.data = {1, 2, 2, 2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(op_type), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -1.0, 2.0, -3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({0, -2, 1, -6}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

@end
