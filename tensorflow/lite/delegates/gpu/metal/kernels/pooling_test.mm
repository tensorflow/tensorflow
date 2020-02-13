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
using ::tflite::gpu::HW;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::Pooling2DAttributes;
using ::tflite::gpu::PoolingType;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface PoolingTest : XCTestCase
@end

@implementation PoolingTest
- (void)setUp {
  [super setUp];
}

- (void)testPoolingMaxKernel2x2Stride2x2WithIndices {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 4, 4, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> indices;
  indices.type = DataType::INT32;
  indices.ref = 2;
  indices.shape = BHWC(1, 2, 2, 1);

  Pooling2DAttributes attr;
  attr.kernel = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.type = PoolingType::MAX;
  attr.output_indices = true;

  SingleOpModel model({ToString(OperationType::POOLING_2D), attr}, {input}, {output, indices});
  XCTAssertTrue(model.PopulateTensor(0, {1, 2, 1, 2, 3, 4, 3, 4, 7, 8, 7, 8, 5, 6, 5, 6}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({4, 4, 8, 8}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({3, 3, 1, 1}, model.GetOutput(1), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testPoolingMaxKernel2x2Stride2x2WithoutIndices {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 4, 4, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  Pooling2DAttributes attr;
  attr.kernel = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.type = PoolingType::MAX;

  SingleOpModel model({ToString(OperationType::POOLING_2D), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 2, 1, 2, 3, 4, 3, 4, 7, 8, 7, 8, 5, 6, 5, 6}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({4, 4, 8, 8}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testPoolingAverageKernel2x2Stride2x2 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 4, 4, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  Pooling2DAttributes attr;
  attr.kernel = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.type = PoolingType::AVERAGE;

  SingleOpModel model({ToString(OperationType::POOLING_2D), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({1, 2, 3, 4}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

@end
