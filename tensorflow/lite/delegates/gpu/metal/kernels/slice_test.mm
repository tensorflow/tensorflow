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

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

using ::tflite::gpu::SliceAttributes;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::DataType;
using ::tflite::gpu::HWC;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::OperationType;

@interface SliceTest : XCTestCase
@end

@implementation SliceTest
- (void)setUp {
  [super setUp];
}

- (void)testSliceIdentity {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 2, 2);

  SliceAttributes attr;
  attr.starts = HWC(0, 0, 0);
  attr.ends = HWC(1, 2, 2);
  attr.strides = HWC(1, 1, 1);

  SingleOpModel model({ToString(OperationType::SLICE), attr}, {input},
                      {output});
  XCTAssertTrue(model.PopulateTensor(0, {1, 2, 3, 4}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.ToString().c_str());
  status = CompareVectors({1, 2, 3, 4}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.ToString().c_str());
}

@end
