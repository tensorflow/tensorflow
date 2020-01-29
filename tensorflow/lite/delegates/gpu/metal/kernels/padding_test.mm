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

using ::tflite::gpu::BHWC;
using ::tflite::gpu::DataType;
using ::tflite::gpu::HWC;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::PadAttributes;
using ::tflite::gpu::PaddingContentType;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface PaddingTest : XCTestCase
- (void)runPadOperation:(const HWC&)prepend
                 append:(const HWC&)append
           output_shape:(const BHWC&)output_shape
               expected:(std::vector<float>&&)expected;
- (void)runPrepending:(const HWC&)prepend
         output_shape:(const BHWC&)output_shape
             expected:(std::vector<float>&&)expected;
- (void)runAppending:(const HWC&)append
        output_shape:(const BHWC&)output_shape
            expected:(std::vector<float>&&)expected;
@end

@implementation PaddingTest
- (void)setUp {
  [super setUp];
}

- (void)runPadOperation:(const HWC&)prepend
                 append:(const HWC&)append
           output_shape:(const BHWC&)output_shape
               expected:(std::vector<float>&&)expected {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = output_shape;

  PadAttributes attr;
  attr.prepended = BHWC(0, prepend.h, prepend.w, prepend.c);
  attr.appended = BHWC(0, append.h, append.w, append.c);
  attr.type = PaddingContentType::ZEROS;

  SingleOpModel model({ToString(OperationType::PAD), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors(expected, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)runPrepending:(const HWC&)prepend
         output_shape:(const BHWC&)output_shape
             expected:(std::vector<float>&&)expected {
  [self runPadOperation:prepend
                 append:HWC(0, 0, 0)
           output_shape:output_shape
               expected:std::move(expected)];
}

- (void)runAppending:(const HWC&)append
        output_shape:(const BHWC&)output_shape
            expected:(std::vector<float>&&)expected {
  [self runPadOperation:HWC(0, 0, 0)
                 append:append
           output_shape:output_shape
               expected:std::move(expected)];
}

- (void)testPadPrependH {
  [self runPrepending:HWC(1, 0, 0) output_shape:BHWC(1, 2, 1, 1) expected:{0, 1}];
}

- (void)testPadPrependW {
  [self runPrepending:HWC(0, 1, 0) output_shape:BHWC(1, 1, 2, 1) expected:{0, 1}];
}

- (void)testPadPrependC {
  [self runPrepending:HWC(0, 0, 1) output_shape:BHWC(1, 1, 1, 2) expected:{0, 1}];
}

- (void)testPadPrependCx4 {
  [self runPrepending:HWC(0, 0, 4) output_shape:BHWC(1, 1, 1, 5) expected:{0, 0, 0, 0, 1}];
}

- (void)testPadPrependHWC {
  [self runPrepending:HWC(1, 1, 1) output_shape:BHWC(1, 2, 2, 2) expected:{0, 0, 0, 0, 0, 0, 0, 1}];
}

- (void)testPadAppendH {
  [self runAppending:HWC(1, 0, 0) output_shape:BHWC(1, 2, 1, 1) expected:{1, 0}];
}

- (void)testPadAppendW {
  [self runAppending:HWC(0, 1, 0) output_shape:BHWC(1, 1, 2, 1) expected:{1, 0}];
}

- (void)testPadAppendC {
  [self runAppending:HWC(0, 0, 1) output_shape:BHWC(1, 1, 1, 2) expected:{1, 0}];
}

- (void)testPadAppendHWC {
  [self runAppending:HWC(1, 1, 1) output_shape:BHWC(1, 2, 2, 2) expected:{1, 0, 0, 0, 0, 0, 0, 0}];
}

- (void)testPadPrependHWCAppendHWC {
  [self runPadOperation:HWC(1, 1, 1)
                 append:HWC(1, 1, 1)
           output_shape:BHWC(1, 3, 3, 3)
               expected:{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}];
}

- (void)testMirrorPadWidthOperation {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 3, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 7, 1);

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 2, 0);
  attr.appended = BHWC(0, 0, 2, 0);
  attr.type = PaddingContentType::REFLECT;

  SingleOpModel model({ToString(OperationType::PAD), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1.0, 2.0, 3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}

- (void)testMirrorPadChannelsOperation {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 3);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 1, 7);

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 0, 2);
  attr.appended = BHWC(0, 0, 0, 2);
  attr.type = PaddingContentType::REFLECT;

  SingleOpModel model({ToString(OperationType::PAD), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {1.0, 2.0, 3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
  status = CompareVectors({3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", status.error_message().c_str());
}


@end
