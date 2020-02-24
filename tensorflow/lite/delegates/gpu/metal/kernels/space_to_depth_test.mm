/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/kernels/space_to_depth.h"

#import <XCTest/XCTest.h>

#include <cmath>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

using ::tflite::gpu::BHWC;
using ::tflite::gpu::DataType;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::SpaceToDepthAttributes;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface SpaceToDepthTest : XCTestCase
@end

@implementation SpaceToDepthTest

- (void)testTensorShape1x2x2x1BlockSize2 {
  const TensorRef<BHWC> input = {.type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 1), .ref = 0};
  const TensorRef<BHWC> output = {.type = DataType::FLOAT32, .shape = BHWC(1, 1, 1, 4), .ref = 1};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  SingleOpModel model({ToString(OperationType::SPACE_TO_DEPTH), attr}, {input}, {output});
  if (!model.PopulateTensor(0, {1.0f, 2.0f, 3.0f, 4.0f})) {
    XCTFail(@"PopulateTensor()");
  }
  const auto status = model.Invoke();
  if (!status.ok()) XCTFail(@"%s", status.error_message().c_str());
  const std::vector<float>& actual = model.GetOutput(0);
  const std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
  XCTAssertEqual(actual[0], expected[0]);
  XCTAssertEqual(actual[1], expected[1]);
  XCTAssertEqual(actual[2], expected[2]);
  XCTAssertEqual(actual[3], expected[3]);
}

- (void)testTensorShape1x2x2x2BlockSize2 {
  const TensorRef<BHWC> input = {.type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 2), .ref = 0};
  const TensorRef<BHWC> output = {.type = DataType::FLOAT32, .shape = BHWC(1, 1, 1, 8), .ref = 1};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  SingleOpModel model({ToString(OperationType::SPACE_TO_DEPTH), attr}, {input}, {output});
  if (!model.PopulateTensor(0, {1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f})) {
    XCTFail(@"PopulateTensor()");
  }
  const auto status = model.Invoke();
  if (!status.ok()) XCTFail(@"%s", status.error_message().c_str());
  const std::vector<float>& actual = model.GetOutput(0);
  const std::vector<float> expected = {1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f};
  XCTAssertEqual(actual[0], expected[0]);
  XCTAssertEqual(actual[1], expected[1]);
  XCTAssertEqual(actual[2], expected[2]);
  XCTAssertEqual(actual[3], expected[3]);
  XCTAssertEqual(actual[4], expected[4]);
  XCTAssertEqual(actual[5], expected[5]);
  XCTAssertEqual(actual[6], expected[6]);
  XCTAssertEqual(actual[7], expected[7]);
}

- (void)testTensorShape1x2x2x3BlockSize2 {
  const TensorRef<BHWC> input = {.type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 3), .ref = 0};
  const TensorRef<BHWC> output = {.type = DataType::FLOAT32, .shape = BHWC(1, 1, 1, 12), .ref = 1};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  SingleOpModel model({ToString(OperationType::SPACE_TO_DEPTH), attr}, {input}, {output});
  if (!model.PopulateTensor(0, {1.0f, 2.0f, 3.0f,  //
                                4.0f, 5.0f, 6.0f,  //
                                7.0f, 8.0f, 9.0f,  //
                                10.0f, 11.0f, 12.0f})) {
    XCTFail(@"PopulateTensor()");
  }
  const auto status = model.Invoke();
  if (!status.ok()) XCTFail(@"%s", status.error_message().c_str());
  const std::vector<float>& actual = model.GetOutput(0);
  const std::vector<float> expected = {1.0f,  2.0f,  3.0f,  //
                                       4.0f,  5.0f,  6.0f,  //
                                       7.0f,  8.0f,  9.0f,  //
                                       10.0f, 11.0f, 12.0f};
  XCTAssertEqual(actual[0], expected[0]);
  XCTAssertEqual(actual[1], expected[1]);
  XCTAssertEqual(actual[2], expected[2]);
  XCTAssertEqual(actual[3], expected[3]);
  XCTAssertEqual(actual[4], expected[4]);
  XCTAssertEqual(actual[5], expected[5]);
  XCTAssertEqual(actual[6], expected[6]);
  XCTAssertEqual(actual[7], expected[7]);
  XCTAssertEqual(actual[8], expected[8]);
  XCTAssertEqual(actual[9], expected[9]);
  XCTAssertEqual(actual[10], expected[10]);
  XCTAssertEqual(actual[11], expected[11]);
}

- (void)testTensorShape1x4x4x1BlockSize2 {
  const TensorRef<BHWC> input = {.type = DataType::FLOAT32, .shape = BHWC(1, 4, 4, 1), .ref = 0};
  const TensorRef<BHWC> output = {.type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 4), .ref = 1};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  SingleOpModel model({ToString(OperationType::SPACE_TO_DEPTH), attr}, {input}, {output});
  if (!model.PopulateTensor(0, {1.0f, 2.0f, 5.0f, 6.0f,     //
                                3.0f, 4.0f, 7.0f, 8.0f,     //
                                9.0f, 10.0f, 13.0f, 14.0f,  //
                                11.0f, 12.0f, 15.0f, 16.0f})) {
    XCTFail(@"PopulateTensor()");
  }
  const auto status = model.Invoke();
  if (!status.ok()) XCTFail(@"%s", status.error_message().c_str());
  const std::vector<float>& actual = model.GetOutput(0);
  const std::vector<float> expected = {1.0f,  2.0f,  3.0f,  4.0f,   //
                                       5.0f,  6.0f,  7.0f,  8.0f,   //
                                       9.0f,  10.0f, 11.0f, 12.0f,  //
                                       13.0f, 14.0f, 15.0f, 16.0f};
  XCTAssertEqual(actual[0], expected[0]);
  XCTAssertEqual(actual[1], expected[1]);
  XCTAssertEqual(actual[2], expected[2]);
  XCTAssertEqual(actual[3], expected[3]);
  XCTAssertEqual(actual[4], expected[4]);
  XCTAssertEqual(actual[5], expected[5]);
  XCTAssertEqual(actual[6], expected[6]);
  XCTAssertEqual(actual[7], expected[7]);
  XCTAssertEqual(actual[8], expected[8]);
  XCTAssertEqual(actual[9], expected[9]);
  XCTAssertEqual(actual[10], expected[10]);
  XCTAssertEqual(actual[11], expected[11]);
  XCTAssertEqual(actual[12], expected[12]);
  XCTAssertEqual(actual[13], expected[13]);
  XCTAssertEqual(actual[14], expected[14]);
  XCTAssertEqual(actual[15], expected[15]);
}

@end
