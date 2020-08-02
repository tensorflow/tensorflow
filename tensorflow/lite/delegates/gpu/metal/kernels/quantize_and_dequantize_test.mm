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
#include "tensorflow/lite/kernels/internal/quantization_util.h"

using ::tflite::NudgeQuantizationRange;
using ::tflite::gpu::DataType;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::QuantizeAndDequantizeAttributes;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

// TODO: Add per-op test if possible.
@interface QuantizeAndDequantizeTest : XCTestCase
@end

@implementation QuantizeAndDequantizeTest
- (void)setUp {
  [super setUp];
}

- (void)testDim2Bits8 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 2, 1);

  // Unlike TFLite's FakeQuant kernel, we assume that the incoming values are
  // pre-nudged, since this should be done during model conversion.
  const int num_bits = 8;
  const int quant_min = 0;
  const int quant_max = (1 << num_bits) - 1;
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ 0.0, /**original_max**/ 1.0, quant_min, quant_max,
                         &attr.min, &attr.max, &attr.scale);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 3, 2, 1);

  SingleOpModel model({ToString(OperationType::QUANTIZE_AND_DEQUANTIZE), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, 1.0, 0.25, 0.50, 0.4444444, 0.00001}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  std::vector<float> expected_output = {0.0f, 1.0f, 0.25098f, 0.498039f, 0.443137f, 0.0f};
  status =
      CompareVectors({0.0f, 1.0f, 0.25098f, 0.498039f, 0.443137f, 0.0f}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDim3Bits8_NegativeRange {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 1, 2);

  // Unlike TFLite's FakeQuant kernel, we assume that the incoming values are
  // pre-nudged, since this should be done during model conversion.
  const int num_bits = 8;
  const int quant_min = 0;
  const int quant_max = (1 << num_bits) - 1;
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ -0.9, /**original_max**/ 0.9, quant_min, quant_max,
                         &attr.min, &attr.max, &attr.scale);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 3, 1, 2);

  SingleOpModel model({ToString(OperationType::QUANTIZE_AND_DEQUANTIZE), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -0.9, 0.25, 0.50, 0.4444444, -0.00001}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0f, -0.896471f, 0.247059f, 0.501176f, 0.444706f, 0.0f},
                          model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDim3Bits16 {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 1, 2);

  // Unlike TFLite's FakeQuant kernel, we assume that the incoming values are
  // pre-nudged, since this should be done during model conversion.
  const int num_bits = 16;
  const int quant_min = 0;
  const int quant_max = (1 << num_bits) - 1;
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ 0.0, /**original_max**/ 1.0, quant_min, quant_max,
                         &attr.min, &attr.max, &attr.scale);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 3, 1, 2);

  SingleOpModel model({ToString(OperationType::QUANTIZE_AND_DEQUANTIZE), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, 1.0, 0.25, 0.50, 0.4444444, 0.00001}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0f, 1.0f, 0.250004f, 0.500008f, 0.44445f, 1.5259e-05f},
                          model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDim2Bits16_NegativeRange {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 2, 1);

  // Unlike TFLite's FakeQuant kernel, we assume that the incoming values are
  // pre-nudged, since this should be done during model conversion.
  const int num_bits = 16;
  const int quant_min = 0;
  const int quant_max = (1 << num_bits) - 1;
  QuantizeAndDequantizeAttributes attr;
  NudgeQuantizationRange(/**original_min**/ -0.9, /**original_max**/ 0.9, quant_min, quant_max,
                         &attr.min, &attr.max, &attr.scale);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 3, 2, 1);

  SingleOpModel model({ToString(OperationType::QUANTIZE_AND_DEQUANTIZE), attr}, {input}, {output});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -0.9, 0.25, 0.50, 0.4444444, -0.00001}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status =
      CompareVectors({0.0f, -0.900014f, 0.249998f, 0.499995f, 0.444431f, 0.0f}, model.GetOutput(0),
                     1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
