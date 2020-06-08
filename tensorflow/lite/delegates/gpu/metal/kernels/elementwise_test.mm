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

using ::tflite::gpu::DataType;
using ::tflite::gpu::HWC;
using ::tflite::gpu::BHWC;
using ::tflite::gpu::OperationType;
using ::tflite::gpu::TensorRef;
using ::tflite::gpu::metal::CompareVectors;
using ::tflite::gpu::metal::SingleOpModel;

@interface ElementwiseTest : XCTestCase
@end

@implementation ElementwiseTest
- (void)setUp {
  [super setUp];
}

TensorRef<BHWC> GetTensorRef(int ref, const BHWC& shape) {
  TensorRef<BHWC> tensor_ref;
  tensor_ref.type = DataType::FLOAT32;
  tensor_ref.ref = ref;
  tensor_ref.shape = shape;
  return tensor_ref;
}

- (void)testAbs {
  OperationType op_type = OperationType::ABS;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.2, 2.0, 4.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, 6.2, 2.0, 4.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testCos {
  OperationType op_type = OperationType::COS;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, 3.1415926, -3.1415926, 1}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1.0, -1.0, -1.0, 0.540302}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testDiv {
  OperationType op_type = OperationType::DIV;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
                      /*outputs=*/{GetTensorRef(2, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.2, 2.0, 4.0}));
  XCTAssertTrue(model.PopulateTensor(1, {1.0, 2.0, -0.5, 4.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, -3.1, -4.0, 1.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testExp {
  OperationType op_type = OperationType::EXP;
  const BHWC shape(1, 1, 1, 7);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 0.01f, -0.01f}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f), std::exp(100.0f),
                           std::exp(-100.0f), std::exp(0.01f), std::exp(-0.01f)},
                          model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testHardSwish {
  OperationType op_type = OperationType::HARD_SWISH;
  const BHWC shape(1, 1, 1, 7);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status =
      CompareVectors({0.0f, 0.0f, -0.375f, 0.0f, 1.125f, 3.f, 4.5f}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testLog {
  OperationType op_type = OperationType::LOG;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {1.0, 3.1415926, 1.0, 1.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, 1.14473, 0.0, 0.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMaximum {
  OperationType op_type = OperationType::MAXIMUM;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
                      /*outputs=*/{GetTensorRef(2, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.2, 2.0, -3.0}));
  XCTAssertTrue(model.PopulateTensor(1, {1.0, 2.0, 3.0, -2.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1.0, 2.0, 3.0, -2.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMaximumWithScalar {
  OperationType op_type = OperationType::MAXIMUM;
  const BHWC shape(1, 2, 2, 1);
  tflite::gpu::ElementwiseAttributes attr;
  attr.param = -1.0f;
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/attr},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.2, 2.0, -3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, -1.0, 2.0, -1.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMaximumWithConstantHWCTensor {
  OperationType op_type = OperationType::MAXIMUM;
  const BHWC shape(1, 2, 1, 2);
  tflite::gpu::ElementwiseAttributes attr;
  tflite::gpu::Tensor<HWC, DataType::FLOAT32> hwc_tensor;
  hwc_tensor.shape = HWC(2, 1, 2);
  hwc_tensor.data = {0.5f, 2.0f, 0.7f, 4.7f};
  attr.param = hwc_tensor;
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/attr},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {1.0f, -6.2f, -2.0f, 3.0f}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1.0f, 2.0f, 0.7f, 4.7f}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMaximumWithConstantHWCTensorBroadcastChannels {
  OperationType op_type = OperationType::MAXIMUM;
  const BHWC shape(1, 2, 1, 2);
  tflite::gpu::ElementwiseAttributes attr;
  tflite::gpu::Tensor<HWC, DataType::FLOAT32> hwc_tensor;
  hwc_tensor.shape = HWC(2, 1, 1);
  hwc_tensor.data = {0.5f, 2.0f};
  attr.param = hwc_tensor;
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/attr},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {1.0f, -6.2f, -2.0f, 3.0f}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1.0f, 0.5f, 2.0f, 3.0f}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMinimum {
  OperationType op_type = OperationType::MINIMUM;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
                      /*outputs=*/{GetTensorRef(2, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.2, 2.0, -3.0}));
  XCTAssertTrue(model.PopulateTensor(1, {1.0, 2.0, 3.0, -2.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, -6.2, 2.0, -3.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMinimumWithScalar {
  OperationType op_type = OperationType::MINIMUM;
  const BHWC shape(1, 2, 2, 1);
  tflite::gpu::ElementwiseAttributes attr;
  attr.param = -1.0f;
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/attr},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.2, 2.0, -3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({-1.0, -6.2, -1.0, -3.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testPow {
  OperationType op_type = OperationType::POW;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
                      /*outputs=*/{GetTensorRef(2, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, 1.0, 2.0, 4.0}));
  XCTAssertTrue(model.PopulateTensor(1, {1.0, 2.0, 3.0, 4.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, 1.0, 8.0, 256.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testRsqrt {
  OperationType op_type = OperationType::RSQRT;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {1.0, 2.0, 4.0, 9.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1.0, 0.707106, 0.5, 0.333333}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSigmoid {
  OperationType op_type = OperationType::SIGMOID;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.0, 2.0, 4.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.5, 0.002473, 0.880797, 0.982014}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSin {
  OperationType op_type = OperationType::SIN;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, 3.1415926, -3.1415926, 1.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, 0.0, 0.0, 0.841471}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSqrt {
  OperationType op_type = OperationType::SQRT;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, 1.0, 2.0, 4.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, 1.0, 1.414213, 2.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSquare {
  OperationType op_type = OperationType::SQUARE;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {1.0, 2.0, 0.5, -3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1.0, 4.0, 0.25, 9.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSquaredDiff {
  OperationType op_type = OperationType::SQUARED_DIFF;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
                      /*outputs=*/{GetTensorRef(2, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, 2.0, 2.0, 4.0}));
  XCTAssertTrue(model.PopulateTensor(1, {1.0, 1.0, 5.0, 4.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({1.0, 1.0, 9.0, 0.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testSub {
  OperationType op_type = OperationType::SUB;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
                      /*outputs=*/{GetTensorRef(2, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.2, 2.0, 4.0}));
  XCTAssertTrue(model.PopulateTensor(1, {1.0, 2.0, 3.0, 4.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({-1.0, -8.2, -1.0, 0.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testTanh {
  OperationType op_type = OperationType::TANH;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {0.0, -6.0, 2.0, 4.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({0.0, -0.999987, 0.964027, 0.999329}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMulBroadcastChannels {
  OperationType op_type = OperationType::MUL;
  const BHWC shape(1, 1, 2, 2);
  const BHWC shape_2(1, 1, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape_2)},
                      /*outputs=*/{GetTensorRef(2, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {1.0, 2.0, 3.0, 4.0}));
  XCTAssertTrue(model.PopulateTensor(1, {2.0, 3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({2.0, 4.0, 9.0, 12.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testMulBroadcastWidthAndHeight {
  OperationType op_type = OperationType::MUL;
  const BHWC shape(1, 1, 2, 2);
  const BHWC shape_2(1, 1, 1, 2);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape_2)},
                      /*outputs=*/{GetTensorRef(2, shape)});
  XCTAssertTrue(model.PopulateTensor(0, {1.0, 2.0, 3.0, 4.0}));
  XCTAssertTrue(model.PopulateTensor(1, {2.0, 3.0}));
  auto status = model.Invoke();
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
  status = CompareVectors({2.0, 6.0, 6.0, 12.0}, model.GetOutput(0), 1e-6f);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
