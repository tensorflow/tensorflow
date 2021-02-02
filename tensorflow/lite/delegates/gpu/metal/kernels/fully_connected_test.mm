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

#import <XCTest/XCTest.h>

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected_test_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

@interface FullyConnectedMetalTest : XCTestCase
@end

@implementation FullyConnectedMetalTest {
  tflite::gpu::metal::MetalExecutionEnvironment exec_env_;
}

namespace tflite {
namespace gpu {
namespace metal {

absl::Status FullyConnectedTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 2);
  src_tensor.data = {1, 2};

  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(4, 1, 1, 2);
  attr.weights.data = {1, 2, 3, 4, 5, 6, 7, 8};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1, 2, 3, 4};

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      FullyConnected operation = CreateFullyConnected(env->GetGpuInfo(), op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<FullyConnected>(std::move(operation)), BHWC(1, 1, 1, 4),
          &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({6, 13, 20, 27}, dst_tensor.data, eps))
          << "Failed using precision " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

- (void)testFullyConnectedMetal {
  auto status = tflite::gpu::metal::FullyConnectedTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testFullyConnected {
  auto status = tflite::gpu::FullyConnectedTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testFullyConnectedLarge {
  auto status = FullyConnectedLargeTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

- (void)testFullyConnectedExtraLarge {
  auto status = FullyConnectedExtraLargeTest(&exec_env_);
  XCTAssertTrue(status.ok(), @"%s", std::string(status.message()).c_str());
}

@end
