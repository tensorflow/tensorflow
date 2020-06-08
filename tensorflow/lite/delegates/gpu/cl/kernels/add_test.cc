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

#include "tensorflow/lite/delegates/gpu/cl/kernels/add.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, AddTwoEqualTensors) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {0.0f, -1.0f, -0.05f, 0.045f};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {0.0f, 1.0f, -0.05f, -0.045f};
  std::vector<int> channels = {2, 2};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Add operation = CreateAdd(op_def, channels, channels[0]);
      ASSERT_OK(ExecuteGPUOperation({src0, src1}, creation_context_, &operation,
                                    BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps), {0.0f, 0.0f, -0.1f, 0.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, AddFirstTensorHasMoreChannelsThanSecond) {
  TensorFloat32 src0, src1;
  src0.shape = BHWC(1, 2, 1, 6);
  src0.data = {0.0f,   -1.0f,  -0.05f, 0.045f, 1.0f,   -2.0f,
               -1.05f, 1.045f, 2.0f,   -3.0f,  -2.05f, 2.045f};
  src1.shape = BHWC(1, 2, 1, 2);
  src1.data = {0.0f, 1.0f, -0.05f, -0.045f};
  std::vector<int> channels = {6, 2};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Add operation = CreateAdd(op_def, channels, channels[0]);
      ASSERT_OK(ExecuteGPUOperation({src0, src1}, creation_context_, &operation,
                                    BHWC(1, 2, 1, 6), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps),
                            {0.0f, 0.0f, -0.05f, 0.045f, 1.0f, -2.0f, -1.1f,
                             1.0f, 2.0f, -3.0f, -2.05f, 2.045f}));
    }
  }
}

TEST_F(OpenCLOperationTest, AddFirstTensorHasLessChannelsThanSecond) {
  TensorFloat32 src0, src1;
  src1.shape = BHWC(1, 2, 1, 6);
  src1.data = {0.0f,   -1.0f,  -0.05f, 0.045f, 1.0f,   -2.0f,
               -1.05f, 1.045f, 2.0f,   -3.0f,  -2.05f, 2.045f};
  src0.shape = BHWC(1, 2, 1, 2);
  src0.data = {0.0f, 1.0f, -0.05f, -0.045f};
  std::vector<int> channels = {2, 6};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Add operation = CreateAdd(op_def, channels, 6);
      ASSERT_OK(ExecuteGPUOperation({src0, src1}, creation_context_, &operation,
                                    BHWC(1, 2, 1, 6), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps),
                            {0.0f, 0.0f, -0.05f, 0.045f, 1.0f, -2.0f, -1.1f,
                             1.0f, 2.0f, -3.0f, -2.05f, 2.045f}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
