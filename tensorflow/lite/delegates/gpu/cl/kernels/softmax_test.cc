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

#include "tensorflow/lite/delegates/gpu/common/tasks/softmax.h"

#include <cmath>
#include <cstdlib>
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

TEST_F(OpenCLOperationTest, Softmax) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {std::log(1.0f), std::log(2.0f), std::log(3.0f),
                     std::log(4.0f)};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateSoftmax(op_def);
      ASSERT_OK(ExecuteGPUOperation(
          src_tensor, creation_context_,
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps), {1.0f / 3.0f, 2.0f / 3.0f,
                                             3.0f / 7.0f, 4.0f / 7.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, SoftmaxBigNumber) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  double doubles[4] = {1.0, 2.0, 3.0, 100.0};
  // exp(100) is inf in float (32 bit) but representable in double (64 bit)
  src_tensor.data.resize(4);
  src_tensor.data[0] = doubles[0];
  src_tensor.data[1] = doubles[1];
  src_tensor.data[2] = doubles[2];
  src_tensor.data[3] = doubles[3];
  EXPECT_TRUE(std::isinf(std::exp(src_tensor.data[3])));
  EXPECT_FALSE(std::isinf(std::exp(doubles[3])));
  double s0 = std::exp(doubles[0]) + std::exp(doubles[1]);
  double s1 = std::exp(doubles[2]) + std::exp(doubles[3]);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateSoftmax(op_def);
      ASSERT_OK(ExecuteGPUOperation(
          src_tensor, creation_context_,
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps),
                    {std::exp(doubles[0]) / s0, std::exp(doubles[1]) / s0,
                     std::exp(doubles[2]) / s1, std::exp(doubles[3]) / s1}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
