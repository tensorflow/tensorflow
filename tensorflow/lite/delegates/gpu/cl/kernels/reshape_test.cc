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

#include "tensorflow/lite/delegates/gpu/cl/kernels/reshape.h"

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

TEST_F(OpenCLOperationTest, Reshape) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 3);
  src_tensor.data = {half(0.5f), half(-1.1f), half(-2.2f),
                     half(3.1f), half(1.2f),  half(2.9f)};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      Reshape operation = CreateReshape(op_def);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 3, 1, 2), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(0.0f), {half(0.5f), half(-1.1f), half(-2.2f),
                                      half(3.1f), half(1.2f), half(2.9f)}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
