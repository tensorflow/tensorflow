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

#include "tensorflow/lite/delegates/gpu/cl/kernels/strided_slice.h"

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

TEST_F(OpenCLOperationTest, StridedSlice) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 2, 4);
  src_tensor.data = {half(0.1f),  half(0.2f),  half(0.3f),  half(0.4),
                     half(1.1f),  half(1.2f),  half(1.3f),  half(1.4),
                     half(10.1f), half(10.2f), half(10.3f), half(10.4),
                     half(11.1f), half(11.2f), half(11.3f), half(11.4),
                     half(20.1f), half(20.2f), half(20.3f), half(20.4),
                     half(21.1f), half(21.2f), half(21.3f), half(21.4)};

  SliceAttributes attr;
  attr.starts = BHWC(0, 1, 0, 1);
  attr.ends = BHWC(src_tensor.shape.b, 2, 2, 3);
  attr.strides = BHWC(1, 1, 2, 2);

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      StridedSlice operation = CreateStridedSlice(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(0.0f), {half(10.2f), half(10.4),
                                              half(20.2f), half(20.4)}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
