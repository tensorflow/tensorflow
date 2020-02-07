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

#include "tensorflow/lite/delegates/gpu/cl/kernels/resize.h"

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

TEST_F(OpenCLOperationTest, ResizeBilinearAligned) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 3, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(4, 4);
  attr.align_corners = true;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 4, 4, 1), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps),
                            {0.0f, 0.666667f, 1.33333f, 2.0f, 1.0f, 1.66667f,
                             2.33333f, 3.0f, 2.0f, 2.66667f, 3.33333f, 4.0f,
                             3.0f, 3.66667f, 4.33333f, 5.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, ResizeBilinearNonAligned) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 3, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(4, 4);
  attr.align_corners = false;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 4, 4, 1), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps),
                    {0.0f, 0.75f, 1.5f, 2.0f, 1.5f, 2.25f, 3.0f, 3.5f, 3.0f,
                     3.75f, 4.5f, 5.0f, 3.0f, 3.75f, 4.5f, 5.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, ResizeNearest) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 1);
  src_tensor.data = {1.0f, 2.0f};

  Resize2DAttributes attr;
  attr.align_corners = false;
  attr.new_shape = HW(2, 4);
  attr.type = SamplingType::NEAREST;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 4, 1), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps),
                            {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
