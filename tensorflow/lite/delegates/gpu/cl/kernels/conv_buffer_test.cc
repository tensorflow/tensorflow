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

#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer.h"

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

TEST_F(OpenCLOperationTest, ConvBufferSimpleWeights) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(1, 2, 2, 2);
  attr.weights.data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.0f};

  for (auto precision : env_.GetSupportedPrecisions()) {
    const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
    OperationDef op_def;
    op_def.precision = precision;
    auto data_type = DeduceDataTypeFromPrecision(precision);
    op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER});
    op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER});
    TensorFloat32 dst_tensor;
    ConvBuffer operation;
    ASSERT_OK(CreateConvBuffer(creation_context_, op_def, attr, &operation));
    ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                  BHWC(1, 2, 2, 1), &dst_tensor));
    EXPECT_THAT(dst_tensor.data,
                Pointwise(FloatNear(eps), {28.0f, 18.0f, 22.0f, 13.0f}));
  }
}

TEST_F(OpenCLOperationTest, ConvBuffer) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  Convolution2DAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 1);
  attr.strides = HW(1, 1);
  attr.dilations = HW(1, 1);
  attr.weights.shape = OHWI(2, 2, 2, 2);
  attr.weights.data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.5f, -0.5f};

  for (auto precision : env_.GetSupportedPrecisions()) {
    const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
    OperationDef op_def;
    op_def.precision = precision;
    auto data_type = DeduceDataTypeFromPrecision(precision);
    op_def.src_tensors.push_back({data_type, TensorStorageType::BUFFER});
    op_def.dst_tensors.push_back({data_type, TensorStorageType::BUFFER});
    TensorFloat32 dst_tensor;
    ConvBuffer operation;
    ASSERT_OK(CreateConvBuffer(creation_context_, op_def, attr, &operation));
    ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                  BHWC(1, 2, 2, 2), &dst_tensor));
    EXPECT_THAT(dst_tensor.data,
                Pointwise(FloatNear(eps), {168.5f, 391.5f, 80.5f, 223.5f, 60.5f,
                                           235.5f, 20.5f, 123.5f}));
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
