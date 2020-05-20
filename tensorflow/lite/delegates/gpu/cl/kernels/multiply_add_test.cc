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

#include "tensorflow/lite/delegates/gpu/cl/kernels/multiply_add.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, MultiplyAddVectorMul) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  MultiplyAttributes attr;
  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> parameters;
  parameters.shape = Linear(2);
  parameters.data = {0.5f, 2.0f};
  attr.param = parameters;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      MultiplyAdd operation;
      ASSERT_OK(CreateMultiplyAdd(creation_context_, op_def, attr, &operation));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps), {0.0f, 2.0f, 1.0f, 6.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, MultiplyAddVectorAdd) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  AddAttributes attr;
  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> parameters;
  parameters.shape = Linear(2);
  parameters.data = {0.5f, 2.0f};
  attr.param = parameters;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      MultiplyAdd operation;
      ASSERT_OK(CreateMultiplyAdd(creation_context_, op_def, attr, &operation));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps), {0.5f, 3.0f, 2.5f, 5.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, MultiplyAddScalarMul) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  MultiplyAttributes attr;
  attr.param = 0.5f;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      MultiplyAdd operation;
      ASSERT_OK(CreateMultiplyAdd(creation_context_, op_def, attr, &operation));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps), {0.0f, 0.5f, 1.0f, 1.5f}));
    }
  }
}

TEST_F(OpenCLOperationTest, MultiplyAddScalarAdd) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  AddAttributes attr;
  attr.param = -0.5f;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      MultiplyAdd operation;
      ASSERT_OK(CreateMultiplyAdd(creation_context_, op_def, attr, &operation));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps), {-0.5f, 0.5f, 1.5f, 2.5f}));
    }
  }
}

TEST_F(OpenCLOperationTest, MultiplyAddVectorMad) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f};

  MultiplyAttributes mul_attr;
  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> parameters;
  parameters.shape = Linear(2);
  parameters.data = {0.5f, 2.0f};
  mul_attr.param = parameters;

  AddAttributes add_attr;
  parameters.data = {-0.5f, 0.5f};
  add_attr.param = parameters;

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      MultiplyAdd operation;
      ASSERT_OK(CreateMultiplyAdd(creation_context_, op_def, mul_attr, add_attr,
                                  &operation));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 2, 1, 2), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps), {-0.5f, 2.5f, 0.5f, 6.5f}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
