/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

TEST_F(OpenCLOperationTest, SliceMulMeanConcat) {
  TestExecutionEnvironment* env = &exec_env_;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {3.0f, 4.0f, 5.0f, 6.0f};

  TensorFloat32 weights_tensor;
  weights_tensor.shape = BHWC(1, 2, 1, 2);
  weights_tensor.data = {1.0f, 2.0f, 1.0f, 2.0f};

  ConvPointwiseAttributes op_attr;
  op_attr.mean = true;
  // Offset dictates the "start" field for the dimensions (width, height)
  // in the slice operation.
  op_attr.offsets.push_back(int2(0, 0));

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateConvPointwise(op_def, op_attr);
      ASSERT_OK(env->ExecuteGPUOperation(
          {src_tensor, weights_tensor},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      ASSERT_OK(PointWiseNear({5.5f, 5.5f, 8.5f, 8.5f}, dst_tensor.data, eps));
    }
  }
}

TEST_F(OpenCLOperationTest, SliceMulSumConcat) {
  TestExecutionEnvironment* env = &exec_env_;
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {3.0f, 4.0f, 5.0f, 6.0f};

  TensorFloat32 weights_tensor;
  weights_tensor.shape = BHWC(1, 2, 1, 2);
  weights_tensor.data = {1.0f, 2.0f, 1.0f, 2.0f};

  ConvPointwiseAttributes op_attr;
  op_attr.mean = false;
  op_attr.offsets.push_back(int2(0, 0));

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateConvPointwise(op_def, op_attr);
      ASSERT_OK(env->ExecuteGPUOperation(
          {src_tensor, weights_tensor},
          std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 2), &dst_tensor));
      ASSERT_OK(
          PointWiseNear({11.0f, 11.0f, 17.0f, 17.0f}, dst_tensor.data, eps));
    }
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
