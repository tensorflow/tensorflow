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

#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed.h"

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

TEST_F(OpenCLOperationTest, ConvolutionTransposedSimpleWeights) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(2, 2, 2, 2);
  attr.weights.data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {0.0f, 0.0f};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      ConvolutionTransposed operation;
      ASSERT_OK(CreateConvolutionTransposed(creation_context_, op_def, attr,
                                            &operation));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 4, 4, 2), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps),
                    {1.0f, 1.0f, 1.0f, 1.0f, 5.0f,  5.0f,  5.0f,  5.0f,
                     1.0f, 1.0f, 1.0f, 1.0f, 5.0f,  5.0f,  5.0f,  5.0f,
                     9.0f, 9.0f, 9.0f, 9.0f, 13.0f, 13.0f, 13.0f, 13.0f,
                     9.0f, 9.0f, 9.0f, 9.0f, 13.0f, 13.0f, 13.0f, 13.0f}));
    }
  }
}

TEST_F(OpenCLOperationTest, ConvolutionTransposed) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

  ConvolutionTransposedAttributes attr;
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.stride = HW(2, 2);
  attr.weights.shape = OHWI(1, 2, 2, 2);
  attr.weights.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  attr.bias.shape = Linear(1);
  attr.bias.data = {0.5f};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 dst_tensor;
      ConvolutionTransposed operation;
      ASSERT_OK(CreateConvolutionTransposed(creation_context_, op_def, attr,
                                            &operation));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &operation,
                                    BHWC(1, 4, 4, 1), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(FloatNear(eps),
                    {2.5f, 4.5f, 8.5f, 18.5f, 6.5f, 8.5f, 28.5f, 38.5f, 14.5f,
                     32.5f, 20.5f, 46.5f, 50.5f, 68.5f, 72.5f, 98.5f}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
