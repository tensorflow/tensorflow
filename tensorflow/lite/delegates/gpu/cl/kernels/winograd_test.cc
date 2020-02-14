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

#include "tensorflow/lite/delegates/gpu/cl/kernels/winograd.h"

#include <cmath>
#include <cstdlib>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, Winograd4x4To36) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 4, 4, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                     8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Padding2D padding;
      padding.prepended = HW(1, 1);
      padding.appended = HW(1, 1);
      Winograd4x4To36 wino_up;
      ASSERT_OK(
          CreateWinograd4x4To36(creation_context_, op_def, padding, &wino_up));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &wino_up,
                                    BHWC(1, 36, 1, 1), &dst_tensor));
      EXPECT_THAT(dst_tensor.data,
                  Pointwise(FloatNear(eps),
                            {-1.8076144457f, 3.0488157272f,   -0.3543013334f,
                             -0.9567713737f, 0.0698715150f,   6.3601350784f,
                             7.9091277122f,  -7.5317668915f,  -0.4988912344f,
                             0.0400028825f,  0.0815277994f,   1.8058515787f,
                             -2.0690131187f, 1.4405870438f,   0.3173895180f,
                             0.3676810265f,  -0.0566446260f,  -3.1750767231f,
                             -4.4264192581f, 3.3195235729f,   0.5952118039f,
                             0.6170299053f,  -0.1053467616f,  -5.5806870461f,
                             0.3939223289f,  -0.2771621346f,  -0.0594099388f,
                             -0.0679424182f, 0.0105922129f,   0.5897778869f,
                             31.1582794189f, -22.9188480377f, -4.3477787971f,
                             -4.6630558968f, 0.7714096308f,   41.5681838989f}));
    }
  }
}

TEST_F(OpenCLOperationTest, Winograd36To4x4) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 36, 1, 1);
  src_tensor.data.resize(36);
  for (int i = 0; i < 36; ++i) {
    src_tensor.data[i] = sin(i);
  }

  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(1);
  biases.data.resize(biases.shape.DimensionsProduct());
  for (int i = 0; i < biases.data.size(); ++i) {
    biases.data[i] = 0.0f;
  }

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Winograd36To4x4 wino_down;
      ASSERT_OK(
          CreateWinograd36To4x4(creation_context_, op_def, biases, &wino_down));
      ASSERT_OK(ExecuteGPUOperation(src_tensor, creation_context_, &wino_down,
                                    BHWC(1, 4, 4, 1), &dst_tensor));
      EXPECT_THAT(
          dst_tensor.data,
          Pointwise(
              FloatNear(eps),
              {5.6982488632f, 4.4291338921f, 7.1398024559f, 8.3108062744f,
               0.2751901150f, 0.6380079389f, -1.6235249043f, 0.6435587406f,
               5.8707995415f, 3.3895490170f, 12.8032960892f, 7.8921923637f,
               1.2864947319f, 1.1310911179f, 1.0033880472f, 1.9512135983f}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
