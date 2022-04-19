/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/resize_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/resize.h"

namespace tflite {
namespace gpu {

absl::Status ResizeBilinearAlignedTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 3, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(4, 4);
  attr.align_corners = true;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Resize>(std::move(operation)),
          BHWC(1, 4, 4, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {0.0f, 0.666667f, 1.33333f, 2.0f, 1.0f, 1.66667f, 2.33333f, 3.0f,
           2.0f, 2.66667f, 3.33333f, 4.0f, 3.0f, 3.66667f, 4.33333f, 5.0f},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ResizeBilinearNonAlignedTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 3, 1);
  src_tensor.data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(4, 4);
  attr.align_corners = false;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Resize>(std::move(operation)),
          BHWC(1, 4, 4, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({0.0f, 0.75f, 1.5f, 2.0f, 1.5f, 2.25f, 3.0f, 3.5f, 3.0f,
                         3.75f, 4.5f, 5.0f, 3.0f, 3.75f, 4.5f, 5.0f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ResizeBilinearWithoutHalfPixelTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(3, 3);
  attr.align_corners = false;
  attr.half_pixel_centers = false;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Resize>(std::move(operation)),
          BHWC(1, 3, 3, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({1.0f, 1.666666f, 2.0f, 2.333333f, 3.0f,
                                     3.333333f, 3.0f, 3.666666f, 4.0f},
                                    dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ResizeBilinearWithHalfPixelTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};

  Resize2DAttributes attr;
  attr.type = SamplingType::BILINEAR;
  attr.new_shape = HW(3, 3);
  attr.align_corners = false;
  attr.half_pixel_centers = true;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Resize>(std::move(operation)),
          BHWC(1, 3, 3, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 1.5f, 2.0f, 2.0f, 2.5f, 3.0f, 3.0f, 3.5f, 4.0f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ResizeNearestTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 1);
  src_tensor.data = {1.0f, 2.0f};

  Resize2DAttributes attr;
  attr.align_corners = false;
  attr.half_pixel_centers = false;
  attr.new_shape = HW(2, 4);
  attr.type = SamplingType::NEAREST;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Resize>(std::move(operation)),
          BHWC(1, 2, 4, 1), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f},
                        dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ResizeNearestAlignCornersTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {3.0f, 6.0f, 9.0f, 12.0f};

  Resize2DAttributes attr;
  attr.align_corners = true;
  attr.half_pixel_centers = false;
  attr.new_shape = HW(3, 3);
  attr.type = SamplingType::NEAREST;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Resize>(std::move(operation)),
          BHWC(1, 3, 3, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {3.0f, 6.0f, 6.0f, 9.0f, 12.0f, 12.0f, 9.0f, 12.0f, 12.0f},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ResizeNearestHalfPixelCentersTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {3.0f, 6.0f, 9.0f, 12.0f};

  Resize2DAttributes attr;
  attr.align_corners = false;
  attr.half_pixel_centers = true;
  attr.new_shape = HW(3, 3);
  attr.type = SamplingType::NEAREST;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-5f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Resize operation = CreateResize(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<Resize>(std::move(operation)),
          BHWC(1, 3, 3, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {3.0f, 6.0f, 6.0f, 9.0f, 12.0f, 12.0f, 9.0f, 12.0f, 12.0f},
          dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
