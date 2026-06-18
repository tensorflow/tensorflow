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

#include "tensorflow/lite/delegates/gpu/common/tasks/tile_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/tile.h"

namespace tflite {
namespace gpu {

absl::Status TileChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 3);
  src_tensor.data = {half(1.0f), half(2.0f), half(3.0f),
                     half(4.0f), half(5.0f), half(6.0f)};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateTile(op_def, src_tensor.shape.c);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 6), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.0f), half(2.0f), half(3.0f), half(1.0f),
                         half(2.0f), half(3.0f), half(4.0f), half(5.0f),
                         half(6.0f), half(4.0f), half(5.0f), half(6.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TileChannelsX4Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 4);
  src_tensor.data = {half(1.0f), half(2.0f), half(3.0f), half(7.0f),
                     half(4.0f), half(5.0f), half(6.0f), half(8.0f)};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateTile(op_def, src_tensor.shape.c);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 1, 8), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.0f), half(2.0f), half(3.0f), half(7.0f),
                         half(1.0f), half(2.0f), half(3.0f), half(7.0f),
                         half(4.0f), half(5.0f), half(6.0f), half(8.0f),
                         half(4.0f), half(5.0f), half(6.0f), half(8.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TileWidthTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 2, 3);
  src_tensor.data = {half(1.0f), half(2.0f), half(3.0f),
                     half(4.0f), half(5.0f), half(6.0f)};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateTile(op_def, src_tensor.shape.c);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 4, 3), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.0f), half(2.0f), half(3.0f), half(4.0f),
                         half(5.0f), half(6.0f), half(1.0f), half(2.0f),
                         half(3.0f), half(4.0f), half(5.0f), half(6.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TileHeightTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 3);
  src_tensor.data = {half(1.0f), half(2.0f), half(3.0f),
                     half(4.0f), half(5.0f), half(6.0f)};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateTile(op_def, src_tensor.shape.c);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 4, 1, 3), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.0f), half(2.0f), half(3.0f), half(4.0f),
                         half(5.0f), half(6.0f), half(1.0f), half(2.0f),
                         half(3.0f), half(4.0f), half(5.0f), half(6.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status TileHWCTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 3);
  src_tensor.data = {half(1.0f), half(2.0f),  half(3.0f),  half(4.0f),
                     half(5.0f), half(6.0f),  half(7.0f),  half(8.0f),
                     half(9.0f), half(10.0f), half(11.0f), half(12.0f)};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateTile(op_def, src_tensor.shape.c);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 4, 4, 6), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {half(1.0f),  half(2.0f),  half(3.0f),  half(1.0f),  half(2.0f),
           half(3.0f),  half(4.0f),  half(5.0f),  half(6.0f),  half(4.0f),
           half(5.0f),  half(6.0f),  half(1.0f),  half(2.0f),  half(3.0f),
           half(1.0f),  half(2.0f),  half(3.0f),  half(4.0f),  half(5.0f),
           half(6.0f),  half(4.0f),  half(5.0f),  half(6.0f),  half(7.0f),
           half(8.0f),  half(9.0f),  half(7.0f),  half(8.0f),  half(9.0f),
           half(10.0f), half(11.0f), half(12.0f), half(10.0f), half(11.0f),
           half(12.0f), half(7.0f),  half(8.0f),  half(9.0f),  half(7.0f),
           half(8.0f),  half(9.0f),  half(10.0f), half(11.0f), half(12.0f),
           half(10.0f), half(11.0f), half(12.0f), half(1.0f),  half(2.0f),
           half(3.0f),  half(1.0f),  half(2.0f),  half(3.0f),  half(4.0f),
           half(5.0f),  half(6.0f),  half(4.0f),  half(5.0f),  half(6.0f),
           half(1.0f),  half(2.0f),  half(3.0f),  half(1.0f),  half(2.0f),
           half(3.0f),  half(4.0f),  half(5.0f),  half(6.0f),  half(4.0f),
           half(5.0f),  half(6.0f),  half(7.0f),  half(8.0f),  half(9.0f),
           half(7.0f),  half(8.0f),  half(9.0f),  half(10.0f), half(11.0f),
           half(12.0f), half(10.0f), half(11.0f), half(12.0f), half(7.0f),
           half(8.0f),  half(9.0f),  half(7.0f),  half(8.0f),  half(9.0f),
           half(10.0f), half(11.0f), half(12.0f), half(10.0f), half(11.0f),
           half(12.0f)},
          dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
