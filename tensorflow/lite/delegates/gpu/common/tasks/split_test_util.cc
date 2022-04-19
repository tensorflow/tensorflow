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

#include "tensorflow/lite/delegates/gpu/common/tasks/split_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/split.h"

namespace tflite {
namespace gpu {

absl::Status SplitChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 3, 2, 5);
  src_tensor.data = {
      half(0.1f),  half(0.2f),  half(0.3f),  half(0.4),  half(0.5),
      half(1.1f),  half(1.2f),  half(1.3f),  half(1.4),  half(1.5),
      half(10.1f), half(10.2f), half(10.3f), half(10.4), half(10.5),
      half(11.1f), half(11.2f), half(11.3f), half(11.4), half(11.5),
      half(20.1f), half(20.2f), half(20.3f), half(20.4), half(20.5),
      half(21.1f), half(21.2f), half(21.3f), half(21.4), half(21.5)};

  SplitAttributes attr;
  attr.axis = Axis::CHANNELS;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor0, dst_tensor1;
      Split operation = CreateSplit(op_def, attr, {2, 3});
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor}, absl::make_unique<Split>(std::move(operation)),
          {BHWC(1, 3, 2, 2), BHWC(1, 3, 2, 3)}, {&dst_tensor0, &dst_tensor1}));
      RETURN_IF_ERROR(
          PointWiseNear({half(0.1f), half(0.2f), half(1.1f), half(1.2f),
                         half(10.1f), half(10.2f), half(11.1f), half(11.2f),
                         half(20.1f), half(20.2f), half(21.1f), half(21.2f)},
                        dst_tensor0.data, 0.0f));
      RETURN_IF_ERROR(PointWiseNear(
          {half(0.3f), half(0.4), half(0.5), half(1.3f), half(1.4), half(1.5),
           half(10.3f), half(10.4), half(10.5), half(11.3f), half(11.4),
           half(11.5), half(20.3f), half(20.4), half(20.5), half(21.3f),
           half(21.4), half(21.5)},
          dst_tensor1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SplitChannelsX4Test(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 8);
  src_tensor.data = {half(0.1f),  half(0.2f),  half(0.3f),  half(0.4),
                     half(1.1f),  half(1.2f),  half(1.3f),  half(1.4),
                     half(10.1f), half(10.2f), half(10.3f), half(10.4),
                     half(11.1f), half(11.2f), half(11.3f), half(11.4),
                     half(20.1f), half(20.2f), half(20.3f), half(20.4),
                     half(21.1f), half(21.2f), half(21.3f), half(21.4),
                     half(30.1f), half(30.2f), half(30.3f), half(30.4),
                     half(31.1f), half(31.2f), half(31.3f), half(31.4)};

  SplitAttributes attr;
  attr.axis = Axis::CHANNELS;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor0, dst_tensor1;
      Split operation = CreateSplit(op_def, attr, {4, 4});
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor}, absl::make_unique<Split>(std::move(operation)),
          {BHWC(1, 2, 2, 4), BHWC(1, 2, 2, 4)}, {&dst_tensor0, &dst_tensor1}));
      RETURN_IF_ERROR(
          PointWiseNear({half(0.1f), half(0.2f), half(0.3f), half(0.4),
                         half(10.1f), half(10.2f), half(10.3f), half(10.4),
                         half(20.1f), half(20.2f), half(20.3f), half(20.4),
                         half(30.1f), half(30.2f), half(30.3f), half(30.4)},
                        dst_tensor0.data, 0.0f));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.1f), half(1.2f), half(1.3f), half(1.4),
                         half(11.1f), half(11.2f), half(11.3f), half(11.4),
                         half(21.1f), half(21.2f), half(21.3f), half(21.4),
                         half(31.1f), half(31.2f), half(31.3f), half(31.4)},
                        dst_tensor1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SplitWidthTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 6, 5, 1);
  src_tensor.data = {
      half(0.1f),  half(0.2f),  half(0.3f),  half(0.4),  half(0.5),
      half(1.1f),  half(1.2f),  half(1.3f),  half(1.4),  half(1.5),
      half(10.1f), half(10.2f), half(10.3f), half(10.4), half(10.5),
      half(11.1f), half(11.2f), half(11.3f), half(11.4), half(11.5),
      half(20.1f), half(20.2f), half(20.3f), half(20.4), half(20.5),
      half(21.1f), half(21.2f), half(21.3f), half(21.4), half(21.5)};

  SplitAttributes attr;
  attr.axis = Axis::WIDTH;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor0, dst_tensor1;
      Split operation = CreateSplit(op_def, attr, {1, 1});
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor}, absl::make_unique<Split>(std::move(operation)),
          {BHWC(1, 6, 2, 1), BHWC(1, 6, 3, 1)}, {&dst_tensor0, &dst_tensor1}));
      RETURN_IF_ERROR(
          PointWiseNear({half(0.1f), half(0.2f), half(1.1f), half(1.2f),
                         half(10.1f), half(10.2f), half(11.1f), half(11.2f),
                         half(20.1f), half(20.2f), half(21.1f), half(21.2f)},
                        dst_tensor0.data, 0.0f));
      RETURN_IF_ERROR(PointWiseNear(
          {half(0.3f), half(0.4), half(0.5), half(1.3f), half(1.4), half(1.5),
           half(10.3f), half(10.4), half(10.5), half(11.3f), half(11.4),
           half(11.5), half(20.3f), half(20.4), half(20.5), half(21.3f),
           half(21.4), half(21.5)},
          dst_tensor1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SplitHeightTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 6, 5, 1);
  src_tensor.data = {
      half(0.1f),  half(0.2f),  half(0.3f),  half(0.4),  half(0.5),
      half(1.1f),  half(1.2f),  half(1.3f),  half(1.4),  half(1.5),
      half(10.1f), half(10.2f), half(10.3f), half(10.4), half(10.5),
      half(11.1f), half(11.2f), half(11.3f), half(11.4), half(11.5),
      half(20.1f), half(20.2f), half(20.3f), half(20.4), half(20.5),
      half(21.1f), half(21.2f), half(21.3f), half(21.4), half(21.5)};

  SplitAttributes attr;
  attr.axis = Axis::HEIGHT;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor0, dst_tensor1;
      Split operation = CreateSplit(op_def, attr, {1, 1});
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor}, absl::make_unique<Split>(std::move(operation)),
          {BHWC(1, 2, 5, 1), BHWC(1, 4, 5, 1)}, {&dst_tensor0, &dst_tensor1}));
      RETURN_IF_ERROR(PointWiseNear(
          {half(0.1f), half(0.2f), half(0.3f), half(0.4), half(0.5), half(1.1f),
           half(1.2f), half(1.3f), half(1.4), half(1.5)},
          dst_tensor0.data, 0.0f));
      RETURN_IF_ERROR(PointWiseNear(
          {half(10.1f), half(10.2f), half(10.3f), half(10.4), half(10.5),
           half(11.1f), half(11.2f), half(11.3f), half(11.4), half(11.5),
           half(20.1f), half(20.2f), half(20.3f), half(20.4), half(20.5),
           half(21.1f), half(21.2f), half(21.3f), half(21.4), half(21.5)},
          dst_tensor1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SplitBatchTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(6, 1, 5, 1);
  src_tensor.data = {
      half(0.1f),  half(0.2f),  half(0.3f),  half(0.4),  half(0.5),
      half(1.1f),  half(1.2f),  half(1.3f),  half(1.4),  half(1.5),
      half(10.1f), half(10.2f), half(10.3f), half(10.4), half(10.5),
      half(11.1f), half(11.2f), half(11.3f), half(11.4), half(11.5),
      half(20.1f), half(20.2f), half(20.3f), half(20.4), half(20.5),
      half(21.1f), half(21.2f), half(21.3f), half(21.4), half(21.5)};

  SplitAttributes attr;
  attr.axis = Axis::BATCH;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 dst_tensor0, dst_tensor1;
      Split operation = CreateSplit(op_def, attr, {1, 1});
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor}, absl::make_unique<Split>(std::move(operation)),
          {BHWC(1, 1, 5, 1), BHWC(5, 1, 5, 1)}, {&dst_tensor0, &dst_tensor1}));
      RETURN_IF_ERROR(PointWiseNear(
          {half(0.1f), half(0.2f), half(0.3f), half(0.4), half(0.5)},
          dst_tensor0.data, 0.0f));
      RETURN_IF_ERROR(PointWiseNear(
          {half(1.1f),  half(1.2f),  half(1.3f),  half(1.4),  half(1.5),
           half(10.1f), half(10.2f), half(10.3f), half(10.4), half(10.5),
           half(11.1f), half(11.2f), half(11.3f), half(11.4), half(11.5),
           half(20.1f), half(20.2f), half(20.3f), half(20.4), half(20.5),
           half(21.1f), half(21.2f), half(21.3f), half(21.4), half(21.5)},
          dst_tensor1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SplitDepthTest(TestExecutionEnvironment* env) {
  Tensor5DFloat32 src_tensor;
  src_tensor.shape = BHWDC(1, 6, 1, 5, 1);
  src_tensor.data = {
      half(0.1f),  half(0.2f),  half(0.3f),  half(0.4),  half(0.5),
      half(1.1f),  half(1.2f),  half(1.3f),  half(1.4),  half(1.5),
      half(10.1f), half(10.2f), half(10.3f), half(10.4), half(10.5),
      half(11.1f), half(11.2f), half(11.3f), half(11.4), half(11.5),
      half(20.1f), half(20.2f), half(20.3f), half(20.4), half(20.5),
      half(21.1f), half(21.2f), half(21.3f), half(21.4), half(21.5)};

  SplitAttributes attr;
  attr.axis = Axis::DEPTH;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWDC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWDC});
      Tensor5DFloat32 dst_tensor0, dst_tensor1;
      Split operation = CreateSplit(op_def, attr, {1, 1});
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor}, absl::make_unique<Split>(std::move(operation)),
          {BHWDC(1, 6, 1, 2, 1), BHWDC(1, 6, 1, 3, 1)},
          {&dst_tensor0, &dst_tensor1}));
      RETURN_IF_ERROR(
          PointWiseNear({half(0.1f), half(0.2f), half(1.1f), half(1.2f),
                         half(10.1f), half(10.2f), half(11.1f), half(11.2f),
                         half(20.1f), half(20.2f), half(21.1f), half(21.2f)},
                        dst_tensor0.data, 0.0f));
      RETURN_IF_ERROR(PointWiseNear(
          {half(0.3f), half(0.4), half(0.5), half(1.3f), half(1.4), half(1.5),
           half(10.3f), half(10.4), half(10.5), half(11.3f), half(11.4),
           half(11.5), half(20.3f), half(20.4), half(20.5), half(21.3f),
           half(21.4), half(21.5)},
          dst_tensor1.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
