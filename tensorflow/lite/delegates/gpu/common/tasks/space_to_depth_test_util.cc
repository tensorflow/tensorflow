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

#include "tensorflow/lite/delegates/gpu/common/tasks/space_to_depth_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.h"

namespace tflite {
namespace gpu {

absl::Status SpaceToDepthTensorShape1x2x2x1BlockSize2Test(
    TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {half(1.0f), half(2.0f), half(3.0f), half(4.0f)};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateSpaceToDepth(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 1, 4), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.0f), half(2.0f), half(3.0f), half(4.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SpaceToDepthTensorShape1x2x2x2BlockSize2Test(
    TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 2);
  src_tensor.data = {half(1.4f), half(2.3f), half(3.2f), half(4.1f),
                     half(5.4f), half(6.3f), half(7.2f), half(8.1f)};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateSpaceToDepth(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 1, 8), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.4f), half(2.3f), half(3.2f), half(4.1f),
                         half(5.4f), half(6.3f), half(7.2f), half(8.1f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SpaceToDepthTensorShape1x2x2x3BlockSize2Test(
    TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 3);
  src_tensor.data = {half(1.0f), half(2.0f),  half(3.0f),  half(4.0f),
                     half(5.0f), half(6.0f),  half(7.0f),  half(8.0f),
                     half(9.0f), half(10.0f), half(11.0f), half(12.0f)};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateSpaceToDepth(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 1, 12), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.0f), half(2.0f), half(3.0f), half(4.0f),
                         half(5.0f), half(6.0f), half(7.0f), half(8.0f),
                         half(9.0f), half(10.0f), half(11.0f), half(12.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status SpaceToDepthTensorShape1x4x4x1BlockSize2Test(
    TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 4, 4, 1);
  src_tensor.data = {half(1.0f),  half(2.0f),  half(5.0f),  half(6.0f),
                     half(3.0f),  half(4.0f),  half(7.0f),  half(8.0f),
                     half(9.0f),  half(10.0f), half(13.0f), half(14.0f),
                     half(11.0f), half(12.0f), half(15.0f), half(16.0f)};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateSpaceToDepth(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 2, 2, 4), &dst_tensor));
      RETURN_IF_ERROR(
          PointWiseNear({half(1.0f), half(2.0f), half(3.0f), half(4.0f),
                         half(5.0f), half(6.0f), half(7.0f), half(8.0f),
                         half(9.0f), half(10.0f), half(11.0f), half(12.0f),
                         half(13.0f), half(14.0f), half(15.0f), half(16.0f)},
                        dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
