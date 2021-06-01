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

#include "tensorflow/lite/delegates/gpu/common/tasks/gather_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/gather.h"

namespace tflite {
namespace gpu {

absl::Status GatherWidthTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 5, 1);
  src_tensor.data = {half(1.5f), half(2.4f), half(3.3f), half(4.2f),
                     half(5.1f)};
  TensorFloat32 src_indices;
  src_indices.shape = BHWC(1, 1, 1, 9);
  src_indices.data = {half(1.1f), half(2.1f), half(3.1f),
                      half(0.1f), half(1.1f), half(4.1f),
                      half(2.1f), half(3.1f), half(1.1f)};
  GatherAttributes attr;
  attr.axis = Axis::WIDTH;
  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateGather(op_def, attr);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor, src_indices},
          absl::make_unique<GPUOperation>(std::move(operation)),
          BHWC(1, 1, 9, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(
          {half(2.4f), half(3.3f), half(4.2f), half(1.5f), half(2.4f),
           half(5.1f), half(3.3f), half(4.2f), half(2.4f)},
          dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
