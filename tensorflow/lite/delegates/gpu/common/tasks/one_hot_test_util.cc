/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/one_hot_test_util.h"

#include "tensorflow/lite/delegates/gpu/common/tasks/one_hot.h"

namespace tflite {
namespace gpu {

absl::Status OneHotTest(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, DataType::INT32> src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 1);
  src_tensor.data = {3};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.src_tensors.push_back({DataType::INT32, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.precision = precision;
      TensorDescriptor src = op_def.src_tensors[0];
      TensorDescriptor dst = op_def.dst_tensors[0];
      OneHotAttributes attr;
      GPUOperation operation = CreateOneHot(op_def, attr);
      src.UploadData(src_tensor);
      dst.SetBHWCShape(BHWC(1, 1, 1, 8));
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src}, {&dst},
          absl::make_unique<GPUOperation>(std::move(operation))));
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);
      RETURN_IF_ERROR(
          PointWiseNear({0, 0, 0, 1.0, 0, 0, 0, 0}, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

absl::Status OneHotBatchTest(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, DataType::INT32> src_tensor;
  src_tensor.shape = BHWC(10, 1, 1, 1);
  src_tensor.data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.src_tensors.push_back({DataType::INT32, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.precision = precision;
      TensorDescriptor src = op_def.src_tensors[0];
      TensorDescriptor dst = op_def.dst_tensors[0];
      OneHotAttributes attr = {/*on_value=*/2.0, /*off_value=*/-2.0};
      GPUOperation operation = CreateOneHot(op_def, attr);
      src.UploadData(src_tensor);
      dst.SetBHWCShape(BHWC(10, 1, 1, 10));
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src}, {&dst},
          absl::make_unique<GPUOperation>(std::move(operation))));
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);
      std::vector<float> expected(100, attr.off_value);
      for (int i = 0; i < 10; i++) {
        expected[i * 10 + i] = attr.on_value;
      }
      RETURN_IF_ERROR(PointWiseNear(expected, dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
