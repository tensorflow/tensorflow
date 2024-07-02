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

#include "tensorflow/lite/delegates/gpu/common/tasks/group_normalization_test_util.h"

#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/group_normalization.h"

namespace tflite {
namespace gpu {

absl::Status GroupNormalizationTest(TestExecutionEnvironment* env,
                              bool constant_idx) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 16);
  src_tensor.data = {half(1.0f), half(2.0f), half(1.0f), half(2.0f),
                     half(1.0f), half(2.0f), half(1.0f), half(2.0f),
                     half(1.0f), half(2.0f), half(1.0f), half(2.0f),
                     half(1.0f), half(2.0f), half(1.0f), half(2.0f)};

  // channels wise mean_sum
  std::vector<float> mean_data{1.0f, 2.0f, 1.0f, 2.0f,
                               1.0f, 2.0f, 1.0f, 2.0f,
                               1.0f, 2.0f, 1.0f, 2.0f,
                               1.0f, 2.0f, 1.0f, 2.0f};
  // channel wise variance_sum
  std::vector<float> var_data{0.25f, 0.25f, 0.25f, 0.25f,
                              0.25f, 0.25f, 0.25f, 0.25f,
                              0.25f, 0.25f, 0.25f, 0.25f,
                              0.25f, 0.25f, 0.25f, 0.25f};

  std::vector<float> gamma_data{1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f};

  std::vector<float> beta_data{0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};

  GroupNormalizationAttributes attr;
  attr.axis = -1;
  attr.groups = 2;
  attr.scale = true;
  attr.centre = true;

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC}); // input
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC}); // mean_sum
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC}); // var_sum
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorDescriptor src_0, src_1, src_2, dst;
      src_0 = op_def.src_tensors[0];
      src_0.UploadData(src_tensor);
      dst.SetBHWDCShape(BHWDC(1, 1, 1, 1, 16));

      // uploading mean_sum and var_sum
      src_1 = op_def.src_tensors[1];
      tflite::gpu::Tensor<BHWC, DataType::FLOAT32> mean_sum;
      mean_sum.shape = BHWC(1, 1, 1, 16);
      mean_sum.data = mean_data;
      src_1.UploadData(mean_sum);

      src_2 = op_def.src_tensors[2];
      tflite::gpu::Tensor<BHWC, DataType::FLOAT32> var_sum;
      var_sum.shape = BHWC(1, 1, 1, 16);
      var_sum.data = var_data;
      src_2.UploadData(var_sum);

      //adding gamma and beta values
      attr.gamma.shape = Linear(16);
      attr.beta.shape = Linear(16);
      attr.gamma.data = gamma_data;
      attr.beta.data = beta_data;

      GPUOperation operation = CreateGroupNormalization(env->GetGpuInfo(), op_def, attr);

      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {&src_0, &src_1, &src_2}, {&dst},
          std::make_unique<GPUOperation>(std::move(operation))));
      TensorFloat32 dst_tensor;
      dst.DownloadData(&dst_tensor);

      // dividing by n-1, as per formul, if n then value would be 0.99998
      RETURN_IF_ERROR(PointWiseNear(
          {half(-0.99998f), half(0.99998f), half(-0.99998f), half(0.99998f), 
           half(-0.99998f), half(0.99998f), half(-0.99998f), half(0.99998f),
           half(-0.99998f), half(0.99998f), half(-0.99998f), half(0.99998f),
           half(-0.99998f), half(0.99998f), half(-0.99998f), half(0.99998f)},
          dst_tensor.data, 0.0f));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
