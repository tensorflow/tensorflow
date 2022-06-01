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

#include "tensorflow/lite/delegates/gpu/common/tasks/resampler_test_util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/resampler.h"

namespace tflite {
namespace gpu {

absl::Status ResamplerIdentityTest(const BHWC& shape,
                                   TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = shape;
  src_tensor.data.resize(src_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_tensor.data.size(); ++i) {
    src_tensor.data[i] = std::sin(i);
  }
  TensorFloat32 warp_tensor;
  warp_tensor.shape = BHWC(1, shape.h, shape.w, 2);
  warp_tensor.data.resize(warp_tensor.shape.DimensionsProduct());
  for (int y = 0; y < shape.h; ++y) {
    for (int x = 0; x < shape.w; ++x) {
      warp_tensor.data[(y * shape.w + x) * 2 + 0] = x;
      warp_tensor.data[(y * shape.w + x) * 2 + 1] = y;
    }
  }

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateResampler(env->GetGpuInfo(), op_def);
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor, warp_tensor},
          absl::make_unique<GPUOperation>(std::move(operation)),
          src_tensor.shape, &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear(src_tensor.data, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
