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

#include "tensorflow/lite/delegates/gpu/common/tasks/lstm_test_util.h"

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/lstm.h"

namespace tflite {
namespace gpu {

absl::Status LstmTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 16);
  src_tensor.data = {
      -std::log(2.0f), -std::log(2.0f), -std::log(2.0f), -std::log(2.0f),
      std::log(3.0f),  std::log(3.0f),  std::log(3.0f),  std::log(3.0f),
      -std::log(4.0f), -std::log(4.0f), -std::log(4.0f), -std::log(4.0f),
      -std::log(5.0f), -std::log(5.0f), -std::log(5.0f), -std::log(5.0f)};
  // input_gate = 1.0 / (1.0 + exp(log(2.0f))) = 1.0 / 3.0;
  // new_input = tanh(log(3.0f)) = (exp(2 * log(3.0f)) - 1) / exp(2 * log(3.0f))
  // + 1 = (9 - 1) / (9 + 1) = 0.8;
  // forget_gate = 1.0 / (1.0 + exp(log(4.0f)))
  //  = 1.0 / 5.0;
  // output_gate = 1.0 / (1.0 + exp(log(5.0f))) = 1.0 / 6.0;
  // new_st = input_gate * new_input + forget_gate * prev_st
  //   = 1.0 / 3.0 * 0.8 + 1.0 / 5.0 * prev_st
  //   = 4.0 / 15.0 + 3.0 / 15.0 = 7.0 / 15.0
  // activation = output_gate * tanh(new_st)
  TensorFloat32 prev_state;
  prev_state.shape = BHWC(1, 1, 1, 4);
  prev_state.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.src_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::BHWC});
      TensorFloat32 new_state;
      TensorFloat32 new_activ;
      GPUOperation operation = CreateLSTM(op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          {src_tensor, prev_state},
          std::make_unique<GPUOperation>(std::move(operation)),
          {BHWC(1, 1, 1, 4), BHWC(1, 1, 1, 4)}, {&new_state, &new_activ}));
      RETURN_IF_ERROR(
          PointWiseNear({7.0 / 15.0, 10.0 / 15.0, 13.0 / 15.0, 16.0 / 15.0},
                        new_state.data, eps))
          << ToString(storage) << ", " << ToString(precision);
      RETURN_IF_ERROR(PointWiseNear(
          {static_cast<float>((1.0 / 6.0) * std::tanh(7.0 / 15.0)),
           static_cast<float>((1.0 / 6.0) * std::tanh(10.0 / 15.0)),
           static_cast<float>((1.0 / 6.0) * std::tanh(13.0 / 15.0)),
           static_cast<float>((1.0 / 6.0) * std::tanh(16.0 / 15.0))},
          new_activ.data, eps))
          << ToString(storage) << ", " << ToString(precision);
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
