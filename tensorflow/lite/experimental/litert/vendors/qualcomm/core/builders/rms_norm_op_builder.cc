// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/rms_norm_op_builder.h"

#include <cstdint>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

static constexpr int kInputIndex = 0;
static constexpr int kGammaIndex = 1;

std::vector<OpWrapper> BuildRmsNormOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const float epsilon) {
  std::vector<OpWrapper> res;

  auto& rms_norm_op = CreateOpWrapper(res, QNN_OP_RMS_NORM);
  for (const auto& input : inputs) {
    rms_norm_op.AddInputTensor(input);
  }

  // Constructs axis param tensor.
  std::vector<std::uint32_t> axis_data;
  axis_data.emplace_back(inputs[kInputIndex].get().GetRank() - 1);
  TensorWrapper& axis_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, inputs[kInputIndex].get().GetQuantParams(), {1},
      sizeof(std::uint32_t) * axis_data.size(), axis_data.data());

  if (inputs[kGammaIndex].get().GetDataType() == QNN_DATATYPE_FLOAT_32) {
    // Construct float beta static all 0 tensor.
    std::vector<float> beta_data(
        inputs[kGammaIndex].get().GetTensorNumElements(), 0);
    TensorWrapper& beta_tensor = tensor_pool.CreateStaticTensor(
        inputs[kGammaIndex].get().GetDataType(),
        inputs[kGammaIndex].get().GetQuantParams(),
        inputs[kGammaIndex].get().GetDims(), sizeof(float) * beta_data.size(),
        beta_data.data());
    rms_norm_op.AddInputTensor(beta_tensor);
  } else {
    // Construct uint8_t beta static all 0 tensor.
    std::vector<uint8_t> beta_data(
        inputs[kGammaIndex].get().GetTensorNumElements(), 0);

    // Offset needs to be 0, scale does not matter since data is 0
    ScaleOffsetQuantizeParamsWrapper q_param(0.00001, 0);

    TensorWrapper& beta_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UFIXED_POINT_8, q_param,
        inputs[kGammaIndex].get().GetDims(), sizeof(uint8_t) * beta_data.size(),
        beta_data.data());
    rms_norm_op.AddInputTensor(beta_tensor);
  }

  rms_norm_op.AddScalarParam<float>(QNN_OP_RMS_NORM_PARAM_EPSILON, epsilon);
  rms_norm_op.AddTensorParam(QNN_OP_RMS_NORM_PARAM_AXES, axis_tensor);
  rms_norm_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
