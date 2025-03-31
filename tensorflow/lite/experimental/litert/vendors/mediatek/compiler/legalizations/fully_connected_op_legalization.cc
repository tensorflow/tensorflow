// Copyright (c) 2025 MediaTek Inc.
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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/fully_connected_op_legalization.h"

#include <cstdint>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

#define GET_RANK(op) ((op).RankedTensorType()->Layout().Rank())
#define GET_DIMENSION(op) ((op).RankedTensorType()->Layout().Dimensions())

namespace litert::mediatek {

Expected<void> LegalizeFullyConnectedOp(
    const NeuronAdapterApi& neuron_adapter_api, NeuronModel* model,
    OperandMap& operand_map, const litert::Op& op) {
  LITERT_LOG(LITERT_INFO, "Legalize Fully Connected");
  std::vector<uint32_t> input_indices;
  for (auto& input : op.Inputs()) {
    auto id = operand_map.GetOperandIndex(input);
    if (!id) {
      return id.Error();
    }
    input_indices.push_back(*id);
  }

  // for beta
  if (input_indices.size() < 3) {
    auto weights_shape = GET_DIMENSION(op.Inputs()[1]);
    std::vector<uint32_t> bias_shape = {
        static_cast<unsigned int>(weights_shape[0])};
    std::vector<int32_t> bias_data(bias_shape[0], 0);
    auto bias_data_operand =
        operand_map.AddTensorByType(NEURON_TENSOR_QUANT8_SYMM, bias_shape,
                                    bias_data.data(), bias_data.size() * 1);
    input_indices.push_back(*bias_data_operand);
  }

  // A NEURON_FULLY_CONNECTED operation takes a 4rd scalar operand, which is
  // used to pass a TfLiteFusedActivation value.
  uint32_t tfl_fused_activation;
  if (auto status = LiteRtGetFullyConnectedFusedActivationOption(
          op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get fused activation");
  }
  auto fused_activation_operand_index =
      operand_map.AddScalarInt32(tfl_fused_activation);
  if (!fused_activation_operand_index) {
    return fused_activation_operand_index.Error();
  }
  input_indices.push_back(*fused_activation_operand_index);

  auto output_operand = OperandType::Create(op.Outputs()[0]);
  std::vector<uint32_t> output_indices;

  if (GET_RANK(op.Outputs()[0]) > 2) {
    // if output_operand shape <B, K, N>, reshape to <B * K, N>
    auto last_dim = output_operand->GetDimension().back();
    auto elements = output_operand->GetElementCount();
    std::vector<uint32_t> new_dimension = {elements / last_dim, last_dim};
    if (auto res = output_operand->Reshape(new_dimension); !res) {
      return res.Error();
    }
    auto intermediate_operand = operand_map.AddOperand(*output_operand);
    output_indices.push_back(*intermediate_operand);
  } else {
    auto output_operand = operand_map.GetOperandIndex(op.Outputs()[0]);
    output_indices.push_back(*output_operand);
    if (!output_operand) {
      return output_operand.Error();
    }
  }

  if (ModelAddOperation(neuron_adapter_api, model,
                        /*type=*/NEURON_FULLY_CONNECTED, input_indices,
                        output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set NEURON_FULLY_CONNECTED operation");
  }

  if (GET_RANK(op.Outputs()[0]) > 2) {
    // intermediate as reshape input
    input_indices = {output_indices.back()};
    auto output_operand = operand_map.GetOperandIndex(op.Outputs()[0]);
    if (!output_operand) {
      return output_operand.Error();
    }

    auto dimension = op.Outputs()[0].RankedTensorType()->Layout().Dimensions();
    std::vector<uint32_t> new_shape(dimension.begin(), dimension.end());
    std::vector<uint32_t> tensor_shape = {(uint32_t)new_shape.size()};
    auto new_shape_operand_index = operand_map.AddTensorInt32(
        tensor_shape, new_shape.data(), new_shape.size() * sizeof(int32_t));
    if (!new_shape_operand_index) {
      return new_shape_operand_index.Error();
    }
    input_indices.push_back(*new_shape_operand_index);
    output_indices = {*output_operand};
    if (ModelAddOperation(neuron_adapter_api, model, /*type=*/NEURON_RESHAPE,
                          input_indices, output_indices) != NEURON_NO_ERROR) {
      return Error(kLiteRtStatusErrorRuntimeFailure,
                   "Failed to add Reshape after FC");
    }
  }

  return {};
}

}  // namespace litert::mediatek
