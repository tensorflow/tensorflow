// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/add_op_legalization.h"

#include <cstdint>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<void> LegalizeAddOp(const NeuronAdapterApi& neuron_adapter_api,
                             NeuronModel* model, OperandMap& operand_map,
                             const litert::Op& op) {
  std::vector<uint32_t> input_indices;
  for (auto& input : op.Inputs()) {
    auto id = operand_map.GetOperandIndex(input);
    if (!id) {
      return id.Error();
    }
    input_indices.push_back(*id);
  }

  // A NEURON_ADD operation takes a 3rd scalar operand, which is used to pass a
  // TfLiteFusedActivation value.
  uint32_t tfl_fused_activation;
  if (auto status =
          LiteRtGetAddFusedActivationOption(op.Get(), &tfl_fused_activation);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get fused activation");
  }
  auto fused_activation_operand_index =
      operand_map.AddScalarInt32(tfl_fused_activation);
  if (!fused_activation_operand_index) {
    return fused_activation_operand_index.Error();
  }
  input_indices.push_back(*fused_activation_operand_index);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (neuron_adapter_api.api().model_add_operation(
          model, /*type=*/NEURON_ADD, input_indices.size(),
          input_indices.data(), output_indices.size(),
          output_indices.data()) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to set value of NEURON_ADD fused activation");
  }

  return {};
}

}  // namespace litert::mediatek
