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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/mean_op_legalization.h"

#include <cstdint>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"

namespace litert::mediatek {

Expected<void> LegalizeMeanOp(const NeuronAdapterApi& neuron_adapter_api,
                              NeuronModel* model, OperandMap& operand_map,
                              const litert::Op& op) {
  LITERT_LOG(LITERT_INFO, "Legalize Mean");
  std::vector<uint32_t> input_indices;
  for (auto& input : op.Inputs()) {
    auto id = operand_map.GetOperandIndex(input);
    if (!id) {
      return id.Error();
    }
    input_indices.push_back(*id);
  }

  // A NEURON_Mean operation takes an additional scalar operand, which is
  // used to pass a keepdims.
  bool keepdims;
  if (auto status = LiteRtGetMeanKeepDimsOption(op.Get(), &keepdims);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get beta");
  }
  LITERT_LOG(LITERT_INFO, "keepdims: %d", keepdims);
  auto keepdims_operand = operand_map.AddScalarInt32(keepdims);
  if (!keepdims_operand) {
    return keepdims_operand.Error();
  }
  input_indices.push_back(*keepdims_operand);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model, /*type=*/NEURON_MEAN,
                        input_indices, output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add NEURON_MEAN operation");
  }

  return {};
}

}  // namespace litert::mediatek
