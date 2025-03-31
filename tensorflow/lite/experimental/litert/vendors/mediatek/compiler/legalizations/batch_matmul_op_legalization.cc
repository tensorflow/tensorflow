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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/batch_matmul_op_legalization.h"

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

Expected<void> LegalizeBatchMatMulOp(const NeuronAdapterApi& neuron_adapter_api,
                                     NeuronModel* model,
                                     OperandMap& operand_map,
                                     const litert::Op& op) {
  LITERT_LOG(LITERT_INFO, "Legalize BatchMatMul");
  std::vector<uint32_t> input_indices;
  for (auto& input : op.Inputs()) {
    auto id = operand_map.GetOperandIndex(input);
    if (!id) {
      return id.Error();
    }
    input_indices.push_back(*id);
  }

  // A NEURON_BATCH_MATMUL operation takes 2 scalar operand, which is used to
  // pass a adjX, adjY value.
  bool tfl_matmul_param_adj_x = 0, tfl_matmul_param_adj_y = 0;
  if (auto status =
          LiteRtGetBatchMatmulAdjXOption(op.Get(), &tfl_matmul_param_adj_x);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get batch matmul adjX");
  }

  if (auto status =
          LiteRtGetBatchMatmulAdjYOption(op.Get(), &tfl_matmul_param_adj_y);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get batch matmul adjY");
  }

  auto adj_x_operand_index = operand_map.AddScalarBool(tfl_matmul_param_adj_x);
  if (!adj_x_operand_index) {
    return adj_x_operand_index.Error();
  }
  input_indices.push_back(*adj_x_operand_index);

  auto adj_j_operand_index = operand_map.AddScalarBool(tfl_matmul_param_adj_y);
  if (!adj_j_operand_index) {
    return adj_j_operand_index.Error();
  }
  input_indices.push_back(*adj_j_operand_index);

  std::vector<uint32_t> output_indices;
  for (auto& output : op.Outputs()) {
    auto id = operand_map.GetOperandIndex(output);
    if (!id) {
      return id.Error();
    }
    output_indices.push_back(*id);
  }

  if (ModelAddOperation(neuron_adapter_api, model, /*type=*/NEURON_BATCH_MATMUL,
                        input_indices, output_indices) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to add NEURON_BATCH_MATMUL op");
  }

  return {};
}

}  // namespace litert::mediatek
