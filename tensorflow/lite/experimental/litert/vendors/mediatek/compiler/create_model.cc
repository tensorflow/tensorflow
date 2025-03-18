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

#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/create_model.h"

#include <cstdint>
#include <string>
#include <vector>

#include "neuron/api/NeuronAdapter.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/add_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/batch_matmul_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/common_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/concat_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/fully_connected_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/gelu_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/mean_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/mul_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/operand_map.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/reshape_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/rsqrt_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/softmax_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/sub_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/compiler/legalizations/transpose_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter_api.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/schema/schema_resolver.h"

namespace litert::mediatek {

Expected<NeuronModelPtr> CreateModel(const NeuronAdapterApi& neuron_adapter_api,
                                     const litert::Subgraph& partition,
                                     const std::string& model_name) {
  auto model = neuron_adapter_api.CreateModel();
  if (!model) {
    return model.Error();
  }

  if (neuron_adapter_api.api().model_set_name(
          model->get(), model_name.c_str()) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to set model name");
  }

  OperandMap operand_map(neuron_adapter_api, model->get());

  std::vector<uint32_t> input_indices;
  for (const auto& input : partition.Inputs()) {
    auto operand_index = operand_map.GetOperandIndex(input);
    if (!operand_index) {
      return operand_index.Error();
    }
    input_indices.push_back(*operand_index);
  }

  std::vector<uint32_t> output_indices;
  for (const auto& output : partition.Outputs()) {
    auto operand_index = operand_map.GetOperandIndex(output);
    if (!operand_index) {
      return operand_index.Error();
    }
    output_indices.push_back(*operand_index);
  }

  if (neuron_adapter_api.api().model_identify_inputs_and_outputs(
          model->get(), input_indices.size(), input_indices.data(),
          output_indices.size(), output_indices.data()) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "Failed to identify model I/Os");
  }

  for (const auto& op : partition.Ops()) {
    Expected<void> status;
    switch (op.Code()) {
      case kLiteRtOpCodeTflAdd:
        status =
            LegalizeAddOp(neuron_adapter_api, model->get(), operand_map, op);
        break;
      case kLiteRtOpCodeTflMul:
        status =
            LegalizeMulOp(neuron_adapter_api, model->get(), operand_map, op);
        break;
      case kLiteRtOpCodeTflBatchMatmul:
        status = LegalizeBatchMatMulOp(neuron_adapter_api, model->get(),
                                       operand_map, op);
        break;
      case kLiteRtOpCodeTflFullyConnected:
        status = LegalizeFullyConnectedOp(neuron_adapter_api, model->get(),
                                          operand_map, op);
        break;
      case kLiteRtOpCodeTflReshape:
        status = LegalizeReshapeOp(neuron_adapter_api, model->get(),
                                   operand_map, op);
        break;
      case kLiteRtOpCodeTflTranspose:
        status = LegalizeTransposeOp(neuron_adapter_api, model->get(),
                                     operand_map, op);
        break;
      case kLiteRtOpCodeTflRsqrt:
        status =
            LegalizeRsqrtOp(neuron_adapter_api, model->get(), operand_map, op);
        break;
      case kLiteRtOpCodeTflConcatenation:
        status =
            LegalizeConcatOp(neuron_adapter_api, model->get(), operand_map, op);
        break;
      case kLiteRtOpCodeTflQuantize:
        status = LegalizeCommonOp(neuron_adapter_api, model->get(), operand_map,
                                  op, NEURON_QUANTIZE);
        break;
      case kLiteRtOpCodeTflSlice:
        status = LegalizeCommonOp(neuron_adapter_api, model->get(), operand_map,
                                  op, NEURON_SLICE);
        break;
      case kLiteRtOpCodeTflTanh:
        status = LegalizeCommonOp(neuron_adapter_api, model->get(), operand_map,
                                  op, NEURON_TANH);
        break;
      case kLiteRtOpCodeTflSub:
        status =
            LegalizeSubOp(neuron_adapter_api, model->get(), operand_map, op);
        break;
      case kLiteRtOpCodeTflSoftmax:
        status = LegalizeSoftmaxOp(neuron_adapter_api, model->get(),
                                   operand_map, op);
        break;
      case kLiteRtOpCodeTflMean:
        status =
            LegalizeMeanOp(neuron_adapter_api, model->get(), operand_map, op);
        break;
      case kLiteRtOpCodeTflGelu:
        status =
            LegalizeGeluOp(neuron_adapter_api, model->get(), operand_map, op);
        break;
      default:
        return Error(kLiteRtStatusErrorRuntimeFailure, "Unsupported op");
    }

    if (!status) {
      return status.Error();
    }
  }

  if (neuron_adapter_api.api().model_finish(model->get()) != NEURON_NO_ERROR) {
    return Error(kLiteRtStatusErrorRuntimeFailure, "Failed to finish model");
  }

  return model;
}

}  // namespace litert::mediatek
