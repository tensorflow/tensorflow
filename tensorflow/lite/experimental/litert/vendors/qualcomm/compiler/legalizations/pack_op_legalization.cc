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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/pack_op_legalization.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

// Pack op config.
static constexpr absl::string_view kQnnPackOpTypeName = "Pack";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kPackOpFmt = "pack_%d";
static constexpr absl::string_view kPackOpAxisParamName = "axis";
static constexpr int kPackOpAxisParamSize = 1;
static constexpr int kPackScalarsOpOutputRank = 2;

// Reshape op config.
static constexpr absl::string_view kReshapeOpTypeName = "Reshape";
static constexpr absl::string_view kReshapeOpFmt = "pack_reshape_%d";
static constexpr int kReshapeOpInputSize = 1;
static constexpr int kReshapeOpOutputSize = 1;
static constexpr int kReshapeParamSize = 0;

LiteRtStatus PackOpLegalization::LegalizeOp(const Op& src, Qnn_OpConfig_t& dest,
                                            GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflPack) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  std::string pack_op_name = absl::StrFormat(kPackOpFmt, op_counter_);
  DumpLegalization(*src.Get());

  // Legalize input tensors, lookup operand tensor in scope.
  const auto op_ins = src.Inputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, op_ins.size(), QNN_TENSOR_INIT);
  Qnn_Tensor_t* cur_qnn_op_in = qnn_op_ins;
  for (const auto& op_in : op_ins) {
    LITERT_RETURN_IF_ERROR(
        graph_mapper.LookupInScope(op_in.Get(), *cur_qnn_op_in));
    ++cur_qnn_op_in;
  }

  // Legalize output tensors.
  const auto op_outs = src.Outputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, op_outs.size(),
                     QNN_TENSOR_INIT);
  LITERT_RETURN_IF_ERROR(
      graph_mapper.LegalizeAndRegister(op_outs.front().Get(), qnn_op_outs[0]));
  LITERT_RETURN_IF_ERROR(
      graph_mapper.PushToScope(op_outs.front().Get(), qnn_op_outs[0]));

  // Get axis option and build QNN scalar param.
  int32_t axis;
  LITERT_RETURN_IF_ERROR(LiteRtGetPackAxisOption(src.Get(), &axis));
  uint32_t axis_value = static_cast<uint32_t>(axis);

  Qnn_Param_t axis_param = BuildDefaultParam();
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildQnnScalarParam<uint32_t>(
      axis_value, QNN_DATATYPE_UINT_32, kPackOpAxisParamName.data(),
      graph_mapper, axis_param));

  // Qnn does not support Packing scalars, scalar value are legalized as 1D
  // tensor with single element. In such case, we need to add a reshape op to
  // convert result packed 2D tensor to 1D tensor.
  auto input_layout = op_ins[0].RankedTensorType()->Layout();
  if (input_layout.Rank() == 0) {
    // prepare Pack op output tensor.
    Qnn_Tensor_t pack_op_out = BuildDefaultTensor();
    uint32_t pack_op_out_rank = kPackScalarsOpOutputRank;
    Qnn_DataType_t PackOpDataType = QNN_DATATYPE_UNDEFINED;

    LITERT_RETURN_IF_ERROR(
        LegalizeElementType(op_ins[0].ElementType(), &PackOpDataType));
    std::vector<uint32_t> pack_op_out_dims = {
        static_cast<uint32_t>(op_ins.size())};

    LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnNativeTensor(
        PackOpDataType, pack_op_out_rank, pack_op_out_dims.data(), graph_mapper,
        pack_op_out));

    // Build Pack op.
    Qnn_OpConfig_t pack_op = BuildDefaultOp();
    LITERT_RETURN_IF_ERROR(SetOpInfo(pack_op_name.c_str(),
                                     kDefaultQnnOpPackageName.data(),
                                     kQnnPackOpTypeName.data(), pack_op));
    LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnOp(
        op_ins.size(), qnn_op_ins, op_outs.size(), &pack_op_out, pack_op,
        kPackOpAxisParamSize, &axis_param, graph_mapper));

    // Build Reshape op.
    std::string reshape_op_name = absl::StrFormat(kReshapeOpFmt, op_counter_);
    LITERT_RETURN_IF_ERROR(SetOpInfo(reshape_op_name.c_str(),
                                     kDefaultQnnOpPackageName.data(),
                                     kReshapeOpTypeName.data(), dest));
    LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnOp(
        kReshapeOpInputSize, &pack_op_out, kReshapeOpOutputSize, qnn_op_outs,
        dest, kReshapeParamSize, nullptr, graph_mapper));
  } else {
    LITERT_RETURN_IF_ERROR(SetOpInfo(pack_op_name.c_str(),
                                     kDefaultQnnOpPackageName.data(),
                                     kQnnPackOpTypeName.data(), dest));
    BuildAndRegisterQnnOp(op_ins.size(), qnn_op_ins, op_outs.size(),
                          qnn_op_outs, dest, kPackOpAxisParamSize, &axis_param,
                          graph_mapper);
  }
  op_counter_++;

  LITERT_LOG(LITERT_INFO, "Legalized pack op", "");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
