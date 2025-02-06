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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/dynamic_update_slice_op_legalization.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";

// Dynamic update slice op info.
static constexpr int kDynamicUpdateSliceOpOperandIndex = 0;
static constexpr int kDynamicUpdateSliceOpUpdateIndex = 1;
static constexpr int kDynamicUpdateSliceOpIndicesIndex = 2;

// ScatterND op config.
static constexpr absl::string_view kQnnScatterNdOpTypeName = "ScatterNd";
static constexpr absl::string_view kScatterNdOpFmt = "dus_scatter_nd_%d";
static constexpr int kScatterNDOpInputSize = 3;
static constexpr int kScatterNDOpOutputSize = 1;
static constexpr int kScatterNDOutputRank = 4;
static constexpr int kScatterNDParamSize = 0;

// Strided slice op config.
static constexpr absl::string_view kStridedSliceOpTypeName = "StridedSlice";
static constexpr absl::string_view kStridedSliceOpFmt = "dus_strided_slice_%d";
static constexpr int kStridedSliceOpInputSize = 1;
static constexpr int kStridedSliceOpOutputSize = 1;
static constexpr int kStridedSliceOpOutputRank = 1;
static constexpr int kStridedSliceParamSize = 1;
static constexpr absl::string_view kRangesParamName = "ranges";
static constexpr int kRangesParamRank = 2;
static constexpr int kRangesParamArgSize = 3;

// Reshape op config.
static constexpr absl::string_view kReshapeOpTypeName = "Reshape";
static constexpr absl::string_view kReshapeOpFmt = "dus_reshape_%d";
static constexpr int kReshapeOpInputSize = 1;
static constexpr int kReshapeOpOutputSize = 1;
static constexpr int kReshapeOpOutputRank = 2;
static constexpr int kReshapeParamSize = 0;

// Transpose op config.
static constexpr absl::string_view kTransposeOpTypeName = "Transpose";
static constexpr absl::string_view kTransposeOperandOpFmt =
    "dus_transpose_operand_%d";
static constexpr absl::string_view kTransposeUpdateOpFmt =
    "dus_transpose_update_%d";
static constexpr absl::string_view kTransposeResultOpFmt =
    "dus_transpose_result_%d";
static constexpr int kTransposeOpInputSize = 1;
static constexpr int kTransposeOpOutputSize = 1;
static constexpr int kTransposeOpOutputRank = 4;
static constexpr int kTransposeParamSize = 1;
static constexpr absl::string_view kPermParamName = "perm";
static constexpr int kPermParamRank = 1;
static constexpr int kPermParamArgSize = 4;

LiteRtStatus DynamicUpdateSliceOpLegalization::LegalizeOp(
    const Op& src, Qnn_OpConfig_t& dest, GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflDynamicUpdateSlice) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  DumpLegalization(*src.Get());

  // Legalize input tensors, lookup operand tensor in scope.
  const auto op_ins = src.Inputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, kScatterNDOpInputSize,
                     QNN_TENSOR_INIT);

  Qnn_Tensor_t* cur_qnn_op_in = qnn_op_ins;
  for (const auto& op_in : op_ins) {
    LITERT_RETURN_IF_ERROR(
        graph_mapper.LookupInScope(op_in.Get(), *cur_qnn_op_in));
    ++cur_qnn_op_in;
  }
  // Legalize op data type.
  Qnn_DataType_t OperandDataType, UpdateDataType;
  LITERT_RETURN_IF_ERROR(LegalizeElementType(
      op_ins[kDynamicUpdateSliceOpOperandIndex].ElementType(),
      &OperandDataType));
  LITERT_RETURN_IF_ERROR(LegalizeElementType(
      op_ins[kDynamicUpdateSliceOpUpdateIndex].ElementType(), &UpdateDataType));

  //===========================================================================
  // Step 1.1 Build strided slice op. Extract slice index from input[2]
  //      input: [0, x, 0, 0] (LiteRT.DUS input[2])
  //      output: [x]
  Qnn_OpConfig_t strided_slice_op = BuildDefaultOp();
  std::string op_name = absl::StrFormat(kStridedSliceOpFmt, op_counter_);
  LITERT_RETURN_IF_ERROR(
      SetOpInfo(op_name.c_str(), kDefaultQnnOpPackageName.data(),
                kStridedSliceOpTypeName.data(), strided_slice_op));

  // Prepare strided slice op params.
  std::vector<int32_t> ranges = {1, 2, 1};
  std::vector<uint32_t> ranges_dims = {1, kRangesParamArgSize};
  Qnn_Param_t range_param = BuildDefaultParam();
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildQnnTesnorParam<int32_t>(
      ranges.data(), ranges_dims.data(), QNN_DATATYPE_INT_32, kRangesParamRank,
      kRangesParamName.data(), graph_mapper, range_param));

  // Prepare strided slice op outputs.
  Qnn_Tensor_t strided_slice_op_out = BuildDefaultTensor();
  std::vector<uint32_t> slice_op_out_dims = {1};
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnNativeTensor(
      QNN_DATATYPE_INT_32, kStridedSliceOpOutputRank, slice_op_out_dims.data(),
      graph_mapper, strided_slice_op_out));

  // Configure strided slice op.
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnOp(
      kStridedSliceOpInputSize, &qnn_op_ins[kDynamicUpdateSliceOpIndicesIndex],
      kStridedSliceOpOutputSize, &strided_slice_op_out, strided_slice_op,
      kStridedSliceParamSize, &range_param, graph_mapper));

  LITERT_LOG(LITERT_INFO, "Added strided slice op for dus");

  //===========================================================================
  // Step 1.2 Build reshape op. Construct input tensor shape for QNN.ScatterND
  // op.
  //      input: [x] (QNN.StridedSlice output)
  //      output: [[x]]
  Qnn_OpConfig_t reshape_op = BuildDefaultOp();
  std::string reshpae_op_name = absl::StrFormat(kReshapeOpFmt, op_counter_);
  LITERT_RETURN_IF_ERROR(SetOpInfo(op_name.c_str(),
                                   kDefaultQnnOpPackageName.data(),
                                   kReshapeOpTypeName.data(), reshape_op));

  // Prepare reshape op output tensor.
  Qnn_Tensor_t reshape_op_out = BuildDefaultTensor();
  std::vector<uint32_t> reshape_op_out_dims = {1, 1};
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnNativeTensor(
      QNN_DATATYPE_INT_32, kReshapeOpOutputRank, reshape_op_out_dims.data(),
      graph_mapper, reshape_op_out));

  // Configure reshape op.
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnOp(
      kReshapeOpInputSize, &strided_slice_op_out, kReshapeOpOutputSize,
      &reshape_op_out, reshape_op, kReshapeParamSize, nullptr, graph_mapper));

  LITERT_LOG(LITERT_INFO, "Added reshape op for dus");

  //===========================================================================
  // Step 2 Build transpose op. Swap the first two dimensions of the input
  // tensor[0] and input tensor[1].
  // op.
  //      input: [a, b, c, d] (LiteRT.DUS input[0]/input[1] )
  //      output: [b, a, c, d]
  Qnn_OpConfig_t transpose_operand_op = BuildDefaultOp();
  Qnn_OpConfig_t transpose_update_op = BuildDefaultOp();
  std::string transpose_operand_op_name =
      absl::StrFormat(kTransposeOperandOpFmt, op_counter_);
  std::string transpose_update_op_name =
      absl::StrFormat(kTransposeUpdateOpFmt, op_counter_);
  LITERT_RETURN_IF_ERROR(SetOpInfo(
      transpose_operand_op_name.c_str(), kDefaultQnnOpPackageName.data(),
      kTransposeOpTypeName.data(), transpose_operand_op));
  LITERT_RETURN_IF_ERROR(SetOpInfo(
      transpose_update_op_name.c_str(), kDefaultQnnOpPackageName.data(),
      kTransposeOpTypeName.data(), transpose_update_op));

  // Prepare transpose op params.
  std::vector<uint32_t> perm = {1, 0, 2, 3};
  std::vector<uint32_t> perm_dims = {kPermParamArgSize};
  Qnn_Param_t perm_param = BuildDefaultParam();
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildQnnTesnorParam<uint32_t>(
      perm.data(), perm_dims.data(), QNN_DATATYPE_UINT_32, kPermParamRank,
      kPermParamName.data(), graph_mapper, perm_param));

  // Prepare transpose op outputs.
  Qnn_Tensor_t transpose_operand_op_output = BuildDefaultTensor();
  Qnn_Tensor_t transpose_update_op_output = BuildDefaultTensor();

  // Cast const int to uint32_t.
  auto cast_f = [](int const_int) { return static_cast<uint32_t>(const_int); };

  std::vector<uint32_t> transpose_operand_op_output_dims(
      kTransposeOpOutputRank);
  std::vector<uint32_t> transpose_update_op_output_dims(kTransposeOpOutputRank);
  auto operand_dims = src.Inputs()[kDynamicUpdateSliceOpOperandIndex]
                          .RankedTensorType()
                          ->Layout()
                          .Dimensions();
  transpose_operand_op_output_dims[0] = cast_f(operand_dims[1]);
  transpose_operand_op_output_dims[1] = cast_f(operand_dims[0]);
  transpose_operand_op_output_dims[2] = cast_f(operand_dims[2]);
  transpose_operand_op_output_dims[3] = cast_f(operand_dims[3]);

  auto update_dims = src.Inputs()[kDynamicUpdateSliceOpUpdateIndex]
                         .RankedTensorType()
                         ->Layout()
                         .Dimensions();
  transpose_update_op_output_dims[0] = cast_f(update_dims[1]);
  transpose_update_op_output_dims[1] = cast_f(update_dims[0]);
  transpose_update_op_output_dims[2] = cast_f(update_dims[2]);
  transpose_update_op_output_dims[3] = cast_f(update_dims[3]);

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnNativeTensor(
      OperandDataType, kTransposeOpOutputRank,
      transpose_operand_op_output_dims.data(), graph_mapper,
      transpose_operand_op_output));
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnNativeTensor(
      UpdateDataType, kTransposeOpOutputRank,
      transpose_update_op_output_dims.data(), graph_mapper,
      transpose_update_op_output));

  // Configure transpose ops.
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnOp(
      kTransposeOpInputSize, &qnn_op_ins[kDynamicUpdateSliceOpOperandIndex],
      kTransposeOpOutputSize, &transpose_operand_op_output,
      transpose_operand_op, kTransposeParamSize, &perm_param, graph_mapper));
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnOp(
      kTransposeOpInputSize, &qnn_op_ins[kDynamicUpdateSliceOpUpdateIndex],
      kTransposeOpOutputSize, &transpose_update_op_output, transpose_update_op,
      kTransposeParamSize, &perm_param, graph_mapper));

  //===========================================================================
  // Step 3 Build ScatterND op.
  Qnn_OpConfig_t scatter_nd_op = BuildDefaultOp();
  std::string scatter_nd_op_name =
      absl::StrFormat(kScatterNdOpFmt, op_counter_);
  LITERT_RETURN_IF_ERROR(
      SetOpInfo(scatter_nd_op_name.c_str(), kDefaultQnnOpPackageName.data(),
                kQnnScatterNdOpTypeName.data(), scatter_nd_op));

  // Prepare scatter nd op output tensor.
  Qnn_Tensor_t scatter_nd_op_output = BuildDefaultTensor();
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      BuildAndRegisterQnnNativeTensor(OperandDataType, kScatterNDOutputRank,
                                      transpose_operand_op_output_dims.data(),
                                      graph_mapper, scatter_nd_op_output));

  // Configure ScatterND op.
  LITERT_STACK_ARRAY(Qnn_Tensor_t, scatter_nd_op_ins, kScatterNDOpInputSize,
                     QNN_TENSOR_INIT);
  scatter_nd_op_ins[0] = transpose_operand_op_output;
  scatter_nd_op_ins[1] = reshape_op_out;
  scatter_nd_op_ins[2] = transpose_update_op_output;
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnOp(
      kScatterNDOpInputSize, scatter_nd_op_ins, kScatterNDOpOutputSize,
      &scatter_nd_op_output, scatter_nd_op, kScatterNDParamSize, nullptr,
      graph_mapper));

  //===========================================================================
  // Step 4 Build final transpose op. Swap back the first two dimensions of the
  // scatter nd op output.
  // op.
  //      input: [b, a, c, d] (QNN.ScatterND output)
  //      output: [a, b, c, d]
  std::string transpose_result_op_name = absl::StrFormat(
      kTransposeResultOpFmt, /*increase counter*/ op_counter_++);
  LITERT_RETURN_IF_ERROR(SetOpInfo(transpose_result_op_name.c_str(),
                                   kDefaultQnnOpPackageName.data(),
                                   kTransposeOpTypeName.data(), dest));

  // Legalize op outputs and update scope.
  const auto op_outs = src.Outputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, op_outs.size(),
                     QNN_TENSOR_INIT);
  LITERT_RETURN_IF_ERROR(
      graph_mapper.LegalizeAndRegister(op_outs.front().Get(), qnn_op_outs[0]));
  LITERT_RETURN_IF_ERROR(
      graph_mapper.PushToScope(op_outs.front().Get(), qnn_op_outs[0]));

  // Configure transpose op.
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(BuildAndRegisterQnnOp(
      kTransposeOpInputSize, &scatter_nd_op_output, kTransposeOpOutputSize,
      &qnn_op_outs[0], dest, kTransposeParamSize, &perm_param, graph_mapper));

  LITERT_LOG(LITERT_INFO, "Legalized dynamic update slice op");

  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
