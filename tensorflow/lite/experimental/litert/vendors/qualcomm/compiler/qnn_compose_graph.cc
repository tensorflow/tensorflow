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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/qnn_compose_graph.h"

#include <alloca.h>
#include <stdbool.h>
#include <stdio.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/tools/dump.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/cast_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/conv2d_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/depthwise_conv2d_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/dynamic_update_slice_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/gather_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/gelu_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/hard_swish_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/leaky_relu_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/mean_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/pack_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/pool2d_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/quantize_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/reduce_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/resize_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/rms_norm_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/select_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/spatial_transform_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/tanh_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

using ::litert::internal::Dump;

LiteRtStatus ConvertPaddingType(const uint32_t litert_padding,
                                ::qnn::PaddingType& qnn_padding) {
  switch (litert_padding) {
    case 0: {
      qnn_padding = ::qnn::PaddingType::Same;
      break;
    }
    case 1: {
      qnn_padding = ::qnn::PaddingType::Valid;
      break;
    }
    default: {
      return kLiteRtStatusErrorUnsupported;
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ConvertDataType(const litert::ElementType litert_type,
                             const bool is_quantized,
                             Qnn_DataType_t& qnn_type) {
  qnn_type = QNN_DATATYPE_UNDEFINED;
  switch (litert_type) {
    case litert::ElementType::Bool:
      qnn_type = QNN_DATATYPE_BOOL_8;
      break;
    case litert::ElementType::Int4:
      qnn_type = QNN_DATATYPE_SFIXED_POINT_4;
      break;
    case litert::ElementType::Int8:
      qnn_type =
          is_quantized ? QNN_DATATYPE_SFIXED_POINT_8 : QNN_DATATYPE_INT_8;
      break;
    case litert::ElementType::Int16:
      qnn_type =
          is_quantized ? QNN_DATATYPE_SFIXED_POINT_16 : QNN_DATATYPE_INT_16;
      break;
    case litert::ElementType::Int32:
      qnn_type =
          is_quantized ? QNN_DATATYPE_SFIXED_POINT_32 : QNN_DATATYPE_INT_32;
      break;
    case litert::ElementType::Int64:
      qnn_type = QNN_DATATYPE_INT_64;
      break;
    case litert::ElementType::UInt8:
      qnn_type =
          is_quantized ? QNN_DATATYPE_UFIXED_POINT_8 : QNN_DATATYPE_UINT_8;
      break;
    case litert::ElementType::UInt16:
      qnn_type =
          is_quantized ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UINT_16;
      break;
    case litert::ElementType::UInt32:
      qnn_type =
          is_quantized ? QNN_DATATYPE_UFIXED_POINT_32 : QNN_DATATYPE_UINT_32;
      break;
    case litert::ElementType::UInt64:
      qnn_type = QNN_DATATYPE_UINT_64;
      break;
    case litert::ElementType::Float16:
      qnn_type = QNN_DATATYPE_FLOAT_16;
      break;
    case litert::ElementType::Float32:
      qnn_type = QNN_DATATYPE_FLOAT_32;
      break;
    case litert::ElementType::Float64:
      qnn_type = QNN_DATATYPE_FLOAT_64;
      break;
    default:
      return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ConvertTensor(const litert::Tensor& litert_tensor,
                           ::qnn::TensorPool& tensor_pool,
                           ::qnn::TensorWrapper*& tensor_wrapper,
                           bool is_tensor_read_and_write) {
  tensor_wrapper = nullptr;

  if (litert_tensor.TypeId() != kLiteRtRankedTensorType) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  const auto ranked_tensor_type = litert_tensor.RankedTensorType();
  if (!ranked_tensor_type) {
    LITERT_LOG(LITERT_ERROR, "%s", ranked_tensor_type.Error().Message().data());
    return ranked_tensor_type.Error().Status();
  }

  Qnn_DataType_t qnn_data_type;
  LITERT_RETURN_IF_ERROR(ConvertDataType(ranked_tensor_type->ElementType(),
                                         litert_tensor.HasQuantization(),
                                         qnn_data_type));

  std::vector<std::uint32_t> dimentions;
  const auto litert_layout = ranked_tensor_type->Layout();
  if (litert_layout.Rank() == 0) {
    dimentions.resize(1, 1);
  } else {
    dimentions.resize(litert_layout.Rank());
    for (size_t i = 0; i < dimentions.size(); ++i) {
      dimentions[i] = litert_layout.Dimensions()[i];
    }
  }

  ::qnn::QuantizeParamsWrapperVariant quantize_params;
  switch (litert_tensor.QTypeId()) {
    case kLiteRtQuantizationPerTensor: {
      const auto per_tensor_quant = litert_tensor.PerTensorQuantization();
      quantize_params.emplace<::qnn::ScaleOffsetQuantizeParamsWrapper>(
          per_tensor_quant.scale, per_tensor_quant.zero_point);
      break;
    }
    case kLiteRtQuantizationPerChannel: {
      const auto per_channel_quant = litert_tensor.PerChannelQuantization();
      // convert zero points from std::int64_t to std::int32_t
      std::vector<std::int32_t> zero_points(per_channel_quant.num_channels);
      for (size_t i = 0; i < zero_points.size(); ++i) {
        zero_points[i] = per_channel_quant.zero_points[i];
      }
      quantize_params.emplace<::qnn::AxisScaleOffsetQuantizeParamsWrapper>(
          per_channel_quant.quantized_dimension,
          absl::Span<const float>{per_channel_quant.scales,
                                  per_channel_quant.num_channels},
          absl::Span<const std::int32_t>{zero_points.data(),
                                         zero_points.size()});
      break;
    }
    case kLiteRtQuantizationBlockWise: {
      LITERT_LOG(LITERT_ERROR, "Unsupported quantization type.");
      return kLiteRtStatusErrorInvalidArgument;
    }
    case kLiteRtQuantizationNone:
    default:
      break;
  }

  if (litert_tensor.IsSubgraphInput()) {
    auto& res = tensor_pool.CreateInputTensor(qnn_data_type, quantize_params,
                                              dimentions);
    tensor_wrapper = &res;
  } else if (litert_tensor.IsSubgraphOutput() || is_tensor_read_and_write) {
    auto& res = tensor_pool.CreateOutpuTensor(qnn_data_type, quantize_params,
                                              dimentions);
    tensor_wrapper = &res;
  } else if (litert_tensor.IsConstant()) {
    LITERT_ENSURE(litert_tensor.HasWeights(),
                  kLiteRtStatusErrorInvalidLegalization,
                  "Empty weights for constant tensor.");
    auto& res = tensor_pool.CreateStaticTensor(
        qnn_data_type, quantize_params, dimentions,
        litert_tensor.Weights().Bytes().size(),
        reinterpret_cast<const void*>(litert_tensor.Weights().Bytes().data()));
    tensor_wrapper = &res;
  } else {
    auto& res = tensor_pool.CreateNativeTensor(qnn_data_type, quantize_params,
                                               dimentions);
    tensor_wrapper = &res;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus ConvertOp(
    const litert::Op& litert_op, ::qnn::TensorPool& tensor_pool,
    const std::vector<::qnn::TensorWrapperRef>& input_tensors,
    const std::vector<::qnn::TensorWrapperRef>& output_tensors,
    std::vector<::qnn::OpWrapper>& op_wrappers) {
  switch (litert_op.Code()) {
    case LiteRtOpCode::kLiteRtOpCodeTflCast: {
      op_wrappers =
          ::qnn::BuildCastOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflConcatenation: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConcatenationAxisOption(litert_op.Get(), &axis));
      op_wrappers = ::qnn::BuildConcatenationOp(tensor_pool, input_tensors,
                                                output_tensors, axis);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflAdd: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetAddFusedActivationOption(
          litert_op.Get(), &fused_activation));
      op_wrappers = ::qnn::BuildElementwiseAddOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLogicalAnd: {
      op_wrappers = ::qnn::BuildElementwiseAndOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflCos: {
      op_wrappers = ::qnn::BuildElementwiseCosOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDiv: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetDivFusedActivationOption(
          litert_op.Get(), &fused_activation));
      op_wrappers = ::qnn::BuildElementwiseDivOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGreater: {
      op_wrappers = ::qnn::BuildElementwiseGreaterOp(tensor_pool, input_tensors,
                                                     output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLess: {
      op_wrappers = ::qnn::BuildElementwiseLessOp(tensor_pool, input_tensors,
                                                  output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMul: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetMulFusedActivationOption(
          litert_op.Get(), &fused_activation));
      op_wrappers = ::qnn::BuildElementwiseMulOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflRsqrt: {
      op_wrappers = ::qnn::BuildElementwiseRsqrtOp(tensor_pool, input_tensors,
                                                   output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSin: {
      op_wrappers = ::qnn::BuildElementwiseSinOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSquaredDifference: {
      op_wrappers = ::qnn::BuildElementwiseSquaredDifferenceOp(
          tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSquare: {
      op_wrappers = ::qnn::BuildElementwiseSquareOp(tensor_pool, input_tensors,
                                                    output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSub: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetSubFusedActivationOption(
          litert_op.Get(), &fused_activation));
      op_wrappers = ::qnn::BuildElementwiseSubOp(tensor_pool, input_tensors,
                                                 output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflEmbeddingLookup: {
      op_wrappers = ::qnn::BuildEmbeddingLookupOp(tensor_pool, input_tensors,
                                                  output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflFullyConnected: {
      uint32_t fused_activation{};
      LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedFusedActivationOption(
          litert_op.Get(), &fused_activation));
      bool keep_num_dims{};
      LITERT_RETURN_IF_ERROR(LiteRtGetFullyConnectedKeepNumDimsOption(
          litert_op.Get(), &keep_num_dims));
      op_wrappers = ::qnn::BuildFullyConnectedOp(tensor_pool, input_tensors,
                                                 output_tensors, keep_num_dims);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGather: {
      int32_t axis{};
      LITERT_RETURN_IF_ERROR(LiteRtGetGatherAxisOption(litert_op.Get(), &axis));
      int32_t batch_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetGatherBatchDimsOption(litert_op.Get(), &batch_dims));
      op_wrappers = ::qnn::BuildGatherOp(tensor_pool, input_tensors,
                                         output_tensors, axis, batch_dims);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflGelu: {
      op_wrappers =
          ::qnn::BuildGeluOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflBatchMatmul: {
      bool adj_x{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetBatchMatmulAdjXOption(litert_op.Get(), &adj_x));
      bool adj_y{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetBatchMatmulAdjYOption(litert_op.Get(), &adj_y));
      op_wrappers = ::qnn::BuildMatmulOp(tensor_pool, input_tensors,
                                         output_tensors, adj_x, adj_y);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflMean: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetMeanKeepDimsOption(litert_op.Get(), &keep_dims));
      op_wrappers = ::qnn::BuildMeanOp(tensor_pool, input_tensors,
                                       output_tensors, keep_dims);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflQuantize: {
      op_wrappers =
          ::qnn::BuildQuantizeOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSum: {
      bool keep_dims{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSumKeepDimsOption(litert_op.Get(), &keep_dims));
      op_wrappers = ::qnn::BuildReduceSumOp(tensor_pool, input_tensors,
                                            output_tensors, keep_dims);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflReshape: {
      op_wrappers =
          ::qnn::BuildReshapeOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSelect:
    case LiteRtOpCode::kLiteRtOpCodeTflSelectV2: {
      op_wrappers =
          ::qnn::BuildSelectOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSlice: {
      op_wrappers =
          ::qnn::BuildSliceOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSoftmax: {
      float beta{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSoftmaxBetaOption(litert_op.Get(), &beta));
      op_wrappers = ::qnn::BuildSoftmaxOp(tensor_pool, input_tensors,
                                          output_tensors, beta);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSplit: {
      int32_t num_splits{};
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSplitNumSplitsOption(litert_op.Get(), &num_splits));
      op_wrappers = ::qnn::BuildSplitOp(tensor_pool, input_tensors,
                                        output_tensors, num_splits);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTanh: {
      op_wrappers =
          ::qnn::BuildTanhOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflTranspose: {
      op_wrappers =
          ::qnn::BuildTransposeOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflPack: {
      int32_t axis{};
      LiteRtGetPackAxisOption(litert_op.Get(), &axis);
      op_wrappers =
          ::qnn::BuildPackOp(tensor_pool, input_tensors, output_tensors, axis);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDynamicUpdateSlice: {
      op_wrappers = ::qnn::BuildDynamicUpdateSliceOp(tensor_pool, input_tensors,
                                                     output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeShloComposite: {
      // TODO(yunandrew): Support custom epsilon for RMS Norm.
      float epsilon = 9.99999997E-7;
      op_wrappers = ::qnn::BuildRmsNormOp(tensor_pool, input_tensors,
                                          output_tensors, epsilon);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflConv2d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dStrideHOption(litert_op.Get(), &stride_h));
      int32_t dilation_w_factor;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dDilationWOption(litert_op.Get(), &dilation_w_factor));
      int32_t dilation_h_factor;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetConv2dDilationWOption(litert_op.Get(), &dilation_h_factor));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));
      op_wrappers = ::qnn::BuildConv2dOp(
          tensor_pool, input_tensors, output_tensors, stride_h, stride_w,
          dilation_h_factor, dilation_w_factor, qnn_padding);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDepthwiseConv2d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthwiseConv2dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthwiseConv2dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthwiseConv2dStrideHOption(litert_op.Get(), &stride_h));
      int32_t dilation_w_factor;
      LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dDilationWOption(
          litert_op.Get(), &dilation_w_factor));
      int32_t dilation_h_factor;
      LITERT_RETURN_IF_ERROR(LiteRtGetDepthwiseConv2dDilationHOptions(
          litert_op.Get(), &dilation_h_factor));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));
      op_wrappers = ::qnn::BuildDepthwiseConv2dOp(
          tensor_pool, input_tensors, output_tensors, stride_h, stride_w,
          dilation_h_factor, dilation_w_factor, qnn_padding);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflAveragePool2d: {
      uint32_t padding;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetAveragePool2dPaddingOption(litert_op.Get(), &padding));
      int32_t stride_w;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetAveragePool2dStrideWOption(litert_op.Get(), &stride_w));
      int32_t stride_h;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetAveragePool2dStrideHOption(litert_op.Get(), &stride_h));
      int32_t filter_width;
      LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dFilterWidthOption(
          litert_op.Get(), &filter_width));
      int32_t filter_height;
      LITERT_RETURN_IF_ERROR(LiteRtGetAveragePool2dFilterHeightOption(
          litert_op.Get(), &filter_height));

      ::qnn::PaddingType qnn_padding;
      LITERT_RETURN_IF_ERROR(ConvertPaddingType(padding, qnn_padding));
      op_wrappers = ::qnn::BuildAveragePoolOp(
          tensor_pool, input_tensors, output_tensors, stride_h, stride_w,
          filter_height, filter_width, qnn_padding);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflDepthToSpace: {
      int32_t block_size;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetDepthToSpaceBlockSizeOption(litert_op.Get(), &block_size));
      op_wrappers = ::qnn::BuildDepthToSpaceOp(tensor_pool, input_tensors,
                                               output_tensors, block_size);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflSpaceToDepth: {
      int32_t block_size;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetSpaceToDepthBlockSizeOption(litert_op.Get(), &block_size));
      op_wrappers = ::qnn::BuildSpaceToDepthOp(tensor_pool, input_tensors,
                                               output_tensors, block_size);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflHardSwish: {
      op_wrappers =
          ::qnn::BuildHardSwishOp(tensor_pool, input_tensors, output_tensors);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflLeakyRelu: {
      float alpha;
      LITERT_RETURN_IF_ERROR(
          LiteRtGetLeakyReluAlphaOption(litert_op.Get(), &alpha));
      op_wrappers = ::qnn::BuildLeakyReluOp(tensor_pool, input_tensors,
                                            output_tensors, alpha);
      break;
    }
    case LiteRtOpCode::kLiteRtOpCodeTflResizeBilinear: {
      bool align_corners;
      LITERT_RETURN_IF_ERROR(LiteRtGetResizeBilinearAlignCornersOption(
          litert_op.Get(), &align_corners));
      bool half_pixel_centers;
      LITERT_RETURN_IF_ERROR(LiteRtGetResizeBilinearHalfPixelCenterOption(
          litert_op.Get(), &half_pixel_centers));
      op_wrappers = ::qnn::BuildResizeBilinearOp(tensor_pool, input_tensors,
                                                 output_tensors, align_corners,
                                                 half_pixel_centers);
      break;
    }
    default: {
      LITERT_LOG(LITERT_ERROR,
                 "LiteRT Op Code: %d is not supported in Qualcomm Compiler.",
                 litert_op.Code());
    }
  }
  return kLiteRtStatusOk;
}

LiteRtStatus MapGraph(QnnManager& qnn, Qnn_ContextHandle_t context_handle,
                      LiteRtSubgraph subgraph,
                      absl::string_view qnn_graph_name) {
  GraphMapper graph_mapper(subgraph, qnn, context_handle);
  LITERT_RETURN_IF_ERROR(graph_mapper.IsLiteRtSubgraphSupported());
  LITERT_RETURN_IF_ERROR(graph_mapper.InitQnnGraph(qnn_graph_name));

  //
  // Legalize subgraph inputs and update tensors in scope
  //

  ::qnn::TensorPool tensor_pool;
  absl::flat_hash_map<LiteRtTensor, ::qnn::TensorWrapper*>
      litert_tensor_to_wrapper;

  for (const auto& subgraph_input : graph_mapper.Graph().Inputs()) {
    ::qnn::TensorWrapper* tensor_wrapper{nullptr};
    LITERT_RETURN_IF_ERROR(
        ConvertTensor(subgraph_input, tensor_pool, tensor_wrapper));
    litert_tensor_to_wrapper.emplace(subgraph_input.Get(), tensor_wrapper);
  }

  for (const auto& subgraph_output : graph_mapper.Graph().Outputs()) {
    graph_mapper.RegisterOutput(subgraph_output.Get());
  }
  //
  // Topologically traverse graph, legalizing and updating tensors in scope
  //

  // TODO: make ConvertOp accept a vector and append OpWrapper in it.
  std::vector<::qnn::OpWrapper> graph_op_wrappers;
  std::ostringstream dump;
  for (const auto& op : graph_mapper.Graph().Ops()) {
    // Dump op info.
    dump.clear();
    Dump(*op.Get(), dump);
    std::string s = dump.str();
    LITERT_LOG(LITERT_INFO, "%s", s.data());

    std::vector<::qnn::TensorWrapperRef> input_tensors;
    for (const auto& input : op.Inputs()) {
      if (const auto it = litert_tensor_to_wrapper.find(input.Get());
          it == litert_tensor_to_wrapper.end()) {
        ::qnn::TensorWrapper* tensor_wrapper{nullptr};
        LITERT_RETURN_IF_ERROR(
            ConvertTensor(input, tensor_pool, tensor_wrapper));
        // add into map to capture re-used static tensor
        litert_tensor_to_wrapper.emplace(input.Get(), tensor_wrapper);
        input_tensors.emplace_back(*tensor_wrapper);
      } else {
        input_tensors.emplace_back(*(it->second));
      }
    }

    std::vector<::qnn::TensorWrapperRef> output_tensors;
    for (const auto& output : op.Outputs()) {
      bool is_tensor_read_and_write = graph_mapper.IsTensorOutput(output.Get());
      ::qnn::TensorWrapper* tensor_wrapper{nullptr};
      LITERT_RETURN_IF_ERROR(ConvertTensor(output, tensor_pool, tensor_wrapper,
                                           is_tensor_read_and_write));
      litert_tensor_to_wrapper.emplace(output.Get(), tensor_wrapper);
      output_tensors.emplace_back(*tensor_wrapper);
    }

    std::vector<::qnn::OpWrapper> op_wrappers;
    LITERT_RETURN_IF_ERROR(
        ConvertOp(op, tensor_pool, input_tensors, output_tensors, op_wrappers));
    std::move(op_wrappers.begin(), op_wrappers.end(),
              std::back_inserter(graph_op_wrappers));
  }
  // Insert all tensors into Qnn graph and update the id of Qnn_Tensor_t inside.
  tensor_pool.ForEach(
      [&qnn, &graph_mapper](::qnn::TensorWrapper& tensor_wrapper) {
        qnn.Api()->tensorCreateGraphTensor(graph_mapper.QnnGraph(),
                                           &tensor_wrapper.GetQnnTensor());
      });
  // Then op can be added into Qnn graph after the tensor ids are updated.
  for (auto& op_wrapper : graph_op_wrappers) {
    qnn.Api()->graphAddNode(graph_mapper.QnnGraph(), op_wrapper.GetOpConfig());
  }

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(graph_mapper.Finalize());

  return kLiteRtStatusOk;
}

//===----------------------------------------------------------------------===//
//
//                                           [WIP] LiteRT SUBGRAPH -> QNN GRAPH
//
// Core driver for IR translation. Traverses LiteRt Subgraph, iteratively
// "legalizing" (mapping) LiteRt entities to their QNN counterpart.
//
// APPROACH:
//
// To support the general case we will need a driver loop that either
// traverses input recursively through edges or just iterates topologically.
//
// The algorithm is pretty straightforward:
// * Store mapping between already evaluated LiteRtTensors and their
//   newly constructed Qnn Tensor counterpart.
// * Look up QNN Tensors when setting QNN Op inputs.
// * Add new QNN Tensor when setting QNN Op outputs.
//
// NOTES ON QNN API:
//
// After QNN Tensors are registered in the context, they need only
// be stored as their ID. QNN Tensor and "id" : uint32_t are used
// interchangeably.
//
//===----------------------------------------------------------------------===//

LiteRtStatus ComposeGraph(QnnManager& qnn, Qnn_ContextHandle_t context_handle,
                          LiteRtSubgraph subgraph,
                          absl::string_view qnn_graph_name) {
  LITERT_RETURN_IF_ERROR(
      MapGraph(qnn, context_handle, subgraph, qnn_graph_name));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
