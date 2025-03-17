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

#include "tensorflow/lite/experimental/litert/c/litert_options.h"

#include <cstdint>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Op Options
//

LiteRtStatus LiteRtGetAddFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorNotFound;
  }
  *fused_activation = opts.AsAddOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAdjXOption(LiteRtOp op, bool* adj_x) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *adj_x = opts.AsBatchMatMulOptions()->adj_x;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAdjYOption(LiteRtOp op, bool* adj_y) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *adj_y = opts.AsBatchMatMulOptions()->adj_y;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input =
      opts.AsBatchMatMulOptions()->asymmetric_quantize_inputs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConcatenationFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsConcatenationOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConcatenationAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->OpCode() != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = opts.AsConcatenationOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDivFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflDiv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsDivOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsFullyConnectedOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedKeepNumDimsOption(LiteRtOp op,
                                                      bool* keep_num_dims) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *keep_num_dims = opts.AsFullyConnectedOptions()->keep_num_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFullyConnectedGetQuantizedBiasTypeOption(
    LiteRtOp op, uint32_t* quantized_bias_type) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *quantized_bias_type = opts.AsFullyConnectedOptions()->quantized_bias_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input =
      opts.AsFullyConnectedOptions()->asymmetric_quantize_inputs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedWeightsFormatOption(
    LiteRtOp op, uint32_t* weights_format) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *weights_format = opts.AsFullyConnectedOptions()->weights_format;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMulFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsMulOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSoftmaxBetaOption(LiteRtOp op, float* beta) {
  if (op->OpCode() != kLiteRtOpCodeTflSoftmax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *beta = opts.AsSoftmaxOptions()->beta;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceBeginMaskOption(LiteRtOp op,
                                                  int32_t* begin_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *begin_mask = opts.AsStridedSliceOptions()->begin_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceEndMaskOption(LiteRtOp op,
                                                int32_t* end_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *end_mask = opts.AsStridedSliceOptions()->end_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceEllipsisMaskOption(LiteRtOp op,
                                                     int32_t* ellipsis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *ellipsis_mask = opts.AsStridedSliceOptions()->ellipsis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceNewAxisMaskOption(LiteRtOp op,
                                                    int32_t* new_axis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *new_axis_mask = opts.AsStridedSliceOptions()->new_axis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceShrinkAxisMaskOption(
    LiteRtOp op, int32_t* shrink_axis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *shrink_axis_mask = opts.AsStridedSliceOptions()->shrink_axis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceOffsetOption(LiteRtOp op, bool* offset) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *offset = opts.AsStridedSliceOptions()->offset;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSubFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflSub) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = opts.AsSubOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetReshapeNewShapeOption(LiteRtOp op,
                                            const int32_t** new_shape,
                                            int32_t* new_shape_size) {
  if (op->OpCode() != kLiteRtOpCodeTflReshape) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    *new_shape_size = -1;
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (opts.AsReshapeOptions() == nullptr) {
    *new_shape_size = -1;
    return kLiteRtStatusOk;
  } else {
    *new_shape = opts.AsReshapeOptions()->new_shape.data();
    *new_shape_size = opts.AsReshapeOptions()->new_shape.size();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSumKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Sum OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetPackAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->OpCode() != kLiteRtOpCodeTflPack) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = opts.AsPackOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGatherAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->OpCode() != kLiteRtOpCodeTflGather) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = opts.AsGatherOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetGatherBatchDimsOption(LiteRtOp op, int32_t* batch_dims) {
  if (op->OpCode() != kLiteRtOpCodeTflGather) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *batch_dims = opts.AsGatherOptions()->batch_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMeanKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflMean) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Mean OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSplitNumSplitsOption(LiteRtOp op, int32_t* num_splits) {
  if (op->OpCode() != kLiteRtOpCodeTflSplit) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_splits = opts.AsSplitOptions()->num_splits;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSHLOCompositeOpName(LiteRtOp op, const char** name) {
  if (op->OpCode() != kLiteRtOpCodeShloComposite) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions2(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *name = opts.AsStableHLOCompositeOptions()->name.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSHLOCompositeOpDecompositionSubgraphIndex(
    LiteRtOp op, int32_t* subgraph_index) {
  if (op->OpCode() != kLiteRtOpCodeShloComposite) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions2(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *subgraph_index =
      opts.AsStableHLOCompositeOptions()->decomposition_subgraph_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dPaddingOption(LiteRtOp op, uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsConv2DOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dStrideWOption(LiteRtOp op, int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsConv2DOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dStrideHOption(LiteRtOp op, int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsConv2DOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsConv2DOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dDilationWOption(LiteRtOp op,
                                            int32_t* dilation_w_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_w_factor = opts.AsConv2DOptions()->dilation_w_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConv2dDilationHOption(LiteRtOp op,
                                            int32_t* dilation_h_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_h_factor = opts.AsConv2DOptions()->dilation_h_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dPaddingOption(LiteRtOp op,
                                                   uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsDepthwiseConv2DOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dStrideWOption(LiteRtOp op,
                                                   int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsDepthwiseConv2DOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dStrideHOption(LiteRtOp op,
                                                   int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsDepthwiseConv2DOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsDepthwiseConv2DOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dDilationWOption(
    LiteRtOp op, int32_t* dilation_w_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_w_factor = opts.AsDepthwiseConv2DOptions()->dilation_w_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthwiseConv2dDilationHOptions(
    LiteRtOp op, int32_t* dilation_h_factor) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthwiseConv2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *dilation_h_factor = opts.AsDepthwiseConv2DOptions()->dilation_h_factor;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dOptions(LiteRtOp op, int8_t* padding,
                                           int32_t* stride_w, int32_t* stride_h,
                                           int32_t* filter_width,
                                           int32_t* filter_height,
                                           int8_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto* options = opts.AsPool2DOptions();
  *padding = options->padding;
  *stride_w = options->stride_w;
  *stride_h = options->stride_h;
  *filter_width = options->filter_width;
  *filter_height = options->filter_height;
  *fused_activation_function = options->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dPaddingOption(LiteRtOp op,
                                                 uint32_t* padding) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *padding = opts.AsPool2DOptions()->padding;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dStrideWOption(LiteRtOp op,
                                                 int32_t* stride_w) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_w = opts.AsPool2DOptions()->stride_w;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dStrideHOption(LiteRtOp op,
                                                 int32_t* stride_h) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *stride_h = opts.AsPool2DOptions()->stride_h;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dFilterWidthOption(LiteRtOp op,
                                                     int32_t* filter_width) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *filter_width = opts.AsPool2DOptions()->filter_width;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dFilterHeightOption(LiteRtOp op,
                                                      int32_t* filter_height) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *filter_height = opts.AsPool2DOptions()->filter_height;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAveragePool2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function) {
  if (op->OpCode() != kLiteRtOpCodeTflAveragePool2d) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation_function =
      opts.AsPool2DOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetResizeBilinearAlignCornersOption(LiteRtOp op,
                                                       bool* align_corners) {
  if (op->OpCode() != kLiteRtOpCodeTflResizeBilinear) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *align_corners = opts.AsResizeBilinearOptions()->align_corners;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetResizeBilinearHalfPixelCenterOption(
    LiteRtOp op, bool* half_pixel_centers) {
  if (op->OpCode() != kLiteRtOpCodeTflResizeBilinear) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *half_pixel_centers = opts.AsResizeBilinearOptions()->half_pixel_centers;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetLeakyReluAlphaOption(LiteRtOp op, float* alpha) {
  if (op->OpCode() != kLiteRtOpCodeTflLeakyRelu) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *alpha = opts.AsLeakyReluOptions()->alpha;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDepthToSpaceBlockSizeOption(LiteRtOp op,
                                                  int32_t* block_size) {
  if (op->OpCode() != kLiteRtOpCodeTflDepthToSpace) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *block_size = opts.AsDepthToSpaceOptions()->block_size;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSpaceToDepthBlockSizeOption(LiteRtOp op,
                                                  int32_t* block_size) {
  if (op->OpCode() != kLiteRtOpCodeTflSpaceToDepth) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto& opts = litert::internal::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *block_size = opts.AsSpaceToDepthOptions()->block_size;
  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}  // extern "C"
#endif
