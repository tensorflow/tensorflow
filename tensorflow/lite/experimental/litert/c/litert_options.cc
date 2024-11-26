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

//
// Op Options
//

LiteRtStatus LiteRtGetAddFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation =
      detail::GetTflOptions(*op).AsAddOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAdjXOption(LiteRtOp op, bool* adj_x) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *adj_x = detail::GetTflOptions(*op).AsBatchMatMulOptions()->adj_x;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAdjYOption(LiteRtOp op, bool* adj_y) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *adj_y = detail::GetTflOptions(*op).AsBatchMatMulOptions()->adj_y;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetBatchMatmulAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input) {
  if (op->OpCode() != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input = detail::GetTflOptions(*op)
                                   .AsBatchMatMulOptions()
                                   ->asymmetric_quantize_inputs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConcatenationFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = detail::GetTflOptions(*op)
                          .AsConcatenationOptions()
                          ->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetConcatenationAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->OpCode() != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = detail::GetTflOptions(*op).AsConcatenationOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetDivFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflDiv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation =
      detail::GetTflOptions(*op).AsDivOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = detail::GetTflOptions(*op)
                          .AsFullyConnectedOptions()
                          ->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedKeepNumDimsOption(LiteRtOp op,
                                                      bool* keep_num_dims) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *keep_num_dims =
      detail::GetTflOptions(*op).AsFullyConnectedOptions()->keep_num_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFullyConnectedGetQuantizedBiasTypeOption(
    LiteRtOp op, uint32_t* quantized_bias_type) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *quantized_bias_type =
      detail::GetTflOptions(*op).AsFullyConnectedOptions()->quantized_bias_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input = detail::GetTflOptions(*op)
                                   .AsFullyConnectedOptions()
                                   ->asymmetric_quantize_inputs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetFullyConnectedWeightsFormatOption(
    LiteRtOp op, uint32_t* weights_format) {
  if (op->OpCode() != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *weights_format =
      detail::GetTflOptions(*op).AsFullyConnectedOptions()->weights_format;
  return kLiteRtStatusOk;
}
LiteRtStatus LiteRtGetMulFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation =
      detail::GetTflOptions(*op).AsMulOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSoftmaxBetaOption(LiteRtOp op, float* beta) {
  if (op->OpCode() != kLiteRtOpCodeTflSoftmax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *beta = detail::GetTflOptions(*op).AsSoftmaxOptions()->beta;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceBeginMaskOption(LiteRtOp op,
                                                  int32_t* begin_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *begin_mask = detail::GetTflOptions(*op).AsStridedSliceOptions()->begin_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceEndMaskOption(LiteRtOp op,
                                                int32_t* end_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *end_mask = detail::GetTflOptions(*op).AsStridedSliceOptions()->end_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceEllipsisMaskOption(LiteRtOp op,
                                                     int32_t* ellipsis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *ellipsis_mask =
      detail::GetTflOptions(*op).AsStridedSliceOptions()->ellipsis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceNewAxisMaskOption(LiteRtOp op,
                                                    int32_t* new_axis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *new_axis_mask =
      detail::GetTflOptions(*op).AsStridedSliceOptions()->new_axis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceShrinkAxisMaskOption(
    LiteRtOp op, int32_t* shrink_axis_mask) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *shrink_axis_mask =
      detail::GetTflOptions(*op).AsStridedSliceOptions()->shrink_axis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetStridedSliceOffsetOption(LiteRtOp op, bool* offset) {
  if (op->OpCode() != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *offset = detail::GetTflOptions(*op).AsStridedSliceOptions()->offset;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSubFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->OpCode() != kLiteRtOpCodeTflSub) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation =
      detail::GetTflOptions(*op).AsSubOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetReshapeNewShapeOption(LiteRtOp op,
                                            const int32_t** new_shape,
                                            int32_t* new_shape_size) {
  if (op->OpCode() != kLiteRtOpCodeTflReshape) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (detail::GetTflOptions(*op).AsReshapeOptions() == nullptr) {
    *new_shape_size = -1;
    return kLiteRtStatusOk;
  } else {
    *new_shape =
        detail::GetTflOptions(*op).AsReshapeOptions()->new_shape.data();
    *new_shape_size =
        detail::GetTflOptions(*op).AsReshapeOptions()->new_shape.size();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSumKeepDimsOption(LiteRtOp op, bool* keepdims) {
  if (op->OpCode() != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Sum OP options is stored as ReducerOptions.
  *keepdims = detail::GetTflOptions(*op).AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}
