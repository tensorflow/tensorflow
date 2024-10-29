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

#include <cstdint>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/core/model.h"

//
// Op Options
//

LiteRtStatus LiteRtAddGetFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->op_code != kLiteRtOpCodeTflAdd) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = op->option.AsAddOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBatchMatmulGetAdjXOption(LiteRtOp op, bool* adj_x) {
  if (op->op_code != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *adj_x = op->option.AsBatchMatMulOptions()->adj_x;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBatchMatmulGetAdjYOption(LiteRtOp op, bool* adj_y) {
  if (op->op_code != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *adj_y = op->option.AsBatchMatMulOptions()->adj_y;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtBatchMatmulGetAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input) {
  if (op->op_code != kLiteRtOpCodeTflBatchMatmul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input =
      op->option.AsBatchMatMulOptions()->asymmetric_quantize_inputs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtConcatenationGetFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation) {
  if (op->op_code != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation =
      op->option.AsConcatenationOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtConcatenationGetAxisOption(LiteRtOp op, int32_t* axis) {
  if (op->op_code != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *axis = op->option.AsConcatenationOptions()->axis;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDivGetFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->op_code != kLiteRtOpCodeTflDiv) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = op->option.AsDivOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFullyConnectedGetFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation) {
  if (op->op_code != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation =
      op->option.AsFullyConnectedOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFullyConnectedGetKeepNumDimsOption(LiteRtOp op,
                                                      bool* keep_num_dims) {
  if (op->op_code != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *keep_num_dims = op->option.AsFullyConnectedOptions()->keep_num_dims;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFullyConnectedGetQuantizedBiasTypeOption(
    LiteRtOp op, uint32_t* quantized_bias_type) {
  if (op->op_code != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *quantized_bias_type =
      op->option.AsFullyConnectedOptions()->quantized_bias_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFullyConnectedGetAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input) {
  if (op->op_code != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input =
      op->option.AsFullyConnectedOptions()->asymmetric_quantize_inputs;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtFullyConnectedGetWeightsFormatOption(
    LiteRtOp op, uint32_t* weights_format) {
  if (op->op_code != kLiteRtOpCodeTflFullyConnected) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *weights_format = op->option.AsFullyConnectedOptions()->weights_format;
  return kLiteRtStatusOk;
}
LiteRtStatus LiteRtMulGetFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->op_code != kLiteRtOpCodeTflMul) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = op->option.AsMulOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSoftmaxGetBetaOption(LiteRtOp op, float* beta) {
  if (op->op_code != kLiteRtOpCodeTflSoftmax) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *beta = op->option.AsSoftmaxOptions()->beta;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtStridedSliceGetBeginMaskOption(LiteRtOp op,
                                                  int32_t* begin_mask) {
  if (op->op_code != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *begin_mask = op->option.AsStridedSliceOptions()->begin_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtStridedSliceGetEndMaskOption(LiteRtOp op,
                                                int32_t* end_mask) {
  if (op->op_code != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *end_mask = op->option.AsStridedSliceOptions()->end_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtStridedSliceGetEllipsisMaskOption(LiteRtOp op,
                                                     int32_t* ellipsis_mask) {
  if (op->op_code != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *ellipsis_mask = op->option.AsStridedSliceOptions()->ellipsis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtStridedSliceGetNewAxisMaskOption(LiteRtOp op,
                                                    int32_t* new_axis_mask) {
  if (op->op_code != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *new_axis_mask = op->option.AsStridedSliceOptions()->new_axis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtStridedSliceGetShrinkAxisMaskOption(
    LiteRtOp op, int32_t* shrink_axis_mask) {
  if (op->op_code != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *shrink_axis_mask = op->option.AsStridedSliceOptions()->shrink_axis_mask;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtStridedSliceGetOffsetOption(LiteRtOp op, bool* offset) {
  if (op->op_code != kLiteRtOpCodeTflStridedSlice) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *offset = op->option.AsStridedSliceOptions()->offset;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSubGetFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation) {
  if (op->op_code != kLiteRtOpCodeTflSub) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *fused_activation = op->option.AsSubOptions()->fused_activation_function;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtReshapeGetNewShapeOption(LiteRtOp op, int32_t** new_shape,
                                            int32_t* new_shape_size) {
  if (op->op_code != kLiteRtOpCodeTflReshape) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (op->option.AsReshapeOptions() == nullptr) {
    *new_shape_size = -1;
    return kLiteRtStatusOk;
  } else {
    *new_shape = op->option.AsReshapeOptions()->new_shape.data();
    *new_shape_size = op->option.AsReshapeOptions()->new_shape.size();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSumGetKeepdimsOption(LiteRtOp op, bool* keepdims) {
  if (op->op_code != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Sum OP options is stored as ReducerOptions.
  *keepdims = op->option.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}
