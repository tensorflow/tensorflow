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
  const auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
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
  auto& opts = detail::GetTflOptions(*op);
  if (opts.value == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Sum OP options is stored as ReducerOptions.
  *keepdims = opts.AsReducerOptions()->keep_dims;
  return kLiteRtStatusOk;
}
