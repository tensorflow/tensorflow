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

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_options.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"

//
// Op Options
//

LrtStatus LrtAddGetFusedActivationOption(LrtOp op, uint32_t* fused_activation) {
  if (op->op_code != kLrtOpCodeTflAdd) {
    return kLrtStatusErrorInvalidArgument;
  }
  *fused_activation = op->option.AsAddOptions()->fused_activation_function;
  return kLrtStatusOk;
}

LrtStatus LrtBatchMatmulGetAdjXOption(LrtOp op, bool* adj_x) {
  if (op->op_code != kLrtOpCodeTflBatchMatmul) {
    return kLrtStatusErrorInvalidArgument;
  }
  *adj_x = op->option.AsBatchMatMulOptions()->adj_x;
  return kLrtStatusOk;
}

LrtStatus LrtBatchMatmulGetAdjYOption(LrtOp op, bool* adj_y) {
  if (op->op_code != kLrtOpCodeTflBatchMatmul) {
    return kLrtStatusErrorInvalidArgument;
  }
  *adj_y = op->option.AsBatchMatMulOptions()->adj_y;
  return kLrtStatusOk;
}

LrtStatus LrtBatchMatmulGetAsymmetricQuantizeInputOption(
    LrtOp op, bool* asymmetric_quantize_input) {
  if (op->op_code != kLrtOpCodeTflBatchMatmul) {
    return kLrtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input =
      op->option.AsBatchMatMulOptions()->asymmetric_quantize_inputs;
  return kLrtStatusOk;
}

LrtStatus LrtConcatenationGetFusedActivationOption(LrtOp op,
                                                   uint32_t* fused_activation) {
  if (op->op_code != kLrtOpCodeTflConcatenation) {
    return kLrtStatusErrorInvalidArgument;
  }
  *fused_activation =
      op->option.AsConcatenationOptions()->fused_activation_function;
  return kLrtStatusOk;
}

LrtStatus LrtConcatenationGetAxisOption(LrtOp op, int32_t* axis) {
  if (op->op_code != kLrtOpCodeTflConcatenation) {
    return kLrtStatusErrorInvalidArgument;
  }
  *axis = op->option.AsConcatenationOptions()->axis;
  return kLrtStatusOk;
}

LrtStatus LrtDivGetFusedActivationOption(LrtOp op, uint32_t* fused_activation) {
  if (op->op_code != kLrtOpCodeTflDiv) {
    return kLrtStatusErrorInvalidArgument;
  }
  *fused_activation = op->option.AsDivOptions()->fused_activation_function;
  return kLrtStatusOk;
}

LrtStatus LrtFullyConnectedGetFusedActivationOption(
    LrtOp op, uint32_t* fused_activation) {
  if (op->op_code != kLrtOpCodeTflFullyConnected) {
    return kLrtStatusErrorInvalidArgument;
  }
  *fused_activation =
      op->option.AsFullyConnectedOptions()->fused_activation_function;
  return kLrtStatusOk;
}

LrtStatus LrtFullyConnectedGetKeepNumDimsOption(LrtOp op, bool* keep_num_dims) {
  if (op->op_code != kLrtOpCodeTflFullyConnected) {
    return kLrtStatusErrorInvalidArgument;
  }
  *keep_num_dims = op->option.AsFullyConnectedOptions()->keep_num_dims;
  return kLrtStatusOk;
}

LrtStatus LrtFullyConnectedGetQuantizedBiasTypeOption(
    LrtOp op, uint32_t* quantized_bias_type) {
  if (op->op_code != kLrtOpCodeTflFullyConnected) {
    return kLrtStatusErrorInvalidArgument;
  }
  *quantized_bias_type =
      op->option.AsFullyConnectedOptions()->quantized_bias_type;
  return kLrtStatusOk;
}

LrtStatus LrtFullyConnectedGetAsymmetricQuantizeInputOption(
    LrtOp op, bool* asymmetric_quantize_input) {
  if (op->op_code != kLrtOpCodeTflFullyConnected) {
    return kLrtStatusErrorInvalidArgument;
  }
  *asymmetric_quantize_input =
      op->option.AsFullyConnectedOptions()->asymmetric_quantize_inputs;
  return kLrtStatusOk;
}

LrtStatus LrtFullyConnectedGetWeightsFormatOption(LrtOp op,
                                                  uint32_t* weights_format) {
  if (op->op_code != kLrtOpCodeTflFullyConnected) {
    return kLrtStatusErrorInvalidArgument;
  }
  *weights_format = op->option.AsFullyConnectedOptions()->weights_format;
  return kLrtStatusOk;
}
LrtStatus LrtMulGetFusedActivationOption(LrtOp op, uint32_t* fused_activation) {
  if (op->op_code != kLrtOpCodeTflMul) {
    return kLrtStatusErrorInvalidArgument;
  }
  *fused_activation = op->option.AsMulOptions()->fused_activation_function;
  return kLrtStatusOk;
}

LrtStatus LrtSoftmaxGetBetaOption(LrtOp op, float* beta) {
  if (op->op_code != kLrtOpCodeTflSoftmax) {
    return kLrtStatusErrorInvalidArgument;
  }
  *beta = op->option.AsSoftmaxOptions()->beta;
  return kLrtStatusOk;
}

LrtStatus LrtStridedSliceGetBeginMaskOption(LrtOp op, int32_t* begin_mask) {
  if (op->op_code != kLrtOpCodeTflStridedSlice) {
    return kLrtStatusErrorInvalidArgument;
  }
  *begin_mask = op->option.AsStridedSliceOptions()->begin_mask;
  return kLrtStatusOk;
}

LrtStatus LrtStridedSliceGetEndMaskOption(LrtOp op, int32_t* end_mask) {
  if (op->op_code != kLrtOpCodeTflStridedSlice) {
    return kLrtStatusErrorInvalidArgument;
  }
  *end_mask = op->option.AsStridedSliceOptions()->end_mask;
  return kLrtStatusOk;
}

LrtStatus LrtStridedSliceGetEllipsisMaskOption(LrtOp op,
                                               int32_t* ellipsis_mask) {
  if (op->op_code != kLrtOpCodeTflStridedSlice) {
    return kLrtStatusErrorInvalidArgument;
  }
  *ellipsis_mask = op->option.AsStridedSliceOptions()->ellipsis_mask;
  return kLrtStatusOk;
}

LrtStatus LrtStridedSliceGetNewAxisMaskOption(LrtOp op,
                                              int32_t* new_axis_mask) {
  if (op->op_code != kLrtOpCodeTflStridedSlice) {
    return kLrtStatusErrorInvalidArgument;
  }
  *new_axis_mask = op->option.AsStridedSliceOptions()->new_axis_mask;
  return kLrtStatusOk;
}

LrtStatus LrtStridedSliceGetShrinkAxisMaskOption(LrtOp op,
                                                 int32_t* shrink_axis_mask) {
  if (op->op_code != kLrtOpCodeTflStridedSlice) {
    return kLrtStatusErrorInvalidArgument;
  }
  *shrink_axis_mask = op->option.AsStridedSliceOptions()->shrink_axis_mask;
  return kLrtStatusOk;
}

LrtStatus LrtStridedSliceGetOffsetOption(LrtOp op, bool* offset) {
  if (op->op_code != kLrtOpCodeTflStridedSlice) {
    return kLrtStatusErrorInvalidArgument;
  }
  *offset = op->option.AsStridedSliceOptions()->offset;
  return kLrtStatusOk;
}

LrtStatus LrtSubGetFusedActivationOption(LrtOp op, uint32_t* fused_activation) {
  if (op->op_code != kLrtOpCodeTflSub) {
    return kLrtStatusErrorInvalidArgument;
  }
  *fused_activation = op->option.AsSubOptions()->fused_activation_function;
  return kLrtStatusOk;
}

LrtStatus LrtReshapeGetNewShapeOption(LrtOp op, int32_t** new_shape,
                                      int32_t* new_shape_size) {
  if (op->op_code != kLrtOpCodeTflReshape) {
    return kLrtStatusErrorInvalidArgument;
  }
  if (op->option.AsReshapeOptions() == nullptr) {
    *new_shape_size = -1;
    return kLrtStatusOk;
  } else {
    *new_shape = op->option.AsReshapeOptions()->new_shape.data();
    *new_shape_size = op->option.AsReshapeOptions()->new_shape.size();
  }
  return kLrtStatusOk;
}
