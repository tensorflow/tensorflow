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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITERT_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITERT_OPTIONS_H_

#include <cstdint>

#include "tensorflow/lite/experimental/lrt/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtOp);

//==============================================================================
//
//  Get option APIs for LiteRt ADD op.
//  Options:
//  - FusedActivationOption : uint32_t
//
//==============================================================================
LiteRtStatus LiteRtAddGetFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation);

//==============================================================================
//
//  Get option APIs for LiteRt BatchMatmul op.
//  Options:
//  - AdjXOption : bool
//  - AdjYOption : bool
//  - AsymmtericQuantizeInputOption : bool
//
//==============================================================================
LiteRtStatus LiteRtBatchMatmulGetAdjXOption(LiteRtOp op, bool* adj_x);
LiteRtStatus LiteRtBatchMatmulGetAdjYOption(LiteRtOp op, bool* adj_y);
LiteRtStatus LiteRtBatchMatmulGetAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input);

//==============================================================================
//
//  Get option APIs for LiteRt Concatenation op.
//  Options:
//  - FusedActivationOption : uint32_t
//  - AxisOption : int32_t
//
//==============================================================================
LiteRtStatus LiteRtConcatenationGetFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation);
LiteRtStatus LiteRtConcatenationGetAxisOption(LiteRtOp op, int32_t* axis);

//==============================================================================
//
// Get option APIs for LiteRt Div op.
//  Options:
//  - FusedActivationOption : uint32_t
//
//==============================================================================
LiteRtStatus LiteRtDivGetFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation);

//==============================================================================
//
// Get option APIs for LiteRt FullyConnected op.
//  Options:
//  - FusedActivationOption : uint32_t
//  - WeightsFormatOption : uint32_t
//  - KeepNumDimsOption : bool
//  - QuantizedBiasTypeOption : uint32_t
//  - AsymmtericQuantizeInputOption : bool
//
//==============================================================================
LiteRtStatus LiteRtFullyConnectedGetFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation);
LiteRtStatus LiteRtFullyConnectedGetWeightsFormatOption(
    LiteRtOp op, uint32_t* weights_format);
LiteRtStatus LiteRtFullyConnectedGetKeepNumDimsOption(LiteRtOp op,
                                                      bool* keep_num_dims);
LiteRtStatus LiteRtFullyConnectedGetQuantizedBiasTypeOption(
    LiteRtOp op, uint32_t* quantized_bias_type);
LiteRtStatus LiteRtFullyConnectedGetAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input);

//==============================================================================
//
// Get option APIs for LiteRt Mul op.
//  Options:
//  - FusedActivationOption : uint32_t
//
//==============================================================================
LiteRtStatus LiteRtMulGetFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation);

//==============================================================================
//
// Get option APIs for LiteRt Softmax op.
//  Options:
//  - BetaOption : float
//
//==============================================================================
LiteRtStatus LiteRtSoftmaxGetBetaOption(LiteRtOp op, float* beta);

//==============================================================================
//
// Get option APIs for LiteRt StridedSlice op.
//  Options:
//  - BeginMaskOption : int32_t
//  - EndMaskOption : int32_t
//  - EllipsisMaskOption : int32_t
//  - NewAxisMaskOption : int32_t
//  - ShrinkAxisMaskOption : int32_t
//  - OffsetOption : bool

//==============================================================================
LiteRtStatus LiteRtStridedSliceGetBeginMaskOption(LiteRtOp op,
                                                  int32_t* begin_mask);
LiteRtStatus LiteRtStridedSliceGetEndMaskOption(LiteRtOp op, int32_t* end_mask);
LiteRtStatus LiteRtStridedSliceGetEllipsisMaskOption(LiteRtOp op,
                                                     int32_t* ellipsis_mask);
LiteRtStatus LiteRtStridedSliceGetNewAxisMaskOption(LiteRtOp op,
                                                    int32_t* new_axis_mask);
LiteRtStatus LiteRtStridedSliceGetShrinkAxisMaskOption(
    LiteRtOp op, int32_t* shrink_axis_mask);
LiteRtStatus LiteRtStridedSliceGetOffsetOption(LiteRtOp op, bool* offset);

//==============================================================================
//
// Get option APIs for LiteRt Sub op.
//  Options:
//  - FusedActivationOption : uint32_t
//  - (Not supported) PotScaleInt16Option : bool
//
//==============================================================================
LiteRtStatus LiteRtSubGetFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation);

//==============================================================================
//
// Get option APIs for LiteRt Reshape op.
//  Options:
//  - new_shape : int32_t[]
//
//==============================================================================
LiteRtStatus LiteRtReshapeGetNewShapeOption(LiteRtOp op, int32_t** new_shape,
                                            int32_t* new_shape_size);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITERT_OPTIONS_H_
