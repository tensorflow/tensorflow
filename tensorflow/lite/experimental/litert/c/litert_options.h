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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_OPTIONS_H_

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stdint.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

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
LiteRtStatus LiteRtGetAddFusedActivationOption(LiteRtOp op,
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
LiteRtStatus LiteRtGetBatchMatmulAdjXOption(LiteRtOp op, bool* adj_x);
LiteRtStatus LiteRtGetBatchMatmulAdjYOption(LiteRtOp op, bool* adj_y);
LiteRtStatus LiteRtGetBatchMatmulAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input);

//==============================================================================
//
//  Get option APIs for LiteRt Concatenation op.
//  Options:
//  - FusedActivationOption : uint32_t
//  - AxisOption : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetConcatenationFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation);
LiteRtStatus LiteRtGetConcatenationAxisOption(LiteRtOp op, int32_t* axis);

//==============================================================================
//
// Get option APIs for LiteRt Div op.
//  Options:
//  - FusedActivationOption : uint32_t
//
//==============================================================================
LiteRtStatus LiteRtGetDivFusedActivationOption(LiteRtOp op,
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
LiteRtStatus LiteRtGetFullyConnectedFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation);
LiteRtStatus LiteRtGetFullyConnectedWeightsFormatOption(
    LiteRtOp op, uint32_t* weights_format);
LiteRtStatus LiteRtGetFullyConnectedKeepNumDimsOption(LiteRtOp op,
                                                      bool* keep_num_dims);
LiteRtStatus LiteRtFullyConnectedGetQuantizedBiasTypeOption(
    LiteRtOp op, uint32_t* quantized_bias_type);
LiteRtStatus LiteRtGetFullyConnectedAsymmetricQuantizeInputOption(
    LiteRtOp op, bool* asymmetric_quantize_input);

//==============================================================================
//
// Get option APIs for LiteRt Mul op.
//  Options:
//  - FusedActivationOption : uint32_t
//
//==============================================================================
LiteRtStatus LiteRtGetMulFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation);

//==============================================================================
//
// Get option APIs for LiteRt Softmax op.
//  Options:
//  - BetaOption : float
//
//==============================================================================
LiteRtStatus LiteRtGetSoftmaxBetaOption(LiteRtOp op, float* beta);

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
LiteRtStatus LiteRtGetStridedSliceBeginMaskOption(LiteRtOp op,
                                                  int32_t* begin_mask);
LiteRtStatus LiteRtGetStridedSliceEndMaskOption(LiteRtOp op, int32_t* end_mask);
LiteRtStatus LiteRtGetStridedSliceEllipsisMaskOption(LiteRtOp op,
                                                     int32_t* ellipsis_mask);
LiteRtStatus LiteRtGetStridedSliceNewAxisMaskOption(LiteRtOp op,
                                                    int32_t* new_axis_mask);
LiteRtStatus LiteRtGetStridedSliceShrinkAxisMaskOption(
    LiteRtOp op, int32_t* shrink_axis_mask);
LiteRtStatus LiteRtGetStridedSliceOffsetOption(LiteRtOp op, bool* offset);

//==============================================================================
//
// Get option APIs for LiteRt Sub op.
//  Options:
//  - FusedActivationOption : uint32_t
//  - (Not supported) PotScaleInt16Option : bool
//
//==============================================================================
LiteRtStatus LiteRtGetSubFusedActivationOption(LiteRtOp op,
                                               uint32_t* fused_activation);

//==============================================================================
//
// Get option APIs for LiteRt Reshape op.
//  Options:
//  - new_shape : int32_t[]
//
//==============================================================================
LiteRtStatus LiteRtGetReshapeNewShapeOption(LiteRtOp op,
                                            const int32_t** new_shape,
                                            int32_t* new_shape_size);

//==============================================================================
//
// Get option APIs for LiteRt Sum op.
//  Options:
// - KeepdimsOption : bool
//
//==============================================================================
LiteRtStatus LiteRtGetSumKeepDimsOption(LiteRtOp op, bool* keepdims);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_OPTIONS_H_
