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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_OPTIONS_H_

#include <cstdint>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITE_RT_DEFINE_HANDLE(LrtOp);

//==============================================================================
//
//  Get option APIs for LRT ADD op.
//  Options:
//  - FusedActivationOption : uint32_t
//
//==============================================================================
LrtStatus LrtAddGetFusedActivationOption(LrtOp op, uint32_t* fused_activation);

//==============================================================================
//
//  Get option APIs for LRT BatchMatmul op.
//  Options:
//  - AdjXOption : bool
//  - AdjYOption : bool
//  - AsymmtericQuantizeInputOption : bool
//
//==============================================================================
LrtStatus LrtBatchMatmulGetAdjXOption(LrtOp op, bool* adj_x);
LrtStatus LrtBatchMatmulGetAdjYOption(LrtOp op, bool* adj_y);
LrtStatus LrtBatchMatmulGetAsymmetricQuantizeInputOption(
    LrtOp op, bool* asymmetric_quantize_input);

//==============================================================================
//
//  Get option APIs for LRT Concatenation op.
//  Options:
//  - FusedActivationOption : uint32_t
//  - AxisOption : int32_t
//
//==============================================================================
LrtStatus LrtConcatenationGetFusedActivationOption(LrtOp op,
                                                   uint32_t* fused_activation);
LrtStatus LrtConcatenationGetAxisOption(LrtOp op, int32_t* axis);

//==============================================================================
//
// Get option APIs for LRT Div op.
//  Options:
//  - FusedActivationOption : uint32_t
//
//==============================================================================
LrtStatus LrtDivGetFusedActivationOption(LrtOp op, uint32_t* fused_activation);

//==============================================================================
//
// Get option APIs for LRT FullyConnected op.
//  Options:
//  - FusedActivationOption : uint32_t
//  - WeightsFormatOption : uint32_t
//  - KeepNumDimsOption : bool
//  - QuantizedBiasTypeOption : uint32_t
//  - AsymmtericQuantizeInputOption : bool
//
//==============================================================================
LrtStatus LrtFullyConnectedGetFusedActivationOption(LrtOp op,
                                                    uint32_t* fused_activation);
LrtStatus LrtFullyConnectedGetWeightsFormatOption(LrtOp op,
                                                  uint32_t* weights_format);
LrtStatus LrtFullyConnectedGetKeepNumDimsOption(LrtOp op, bool* keep_num_dims);
LrtStatus LrtFullyConnectedGetQuantizedBiasTypeOption(
    LrtOp op, uint32_t* quantized_bias_type);
LrtStatus LrtFullyConnectedGetAsymmetricQuantizeInputOption(
    LrtOp op, bool* asymmetric_quantize_input);

//==============================================================================
//
// Get option APIs for LRT Mul op.
//  Options:
//  - FusedActivationOption : uint32_t
//
//==============================================================================
LrtStatus LrtMulGetFusedActivationOption(LrtOp op, uint32_t* fused_activation);

//==============================================================================
//
// Get option APIs for LRT Softmax op.
//  Options:
//  - BetaOption : float
//
//==============================================================================
LrtStatus LrtSoftmaxGetBetaOption(LrtOp op, float* beta);

//==============================================================================
//
// Get option APIs for LRT StridedSlice op.
//  Options:
//  - BeginMaskOption : int32_t
//  - EndMaskOption : int32_t
//  - EllipsisMaskOption : int32_t
//  - NewAxisMaskOption : int32_t
//  - ShrinkAxisMaskOption : int32_t
//  - OffsetOption : bool

//==============================================================================
LrtStatus LrtStridedSliceGetBeginMaskOption(LrtOp op, int32_t* begin_mask);
LrtStatus LrtStridedSliceGetEndMaskOption(LrtOp op, int32_t* end_mask);
LrtStatus LrtStridedSliceGetEllipsisMaskOption(LrtOp op,
                                               int32_t* ellipsis_mask);
LrtStatus LrtStridedSliceGetNewAxisMaskOption(LrtOp op, int32_t* new_axis_mask);
LrtStatus LrtStridedSliceGetShrinkAxisMaskOption(LrtOp op,
                                                 int32_t* shrink_axis_mask);
LrtStatus LrtStridedSliceGetOffsetOption(LrtOp op, bool* offset);

//==============================================================================
//
// Get option APIs for LRT Sub op.
//  Options:
//  - FusedActivationOption : uint32_t
//  - (Not supported) PotScaleInt16Option : bool
//
//==============================================================================
LrtStatus LrtSubGetFusedActivationOption(LrtOp op, uint32_t* fused_activation);

//==============================================================================
//
// Get option APIs for LRT Reshape op.
//  Options:
//  - new_shape : int32_t[]
//
//==============================================================================
LrtStatus LrtReshapeGetNewShapeOption(LrtOp op, int32_t** new_shape,
                                      int32_t* new_shape_size);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_OPTIONS_H_
