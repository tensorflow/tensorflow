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

//==============================================================================
//
// Get option APIs for LiteRt Pack op.
//  Options:
// - axisOption : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetPackAxisOption(LiteRtOp op, int32_t* axis);

//==============================================================================
//
// Get option APIs for LiteRt Gather op.
//  Options:
// - axisOption : int32_t
// - batch_dims : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetGatherAxisOption(LiteRtOp op, int32_t* axis);
LiteRtStatus LiteRtGetGatherBatchDimsOption(LiteRtOp op, int32_t* batch_dims);

//==============================================================================
//
// Get option APIs for LiteRt Mean op.
//  Options:
// - keepdimsOption : bool
//
//==============================================================================
LiteRtStatus LiteRtGetMeanKeepDimsOption(LiteRtOp op, bool* keepdims);

//==============================================================================
//
// Get option APIs for LiteRt Split op.
//  Options:
// - num_splits : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetSplitNumSplitsOption(LiteRtOp op, int32_t* num_splits);

//==============================================================================
//
// Get option APIs for LiteRt SHLO Composite op.
//  Options:
// - name : string
// - decomposition_subgraph_index : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetSHLOCompositeOpName(LiteRtOp op, const char** name);
LiteRtStatus LiteRtGetSHLOCompositeOpDecompositionSubgraphIndex(
    LiteRtOp op, int32_t* decomposition_subgraph_index);

//==============================================================================
//
// Get option APIs for LiteRt Conv2d op.
//  Options:
// - padding : uint32_t
// - stride_w : int32_t
// - stride_h : int32_t
// - fused_activation_function : uint32_t
// - dilation_w_factor : int32_t
// - dilation_h_factor : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetConv2dPaddingOption(LiteRtOp op, uint32_t* padding);
LiteRtStatus LiteRtGetConv2dStrideWOption(LiteRtOp op, int32_t* stride_w);
LiteRtStatus LiteRtGetConv2dStrideHOption(LiteRtOp op, int32_t* stride_h);
LiteRtStatus LiteRtGetConv2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function);
LiteRtStatus LiteRtGetConv2dDilationWOption(LiteRtOp op,
                                            int32_t* dilation_w_factor);
LiteRtStatus LiteRtGetConv2dDilationHOption(LiteRtOp op,
                                            int32_t* dilation_h_factor);

//==============================================================================
//
// Get option APIs for LiteRt DepthwiseConv2d op.
//  Options:
// - padding : uint32_t
// - stride_w : int32_t
// - stride_h : int32_t
// - fused_activation_function : uint32_t
// - dilation_w_factor : int32_t
// - dilation_h_factor : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetDepthwiseConv2dPaddingOption(LiteRtOp op,
                                                   uint32_t* padding);
LiteRtStatus LiteRtGetDepthwiseConv2dStrideWOption(LiteRtOp op,
                                                   int32_t* stride_w);
LiteRtStatus LiteRtGetDepthwiseConv2dStrideHOption(LiteRtOp op,
                                                   int32_t* stride_h);
LiteRtStatus LiteRtGetDepthwiseConv2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function);
LiteRtStatus LiteRtGetDepthwiseConv2dDilationWOption(
    LiteRtOp op, int32_t* dilation_w_factor);
LiteRtStatus LiteRtGetDepthwiseConv2dDilationHOptions(
    LiteRtOp op, int32_t* dilation_h_factor);

//==============================================================================
//
// Get option APIs for LiteRt AveragePool2d op.
//  Options:
// - padding : uint32_t
// - stride_w : int32_t
// - stride_h : int32_t
// - filter_width : int32_t
// - filter_height : int32_t
// - fused_activation_function : uint32_t
//
//==============================================================================
LiteRtStatus LiteRtGetAveragePool2dPaddingOption(LiteRtOp op,
                                                 uint32_t* padding);
LiteRtStatus LiteRtGetAveragePool2dStrideWOption(LiteRtOp op,
                                                 int32_t* stride_w);
LiteRtStatus LiteRtGetAveragePool2dStrideHOption(LiteRtOp op,
                                                 int32_t* stride_h);
LiteRtStatus LiteRtGetAveragePool2dFilterWidthOption(LiteRtOp op,
                                                     int32_t* filter_width);
LiteRtStatus LiteRtGetAveragePool2dFilterHeightOption(LiteRtOp op,
                                                      int32_t* filter_height);
LiteRtStatus LiteRtGetAveragePool2dFusedActivationOption(
    LiteRtOp op, uint32_t* fused_activation_function);

//==============================================================================
//
// Get option APIs for LiteRt ResizeBilinear op.
//  Options:
// - align_corners : bool
// - half_pixel_centers : bool
//
//==============================================================================
LiteRtStatus LiteRtGetResizeBilinearAlignCornersOption(LiteRtOp op,
                                                       bool* align_corners);
LiteRtStatus LiteRtGetResizeBilinearHalfPixelCenterOption(
    LiteRtOp op, bool* half_pixel_centers);

//==============================================================================
//
// Get option APIs for LiteRt LeakyRelu op.
//  Options:
// - alpha : float
//
//==============================================================================
LiteRtStatus LiteRtGetLeakyReluAlphaOption(LiteRtOp op, float* alpha);

//==============================================================================
//
// Get option APIs for LiteRt DepthToSpace op.
//  Options:
// - block_size : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetDepthToSpaceBlockSizeOption(LiteRtOp op,
                                                  int32_t* block_size);

//==============================================================================
//
// Get option APIs for LiteRt SpaceToDepth op.
//  Options:
// - block_size : int32_t
//
//==============================================================================
LiteRtStatus LiteRtGetSpaceToDepthBlockSizeOption(LiteRtOp op,
                                                  int32_t* block_size);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_OPTIONS_H_
