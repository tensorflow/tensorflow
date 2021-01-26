
/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// API header for CEVA TFLM optimized kernel library

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_CEVA_CEVA_TFLM_LIB_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_CEVA_CEVA_TFLM_LIB_H_

#include "tensorflow/lite/micro/kernels/ceva/types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

void CEVA_TFLM_ResizeNearestNeighbor_float32(
    const bool align_corners, int32_t output_height, int32_t output_width,
    int32_t row_offset, int32_t input_height, int32_t input_width,
    int32_t col_offset, int32_t depth, const int32_t* input_ptr,
    int32_t* output_ptr, const bool half_pixel_centers, int32_t* scratch);
void CEVA_TFLM_ResizeNearestNeighbor_int8(
    const bool align_corners, int32_t output_height, int32_t output_width,
    int32_t row_offset, int32_t input_height, int32_t input_width,
    int32_t col_offset, int32_t depth, const int8_t* input_ptr,
    int8_t* output_ptr, const bool half_pixel_centers, int32_t* scratch);

void CEVA_TFLM_Abs_Float32(const float* input_data, float* output_data,
                           int flat_size);
void CEVA_TFLM_Sqrt_Float32(const float* input_data, float* output_data,
                            int flat_size);
void CEVA_TFLM_Rsqrt_Float32(const float* input_data, float* output_data,
                             int flat_size);
void CEVA_TFLM_Square_Float32(const float* input_data, float* output_data,
                              int flat_size);

void CEVA_TFLM_Cos_Float32(const float* input_data, float* output_data,
                           int flat_size);
void CEVA_TFLM_Sin_Float32(const float* input_data, float* output_data,
                           int flat_size);
void CEVA_TFLM_Tanh_Float32(const float* input_data, float* output_data,
                            int flat_size);

void CEVA_TFLM_Sigmoid_Float32(const float* input_data, float* output_data,
                               int flat_size);
void CEVA_TFLM_Log_Float32(const float* input_data, float* output_data,
                           int flat_size);

void CEVA_TFLM_LogicalNot(const bool* input_data, bool* output_data,
                          int flat_size);

void CEVA_TFLM_AffineQuantize_Int8(const float_32* input_data,
                                   int8_t* output_data, int flat_size,
                                   float_32 scale, int zero_point);

void CEVA_TFLM_Softmax_Float32(const float* input_data, float* output_data,
                               const float beta, const int depth);

void CEVA_TFLM_Neg_Float32(const float_32* input_data, float_32* output_data,
                           const int flat_size);

void CEVA_TFLM_RoundToNearest_asm(const float* input_arr, float* output_arr,
                                  const int size);
float RoundToNearest(float value);

void CEVA_TFLM_Round_float32(const float* input_data, float* output_data,
                             const int flat_size);

void CEVA_TFLM_Softmax_Int8(const int8_t* input_data, int8_t* output_data,
                            const int32_t input_beta_multiplier,
                            const int32_t input_beta_left_shift,
                            const int32_t depth, void* scratch);

void CEVA_TFLM_Min_Max_Float32(const float* input_data,
                               const float float_activation_min,
                               const float float_activation_max,
                               const int flat_size, float* output_data);

void CEVA_TFLM_Add_Float32(const void* params_inp, const float* input1_data,
                           const float* input2_data, float* output_data,
                           const int flat_size);

void CEVA_TFLM_BroadcastAdd4DSlow_Float32(const void* params_inp,
                                          const float* input1_data,
                                          const float* input2_data,
                                          float* output_data, const int* Dims,
                                          const int* desc1, const int* desc2);

void CEVA_TFLM_BroadcastSubSlow_Float32(
    const void* params_inp, const float* input1_data, const float* input2_data,
    float* output_data, const int* strides1, const int* strides2,
    const int* output_strides, const int* output_extents);

void CEVA_TFLM_BroadcastSubSlow_Float32_loop(
    const void* params_inp, const float* input1_data, const float* input2_data,
    float* output_data, const int* output_extents, const int* strides1,
    const int* strides2, const int* output_strides);

void CEVA_TFLM_SubWithActivation_Float32(const void* params_inp,
                                         const float* input1_data,
                                         const float* input2_data,
                                         float* output_data,
                                         const int flat_size);

void CEVA_TFLM_MaximumBroadcastSlow_Float32(
    const float* input1_data, const float* input2_data, float* output_data,
    const int* strides1, const int* strides2, const int* output_strides,
    const int* output_extents);
void CEVA_TFLM_MinimumBroadcastSlow_Float32(
    const float* input1_data, const float* input2_data, float* output_data,
    const int* strides1, const int* strides2, const int* output_strides,
    const int* output_extents);

void CEVA_TFLM_Maximum_Float32(const float* input1_data,
                               const float* input2_data, float* output_data,
                               const int flat_size);
void CEVA_TFLM_Minimum_Float32(const float* input1_data,
                               const float* input2_data, float* output_data,
                               const int flat_size);
void CEVA_TFLM_Maximum_Float32_asm(const float* input1_data,
                                   const float* input2_data, float* output_data,
                                   const int flat_size);
void CEVA_TFLM_Minimum_Float32_asm(const float* input1_data,
                                   const float* input2_data, float* output_data,
                                   const int flat_size);
void CEVA_TFLM_DepthwiseConv_Float32(
    // const DepthwiseParams& params,
    // const int batches, // always 1
    const int stride_width, const int stride_height, const int pad_width,
    const int pad_height, const int depth_multiplier, const int input_height,
    const int input_width, const int input_depth, const float* input_data,
    const int filter_height, const int filter_width, const int filter_depth,
    const float* filter_data, const float* bias_data, const int output_height,
    const int output_width, const int output_depth, float* output_data,
    const int dilation_width_factor, const int dilation_height_factor,
    const float output_activation_min, const float output_activation_max

);
void CEVA_TFLM_DepthwiseConvPerChannel_int8(
    const int stride_width, const int stride_height, const int pad_width,
    const int pad_height, const int depth_multiplier_,
    const int32_t input_offset_, const int32_t output_offset,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const int input_height, const int input_width_, const int input_depth_,
    const int8_t* input_data, const int filter_height, const int filter_width,
    const int filter_depth_, const int8_t* filter_data,
    const int32_t* bias_data, const int output_height, const int output_width,
    const int output_depth,

    int8_t* output_data, int32_t* scratch_

    ,
    const int dilation_width_factor_, const int dilation_height_factor,
    const int32_t output_activation_min, const int32_t output_activation_max);

void CEVA_TFLM_ConvPerChannel_Int8(
    const int stride_width, const int stride_height, const int pad_width,
    const int pad_height,  // const int depth_multiplier,
    const int32_t input_offset, const int32_t output_offset,
    const int32_t* output_multiplier, const int32_t* output_shift,
    const int input_height, const int input_width, const int input_depth_Dims3,
    const int input_depth, const int8_t* input_data, const int filter_height,
    const int filter_width, const int filter_depth, const int8_t* filter_data,
    const int32_t* bias_data, const int output_height, const int output_width,
    const int output_depth_Dims3, const int output_depth, int8_t* output_data,
    int32_t* scratch, const int dilation_width_factor,
    const int dilation_height_factor, const int32_t output_activation_min,
    const int32_t output_activation_max);

void CEVA_TFLM_Conv_Float32(
    // const int batches,
    const int stride_width, const int stride_height, const int pad_width,
    const int pad_height,  // const int depth_multiplier,
    const int input_height, const int input_width, const int input_depth_Dims3,
    const int input_depth, const float* input_data, const int filter_height,
    const int filter_width, const int filter_depth, const float* filter_data,
    const float* bias_data, const int output_height, const int output_width,
    const int output_depth_Dims3, const int output_depth, float* output_data,
    const int dilation_width_factor, const int dilation_height_factor,
    const float output_activation_min, const float output_activation_max

);

///////////////////
void CEVA_TFLM_MaximumBroadcastSlow_Int8(
    const int8_t* input1_data, const int8_t* input2_data, int8_t* output_data,
    const int* strides1, const int* strides2, const int* output_strides,
    const int* output_extents);
void CEVA_TFLM_MinimumBroadcastSlow_Int8(
    const int8_t* input1_data, const int8_t* input2_data, int8_t* output_data,
    const int* strides1, const int* strides2, const int* output_strides,
    const int* output_extents);

void CEVA_TFLM_Maximum_Int8(const int8_t* input1_data,
                            const int8_t* input2_data, int8_t* output_data,
                            const int flat_size);
void CEVA_TFLM_Minimum_Int8(const int8_t* input1_data,
                            const int8_t* input2_data, int8_t* output_data,
                            const int flat_size);

void CEVA_TFLM_BroadcastSubSlow_Int8(
    const void* params_inp, const int8_t* input1_data,
    const int8_t* input2_data, int8_t* output_data, const int* strides1,
    const int* strides2, const int* output_strides, const int* output_extents);

void CEVA_TFLM_BroadcastSubSlow_Int8_loop(
    const void* params_inp, const int8_t* input1_data,
    const int8_t* input2_data, int8_t* output_data, const int* output_extents,
    const int* strides1, const int* strides2, const int* output_strides);

void CEVA_TFLM_BroadcastAddSlow_Int8(const void* params_inp,
                                     const int8_t* input1_data,
                                     const int8_t* input2_data,
                                     int8_t* output_data, const int* strides1,
                                     const int* strides2,
                                     const int* output_extents);

void CEVA_TFLM_BroadcastAddSlow_Int8_loop(
    const void* params_inp, const int8_t* input1_data,
    const int8_t* input2_data, int8_t* output_data, const int* output_extents,
    const int* strides1, const int* strides2);

void CEVA_TFLM_Sub_Int8(const void* params_inp, const int8_t* input1_data,
                        const int8_t* input2_data, int8_t* output_data,
                        const int flat_size);

void CEVA_TFLM_Sub_Uint8(const void* params_inp, const uint8_t* input1_data,
                         const uint8_t* input2_data, uint8_t* output_data,
                         const int flat_size);

void CEVA_TFLM_Add_Uint8(const void* params, const uint8_t* input1_data,
                         const uint8_t* input2_data, uint8_t* output_data,
                         const int flat_size);

void CEVA_TFLM_Add_Int8(const void* params_inp, const int8_t* input1_data,
                        const int8_t* input2_data, int8_t* output_data,
                        const int flat_size);

void CEVA_TFLM_BroadcastAdd4DSlow_Uint8(const void* params,
                                        const uint8_t* input1_data,
                                        const uint8_t* input2_data,
                                        uint8_t* output_data, const int* Dims,
                                        const int* desc1, const int* desc2,
                                        const int* dims_data);
void CEVA_TFLM_svdf_Float32(float_32* vector1_ptr, float_32* vector2_ptr,
                            int32_t num_units, int32_t memory_size_rank,
                            float_32* output_ptr_batch);
void CEVA_TFLM_svdf_Int8(int n_memory, const int8_t* matrix_ptr,
                         const int8_t* vector_in_batch_t,
                         int16_t* result_in_batch, int input_zp, int n_input,
                         int effective_scale_1_a, int effective_scale_1_b,
                         int n_filter, int* scratch);
void CEVA_TFLM_AffineQuantize_Int8(const float_32* input_data,
                                   int8_t* output_data, int flat_size,
                                   float_32 scale, int zero_point);

// int32_t MultiplyByQuantizedMultiplier_t(int32_t x, int32_t
// quantized_multiplier, int shift); int32_t
// MultiplyByQuantizedMultiplier_t1(int32_t x, int32_t quantized_multiplier, int
// shift);

void CEVA_TFLM_L2Normalization_Float32(const float* input_data,
                                       float* output_data, float epsilon,
                                       const int outer_size, const int depth);
void CEVA_TFLM_L2Normalization_Int8(int32_t input_zero_point,
                                    int32_t outer_size, int32_t depth,
                                    const int8_t* input_data,
                                    int8_t* output_data);

void CEVA_TFLM_prelu_Float32(const float* in1_data, const int32_t* in1_strides,
                             const float* in2_data, const int32_t* in2_strides,
                             float* out_data, const int32_t* out_strides,
                             const int32_t* dims);

void CEVA_TFLM_prelu_Int8(const int8_t* in1_data, const int32_t* in1_strides,
                          const int8_t* alpha_data,
                          const int32_t* alpha_strides, int8_t* out_data,
                          const int32_t* out_strides, const int32_t* dims,
                          const int32_t* params);
void CEVA_TFLM_FullyConnected_Float32(
    const void* params_inp, const int input_shape, const float* input_data,
    const int weights_shape_DimensionsCount, const int* weights_shape_DimsData,
    const float* weights_data, const int bias_shape, const float* bias_data,
    const int output_shape_DimensionsCount, const int* output_shape_DimsData,
    float* output_data);
void CEVA_TFLM_FullyConnected_int8(
    const void* params_inp, const int input_shape, const int8_t* input_data,
    const int filter_shape_DimensionsCount, const int* filter_shape_DimsData,
    const int8_t* filter_data, const int bias_shape, const int32_t* bias_data,
    const int output_shape_DimensionsCount, const int* output_shape_DimsData,
    int8_t* output_data, int* scratch);

void CEVA_TFLM_tanh_Int8(int32_t input_zero_point, int32_t input_range_radius,
                         int32_t input_multiplier, int32_t input_shift,
                         int32_t input_size, const int8_t* input_data,
                         int8_t* output_data);

void CEVA_TFLM_Logistic_Int8(int32_t input_zero_point,
                             int32_t input_range_radius,
                             int32_t input_multiplier, int32_t input_left_shift,
                             int32_t input_size, const int8_t* input_data,
                             int8_t* output_data);

void CEVA_TFLM_Tanh_float32(const float_32* input_data, float_32* output_data,
                            const int flat_size);
void CEVA_TFLM_Logistic_float32(const float_32* input_data,
                                float_32* output_data, const int flat_size);

void CEVA_TFLM_PackImplLoop_float(const float* input_ptr, float* output_ptr,
                                  int outer_size, int copy_size,
                                  int step_vcount_copy_size);
void CEVA_TFLM_PackUnpackImplLoopInitSizes(int* const copy_size,
                                           int* const outer_size,
                                           const int* const outputDimsData,
                                           const int dimensions, int axis);
void CEVA_TFLM_PackImplLoop_Int8(const int8_t* input_ptr, int8_t* output_ptr,
                                 int outer_size, int copy_size,
                                 int step_vcount_copy_size);
void CEVA_TFLM_UnpackImplLoop_float(const float* input_ptr, float* output_ptr,
                                    int outer_size, int copy_size,
                                    int step_vcount_copy_size);
void CEVA_TFLM_UnpackImplLoop_Int8(const int8_t* input_ptr, int8_t* output_ptr,
                                   int outer_size, int copy_size,
                                   int step_vcount_copy_size);

void CEVA_TFLM_ComparisonEqual_Float32(const float* input1, const float* input2,
                                       bool* output, const int32_t size);
void CEVA_TFLM_ComparisonNotEqual_Float32(const float* input1,
                                          const float* input2, bool* output,
                                          const int32_t size);
void CEVA_TFLM_ComparisonGreater_Float32(const float* input1,
                                         const float* input2, bool* output,
                                         const int32_t size);
void CEVA_TFLM_ComparisonGreaterEqual_Float32(const float* input1,
                                              const float* input2, bool* output,
                                              const int32_t size);
void CEVA_TFLM_ComparisonLess_Float32(const float* input1, const float* input2,
                                      bool* output, const int32_t size);
void CEVA_TFLM_ComparisonLessEqual_Float32(const float* input1,
                                           const float* input2, bool* output,
                                           const int32_t size);

void CEVA_TFLM_ComparisonEqual_Float32_Broadcast(const float* input1,
                                                 const float* input2,
                                                 bool* output,
                                                 const int32_t* dims,
                                                 const int32_t** op_param);

void CEVA_TFLM_ComparisonNotEqual_Float32_Broadcast(const float* input1,
                                                    const float* input2,
                                                    bool* output,
                                                    const int32_t* dims,
                                                    const int32_t** op_param);

void CEVA_TFLM_ComparisonGreater_Float32_Broadcast(const float* input1,
                                                   const float* input2,
                                                   bool* output,
                                                   const int32_t* dims,
                                                   const int32_t** op_param);
void CEVA_TFLM_ComparisonGreaterEqual_Float32_Broadcast(
    const float* input1, const float* input2, bool* output, const int32_t* dims,
    const int32_t** op_param);

void CEVA_TFLM_ComparisonLess_Float32_Broadcast(const float* input1,
                                                const float* input2,
                                                bool* output,
                                                const int32_t* dims,
                                                const int32_t** op_param);

void CEVA_TFLM_ComparisonLessEqual_Float32_Broadcast(const float* input1,
                                                     const float* input2,
                                                     bool* output,
                                                     const int32_t* dims,
                                                     const int32_t** op_param);

void CEVA_TFLM_ComparisonEqual_Int8(const int8_t* input1, const int8_t* input2,
                                    bool* output, const int32_t flatsize,
                                    void* op_params);
void CEVA_TFLM_ComparisonNotEqual_Int8(const int8_t* input1,
                                       const int8_t* input2, bool* output,
                                       const int32_t flatsize, void* op_params);
void CEVA_TFLM_ComparisonGreater_Int8(const int8_t* input1,
                                      const int8_t* input2, bool* output,
                                      const int32_t flatsize, void* op_params);
void CEVA_TFLM_ComparisonGreaterEqual_Int8(const int8_t* input1,
                                           const int8_t* input2, bool* output,
                                           const int32_t flatsize,
                                           void* op_params);
void CEVA_TFLM_ComparisonLess_Int8(const int8_t* input1, const int8_t* input2,
                                   bool* output, const int32_t flatsize,
                                   void* op_params);
void CEVA_TFLM_ComparisonLessEqual_Int8(const int8_t* input1,
                                        const int8_t* input2, bool* output,
                                        const int32_t flatsize,
                                        void* op_params);

void CEVA_TFLM_ComparisonEqual_Int8_Broadcast(const int8_t* input1,
                                              const int8_t* input2,
                                              bool* output, const int32_t* dims,
                                              void* op_params);
void CEVA_TFLM_ComparisonNotEqual_Int8_Broadcast(const int8_t* input1,
                                                 const int8_t* input2,
                                                 bool* output,
                                                 const int32_t* dims,
                                                 void* op_params);
void CEVA_TFLM_ComparisonGreater_Int8_Broadcast(const int8_t* input1,
                                                const int8_t* input2,
                                                bool* output,
                                                const int32_t* dims,
                                                void* op_params);
void CEVA_TFLM_ComparisonGreaterEqual_Int8_Broadcast(const int8_t* input1,
                                                     const int8_t* input2,
                                                     bool* output,
                                                     const int32_t* dims,
                                                     void* op_params);
void CEVA_TFLM_ComparisonLess_Int8_Broadcast(const int8_t* input1,
                                             const int8_t* input2, bool* output,
                                             const int32_t* dims,
                                             void* op_params);
void CEVA_TFLM_ComparisonLessEqual_Int8_Broadcast(const int8_t* input1,
                                                  const int8_t* input2,
                                                  bool* output,
                                                  const int32_t* dims,
                                                  void* op_params);

void CEVA_TFLM_Mul_Float32(const void* params_inp, const float* input1_data,
                           const float* input2_data, float* output_data,
                           const int flat_size);

void CEVA_TFLM_BroadcastMul4DSlow_Float32(const void* params_inp,
                                          const float* input1_data,
                                          const float* input2_data,
                                          float* output_data, const int* Dims,
                                          const int* desc1, const int* desc2);

void CEVA_TFLM_AveragePool_Float32(const void* params, const int* input_shape,
                                   const float* input_data,
                                   const int* output_shape, float* output_data);

void CEVA_TFLM_AveragePool_Int8(const void* params_inp, const int* input_shape,
                                const int8_t* input_data,
                                const int* output_shape, int8_t* output_data);

void CEVA_TFLM_AveragePool_Int8_Loop(
    const int* input_shape, const int8_t* input_data, int8_t* output_data,
    const int depth, int batch, int in_y, const int filter_y_start,
    const int filter_y_end, const int in_x_origin, const int filter_x_start,
    const int filter_x_end, int filter_count, int32_t quantized_activation_min,
    int32_t quantized_activation_max, int indx_out);

void CEVA_TFLM_MaxPool_Float32(const void* params_inp, const int* input_shape,
                               const float* input_data, const int* output_shape,
                               float* output_data);

void CEVA_TFLM_MaxPool_Int8(const void* params_inp, const int* input_shape,
                            const int8_t* input_data, const int* output_shape,
                            int8_t* output_data);

void CEVA_TFLM_MaxPool_Int8_Loop(
    const int* input_shape, const int8_t* input_data, int8_t* output_data,
    const int depth, int batch, int in_y, const int filter_y_start,
    const int filter_y_end, const int in_x_origin, const int filter_x_start,
    const int filter_x_end, int32_t quantized_activation_min,
    int32_t quantized_activation_max, int indx_out);

void CEVA_TFLM_Mul_Int8(const void* params_inp, const int8_t* input1_data,
                        const int8_t* input2_data, int8_t* output_data,
                        const int flat_size);

void CEVA_TFLM_BroadcastMul4DSlow_Int8(const void* params_inp,
                                       const int8_t* input1_data,
                                       const int8_t* input2_data,
                                       int8_t* output_data, const int* Dims,
                                       const int* desc1, const int* desc2);

void CEVA_TFLM_Dequantize_Float32(const int8_t* input_data,
                                  float_32* output_data, int flat_size,
                                  float_32 scale, int zero_point);

void CEVA_TFLM_Ceil_Float32(const float* input_data, float* output_data,
                            const int flat_size);

void CEVA_TFLM_Logical_And_Int8(const int8_t* input1_data,
                                const int8_t* input2_data, int8_t* output_data,
                                const int flat_size);

void CEVA_TFLM_BroadcastLogicalAnd4DSlow_Int8(const int8_t* input1_data,
                                              const int8_t* input2_data,
                                              int8_t* output_data,
                                              const int* Dims, const int* desc1,
                                              const int* desc2);

void CEVA_TFLM_Logical_Or_Int8(const int8_t* input1_data,
                               const int8_t* input2_data, int8_t* output_data,
                               const int flat_size);

void CEVA_TFLM_BroadcastLogicalOr4DSlow_Int8(const int8_t* input1_data,
                                             const int8_t* input2_data,
                                             int8_t* output_data,
                                             const int* Dims, const int* desc1,
                                             const int* desc2);

void CEVA_TFLM_SplitLoops_Float32(float** out_ptrs, const int* dataIndex,
                                  const float* input_ptr, int outer_size,
                                  int output_count, int copy_size);
void CEVA_TFLM_SplitLoops_int8(int8_t** out_ptrs, const int* dataIndex,
                               const int8_t* input_ptr, int outer_size,
                               int output_count, int copy_size);

void CEVA_TFLM_Relu_Float32(const float* input_data, float* output_data,
                            const int flat_size);
void CEVA_TFLM_Relu6_Float32(const float* input_data, float* output_data,
                             const int flat_size);
void CEVA_TFLM_Relu_int8(const void* params, const int8_t* input_data,
                         int8_t* output_data, const int flat_size);
void CEVA_TFLM_Relu6_int8(const int8_t lower, const int8_t upper,
                          const int8_t* input_data, int8_t* output_data,
                          const int flat_size);
void CEVA_TFLM_Floor_float32(const float* input_data, float* output_data,
                             const int flat_size);

void CEVA_TFLM_Concatenation_Float32(const void* params_inp,
                                     const int** input_shape,
                                     const float** input_data,
                                     const int output_shape_DimensionsCount,
                                     const int* output_shape_DimsData,
                                     float* output_data);

void CEVA_TFLM_Concatenation_int8(const void* params_inp,
                                  const int** input_shape,
                                  const int8_t** input_data,
                                  const int output_shape_DimensionsCount,
                                  const int* output_shape_DimsData,
                                  int8_t* output_data);

void CEVA_TFLM_Mean4D_Float32(const float* input_data, float* output_data,
                              const int* Dims, const int* Dims_inp,
                              const int* dims_data, const int* dims_data_inp);
bool CEVA_TFLM_Mean_Float32(const float* input_data, const int* input_dims,
                            const int input_num_dims, float* output_data,
                            const int* output_dims, const int output_num_dims,
                            const int* axis, const int num_axis_dimensions,
                            bool keep_dims, int* temp_index, int* resolved_axis,
                            float* temp_sum);
void CEVA_TFLM_Mean_Float32_loop(float* temp_sum, float* output_data,
                                 int num_elements_in_axis, size_t num_outputs);
void CEVA_TFLM_Mean4D_Int8(int32_t multiplier, int32_t shift,
                           const int8_t* input_data, int32_t input_zero_point,
                           int8_t* output_data, int32_t output_zero_point,
                           int* input_shape, int* output_shape);
bool CEVA_TFLM_Mean_Int8(const int8_t* input_data, const int* input_dims,
                         const int input_num_dims, int8_t* output_data,
                         const int* output_dims, const int output_num_dims,
                         const int* axis, const int num_axis_dimensions,
                         bool keep_dims, int* temp_index, int* resolved_axis,
                         int32_t* temp_sum);
void CEVA_TFLM_Mean_Int8_loop(int32_t* temp_sum, int8_t* output_data,
                              int num_elements_in_axis, size_t num_outputs);
void CEVA_TFLM_StridedSlice_Float32(void* op_params,
                                    int unextended_input_shape_DimensionsCount,
                                    int* unextended_input_shape_DimsData,
                                    float* input_data,

                                    float* output_data);

void CEVA_TFLM_StridedSlice_Float32(void* op_params,
                                    int unextended_input_shape_DimensionsCount,
                                    int* unextended_input_shape_DimsData,
                                    float* input_data, float* output_data);

void CEVA_TFLM_StridedSlice_loop_Float32(float* input_data, float* output_data,
                                         void* params);

void CEVA_TFLM_StridedSlice_int8(void* op_params,
                                 int unextended_input_shape_DimensionsCount,
                                 int* unextended_input_shape_DimsData,
                                 int8_t* input_data, int8_t* output_data);

void CEVA_TFLM_StridedSlice_loop_int8(int8_t* input_data, int8_t* output_data,
                                      void* params);

void CEVA_TFLM_Pad_Float32(void* op_params, int input_shape, int* output_shape,
                           const float* input_data, const float* pad_value_ptr,
                           float* output_data);

void CEVA_TFLM_Pad_Int8(void* op_params, int input_shape, int* output_shape,
                        const int8_t* input_data, const int8_t* pad_value_ptr,
                        int8_t* output_data);

int CEVA_TFLM_ReshapeOutput(int input_type, const int input_size,
                            const int* input_data, int output_type,
                            int* output_size, int* output_data,
                            int node_in_size);

int CEVA_TFLM_EvalRashape(const int8_t* input, int8_t* output,
                          unsigned int N_cnt);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_CEVA_CEVA_TFLM_LIB_H_
