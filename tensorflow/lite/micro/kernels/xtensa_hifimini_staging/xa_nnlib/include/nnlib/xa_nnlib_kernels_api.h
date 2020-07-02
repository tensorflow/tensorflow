/*******************************************************************************
 * Copyright (c) 2019-2020 Cadence Design Systems, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to use this Software with Cadence processor cores only and
 * not with any other processors and platforms, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef __XA_NNLIB_KERNELS_API_H__
#define __XA_NNLIB_KERNELS_API_H__

/**
 * @file xa_nnlib_kernels_api.h
 * @brief This file gives the API definition for the HiFi NNLIB
 *
 * matXvec KERNELS API NAMING CONVENTION <br>
 * <br>
 * xa_nn_matXvec_<batch>_[m]x[n]_[p]_<activation>, where
 * - <batch>: Optional 'batch' tag to indicate time batching routine
 * - [m]: Matrix precision in bits
 * - [n]: Vector (and bias for non-activation routines) precision in bits
 * - [p]: Output precision in bits
 * - <activation>: optional activation tag 'sigmoid' / 'tanh'
 *
 * These set of kernels perform dual matXvec followed by optional
 * activation function. There are several variants based on the input,
 * output precision and use of activation functions.
 *
 * Restriction,
 * - All pointers (p_out, p_mat1, p_mat2, p_vec1, p_vec2, p_bias, p_scratch)
 * must be SIMD (64-bit) aligned and should not overlap.
 * - p_mat2, p_vec2 can be 'NULL', but other pointers cannot be 'NULL'
 * - Variables cols1, cols2, row_stride1, row_stride2 must be multiple of 4
 *
 * Usage of few critical variables,
 * - acc_shift:
 *   -# In case of valid activation tag i.e. <activation>: shift to be
 *   applied on accumulator to match accumulator's Q format with activation
 *   function's input's Q format
 *   -# In case of bypass i.e. no activation tag: shift to be applied on
 *   accumulator.
 *   -# Positive value denotes left shift, and negative value denotes right
 * shift.
 * - bias_shift: shift which is to be applied on bias to match bias's
 *   Q format with accumulator's Q format. Positive value denotes left shift,
 *   and negative value denotes right shift.
 * - bias_precision: This represents bias precision
 *   -# For 16x16, and 8x16 apis, valid values are '16' and '64'
 *   -# For 8x8 apis, valid values are '8' and '32'
 *
 * Output 8b, 16b, 32b of fixed point apis (only for bypass variants) is
 * extracted from 64b accumulator with symmetric rounding. Output 64b of fixed
 * point apis (only for bypass variants) is extracted from 64b accumulator.
 * Output 8b, 16b of fixed point apis (only for activation variants) is
 * symmetrically rounded.
 *
 * matXvec 16x16 Kernels,
 * - Bypass kernels with 16, 32, 64 bit output: 3
 * - Fused kernel with 2 activation variants:   2
 * - Time batching kernel:                      1 (Not implemented)
 * - Total:                                     6
 *
 * matXvec 8x16 Kernels,
 * - Bypass kernels with 16, 32, 64 bit output: 3
 * - Fused kernel with 2 activation variants:   2
 * - Time batching kernel:                      1 (Not implemented)
 * - Total:                                     6
 *
 * matXvec 8x8 Kernels,
 * - Bypass kernels with 8, 16, 32 bit output: 3
 * - Fused kernel with 2 activation variants:  2
 * - Time batching kernel:                     1 (Not implemented)
 * - Total:                                    6
 *
 * matXvec float32 x float32 Kernels,
 * - Bypass kernels 32 bit output:            1
 * - Fused kernel with 2 activation variants: 2
 * - Time batching kernel:                    1 (Not implemented)
 * - Total:                                   4
 *
 * ACTIVATION KERNELS API NAMING CONVENTION <br>
 * <br>
 * xa_nn_vec_[activation]_[n]_[p] for fixed point <br>
 * xa_nn_vec_[activation]_f32_f32 for floating point, where
 * - [activation]: One of activations - sigmoid/tanh/relu/relu1/relu6/softmax
 * - [n]:          Input precision in bits
 * - [p]:          Output precision in bits
 *
 * Possible values,
 * - 'n' takes value '32', and expects input in Q6.25 format.
 * - 'p' takes values '32' and '16', gives output in Q16.15 and Q0.15 formats
 * respectively.
 *
 * There is WORD32 datatype variable 'threshold' for 'relu' related apis, which
 * expects value in Q16.15 format.
 *
 * Restriction,
 * - All pointers (p_out, p_vec) must be 32-bit aligned and should not overlap.
 *
 * activation 32_32 kernels,
 * - Vector activation kernels: 6
 * - Total:                     6
 *
 * activation f32_f32 kernels,
 * - Vector activation kernels: 6
 * - Total:                     6
 *
 * activation 32_16 kernels,
 * - Vector activation kernels: 2
 * - Total:                     2
 */

#if defined(__cplusplus)
extern "C" {
#endif

WORD32 xa_nn_conv2d_depthwise_getsize(
    WORD32 input_height, WORD32 input_width, WORD32 input_channels,
    WORD32 kernel_height, WORD32 kernel_width, WORD32 channels_multiplier,
    WORD32 x_stride, WORD32 y_stride, WORD32 x_padding, WORD32 y_padding,
    WORD32 output_height, WORD32 output_width, WORD32 circ_buf_precision,
    WORD32 inp_data_format);

WORD32 xa_nn_vec_activation_min_max_asym8u_asym8u(
    UWORD8 *__restrict__ p_out, const UWORD8 *__restrict__ p_vec,
    int activation_min, int activation_max, WORD32 vec_length);

WORD32 xa_nn_vec_activation_min_max_asym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_vec,
    int activation_min, int activation_max, WORD32 vec_length);

WORD32 xa_nn_conv2d_std_getsize(WORD32 input_height, WORD32 input_channels,
                                WORD32 kernel_height, WORD32 kernel_width,
                                WORD32 y_stride, WORD32 y_padding,
                                WORD32 out_height, WORD32 input_precision);

WORD32 xa_nn_conv2d_std_asym8uxasym8u(
    UWORD8 *__restrict__ p_out, const UWORD8 *__restrict__ p_inp,
    const UWORD8 *__restrict__ p_kernel, const WORD32 *__restrict__ p_bias,
    WORD32 input_height, WORD32 input_width, WORD32 input_channels,
    WORD32 kernel_height, WORD32 kernel_width, WORD32 out_channels,
    WORD32 x_stride, WORD32 y_stride, WORD32 x_padding, WORD32 y_padding,
    WORD32 out_height, WORD32 out_width, WORD32 input_zero_bias,
    WORD32 kernel_zero_bias, WORD32 out_multiplier, WORD32 out_shift,
    WORD32 out_zero_bias, WORD32 out_data_format, VOID *p_scratch);

WORD32 xa_nn_conv2d_std_per_chan_sym8sxasym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_inp,
    const WORD8 *__restrict__ p_kernel, const WORD32 *__restrict__ p_bias,
    WORD32 input_height, WORD32 input_width, WORD32 input_channels,
    WORD32 kernel_height, WORD32 kernel_width, WORD32 out_channels,
    WORD32 x_stride, WORD32 y_stride, WORD32 x_padding, WORD32 y_padding,
    WORD32 out_height, WORD32 out_width, WORD32 input_zero_bias,
    WORD32 *p_out_multiplier, WORD32 *p_out_shift, WORD32 out_zero_bias,
    WORD32 out_data_format, VOID *p_scratch);

WORD32 xa_nn_conv2d_depthwise_asym8uxasym8u(
    pUWORD8 __restrict__ p_out, const UWORD8 *__restrict__ p_kernel,
    const UWORD8 *__restrict__ p_inp, const WORD32 *__restrict__ p_bias,
    WORD32 input_height, WORD32 input_width, WORD32 input_channels,
    WORD32 kernel_height, WORD32 kernel_width, WORD32 channels_multiplier,
    WORD32 x_stride, WORD32 y_stride, WORD32 x_padding, WORD32 y_padding,
    WORD32 out_height, WORD32 out_width, WORD32 input_zero_bias,
    WORD32 kernel_zero_bias, WORD32 out_multiplier, WORD32 out_shift,
    WORD32 out_zero_bias, WORD32 inp_data_format, WORD32 out_data_format,
    pVOID p_scratch);

WORD32 xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_kernel,
    const WORD8 *__restrict__ p_inp, const WORD32 *__restrict__ p_bias,
    WORD32 input_height, WORD32 input_width, WORD32 input_channels,
    WORD32 kernel_height, WORD32 kernel_width, WORD32 channels_multiplier,
    WORD32 x_stride, WORD32 y_stride, WORD32 x_padding, WORD32 y_padding,
    WORD32 out_height, WORD32 out_width, WORD32 input_zero_bias,
    const WORD32 *p_out_multiplier, const WORD32 *p_out_shift,
    WORD32 out_zero_bias, WORD32 inp_data_format, WORD32 out_data_format,
    pVOID p_scratch);

WORD32 xa_nn_fully_connected_asym8uxasym8u_asym8u(
    pUWORD8 __restrict__ p_out, const UWORD8 *__restrict__ p_weight,
    const UWORD8 *__restrict__ p_inp, const WORD32 *__restrict__ p_bias,
    WORD32 weight_depth, WORD32 out_depth, WORD32 input_zero_bias,
    WORD32 weight_zero_bias, WORD32 out_multiplier, WORD32 out_shift,
    WORD32 out_zero_bias);

WORD32 xa_nn_fully_connected_sym8sxasym8s_asym8s(
    pWORD8 __restrict__ p_out, const WORD8 *__restrict__ p_weight,
    const WORD8 *__restrict__ p_inp, const WORD32 *__restrict__ p_bias,
    WORD32 weight_depth, WORD32 out_depth, WORD32 input_zero_bias,
    WORD32 out_multiplier, WORD32 out_shift, WORD32 out_zero_bias);

WORD32 xa_nn_fully_connected_asym8sxasym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_weight,
    const WORD8 *__restrict__ p_inp, const WORD32 *__restrict__ p_bias,
    WORD32 weight_depth, WORD32 out_depth, WORD32 weight_zero_bias,
    WORD32 input_zero_bias, WORD32 out_multiplier, WORD32 out_shift,
    WORD32 out_zero_bias);

WORD32 xa_nn_vec_softmax_asym8u_8(UWORD8 *__restrict__ p_out,
                                  const UWORD8 *__restrict__ p_vec,
                                  WORD32 diffmin, WORD32 input_left_shift,
                                  WORD32 input_multiplier, WORD32 vec_length,
                                  pVOID p_scratch);

WORD32 xa_nn_vec_softmax_asym8s_16(WORD16 *__restrict__ p_out,
                                   const WORD8 *__restrict__ p_vec,
                                   WORD32 diffmin, WORD32 input_left_shift,
                                   WORD32 input_multiplier, WORD32 vec_length,
                                   pVOID p_scratch);

WORD32 xa_nn_vec_softmax_asym8s_8(WORD8 *__restrict__ p_out,
                                  const WORD8 *__restrict__ p_vec,
                                  WORD32 diffmin, WORD32 input_left_shift,
                                  WORD32 input_multiplier, WORD32 vec_length,
                                  pVOID p_scratch);

int xa_nn_get_softmax_scratch_size(int inp_precision, int out_precision,
                                   int length);

WORD32 xa_nn_matXvec_out_stride_asym8uxasym8u_asym8u(
    UWORD8 *__restrict__ p_out, const UWORD8 *__restrict__ p_mat1,
    const UWORD8 *__restrict__ p_vec1, const WORD32 *__restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 row_stride1, WORD32 out_stride,
    WORD32 mat1_zero_bias, WORD32 vec1_zero_bias, WORD32 out_multiplier,
    WORD32 out_shift, WORD32 out_zero_bias);

WORD32 xa_nn_matXvec_out_stride_sym8sxasym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_mat1,
    const WORD8 *__restrict__ p_vec1, const WORD32 *__restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 row_stride1, WORD32 out_stride,
    WORD32 vec1_zero_bias, WORD32 out_multiplier, WORD32 out_shift,
    WORD32 out_zero_bias);

WORD32 xa_nn_matXvec_out_stride_asym8sxasym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_mat1,
    const WORD8 *__restrict__ p_vec1, const WORD32 *__restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 row_stride1, WORD32 out_stride,
    WORD32 mat1_zero_bias, WORD32 vec1_zero_bias, WORD32 out_multiplier,
    WORD32 out_shift, WORD32 out_zero_bias);

WORD32 xa_nn_matXvec_out_stride_sym8sxasym8s_16(
    WORD16 *__restrict__ p_out, const WORD8 *__restrict__ p_mat1,
    const WORD8 *__restrict__ p_vec1, const WORD32 *__restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 row_stride1, WORD32 out_stride,
    WORD32 vec1_zero_bias, WORD32 out_multiplier, WORD32 out_shift);

WORD32 xa_nn_dot_prod_16x16_asym8s(
    WORD8 *__restrict__ p_out,               /* pointer to output */
    const WORD16 *__restrict__ p_inp1_start, /* pointer to input1 */
    const WORD16 *__restrict__ p_inp2_start, /* pointer to input2 */
    const WORD32 *bias_ptr, WORD32 vec_length, WORD32 out_multiplier,
    WORD32 out_shift, WORD32 out_zero_bias, WORD32 vec_count);

/* Mapping the functions names from previous naming convension for backward
 * compatibility */
#define xa_nn_vec_activation_min_max_asym8_asym8 \
  xa_nn_vec_activation_min_max_asym8u_asym8u
#define xa_nn_conv2d_std_asym8xasym8 xa_nn_conv2d_std_asym8uxasym8u
#define xa_nn_conv2d_depthwise_asym8xasym8 xa_nn_conv2d_depthwise_asym8uxasym8u
#define xa_nn_fully_connected_asym8xasym8_asym8 \
  xa_nn_fully_connected_asym8uxasym8u_asym8u
#define xa_nn_vec_softmax_asym8_asym8 xa_nn_vec_softmax_asym8u_asym8u
#define xa_nn_dot_prod_asym8xasym8_asym8 xa_nn_dot_prod_asym8uxasym8u_asym8u
#define xa_nn_matXvec_out_stride_asym8xasym8_asym8 \
  xa_nn_matXvec_out_stride_asym8uxasym8u_asym8u

#if defined(__cplusplus)
}
#endif
#endif /* __XA_NNLIB_KERNELS_API_H__ */
