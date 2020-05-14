/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#ifndef  __XA_NN_CONV2D_STD_STATE_H__
#define  __XA_NN_CONV2D_STD_STATE_H__

#include "xa_type_def.h"

#define ALIGNED_SIZE( size, align ) \
  ( (size_t)(size) + (align) - 1 )

#define ALIGNED_ADDR( addr, align ) \
  (void*)( ( (UWORD32)(addr) + ( (align) - 1 ) ) & ~( (align) - 1 ) )

#define PADDED_SIZE( size, align ) \
  ( ( (size_t)(size) + (align) - 1 ) & ~( (align) - 1 ) )

#define BUS_WIDTH (8)
#define BUS_WIDTH_MASK (0xf)

typedef enum xa_nn_conv_datafmt_t{
  HWC=0
} xa_nn_conv_datafmt_t;

typedef struct _circular_buf_t{
  VOID *p_begin;
  VOID *p_end;
  VOID *p_curr;
} circular_buf_t;


typedef struct _xa_nn_conv_state_t{
  circular_buf_t cir_buf;
} xa_nn_conv_state_t;

VOID xa_nn_conv2d_std_init_state(
    VOID *p_handle,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 input_precision);

WORD32 xa_nn_matXvec_8x16_16_circ(
    WORD16 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat,
    WORD8  * __restrict__ p_vec,
    WORD16 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_8x8_8_circ(
    WORD8  * __restrict__ p_out,
    WORD8  * __restrict__ p_mat,
    WORD8  * __restrict__ p_vec,
    WORD8  * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_16x16_16_circ(
    WORD16 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat,
    WORD16 * __restrict__ p_vec,
    WORD16 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 bias_shift,
    WORD32 acc_shift);

WORD32 xa_nn_matXvec_f32_circ(
    FLOAT32 * __restrict__ p_out,
    FLOAT32 * __restrict__ p_mat,
    const FLOAT32 * __restrict__ p_vec,
    const FLOAT32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset);

WORD32 xa_nn_matXvec_asym8xasym8_asym8_circ(
    UWORD8 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat1,
    const UWORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 mat1_offset,
    WORD32 vec1_offset,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_offset,
    WORD16 *p_begin,
    WORD16 *p_end);

WORD32 xa_nn_matXvec_sym8xsym8_sym8_circ(
    WORD8 * __restrict__ p_out,
    WORD16 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_offset,
    WORD16 *p_begin,
    WORD16 *p_end);

VOID conv2d_std_init_cir_buf(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state);

VOID conv2d_std_update_cir_buf(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state);

VOID conv2d_std_init_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 cir_buf_bytewidth,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val);

VOID conv2d_std_update_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 cir_buf_bytewidth,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val);

VOID conv2d_std_init_cir_buf_sym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 cir_buf_bytewidth,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val);

VOID conv2d_std_update_cir_buf_sym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 cir_buf_bytewidth,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val);

#endif /* __XA_NN_CONV2D_STD_STATE_H__ */

