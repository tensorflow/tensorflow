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
#include "xa_nnlib_common.h"
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_common_macros.h"


static WORD32 conv_x_left_pad(
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD32* __restrict__ p_bias,
    UWORD8 *p_out,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;

  WORD32 left_shift, right_shift;
  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i=0;i<out_height;i++)
  {
    for(j=0;j<out_width_over_x_pad;j++)
    {
      for(k=0;k<out_channels;k++)
      {
        ae_q56s d_acc0;
        MULTIPLY_BY_QUANTIZED_MULTIPLIER(d_acc0, p_bias[k], out_multiplier, left_shift, right_shift);
        d_acc0 = AE_ADDSQ56S(d_acc0, AE_CVTQ48A32S(out_zero_bias));
        d_acc0 = AE_MAXQ56S(AE_MINQ56S(d_acc0, AE_CVTQ48A32S(255)), AE_ZEROQ56());
        p_out[i*out_height_offset+j*out_width_offset+k*out_channels_offset] = (UWORD8)AE_TRUNCA32Q48(d_acc0);
      }
    }
  }
  return out_width_over_x_pad;
}

static WORD32 conv_x_right_pad(
    WORD32 x_padding,
    WORD32 input_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD32* __restrict__ p_bias,
    UWORD8 *p_out,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride + 1;
  WORD32 left_shift, right_shift;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;

  left_shift = out_shift<0?0:out_shift;
  right_shift = out_shift>0?0:-out_shift;
  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i=0;i<out_height;i++)
  {
    for(j=idx_out_width_over_x_r_pad;j<out_width;j++)
    {
      for(k=0;k<out_channels;k++)
      {
        ae_q56s d_acc0;
        MULTIPLY_BY_QUANTIZED_MULTIPLIER(d_acc0, p_bias[k], out_multiplier, left_shift, right_shift);
        d_acc0 = AE_ADDSQ56S(d_acc0, AE_CVTQ48A32S(out_zero_bias));
        d_acc0 = AE_MAXQ56S(AE_MINQ56S(d_acc0, AE_CVTQ48A32S(255)), AE_ZEROQ56());
        p_out[i*out_height_offset+j*out_width_offset+k*out_channels_offset] = (UWORD8)AE_TRUNCA32Q48(d_acc0);
      }
    }
  }
  return out_width_over_x_r_pad;
}

WORD32 xa_nn_conv2d_std_asym8uxasym8u(
    UWORD8* __restrict__ p_out,
    const UWORD8* __restrict__ p_inp,
    const UWORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 kernel_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch)
{
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -255 || input_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_zero_bias < -255 || kernel_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < 0 || out_zero_bias > 255), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  WORD32 j;
  WORD32 input_bytewidth = 1;
  WORD32 cir_buf_bytewidth = 2;
  VOID *pp_inp = (VOID *)p_inp;

  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;

  xa_nn_conv2d_std_init_state((void*)p_state,(void*)p_kernel,input_height,input_channels,kernel_height,kernel_width,x_stride,y_stride,y_padding,out_height,-3);

  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_padding;
  WORD32 input_channels_pad = PADDED_SIZE(input_channels, (ALIGNMENT>>2));

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= kernel_width)
  {
    out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, out_multiplier, out_shift, out_zero_bias);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = kernel_width + (out_width - 1) * x_stride - (x_padding + input_width);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= kernel_width)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_padding, input_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, out_multiplier, out_shift, out_zero_bias);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  conv2d_std_init_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, cir_buf_bytewidth, (VOID**)&pp_inp, p_state, -input_zero_bias);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = kernel_width - x_stride;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;


  // Process Loop to compute one output plane [out_height x out_channels] per iteration
  for(j=0;j<out_width-out_width_over_x_pad-out_width_over_x_r_pad;j++)
  {
    // Add x_stride x (input_height x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, cir_buf_bytewidth, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_zero_bias);

    // Update index to input width padded
    idx_beg_inp_width_pad += x_stride;

    // Convolution using matXvec with matrix as circular buffer
    xa_nn_matXvec_asym8xasym8_asym8_circ
      (p_out /* output */
       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
       ,p_kernel /* vec: cols */
       ,p_bias /* bias */
       ,out_height /* rows */
       ,input_channels_pad * kernel_width * kernel_height /* cols */
       ,input_channels_pad * kernel_width * y_stride/* row_offset */
       ,out_channels /* vec_count */
       ,input_channels_pad * kernel_width * kernel_height /* vec_stride */
       ,out_channels_offset /* out_col_offset */
       ,out_height_offset /* out_row_offset */
       ,input_zero_bias
       ,kernel_zero_bias
       ,out_multiplier
       ,out_shift
       ,out_zero_bias
       ,(WORD16 *)p_state->cir_buf.p_begin
       ,(WORD16 *)p_state->cir_buf.p_end
      );

    p_out += out_width_offset;
  }

  return 0;
}

