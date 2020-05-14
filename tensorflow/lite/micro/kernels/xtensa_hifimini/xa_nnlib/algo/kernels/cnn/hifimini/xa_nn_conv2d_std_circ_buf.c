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
#include <string.h>
#include "xa_nnlib_common.h"
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_common_macros.h"

WORD32 xa_nn_conv2d_std_getsize(
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 input_precision)
{
  XA_NNLIB_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_CHK_COND((y_stride <= 0), -1);
  XA_NNLIB_CHK_COND((y_padding < 0), -1);
  XA_NNLIB_CHK_COND((out_height <= 0), -1);

  WORD32 mem_req = 0;
  WORD32 input_size;
  WORD32 align_size;

  mem_req += ALIGNED_SIZE(sizeof(xa_nn_conv_state_t), ALIGNMENT);
  /* Input precision is checked here */
  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = 2*sizeof(UWORD8);
      align_size = ALIGNMENT>>2;
      break;
    case -4:
      input_size = 2*sizeof(WORD8);
      align_size = ALIGNMENT>>2;
      break;
    default:
      return -1;
      break;
  }

  // Computing circular buffer size
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;
  WORD32 input_channels_pad = PADDED_SIZE(input_channels, align_size);
  WORD32 cir_buf_size_bytes = (y_padding + input_height + y_b_pad) * kernel_width * input_channels_pad * input_size;

  /* scratch memory for convolution using matrix multiplication */
  mem_req += cir_buf_size_bytes;
  mem_req += BUS_WIDTH;

  return mem_req;
}

VOID xa_nn_conv2d_std_init_state(
    VOID *p_scratch,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 input_precision)
{
  WORD8 *p_mem = (WORD8 *)p_scratch;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_mem;

  size_t input_size;
  UWORD32 align_size;

  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = 2*sizeof(UWORD8);
      align_size = ALIGNMENT>>2;
      break;
    case -4:
      input_size = 2*sizeof(WORD8);
      align_size = ALIGNMENT>>2;
      break;
    default:
      input_size = 0;
      align_size = 0;
      break;
  }

  p_mem += sizeof(xa_nn_conv_state_t);
  p_mem = ALIGNED_ADDR(p_mem, ALIGNMENT);


  if(((UWORD32)p_kernel & BUS_WIDTH_MASK) == ((UWORD32)p_mem & BUS_WIDTH_MASK))
  {
    p_mem += BUS_WIDTH; /* Add a offset to avoid banking stall */
  }

  p_state->cir_buf.p_begin = p_mem;
  p_state->cir_buf.p_curr = p_mem;

  // Computing circular buffer size
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;
  WORD32 input_channels_pad = PADDED_SIZE(input_channels, align_size);
  WORD32 cir_buf_size_bytes = (y_padding + input_height + y_b_pad) * kernel_width * input_channels_pad * input_size;

  p_mem += cir_buf_size_bytes;
  p_state->cir_buf.p_end = p_mem;

  //AE_SETCBEGIN0(p_state->cir_buf.p_begin);
  //AE_SETCEND0(p_state->cir_buf.p_end);
}

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
    WORD32 pad_val)
{
  WORD32 i,j,k;
  UWORD8 *p_inp = (UWORD8 *)*pp_inp;
  WORD32 planes_to_add = x_stride > kernel_width ? 0 : kernel_width - x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;
  WORD16 *p_dst = (WORD16 *)p_state->cir_buf.p_curr;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  WORD8 *p_begin, *p_end;
  WORD8 *p_temp;
  int size;

  p_begin = (WORD8 *)p_state->cir_buf.p_begin;
  p_end   = (WORD8 *)p_state->cir_buf.p_end;
  size = p_end - p_begin;

  p_temp = (WORD8 *)p_dst;
  HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
  p_dst = (WORD16 *)p_temp;

  // Initialize circular buffer
  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
  }

  // Set next 'input_height' rows of cir_buf with zero and/or input data
  WORD32 copy_x_pad_width = x_padding;
  WORD32 copy_inp_width = 0;
  if(planes_to_add <= x_padding)
  {
    copy_x_pad_width = planes_to_add;
  }
  else
  {
    copy_inp_width = planes_to_add - x_padding;
  }
  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    for(k=0;k<copy_inp_width;k++)
    {
      //memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      for(j=0; j < input_channels; j++){
          p_dst[j] = p_inp[j] - pad_val_u8;
      }
      memset(&p_dst[input_channels * input_bytewidth], 0, (input_channels_pad - input_channels) * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
      p_inp += input_channels * input_bytewidth;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;
  *pp_inp = (VOID *)p_inp;
}

// Add x_stride (but not more than kernel_width) x (input_height x input_channels) new planes to circular buffer
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
    WORD32 pad_val)
{
  WORD32 i,j,k;
  UWORD8 *p_inp = (UWORD8 *)*pp_inp;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  WORD32 planes_to_add = x_stride > kernel_width ? kernel_width : x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;
  WORD8 *p_temp;

  WORD8 *p_begin, *p_end;
  int size;
  p_begin = (WORD8 *)p_state->cir_buf.p_begin;
  p_end   = (WORD8 *)p_state->cir_buf.p_end;
  size = p_end - p_begin;

  // Copy 'planes_to_add' planes of data to circular buffer
  p_temp = (WORD8 *)p_state->cir_buf.p_curr;
  HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_add * input_channels_pad * cir_buf_bytewidth);
  p_state->cir_buf.p_curr = (WORD16 *)p_temp;
  WORD16 *p_dst = (WORD16 *)p_state->cir_buf.p_curr;
  p_temp = (WORD8 *)p_dst;
  HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
  p_dst = (WORD16 *)p_temp;

  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
  }

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + planes_to_add;
  WORD32 copy_x_pad_width = 0;
  WORD32 copy_inp_width = 0;
  WORD32 to_skip_inp_width = x_stride - planes_to_add;     // Non-zero for x_stride > kernel_width
  WORD32 copy_x_r_pad_width = 0;
  if(idx_beg_inp_width_pad < x_padding)
  {
    copy_x_pad_width = x_padding - idx_beg_inp_width_pad;
    copy_inp_width = idx_end_inp_width_pad - x_padding;
  }
  else if(idx_end_inp_width_pad <= x_padding + input_width)
  {
    copy_inp_width = planes_to_add;
  }
  else if(idx_beg_inp_width_pad < x_padding + input_width)
  {
    copy_inp_width = x_padding + input_width - idx_beg_inp_width_pad;
    copy_x_r_pad_width = idx_end_inp_width_pad - (x_padding + input_width);
  }
  else
  {
    copy_x_r_pad_width = planes_to_add;
  }

  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    for(k=0;k<copy_inp_width;k++)
    {
      //memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      for(j=0; j < input_channels; j++){
          p_dst[j] = p_inp[j] - pad_val_u8;
      }
      memset(&p_dst[input_channels * input_bytewidth], 0, (input_channels_pad - input_channels) * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
      p_inp += input_channels * input_bytewidth;
    }
    for(k=0;k<copy_x_r_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }
  p_inp += (-input_height * input_width + copy_inp_width + to_skip_inp_width) * input_channels * input_bytewidth;

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
  }
  *pp_inp = (VOID *)p_inp;
}

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
    WORD32 pad_val)
{
  WORD32 i,j,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD32 planes_to_add = x_stride > kernel_width ? 0 : kernel_width - x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;
  WORD16 *p_dst = (WORD16 *)p_state->cir_buf.p_curr;
  WORD8 pad_val_u8 = (WORD8)pad_val;
  WORD8 *p_begin, *p_end;
  WORD8 *p_temp;
  int size;

  p_begin = (WORD8 *)p_state->cir_buf.p_begin;
  p_end   = (WORD8 *)p_state->cir_buf.p_end;
  size = p_end - p_begin;

  p_temp = (WORD8 *)p_dst;
  HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
  p_dst = (WORD16 *)p_temp;

  // Initialize circular buffer
  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
  }

  // Set next 'input_height' rows of cir_buf with zero and/or input data
  WORD32 copy_x_pad_width = x_padding;
  WORD32 copy_inp_width = 0;
  if(planes_to_add <= x_padding)
  {
    copy_x_pad_width = planes_to_add;
  }
  else
  {
    copy_inp_width = planes_to_add - x_padding;
  }
  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    for(k=0;k<copy_inp_width;k++)
    {
      //memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      for(j=0; j < input_channels; j++){
          p_dst[j] = p_inp[j] - pad_val_u8;
      }
      memset(&p_dst[input_channels * input_bytewidth], 0, (input_channels_pad - input_channels) * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
      p_inp += input_channels * input_bytewidth;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;
  *pp_inp = (VOID *)p_inp;
}

// Add x_stride (but not more than kernel_width) x (input_height x input_channels) new planes to circular buffer
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
    WORD32 pad_val)
{
  WORD32 i,j,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD8 pad_val_u8 = (WORD8)pad_val;
  WORD32 planes_to_add = x_stride > kernel_width ? kernel_width : x_stride;
  WORD32 planes_to_keep = kernel_width - planes_to_add;
  WORD8 *p_temp;

  WORD8 *p_begin, *p_end;
  int size;
  p_begin = (WORD8 *)p_state->cir_buf.p_begin;
  p_end   = (WORD8 *)p_state->cir_buf.p_end;
  size = p_end - p_begin;

  // Copy 'planes_to_add' planes of data to circular buffer
  p_temp = (WORD8 *)p_state->cir_buf.p_curr;
  HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_add * input_channels_pad * cir_buf_bytewidth);
  p_state->cir_buf.p_curr = (WORD16 *)p_temp;
  WORD16 *p_dst = (WORD16 *)p_state->cir_buf.p_curr;
  p_temp = (WORD8 *)p_dst;
  HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
  p_dst = (WORD16 *)p_temp;

  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
  }

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + planes_to_add;
  WORD32 copy_x_pad_width = 0;
  WORD32 copy_inp_width = 0;
  WORD32 to_skip_inp_width = x_stride - planes_to_add;     // Non-zero for x_stride > kernel_width
  WORD32 copy_x_r_pad_width = 0;
  if(idx_beg_inp_width_pad < x_padding)
  {
    copy_x_pad_width = x_padding - idx_beg_inp_width_pad;
    copy_inp_width = idx_end_inp_width_pad - x_padding;
  }
  else if(idx_end_inp_width_pad <= x_padding + input_width)
  {
    copy_inp_width = planes_to_add;
  }
  else if(idx_beg_inp_width_pad < x_padding + input_width)
  {
    copy_inp_width = x_padding + input_width - idx_beg_inp_width_pad;
    copy_x_r_pad_width = idx_end_inp_width_pad - (x_padding + input_width);
  }
  else
  {
    copy_x_r_pad_width = planes_to_add;
  }

  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    for(k=0;k<copy_inp_width;k++)
    {
      //memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      for(j=0; j < input_channels; j++){
          p_dst[j] = p_inp[j] - pad_val_u8;
      }
      memset(&p_dst[input_channels * input_bytewidth], 0, (input_channels_pad - input_channels) * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
      p_inp += input_channels * input_bytewidth;
    }
    for(k=0;k<copy_x_r_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }
  p_inp += (-input_height * input_width + copy_inp_width + to_skip_inp_width) * input_channels * input_bytewidth;

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<planes_to_add;k++)
    {
      memset(p_dst, 0, input_channels_pad * cir_buf_bytewidth);
      p_temp = (WORD8 *)p_dst;
      HF2_AE_ADDCIRC16X4_XC(p_temp, input_channels_pad * cir_buf_bytewidth);
      p_dst = (WORD16 *)p_temp;
    }
    p_temp = (WORD8 *)p_dst;
    HF2_AE_ADDCIRC16X4_XC(p_temp, planes_to_keep * input_channels_pad * cir_buf_bytewidth);
    p_dst = (WORD16 *)p_temp;
  }
  *pp_inp = (VOID *)p_inp;
}

