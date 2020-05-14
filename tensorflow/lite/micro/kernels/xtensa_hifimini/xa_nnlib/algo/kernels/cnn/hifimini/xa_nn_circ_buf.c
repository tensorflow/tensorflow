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
#include <string.h>
#include "xa_nn_circ_buf.h"
#include "xa_nnlib_common_macros.h"

int xa_nn_circ_buf_nchw_getsize(
    WORD32 bytewidth,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 circ_buf_height,
    WORD32 output_width)
{
  int circ_buf_width;
  int size_in_bytes;
  /* Some optimization is unrolling output_width by 2 without reminder loop so circular
  buffer need to have enough width to take care of it */
  int output_width_m2 = (output_width + 1)&(~1);

  circ_buf_width = kernel_width + ((output_width_m2 - 1) * x_stride);
  circ_buf_width = XT_MAX(circ_buf_width, x_padding + input_width);

  /* Aligned size independent of bytewidth */
  circ_buf_width = ALIGNED_SIZE(circ_buf_width, 2);

  size_in_bytes = bytewidth*circ_buf_height*circ_buf_width;

  if (0 > size_in_bytes)
  {
    /* If callee of this function interprets received value from here in
     * unsigned value then negative returned value will be interpreted as
     * large positive number which will explode the memory allocations.
     * Callee of this function should take care of the negative returned
     * values. */
    return -3;
  }
  else
  {
    return size_in_bytes;
  }
}

VOID xa_nn_circ_buf_nchw_init(
    xa_nn_circ_buf_t *p_circ_buf,
    pVOID p_mem,
    WORD32 bytewidth,
    WORD32 inp_precision,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 circ_buf_height,
    WORD32 output_width)
{
  int output_width_m2 = (output_width + 1)&(~1);
  /* No. of row in circular buf */
  p_circ_buf->rows       = circ_buf_height;
  p_circ_buf->row_offset = kernel_width + ((output_width_m2 - 1) * x_stride);
  p_circ_buf->row_offset = XT_MAX(p_circ_buf->row_offset, x_padding + input_width);
  /* Aligned independent of bytewidth */
  p_circ_buf->row_offset = ALIGNED_SIZE(p_circ_buf->row_offset, 2);
  p_circ_buf->bytewidth  = bytewidth;
  p_circ_buf->inp_precision = inp_precision;
  /* Initialize circular buffer pointers */
  p_circ_buf->p_begin    = p_mem;
  p_circ_buf->p_curr     = p_mem;
  p_circ_buf->p_end      = (((char *)p_mem) + p_circ_buf->rows*p_circ_buf->row_offset*bytewidth);
}

void xa_nn_circ_buf_nchw_add_rows_with_pad_val(
    xa_nn_circ_buf_t *p_circ_buf,
    const VOID *p_inp,
    WORD32 left_padding,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 n_rows,
    WORD32 top_pad,
    WORD32 bottom_pad,
    pVOID p_pad_val)
{
    int i;
    int bytewidth = p_circ_buf->bytewidth;
    WORD8 *p_begin, *p_end;
    int size;
    p_begin = (WORD8 *)p_circ_buf->p_begin;
    p_end = (WORD8 *)p_circ_buf->p_end;
    size = p_end - p_begin;

    /* Error checks */
    if (n_rows < (top_pad + bottom_pad))
    {
      return;
    }
    if (p_circ_buf->row_offset < (left_padding + input_width))
    {
      return;
    }

    if(bytewidth == 2 && p_circ_buf->inp_precision == -3)
    {
        const UWORD8 *p_src = (const UWORD8 *)p_inp;
        pWORD8 p_dst = (pWORD8)p_circ_buf->p_curr;
        WORD32 pad_val = *(WORD32 *)p_pad_val;
        /* Add top padding rows */
        for(i = 0; i < top_pad; i++)
        {
            memset(p_dst, 0, p_circ_buf->row_offset*sizeof(WORD16));
            HF2_AE_ADDCIRC16X4_XC(p_dst, p_circ_buf->row_offset*sizeof(WORD16));
        }
        /* Add input rows with left and right padding */
        for(i = 0; i < (n_rows - top_pad - bottom_pad); i++)
        {
            WORD16 *pt_dst = (WORD16 *)p_dst;
            int j;
            for(j = 0; j < left_padding; j++)
            {
                pt_dst[j] = 0;
            }
            for(; j < (left_padding + input_width); j++)
            {
                pt_dst[j] = (WORD16)p_src[i*input_width*input_channels+(j-left_padding)*input_channels] - pad_val;
            }
            for(; j < p_circ_buf->row_offset; j++)
            {
                pt_dst[j] = 0;
            }
            HF2_AE_ADDCIRC16X4_XC(p_dst, p_circ_buf->row_offset*sizeof(WORD16));
        }
        /* Add bottom padding rows */
        for(i = 0; i < bottom_pad; i++)
        {
            memset(p_dst, 0, p_circ_buf->row_offset*sizeof(WORD16));
            HF2_AE_ADDCIRC16X4_XC(p_dst, p_circ_buf->row_offset*sizeof(WORD16));
        }
        /* Update current pointer for circular buffer */
        p_circ_buf->p_curr = (pVOID)p_dst;
    }
    else if(bytewidth == 2 && p_circ_buf->inp_precision == -4)
    {
        const WORD8 *p_src = (const WORD8 *)p_inp;
        pWORD8 p_dst = (pWORD8)p_circ_buf->p_curr;
        WORD32 pad_val = *(WORD32 *)p_pad_val;
        /* Add top padding rows */
        for(i = 0; i < top_pad; i++)
        {
            memset(p_dst, 0, p_circ_buf->row_offset*sizeof(WORD16));
            HF2_AE_ADDCIRC16X4_XC(p_dst, p_circ_buf->row_offset*sizeof(WORD16));
        }
        /* Add input rows with left and right padding */
        for(i = 0; i < (n_rows - top_pad - bottom_pad); i++)
        {
            WORD16 *pt_dst = (WORD16 *)p_dst;
            int j;
            for(j = 0; j < left_padding; j++)
            {
                pt_dst[j] = 0;
            }
            for(; j < (left_padding + input_width); j++)
            {
                pt_dst[j] = (WORD16)p_src[i*input_width*input_channels+(j-left_padding)*input_channels] + pad_val;
            }
            for(; j < p_circ_buf->row_offset; j++)
            {
                pt_dst[j] = 0;
            }
            HF2_AE_ADDCIRC16X4_XC(p_dst, p_circ_buf->row_offset*sizeof(WORD16));
        }
        /* Add bottom padding rows */
        for(i = 0; i < bottom_pad; i++)
        {
            memset(p_dst, 0, p_circ_buf->row_offset*sizeof(WORD16));
            HF2_AE_ADDCIRC16X4_XC(p_dst, p_circ_buf->row_offset*sizeof(WORD16));
        }
        /* Update current pointer for circular buffer */
        p_circ_buf->p_curr = (pVOID)p_dst;
    }
}
