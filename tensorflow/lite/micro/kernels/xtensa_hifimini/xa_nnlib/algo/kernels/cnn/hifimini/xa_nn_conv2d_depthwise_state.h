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

#ifndef __XA_NN_CONV2D_DEPTHWISE_STATE_H__
#define __XA_NN_CONV2D_DEPTHWISE_STATE_H__

#include "xa_nn_circ_buf.h"

typedef struct _xa_nn_conv2d_dw_state_t
{
    xa_nn_circ_buf_t circ_buf;
    pVOID p_scratch;
} xa_nn_conv2d_dw_state_t;

VOID xa_nn_conv2d_depthwise_init
(pVOID p_scratch
 ,WORD32 input_height
 ,WORD32 input_width
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 circ_buf_precision
 );

#endif /* #ifndef __XA_NN_CONV2D_DEPTHWISE_STATE_H__ */
