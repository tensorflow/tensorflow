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
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros.h"

static WORD32 xa_nn_conv2d_depthwise_nchw_getsize
(WORD32 input_width
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 output_width
 ,WORD32 circ_buf_bytewidth
 ,WORD32 scratch_bytewidth
 ,WORD32 kernel_bytewidth
 )
{
    WORD32 circ_buf_height = (kernel_height + ((OUT_HEIGHT_PER_ITER - 1) * y_stride));

    int total_size, kernel_tmp_size, state_size, circ_buf_size, scratch_size;
    int output_width_m2 = (output_width + 1)&(~1);
    kernel_tmp_size = kernel_height*ALIGNED_SIZE(kernel_width, 2)*kernel_bytewidth;
    state_size = ALIGNED_SIZE(sizeof(xa_nn_conv2d_dw_state_t), ALIGNMENT);

    circ_buf_size =
        xa_nn_circ_buf_nchw_getsize
        (circ_buf_bytewidth
         ,input_width
         ,kernel_height
         ,kernel_width
         ,x_stride
         ,y_stride
         ,x_padding
         ,circ_buf_height
         ,output_width
        );

    if (0 > circ_buf_size)
    {
        /* Returning negative error value as is to callee function to notify it.
         * Callee function should handle this negative value with care to avoid
         * any memory alloc issues. */
        return -1;
    }

    /* Get aligned size so as to have next memory pointer aligned */
    circ_buf_size = ALIGNED_SIZE(circ_buf_size, ALIGNMENT);

    scratch_size = (OUT_HEIGHT_PER_ITER * output_width_m2 * scratch_bytewidth);
    /* Get aligned size so as to have next memory pointer aligned */
    scratch_size = ALIGNED_SIZE(scratch_size, ALIGNMENT);

    total_size = kernel_tmp_size + state_size + circ_buf_size + scratch_size;

    if (0 > total_size)
    {
        return -1;
    }
    else
    {
        return total_size;
    }
}

static VOID xa_nn_conv2d_depthwise_nchw_init
(pVOID p_scratch
 ,WORD32 input_width
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 output_width
 ,WORD32 circ_buf_bytewidth
 ,WORD32 inp_precision
 )

{
    pWORD8 p_mem = p_scratch;
    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_mem;
    int state_size, circ_buf_size;
    state_size = ALIGNED_SIZE(sizeof(xa_nn_conv2d_dw_state_t), ALIGNMENT);
    p_mem = (p_mem + state_size);
    WORD32 circ_buf_height = (kernel_height + ((OUT_HEIGHT_PER_ITER - 1) * y_stride));

    xa_nn_circ_buf_nchw_init(&(p_state->circ_buf)
            ,p_mem
            ,circ_buf_bytewidth
            ,inp_precision
            ,input_width
            ,kernel_height
            ,kernel_width
            ,x_stride
            ,y_stride
            ,x_padding
            ,circ_buf_height
            ,output_width
            );

    circ_buf_size = (int)((unsigned)p_state->circ_buf.p_end - (unsigned)p_state->circ_buf.p_begin);
    /* Get aligned size so as to have next memory pointer aligned */
    circ_buf_size = ALIGNED_SIZE(circ_buf_size, ALIGNMENT);

    /* Every row of circular buffer is 8 byte aligned so don't need ALIGNED_SIZE for circular
       buffer size */
    p_mem = (p_mem + circ_buf_size);
    p_state->p_scratch = (pVOID)p_mem;
}

WORD32 xa_nn_conv2d_depthwise_getsize
(WORD32 input_height
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
 ,WORD32 inp_precision
 ,WORD32 inp_data_format
 )
{
    XA_NNLIB_CHK_COND((input_height <= 0), -1);
    XA_NNLIB_CHK_COND((input_width <= 0), -1);
    XA_NNLIB_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
    XA_NNLIB_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_CHK_COND((x_stride <= 0 || x_stride > kernel_width), -1);
    XA_NNLIB_CHK_COND((y_stride <= 0 || y_stride > kernel_height), -1);
    XA_NNLIB_CHK_COND((x_padding < 0), -1);
    XA_NNLIB_CHK_COND((y_padding < 0), -1);
    XA_NNLIB_CHK_COND((output_height <= 0), -1);
    XA_NNLIB_CHK_COND((output_width <= 0), -1);
    XA_NNLIB_CHK_COND((inp_data_format != 0), -1);

    WORD32 scratch_bytewidth = 0;
    WORD32 circ_buf_bytewidth = 0;
    WORD32 kernel_tmp_bytewidth = 0;
    WORD32 total_size = 0;

    switch (inp_precision)
    {
        case 8: /* For 8b */
        case 16: /* For 16b */
            scratch_bytewidth = 8; /* 64b scratch */
            circ_buf_bytewidth = (inp_precision/8); /* bytewidth as per precision */
            break;

        case -1: /* For f32 */
            scratch_bytewidth = 4; /* f32 scratch */
            circ_buf_bytewidth = 4; /* bytewidth for f32 */
            break;

        case -3: /* For asym8 */
        case -4: /* For sym8 */
            scratch_bytewidth = 4;
            circ_buf_bytewidth = 2;
            kernel_tmp_bytewidth = 2;
            break;

        default:
            return -1; /* Retunrning due to invalid input */
            break;
    }

    total_size = xa_nn_conv2d_depthwise_nchw_getsize(input_width
                ,kernel_height
                ,kernel_width
                ,x_stride
                ,y_stride
                ,x_padding
                ,output_width
                ,circ_buf_bytewidth
                ,scratch_bytewidth
                ,kernel_tmp_bytewidth);

    return total_size;
}

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
 ,WORD32 inp_precision
 )

{
    WORD32 circ_buf_bytewidth = 0;

    switch (inp_precision)
    {
        case 8: /* For 8b */
        case 16: /* For 16b */
            circ_buf_bytewidth = (inp_precision/8);
            break;

        case -1: /* For f32 */
            circ_buf_bytewidth = 4;
            break;

        case -3: /* For asym8 */
        case -4: /* For sym8 */
            circ_buf_bytewidth = 2;

        default:
            break;
    }

    xa_nn_conv2d_depthwise_nchw_init(p_scratch
                ,input_width
                ,kernel_height
                ,kernel_width
                ,x_stride
                ,y_stride
                ,x_padding
                ,output_width
                ,circ_buf_bytewidth
                ,inp_precision);
}
