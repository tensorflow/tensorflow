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

/*************************************************************************/
/* 2D Convolution implementation */
static inline void conv2d_nchw_asym8uxasym8u_hf4_convmul
(pWORD8 __restrict__ p_out  /* Output:  [Stream] [(out_stride): (actual_out_height): (actual_out_width)] */
 ,const WORD16 *__restrict__ p_ker  /* Kernel:  [Block] [1:             kernel_height:       kernel_width_pad] */
 ,const WORD16 *__restrict__ p_inp  /* Input:   [Block] [1:             input_height:        input_width] */
 ,WORD32 bias
 ,int input_height
 ,int input_width
 ,int kernel_height
 ,int kernel_width
 ,int actual_out_height      /* This is the actual output height, processing should be limited to it. */
 ,int actual_out_width       /* This is the actual output width, processing should be limited to it. */
 ,int out_stride
 ,int x_stride
 ,int y_stride
 ,WORD32  out_multiplier
 ,WORD32  out_shift
 ,WORD32  out_zero_bias
 ,pWORD32 __restrict__ p_scratch /* Scratch: [Block] [1:             (actual_out_height): (out_width)] */
 ,WORD8 *p_begin
 ,WORD8 *p_end
 )
{
    int kernel_width_pad = (kernel_width+1)&(~1);

    /* Generic case */
    int i, j, k, l;
    int output_width_m2 = (actual_out_width + 1)&(~1);
    int size = p_end - p_begin;

    ae_q56s accu_int64_0, accu_int64_1;
    ae_q32s *scratch_ptr = (ae_q32s *)p_scratch;

    ae_q56s _ae_int32_sat_bias;
    _ae_int32_sat_bias = AE_CVTQ48A32S(bias);

    if(x_stride == 1)
    {
        for(i = 0; i < actual_out_height; i++)
        {
            scratch_ptr = (ae_q32s *) (p_scratch + (i * output_width_m2));
            for(j = 0; j < output_width_m2; j+=2)
            {
                accu_int64_0 = AE_ZEROQ56();
                accu_int64_1 = AE_ZEROQ56();
#pragma loop_count min=1
                for(k = 0; k < kernel_height; k++)
                {
                    const WORD8 *pt_inp = (const WORD8 *)(p_inp);
                    HF2_AE_ADDCIRC16X4_XC
                        (pt_inp
                        ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width))
                        );
                    const ae_p16x2s *pt_ker = (const ae_p16x2s *)(p_ker + k*kernel_width_pad);
                    const ae_p16x2s *ptt_inp = (const ae_p16x2s *)pt_inp;
                    ae_p24x2s d_inp, d_ker;
                    ae_p24x2s d_inp0, d_inp1;
                    d_inp0 = *ptt_inp++;
#pragma loop_count min=1
#pragma no_unroll
                    for(l = 0; l < (kernel_width_pad>>1); l++)
                    {
                        d_inp = *ptt_inp++;
                        d_ker = *pt_ker++;
                        d_inp1 = AE_SELP24_LH(d_inp0, d_inp);
                        AE_MULAAP24S_HH_LL(accu_int64_0, d_inp0, d_ker);
                        AE_MULAAP24S_HH_LL(accu_int64_1, d_inp1, d_ker);
                        d_inp0 = d_inp;
                    }
                }
                accu_int64_0 = AE_ADDSQ56S(accu_int64_0, _ae_int32_sat_bias);
                accu_int64_1 = AE_ADDSQ56S(accu_int64_1, _ae_int32_sat_bias);
                accu_int64_0 = AE_SATQ48S(accu_int64_0);
                accu_int64_1 = AE_SATQ48S(accu_int64_1);
                *scratch_ptr++ = (accu_int64_0);
                *scratch_ptr++ = (accu_int64_1);
            }
        }
    }
    else if(x_stride == 2)
    {
        for(i = 0; i < actual_out_height; i++)
        {
            scratch_ptr = (ae_q32s *) (p_scratch + (i * output_width_m2));
            for(j = 0; j < output_width_m2; j+=2)
            {
                accu_int64_0 = AE_ZEROQ56();
                accu_int64_1 = AE_ZEROQ56();
#pragma loop_count min=1
                for(k = 0; k < kernel_height; k++)
                {
                    const WORD8 *pt_inp = (const WORD8 *)(p_inp);
                    HF2_AE_ADDCIRC16X4_XC
                        (pt_inp
                        ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*2 + k*input_width))
                        );
                    const ae_p16x2s *pt_ker = (const ae_p16x2s *)(p_ker + k*kernel_width_pad);
                    const ae_p16x2s *ptt_inp = (const ae_p16x2s *)pt_inp;
                    ae_p24x2s d_inp, d_ker;
                    ae_p24x2s d_inp0;
                    d_inp0 = *ptt_inp++;
#pragma loop_count min=1
#pragma no_unroll
                    for(l = 0; l < (kernel_width_pad>>1); l++)
                    {
                        d_inp = *ptt_inp++;
                        d_ker = *pt_ker++;
                        AE_MULAAP24S_HH_LL(accu_int64_0, d_inp0, d_ker);
                        AE_MULAAP24S_HH_LL(accu_int64_1, d_inp, d_ker);
                        d_inp0 = d_inp;
                    }
                }
                accu_int64_0 = AE_ADDSQ56S(accu_int64_0, _ae_int32_sat_bias);
                accu_int64_1 = AE_ADDSQ56S(accu_int64_1, _ae_int32_sat_bias);
                accu_int64_0 = AE_SATQ48S(accu_int64_0);
                accu_int64_1 = AE_SATQ48S(accu_int64_1);
                *scratch_ptr++ = (accu_int64_0);
                *scratch_ptr++ = (accu_int64_1);
            }
        }
    }
    else if((x_stride&1) == 0)
    {
        for(i = 0; i < actual_out_height; i++)
        {
            scratch_ptr = (ae_q32s *) (p_scratch + (i * output_width_m2));
            for(j = 0; j < actual_out_width; j+=2)
            {
                accu_int64_0 = AE_ZEROQ56();
                accu_int64_1 = AE_ZEROQ56();
#pragma loop_count min=1
                for(k = 0; k < kernel_height; k++)
                {
                    const WORD8 *pt_inp = (const WORD8 *)(p_inp);
                    const WORD8 *pt_inp1 = (const WORD8 *)(p_inp);
                    HF2_AE_ADDCIRC16X4_XC
                        (pt_inp
                        ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*x_stride + k*input_width))
                        );
                    pt_inp1 = pt_inp + x_stride*sizeof(WORD16);
                    const ae_p16x2s *pt_ker = (const ae_p16x2s *)(p_ker + k*kernel_width_pad);
                    const ae_p16x2s *ptt_inp = (const ae_p16x2s *)pt_inp;
                    const ae_p16x2s *ptt_inp1 = (const ae_p16x2s *)pt_inp1;
                    ae_p24x2s d_ker;
                    ae_p24x2s d_inp0, d_inp1;
#pragma loop_count min=1
#pragma no_unroll
                    for(l = 0; l < (kernel_width_pad>>1); l++)
                    {
                        d_inp0 = *ptt_inp++;
                        d_inp1 = *ptt_inp1++;
                        d_ker = *pt_ker++;
                        AE_MULAAP24S_HH_LL(accu_int64_0, d_inp0, d_ker);
                        AE_MULAAP24S_HH_LL(accu_int64_1, d_inp1, d_ker);
                    }
                }
                accu_int64_0 = AE_ADDSQ56S(accu_int64_0, _ae_int32_sat_bias);
                accu_int64_0 = AE_SATQ48S(accu_int64_0);
                *scratch_ptr++ = (accu_int64_0);
                accu_int64_1 = AE_ADDSQ56S(accu_int64_1, _ae_int32_sat_bias);
                accu_int64_1 = AE_SATQ48S(accu_int64_1);
                *scratch_ptr++ = (accu_int64_1);
            }
        }
    }
    else
    {
        for(i = 0; i < actual_out_height; i++)
        {
            scratch_ptr = (ae_q32s *) (p_scratch + (i * output_width_m2));
            for(j = 0; j < output_width_m2; j+=2)
            {
                accu_int64_0 = AE_ZEROQ56();
                accu_int64_1 = AE_ZEROQ56();
#pragma loop_count min=1
                for(k = 0; k < kernel_height; k++)
                {
                    const WORD8 *pt_inp = (const WORD8 *)(p_inp);
                    const WORD8 *pt_inp1;
                    HF2_AE_ADDCIRC16X4_XC
                        (pt_inp
                        ,((sizeof(WORD16)) * ((i * y_stride * input_width) + j*x_stride + k*input_width))
                        );
                    pt_inp1 = pt_inp + x_stride*sizeof(WORD16);
                    const ae_p16x2s *pt_ker = (const ae_p16x2s *)(p_ker + k*kernel_width_pad);
                    const ae_p16x2s *ptt_inp = (const ae_p16x2s *)pt_inp;
                    const ae_p16s *ptt_inp1 = (const ae_p16s *)pt_inp1;
                    ae_p24x2s d_inp, d_ker;
                    ae_p24x2s d_inp0, d_inp1;
#pragma loop_count min=1
#pragma no_unroll
                    for(l = 0; l < (kernel_width_pad>>1); l++)
                    {
                        d_inp0 = *ptt_inp++;
                        d_inp1 = *ptt_inp1++;
                        d_inp = *ptt_inp1++;
                        d_inp1 = AE_SELP24_LL(d_inp1, d_inp);
                        d_ker = *pt_ker++;
                        AE_MULAAP24S_HH_LL(accu_int64_0, d_inp0, d_ker);
                        AE_MULAAP24S_HH_LL(accu_int64_1, d_inp1, d_ker);
                    }
                }
                accu_int64_0 = AE_ADDSQ56S(accu_int64_0, _ae_int32_sat_bias);
                accu_int64_0 = AE_SATQ48S(accu_int64_0);
                *scratch_ptr++ = accu_int64_0;
                accu_int64_1 = AE_ADDSQ56S(accu_int64_1, _ae_int32_sat_bias);
                accu_int64_1 = AE_SATQ48S(accu_int64_1);
                *scratch_ptr++ = accu_int64_1;
            }
        }
    }

    WORD32 *scratch_ptr1 = (WORD32 *) p_scratch;
    int left_shift = out_shift;
    int right_shift = -out_shift;
    left_shift = left_shift < 0 ? 0 : left_shift;
    right_shift = right_shift < 0 ? 0 : right_shift;

    for(i = 0; i < actual_out_height; i++)
    {
        scratch_ptr1 = (WORD32 *) p_scratch + (i * output_width_m2);
        UWORD8 *out_ptr  = (UWORD8 *) p_out + (i * out_stride * actual_out_width);
        ae_q56s accu_int32_0;

        for(j = 0; j < actual_out_width; j++)
        {
            MULTIPLY_BY_QUANTIZED_MULTIPLIER(accu_int32_0, scratch_ptr1[j], out_multiplier, left_shift, right_shift);
            /* Doing MAC here reduces Q register pressure */
            AE_MULAP24S_LL(accu_int32_0, AE_CVTP24A16(1), AE_CVTP24A16(out_zero_bias));
            int accu_32 = AE_TRUNCA32Q48(AE_SATQ48S(accu_int32_0));
            /* Doing these min/max in ar registers reduces spill in Q registers which reduces loop cycles */
            accu_32 = accu_32 < 0 ? 0 : accu_32;
            accu_32 = accu_32 > 255 ? 255 : accu_32;

            out_ptr[(j * out_stride)] = (UWORD8)accu_32;
        }
    }
}

static void xa_nn_conv2d_depthwise_nchw_asym8uxasym8u
(pUWORD8 __restrict__ p_out
 ,const UWORD8 *__restrict__ p_kernel
 ,const UWORD8 *__restrict__ p_inp
 ,const WORD32 *__restrict__ p_bias
 ,WORD32  input_height
 ,WORD32  input_width
 ,WORD32  input_channels
 ,WORD32  kernel_height
 ,WORD32  kernel_width
 ,WORD32  channels_multiplier
 ,WORD32  x_stride
 ,WORD32  y_stride
 ,WORD32  x_padding
 ,WORD32  y_padding
 ,WORD32  out_height
 ,WORD32  out_width
 ,WORD32  input_zero_bias
 ,WORD32  kernel_zero_bias
 ,WORD32  out_multiplier
 ,WORD32  out_shift
 ,WORD32  out_zero_bias
,WORD32  out_data_format
,pVOID p_scratch
)
{
    WORD16 *p_kernel_tmp = p_scratch;
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 2);
    p_scratch = ((WORD8 *)p_scratch) + kernel_height*kernel_width_pad*sizeof(WORD16);
    xa_nn_conv2d_depthwise_init
        (p_scratch
         ,input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,out_height
         ,out_width
         ,-3
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh;
    int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    int input_zero_bias_neg = -input_zero_bias;
    const WORD16 *pt_ker;
    const WORD8 *pt_inp;
    pWORD16 p_inp_circ;
    p_scratch = (pWORD64)(p_state->p_scratch);

    WORD32 bias = 0;

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = (const WORD8 *)&p_inp[itr_ic];

        CIRC_BUF_ADD_ROWS_INIT_WITH_PAD_VAL(rows_added
                ,rows_to_add
                ,top_pad
                ,bottom_pad
                ,input_row
                ,input_height
                ,input_width
                ,input_channels
                ,kernel_height
                ,y_stride
                ,x_padding
                ,y_padding
                ,p_circ_buf
                ,pt_inp
                ,&input_zero_bias_neg
                );

        for(itr_oh = 0; itr_oh < out_height - (circ_out_height - 1); itr_oh += circ_out_height)
        {
            CIRC_BUF_ADD_ROWS_WITH_PAD_VAL(rows_added
                    ,rows_to_add
                    ,top_pad
                    ,bottom_pad
                    ,input_row
                    ,input_height
                    ,input_width
                    ,input_channels
                    ,circ_out_height
                    ,y_stride
                    ,x_padding
                    ,y_padding
                    ,p_circ_buf
                    ,pt_inp
                    ,&input_zero_bias_neg
                    );

            p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

            for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
            {
                int kh, kw;
                for(kh = 0; kh < kernel_height; kh++)
                {
                    for(kw = 0; kw < kernel_width; kw++)
                    {
                        WORD16 tmp = p_kernel[(kh*kernel_width + kw)*(input_channels*channels_multiplier) + itr_ic*channels_multiplier + itr_cm];
                        p_kernel_tmp[kh*kernel_width_pad + kw] = tmp + kernel_zero_bias;
                    }
                    for(; kw < kernel_width_pad; kw++)
                    {
                        p_kernel_tmp[kh*kernel_width_pad + kw] = 0;
                    }
                }
                pt_ker = (const WORD16*)p_kernel_tmp;
                bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

                conv2d_nchw_asym8uxasym8u_hf4_convmul
                    ((pWORD8)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                     ,pt_ker
                     ,p_inp_circ
                     ,bias
                     ,p_circ_buf->rows
                     ,p_circ_buf->row_offset
                     ,kernel_height
                     ,kernel_width
                     ,circ_out_height
                     ,out_width
                     ,(input_channels * channels_multiplier)
                     ,x_stride
                     ,y_stride
                     ,out_multiplier
                     ,out_shift
                     ,out_zero_bias
                     ,p_scratch
                     ,(WORD8 *)p_circ_buf->p_begin
                     ,(WORD8 *)p_circ_buf->p_end
                    );
            }
        }

        CIRC_BUF_ADD_ROWS_WITH_PAD_VAL(rows_added
                ,rows_to_add
                ,top_pad
                ,bottom_pad
                ,input_row
                ,input_height
                ,input_width
                ,input_channels
                ,circ_out_height
                ,y_stride
                ,x_padding
                ,y_padding
                ,p_circ_buf
                ,pt_inp
                ,&input_zero_bias_neg
                );

        p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            int kh, kw;
            for(kh = 0; kh < kernel_height; kh++)
            {
                for(kw = 0; kw < kernel_width; kw++)
                {
                    WORD16 tmp = p_kernel[(kh*kernel_width + kw)*(input_channels*channels_multiplier) + itr_ic*channels_multiplier + itr_cm];
                    p_kernel_tmp[kh*kernel_width_pad + kw] = tmp + kernel_zero_bias;
                }
                for(; kw < kernel_width_pad; kw++)
                {
                    p_kernel_tmp[kh*kernel_width_pad + kw] = 0;
                }
            }
            pt_ker = (const WORD16*)p_kernel_tmp;
            bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

            conv2d_nchw_asym8uxasym8u_hf4_convmul
                ((pWORD8)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                 ,pt_ker
                 ,p_inp_circ
                 ,bias
                 ,p_circ_buf->rows
                 ,p_circ_buf->row_offset
                 ,kernel_height
                 ,kernel_width
                 ,(out_height - itr_oh)
                 ,out_width
                 ,(input_channels * channels_multiplier)
                 ,x_stride
                 ,y_stride
                 ,out_multiplier
                 ,out_shift
                 ,out_zero_bias
                 ,p_scratch
                 ,(WORD8 *)p_circ_buf->p_begin
                 ,(WORD8 *)p_circ_buf->p_end
                );
        }
    }
}
/*******************************************************************************/

WORD32 xa_nn_conv2d_depthwise_asym8uxasym8u
(pUWORD8 __restrict__ p_out
 ,const UWORD8 *__restrict__ p_kernel
 ,const UWORD8 *__restrict__ p_inp
 ,const WORD32 *__restrict__ p_bias
 ,WORD32  input_height
 ,WORD32  input_width
 ,WORD32  input_channels
 ,WORD32  kernel_height
 ,WORD32  kernel_width
 ,WORD32  channels_multiplier
 ,WORD32  x_stride
 ,WORD32  y_stride
 ,WORD32  x_padding
 ,WORD32  y_padding
 ,WORD32  out_height
 ,WORD32  out_width
 ,WORD32  input_zero_bias
 ,WORD32  kernel_zero_bias
 ,WORD32  out_multiplier
 ,WORD32  out_shift
 ,WORD32  out_zero_bias
,WORD32  inp_data_format
,WORD32  out_data_format
,pVOID p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, sizeof(WORD32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_zero_bias < -255 || input_zero_bias > 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_zero_bias < -255 || kernel_zero_bias > 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((out_zero_bias < 0 || out_zero_bias > 255), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND((y_stride > kernel_height), -1);
    XA_NNLIB_ARG_CHK_COND((x_stride > kernel_width), -1);

    xa_nn_conv2d_depthwise_nchw_asym8uxasym8u
        (p_out
         ,p_kernel
         ,p_inp
         ,p_bias
         ,input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,out_height
         ,out_width
         ,input_zero_bias
         ,kernel_zero_bias
         ,out_multiplier
         ,out_shift
         ,out_zero_bias
         ,out_data_format
         ,p_scratch);

    return 0;
}
