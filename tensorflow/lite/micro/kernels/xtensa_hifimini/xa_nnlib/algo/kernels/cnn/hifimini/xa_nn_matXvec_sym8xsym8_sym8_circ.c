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

#define ZERO24X2  AE_ZEROP48()
#define ZERO56    AE_ZEROQ56()

#undef ROW_UNROLL
#undef VEC_UNROLL
#define ROW_UNROLL  2
#define VEC_UNROLL  2

#define CIR_BUF_CHECK(ptr, offset)\
  p_check = ptr + offset; \
  int cbuf_ovflow_ ##ptr = p_check >= p_end;


static ae_q56s MultiplyByQuantizedMultiplier(WORD32 acc, WORD32 out_multiplier, WORD32 left_shift, WORD32 right_shift)
{
    ae_q56s d_acc, d1;
    ae_p24x2s d_mul;
    d_mul = AE_CVTP24A16X2_HL(out_multiplier, out_multiplier);
    d1 = AE_CVTQ48A32S(acc);
    d1 = AE_SLLAQ56(d1, left_shift);
    d_acc = AE_MULFQ32SP16U_L(d1, d_mul);
    d_acc = AE_SRAIQ56(d_acc, 16);
    AE_MULAFQ32SP16S_H(d_acc, d1, d_mul);
    d_acc = AE_ROUNDSQ32ASYM(d_acc);
    d_acc = AE_SRAAQ56(d_acc, right_shift);
    d_acc = AE_ROUNDSQ32SYM(d_acc);
    return d_acc;
}

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
    WORD16 *p_end)
{
  /* Iterators used in for loops */
  int i, m_itr, c_itr, vec_itr;
  /* Shifts to match with Tensorflow */
  int left_shift_0, right_shift_0, out_shift_0;
  int left_shift_1, right_shift_1, out_shift_1;
  WORD16 *p_check;
  int size = p_end - p_begin;

  for(i = 0; i < vec_count; i++)
  {
    if((p_out_shift[i] > 31) || (p_out_shift[i] < -31))
    {
        return -1;
    }
  }

  if (!p_bias)
  {
    return -1;
  }

  if(p_mat1 && p_vec1)
  {
    for(vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr+= VEC_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL - 1)); m_itr += ROW_UNROLL)
      {
        //SETUP_BIAS_BATCH;
        ae_q56s bias_0 = AE_CVTQ48A32S(p_bias[vec_itr]);
        ae_q56s bias_1 = AE_CVTQ48A32S(p_bias[vec_itr + 1]);

        //SETUP_ACC_BATCH;
        ae_q56s acc_0_0 = ZERO56;
        ae_q56s acc_0_1 = ZERO56;
        ae_q56s acc_1_0 = ZERO56;
        ae_q56s acc_1_1 = ZERO56;

        ae_p24x2s acc24_0_0 = ZERO24X2;
        ae_p24x2s acc24_0_1 = ZERO24X2;
        ae_p24x2s acc24_1_0 = ZERO24X2;
        ae_p24x2s acc24_1_1 = ZERO24X2;

        //SETUP_VEC_BATCH;
        ae_p24x2s vec_0  = ZERO24X2;
        ae_p24x2s vec_1  = ZERO24X2;
        WORD8 *p_vec_0  = (WORD8 *)(&p_vec1[(vec_itr)*vec_stride]);
        WORD8 *p_vec_1  = (WORD8 *)(&p_vec1[(vec_itr+1)*vec_stride]);

        //SETUP_MAT1;
        ae_p24x2s mat1_0 = ZERO24X2;
        ae_p24x2s mat1_1 = ZERO24X2;

        WORD16 *p_mat1_0 = (WORD16 *) p_mat1;
        WORD16 *p_mat1_1 = (WORD16 *) p_mat1;

        HF2_AE_ADDCIRC16X4_XC(p_mat1_0, (m_itr)*row_stride1);
        HF2_AE_ADDCIRC16X4_XC(p_mat1_1, (m_itr+1)*row_stride1);

        CIR_BUF_CHECK(p_mat1_0, cols1);
        CIR_BUF_CHECK(p_mat1_1, cols1);

        if(!cbuf_ovflow_p_mat1_0 && !cbuf_ovflow_p_mat1_1)
        {
          p_vec_0 = p_vec_0 - 2;
          p_vec_1 = p_vec_1 - 2;
          p_mat1_0 = p_mat1_0 - 2;
          p_mat1_1 = p_mat1_1 - 2;

          for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
          {
            //LOAD_VEC_BATCH;
            AE_LP8X2F_IU(vec_0, p_vec_0, 2*sizeof(WORD8));
            AE_LP8X2F_IU(vec_1, p_vec_1, 2*sizeof(WORD8));

            //LOAD_MAT1;
            AE_LP16X2F_IU(mat1_0, (ae_p16x2s *)p_mat1_0, 2*sizeof(WORD16));
            AE_LP16X2F_IU(mat1_1, (ae_p16x2s *)p_mat1_1, 2*sizeof(WORD16));

            //KERNEL_MAT1_VEC_BATCH;
            AE_MULAAP24S_HH_LL(acc_0_0, vec_0, mat1_0);
            AE_MULAAP24S_HH_LL(acc_0_1, vec_1, mat1_0);
            AE_MULAAP24S_HH_LL(acc_1_0, vec_0, mat1_1);
            AE_MULAAP24S_HH_LL(acc_1_1, vec_1, mat1_1);
          }
        }
        else
        {
          p_vec_0 = p_vec_0 - 2;
          p_vec_1 = p_vec_1 - 2;

          for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
          {
            //LOAD_VEC_BATCH;
            AE_LP8X2F_IU(vec_0, p_vec_0, 2*sizeof(WORD8));
            AE_LP8X2F_IU(vec_1, p_vec_1, 2*sizeof(WORD8));

            //LOAD_MAT1;
            mat1_0 = AE_LP16X2F_I((ae_p16x2s *)p_mat1_0, 0);
            mat1_1 = AE_LP16X2F_I((ae_p16x2s *)p_mat1_1, 0);

            HF2_AE_ADDCIRC16X4_XC(p_mat1_0, 2);
            HF2_AE_ADDCIRC16X4_XC(p_mat1_1, 2);

            //KERNEL_MAT1_VEC_BATCH;
            AE_MULAAP24S_HH_LL(acc_0_0, vec_0, mat1_0);
            AE_MULAAP24S_HH_LL(acc_0_1, vec_1, mat1_0);
            AE_MULAAP24S_HH_LL(acc_1_0, vec_0, mat1_1);
            AE_MULAAP24S_HH_LL(acc_1_1, vec_1, mat1_1);
          }
        }

        // Adjusting Shift for accumulators
        acc_0_0  = AE_SRAIQ56(acc_0_0, 8);
        acc_0_1  = AE_SRAIQ56(acc_0_1, 8);
        acc_1_0  = AE_SRAIQ56(acc_1_0, 8);
        acc_1_1  = AE_SRAIQ56(acc_1_1, 8);

        //ADD_BIAS_ACC_BATCH;
        acc_0_0 = AE_ADDSQ56S(acc_0_0, bias_0);
        acc_1_0 = AE_ADDSQ56S(acc_1_0, bias_0);
        acc_0_1 = AE_ADDSQ56S(acc_0_1, bias_1);
        acc_1_1 = AE_ADDSQ56S(acc_1_1, bias_1);

        //ADJUST_ACC_BATCH;
        out_shift_0 = p_out_shift[vec_itr];
        out_shift_1 = p_out_shift[vec_itr + 1];

        left_shift_0 = out_shift_0<0?0:out_shift_0;
        left_shift_1 = out_shift_1<0?0:out_shift_1;
        right_shift_0 = out_shift_0>0?0:-out_shift_0;
        right_shift_1 = out_shift_1>0?0:-out_shift_1;

        acc_0_0 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_0_0), p_out_multiplier[vec_itr], left_shift_0, right_shift_0);
        acc_1_0 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_1_0), p_out_multiplier[vec_itr], left_shift_0, right_shift_0);
        acc_0_1 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_0_1), p_out_multiplier[vec_itr + 1], left_shift_1, right_shift_1);
        acc_1_1 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_1_1), p_out_multiplier[vec_itr + 1], left_shift_1, right_shift_1);

        AE_MULAP24S_LL(acc_0_0, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));
        AE_MULAP24S_LL(acc_0_1, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));
        AE_MULAP24S_LL(acc_1_0, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));
        AE_MULAP24S_LL(acc_1_1, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));

        //STORE_ACC_BATCH;
        acc24_0_0 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_0_0, 24));
        (*((WORD8 *) (&p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_0_0, 8));\
        acc24_0_1 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_0_1, 24));
        (*((WORD8 *) (&p_out[(vec_itr + 1)*out_col_offset + (m_itr)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_0_1, 8));\
        acc24_1_0 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_1_0, 24));
        (*((WORD8 *) (&p_out[(vec_itr)*out_col_offset + (m_itr + 1)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_1_0, 8));\
        acc24_1_1 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_1_1, 24));
        (*((WORD8 *) (&p_out[(vec_itr + 1)*out_col_offset + (m_itr + 1)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_1_1, 8));\
      }

      for(; m_itr < rows; m_itr++)
      {
        //UNROLL_ROW_SETUP_BIAS_BATCH(0);
        ae_q56s bias_0 = AE_CVTQ48A32S(p_bias[vec_itr]);
        ae_q56s bias_1 = AE_CVTQ48A32S(p_bias[vec_itr + 1]);

        //UNROLL_ROW_SETUP_ACC_BATCH(0);
        ae_q56s acc_0_0 = ZERO56;
        ae_q56s acc_0_1 = ZERO56;
        ae_p24x2s acc24_0_0 = ZERO24X2;
        ae_p24x2s acc24_0_1 = ZERO24X2;

        //SETUP_VEC_BATCH;
        ae_p24x2s vec_0  = ZERO24X2;
        ae_p24x2s vec_1  = ZERO24X2;
        WORD8 *p_vec_0  = (WORD8 *)(&p_vec1[(vec_itr)*vec_stride]);
        WORD8 *p_vec_1  = (WORD8 *)(&p_vec1[(vec_itr+1)*vec_stride]);

        //UNROLL_SETUP_MAT1(0);
        ae_p24x2s mat1_0 = ZERO24X2;
        WORD16 *p_mat1_0 = (WORD16 *) p_mat1;

        HF2_AE_ADDCIRC16X4_XC(p_mat1_0, (m_itr)*row_stride1);

        p_vec_0 = p_vec_0 - 2;
        p_vec_1 = p_vec_1 - 2;
        for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
        {
          //LOAD_VEC_BATCH;
          AE_LP8X2F_IU(vec_0, p_vec_0, 2*sizeof(WORD8));
          AE_LP8X2F_IU(vec_1, p_vec_1, 2*sizeof(WORD8));

          //UNROLL_LOAD_ROW_MAT1(0);
          mat1_0 = AE_LP16X2F_I((ae_p16x2s *)p_mat1_0, 0);
          HF2_AE_ADDCIRC16X4_XC(p_mat1_0, 2);

          //UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0);
          AE_MULAAP24S_HH_LL(acc_0_0, vec_0, mat1_0);
          AE_MULAAP24S_HH_LL(acc_0_1, vec_1, mat1_0);
        }
        // Adjusting Shift for accumulators
        acc_0_0  = AE_SRAIQ56(acc_0_0, 8);
        acc_0_1  = AE_SRAIQ56(acc_0_1, 8);

        //UNROLL_ROW_ADD_BIAS_ACC(0);
        acc_0_0 = AE_ADDSQ56S(acc_0_0, bias_0);
        acc_0_1 = AE_ADDSQ56S(acc_0_1, bias_1);

        //UNROLL_ROW_ADJUST_ACC(0);
        out_shift_0 = p_out_shift[vec_itr];
        out_shift_1 = p_out_shift[vec_itr + 1];
        left_shift_0 = out_shift_0<0?0:out_shift_0;
        left_shift_1 = out_shift_1<0?0:out_shift_1;
        right_shift_0 = out_shift_0>0?0:-out_shift_0;
        right_shift_1 = out_shift_1>0?0:-out_shift_1;

        acc_0_0 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_0_0), p_out_multiplier[vec_itr], left_shift_0, right_shift_0);
        acc_0_1 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_0_1), p_out_multiplier[vec_itr + 1], left_shift_1, right_shift_1);

        AE_MULAP24S_LL(acc_0_0, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));
        AE_MULAP24S_LL(acc_0_1, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));

        //UNROLL_ROW_STORE_ACC(0);
        acc24_0_0 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_0_0, 24));
        (*((WORD8 *) (&p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_0_0, 8));\
        acc24_0_1 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_0_1, 24));
        (*((WORD8 *) (&p_out[(vec_itr + 1)*out_col_offset + (m_itr)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_0_1, 8));\
      }
    }
    /* Tail loop for vec unroll */
    for(; vec_itr < vec_count; vec_itr++)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        //SETUP_BIAS_BATCH_TAIL;
        ae_q56s bias_0 = AE_CVTQ48A32S(p_bias[vec_itr]);
        //SETUP_ACC_BATCH_TAIL;
        ae_q56s acc_0_0 = ZERO56;
        ae_q56s acc_1_0 = ZERO56;
        ae_p24x2s acc24_0_0 = ZERO24X2;
        ae_p24x2s acc24_1_0 = ZERO24X2;

        //UNROLL_SETUP_VEC_BATCH(0);
        ae_p24x2s vec_0  = ZERO24X2;
        WORD8 *p_vec_0  = (WORD8 *)(&p_vec1[(vec_itr)*vec_stride]);

        //SETUP_MAT1;
        ae_p24x2s mat1_0 = ZERO24X2;
        ae_p24x2s mat1_1 = ZERO24X2;
        WORD16 *p_mat1_0 = (WORD16 *) p_mat1;
        WORD16 *p_mat1_1 = (WORD16 *) p_mat1;

        HF2_AE_ADDCIRC16X4_XC(p_mat1_0, (m_itr)*row_stride1);
        HF2_AE_ADDCIRC16X4_XC(p_mat1_1, (m_itr+1)*row_stride1);

        p_vec_0 = p_vec_0 - 2;

        for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
        {
          //UNROLL_LOAD_VEC_BATCH(0);
          AE_LP8X2F_IU(vec_0, p_vec_0, 2*sizeof(WORD8));
          //LOAD_MAT1;
          mat1_0 = AE_LP16X2F_I((ae_p16x2s *)p_mat1_0, 0);
          mat1_1 = AE_LP16X2F_I((ae_p16x2s *)p_mat1_1, 0);
          HF2_AE_ADDCIRC16X4_XC(p_mat1_0, 2);
          HF2_AE_ADDCIRC16X4_XC(p_mat1_1, 2);

          //KERNEL_MAT1_VEC_BATCH_TAIL;
          AE_MULAAP24S_HH_LL(acc_0_0, vec_0, mat1_0);
          AE_MULAAP24S_HH_LL(acc_1_0, vec_0, mat1_1);
        }
        // Adjusting Shift for accumulators
        acc_0_0  = AE_SRAIQ56(acc_0_0, 8);
        acc_1_0  = AE_SRAIQ56(acc_1_0, 8);

        //ADD_BIAS_ACC_BATCH_TAIL;
        acc_0_0 = AE_ADDSQ56S(acc_0_0, bias_0);
        acc_1_0 = AE_ADDSQ56S(acc_1_0, bias_0);

        //ADJUST_ACC_BATCH_TAIL;
        out_shift_0 = p_out_shift[vec_itr];
        left_shift_0 = out_shift_0<0?0:out_shift_0;
        right_shift_0 = out_shift_0>0?0:-out_shift_0;

        acc_0_0 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_0_0), p_out_multiplier[vec_itr], left_shift_0, right_shift_0);
        acc_1_0 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_1_0), p_out_multiplier[vec_itr], left_shift_0, right_shift_0);

        AE_MULAP24S_LL(acc_0_0, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));
        AE_MULAP24S_LL(acc_1_0, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));

        //STORE_ACC_BATCH_TAIL;
        acc24_0_0 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_0_0, 24));
        (*((WORD8 *) (&p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_0_0, 8));\
        acc24_1_0 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_1_0, 24));
        (*((WORD8 *) (&p_out[(vec_itr)*out_col_offset + (m_itr + 1)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_1_0, 8));\
      }
      for(; m_itr < rows; m_itr++)
      {
        //UNROLL_SETUP_BIAS_BATCH(0,0);
        ae_q56s bias_0 = AE_CVTQ48A32S(p_bias[vec_itr]);

        //UNROLL_SETUP_ACC_BATCH(0,0);
        ae_q56s acc_0_0 = ZERO56;
        ae_p24x2s acc24_0_0 = ZERO24X2;

        //UNROLL_SETUP_VEC_BATCH(0);
        ae_p24x2s vec_0  = ZERO24X2;
        WORD8 *p_vec_0  = (WORD8 *)(&p_vec1[(vec_itr)*vec_stride]);

        //UNROLL_SETUP_MAT1(0);
        ae_p24x2s mat1_0 = ZERO24X2;
        WORD16 *p_mat1_0 = (WORD16 *) p_mat1;
        HF2_AE_ADDCIRC16X4_XC(p_mat1_0, (m_itr)*row_stride1);

        p_vec_0 = p_vec_0 - 2;

        for(c_itr = 0; c_itr < (cols1 >> 1); c_itr++)
        {
            //UNROLL_LOAD_VEC_BATCH(0);
            AE_LP8X2F_IU(vec_0, p_vec_0, 2*sizeof(WORD8));

            //UNROLL_LOAD_ROW_MAT1(0);
            mat1_0 = AE_LP16X2F_I((ae_p16x2s *)p_mat1_0, 0);
            HF2_AE_ADDCIRC16X4_XC(p_mat1_0, 2);

            //UNROLL_KERNEL_MAT1_VEC_BATCH(0,0);
            AE_MULAAP24S_HH_LL(acc_0_0, vec_0, mat1_0);
        }
        // Adjusting Shift for accumulators
        acc_0_0  = AE_SRAIQ56(acc_0_0, 8);

        //UNROLL_ADD_BIAS_ACC_BATCH(0,0);
        acc_0_0 = AE_ADDSQ56S(acc_0_0, bias_0);

        //UNROLL_ADJUST_ACC_BATCH(0,0);
        out_shift_0 = p_out_shift[vec_itr];
        left_shift_0 = out_shift_0<0?0:out_shift_0;
        right_shift_0 = out_shift_0>0?0:-out_shift_0;
        acc_0_0 = MultiplyByQuantizedMultiplier(AE_TRUNCA32Q48(acc_0_0), p_out_multiplier[vec_itr], left_shift_0, right_shift_0);
        AE_MULAP24S_LL(acc_0_0, AE_CVTP24A16(1), AE_CVTP24A16(out_offset));

        //UNROLL_STORE_ACC_BATCH(0,0);
        acc24_0_0 = AE_ROUNDSP24Q48SYM(AE_SLLISQ56S(acc_0_0, 24));
        (*((WORD8 *) (&p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]))) = (WORD8)AE_TRUNCA16P24S_L(AE_SRAIP24(acc24_0_0, 8));\
      }
    }
  }
  else
  {
    return -1;
  }
  return 0;
}
