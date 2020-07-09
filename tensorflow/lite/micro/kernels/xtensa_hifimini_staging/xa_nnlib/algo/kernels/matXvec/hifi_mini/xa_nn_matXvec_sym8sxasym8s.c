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

#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros.h"

#define ADD_OUT_OFFSET_STORE_INT8(ptr, data, out_offset) \
  {                                                      \
    data = AE_ADDSQ56S(data, AE_CVTQ48A32S(out_offset)); \
    int out_i32 = AE_TRUNCA32Q48(AE_SATQ48S(data));      \
    out_i32 = out_i32 < -128 ? -128 : out_i32;           \
    out_i32 = out_i32 > 127 ? 127 : out_i32;             \
    *(ptr) = (WORD8)out_i32;                             \
  }

WORD32 xa_nn_matXvec_out_stride_sym8sxasym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_mat1,
    const WORD8 *__restrict__ p_vec1, const WORD32 *__restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 row_stride1, WORD32 out_stride,
    WORD32 vec1_zero_bias, WORD32 out_multiplier, WORD32 out_shift,
    WORD32 out_zero_bias) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  /* Iterators used in for loops */
  int m_itr, c_itr, i;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;

  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;

  const WORD8 *p_mat1_0, *p_mat1_1, *p_mat1_2, *p_mat1_3;
  const WORD8 *p_vec1_0;
  ae_p24x2s dp_mat1_0, dp_mat1_1, dp_mat1_2, dp_mat1_3, dp_vec1_0;
  ae_p24x2s dp_vec1_zb;
  ae_q56s dq_acc[4];
  ae_q56s dq_out32, dq_out;

  dp_vec1_zb = AE_MOVPA24(vec1_zero_bias);
  if (((((unsigned)p_mat1) & 1) == 0) && ((((unsigned)p_vec1) & 1) == 0) &&
      ((row_stride1 & 1) == 0)) {
    for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
      p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1 - 2];
      p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1 - 2];
      p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1 - 2];
      p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1 - 2];
      p_vec1_0 = p_vec1 - 2;

      dq_acc[0] = dq_acc[1] = dq_acc[2] = dq_acc[3] = AE_ZEROQ56();

      /* AE_LP8X2F* instruction loads in upper 8 bits of P register, so shifting
      vector right by 16 to get multiplication result in middle 32 bits of Q
      register (lower 16 bits 0) */
      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_mat1_1, p_mat1_1, 2);
        AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
        AE_LP8X2F_IU(dp_mat1_3, p_mat1_3, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc[1], dp_mat1_1, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc[2], dp_mat1_2, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc[3], dp_mat1_3, dp_vec1_0);
      }
      /* Pointers are aligned so can do 8X2 loads and ignore L parts of
       * registers */
      if (cols1 & 1) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_mat1_1, p_mat1_1, 2);
        AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
        AE_LP8X2F_IU(dp_mat1_3, p_mat1_3, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAP24S_HH(dq_acc[0], dp_mat1_0, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc[1], dp_mat1_1, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc[2], dp_mat1_2, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc[3], dp_mat1_3, dp_vec1_0);
      }

      if (p_bias != NULL) {
        for (i = 0; i < 4; i++)
          dq_acc[i] = AE_ADDSQ56S(dq_acc[i], *(ae_q32s *)(&p_bias[m_itr + i]));
      }

      for (i = 0; i < 4; i++) {
        dq_out32 = AE_SATQ48S(dq_acc[i]);
        MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                         out_multiplier, left_shift,
                                         right_shift);
        ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                  out_zero_bias);
      }
    }
    for (; m_itr < rows; m_itr++) {
      p_mat1_0 = &p_mat1[m_itr * row_stride1 - 2];
      p_vec1_0 = p_vec1 - 2;

      dq_acc[0] = AE_ZEROQ56();

      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }
      /* Pointers are aligned so can do 8X2 loads and ignore L parts of
       * registers */
      if (cols1 & 1) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAP24S_HH(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }

      if (p_bias != NULL)
        dq_acc[0] = AE_ADDSQ56S(dq_acc[0], *(ae_q32s *)(&p_bias[m_itr]));

      dq_out32 = AE_SATQ48S(dq_acc[0]);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      ADD_OUT_OFFSET_STORE_INT8(&p_out[m_itr * out_stride], dq_out,
                                out_zero_bias);
    }
  } else {
    if ((((unsigned)p_mat1) & 1) == 0) {
      for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
        p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1 - 2];
        p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1];
        p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1 - 2];
        p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1];
        p_vec1_0 = p_vec1;

        dq_acc[0] = dq_acc[1] = dq_acc[2] = dq_acc[3] = AE_ZEROQ56();

        /* Matrix elements are kept in upper 8 bits of P registers, vector
        elements are kept in lower 8 bits of P registers, typecasting to UWORD8
        is to avoid extra extui instructions since signed 8-bit load in not
        there in HiFiMini */
        for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
          AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
          dp_mat1_1 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_1[c_itr],
                                        (UWORD8)p_mat1_1[c_itr + 1]);
          AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
          dp_mat1_3 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_3[c_itr],
                                        (UWORD8)p_mat1_3[c_itr + 1]);
          dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                        (UWORD8)p_vec1_0[c_itr + 1]);
          dp_mat1_1 = AE_SLLIP24(dp_mat1_1, 8);
          dp_mat1_3 = AE_SLLIP24(dp_mat1_3, 8);
          dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
          dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
          AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[1], dp_mat1_1, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[2], dp_mat1_2, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[3], dp_mat1_3, dp_vec1_0);
        }
        if (cols1 & 1) {
          ae_p24x2s dp_mat1_01, dp_mat1_23;
          dp_mat1_01 =
              AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[2], (UWORD8)p_mat1_1[c_itr]);
          dp_mat1_23 =
              AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[2], (UWORD8)p_mat1_3[c_itr]);
          dp_vec1_0 = AE_MOVPA24(p_vec1_0[c_itr]);
          dp_mat1_01 = AE_SLLIP24(dp_mat1_01, 8);
          dp_mat1_23 = AE_SLLIP24(dp_mat1_23, 8);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
          AE_MULAP24S_HH(dq_acc[0], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[1], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_HH(dq_acc[2], dp_mat1_23, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[3], dp_mat1_23, dp_vec1_0);
        }

        if (p_bias != NULL) {
          for (i = 0; i < 4; i++)
            dq_acc[i] =
                AE_ADDSQ56S(dq_acc[i], *(ae_q32s *)(&p_bias[m_itr + i]));
        }

        for (i = 0; i < 4; i++) {
          dq_out32 = AE_SATQ48S(dq_acc[i]);
          MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                           out_multiplier, left_shift,
                                           right_shift);
          ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                    out_zero_bias);
        }
      }
    } else {
      for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
        p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1];
        p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1];
        p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1];
        p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1];
        p_vec1_0 = p_vec1;

        dq_acc[0] = dq_acc[1] = dq_acc[2] = dq_acc[3] = AE_ZEROQ56();

        /* Matrix elements are kept in upper 8 bits of P registers, vector
        elements are kept in lower 8 bits of P registers, typecasting to UWORD8
        is to avoid extra extui instructions since signed 8-bit load in not
        there in HiFiMini */
        for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
          dp_mat1_0 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                        (UWORD8)p_mat1_0[c_itr + 1]);
          dp_mat1_1 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_1[c_itr],
                                        (UWORD8)p_mat1_1[c_itr + 1]);
          dp_mat1_2 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[c_itr],
                                        (UWORD8)p_mat1_2[c_itr + 1]);
          dp_mat1_3 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_3[c_itr],
                                        (UWORD8)p_mat1_3[c_itr + 1]);
          dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                        (UWORD8)p_vec1_0[c_itr + 1]);
          dp_mat1_0 = AE_SLLIP24(dp_mat1_0, 8);
          dp_mat1_1 = AE_SLLIP24(dp_mat1_1, 8);
          dp_mat1_2 = AE_SLLIP24(dp_mat1_2, 8);
          dp_mat1_3 = AE_SLLIP24(dp_mat1_3, 8);
          dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
          dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
          AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[1], dp_mat1_1, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[2], dp_mat1_2, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[3], dp_mat1_3, dp_vec1_0);
        }
        if (cols1 & 1) {
          ae_p24x2s dp_mat1_01, dp_mat1_23;
          dp_mat1_01 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                         (UWORD8)p_mat1_1[c_itr]);
          dp_mat1_23 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[c_itr],
                                         (UWORD8)p_mat1_3[c_itr]);
          dp_vec1_0 = AE_MOVPA24(p_vec1_0[c_itr]);
          dp_mat1_01 = AE_SLLIP24(dp_mat1_01, 8);
          dp_mat1_23 = AE_SLLIP24(dp_mat1_23, 8);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
          AE_MULAP24S_HH(dq_acc[0], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[1], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_HH(dq_acc[2], dp_mat1_23, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[3], dp_mat1_23, dp_vec1_0);
        }

        if (p_bias != NULL) {
          for (i = 0; i < 4; i++)
            dq_acc[i] =
                AE_ADDSQ56S(dq_acc[i], *(ae_q32s *)(&p_bias[m_itr + i]));
        }

        for (i = 0; i < 4; i++) {
          dq_out32 = AE_SATQ48S(dq_acc[i]);
          MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                           out_multiplier, left_shift,
                                           right_shift);
          ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                    out_zero_bias);
        }
      }
    }
    for (; m_itr < rows; m_itr++) {
      p_mat1_0 = &p_mat1[m_itr * row_stride1];
      p_vec1_0 = p_vec1;

      dq_acc[0] = AE_ZEROQ56();

      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        dp_mat1_0 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                      (UWORD8)p_mat1_0[c_itr + 1]);
        dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                      (UWORD8)p_vec1_0[c_itr + 1]);
        dp_mat1_0 = AE_SLLIP24(dp_mat1_0, 8);
        dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }
      if (cols1 & 1) {
        dp_mat1_0 = AE_CVTP24A16(p_mat1_0[c_itr]);
        dp_vec1_0 = AE_CVTP24A16(p_vec1_0[c_itr]);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, AE_CVTP24A16(vec1_zero_bias));
        AE_MULAP24S_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }

      if (p_bias != NULL)
        dq_acc[0] = AE_ADDSQ56S(dq_acc[0], *(ae_q32s *)(&p_bias[m_itr]));

      dq_out32 = AE_SATQ48S(dq_acc[0]);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      ADD_OUT_OFFSET_STORE_INT8(&p_out[m_itr * out_stride], dq_out,
                                out_zero_bias);
    }
  }

  return 0;
}

WORD32 xa_nn_matXvec_out_stride_asym8sxasym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_mat1,
    const WORD8 *__restrict__ p_vec1, const WORD32 *__restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 row_stride1, WORD32 out_stride,
    WORD32 mat1_zero_bias, WORD32 vec1_zero_bias, WORD32 out_multiplier,
    WORD32 out_shift, WORD32 out_zero_bias) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -127 || mat1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  /* Iterators used in for loops */
  int m_itr, c_itr, i;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;

  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;

  const WORD8 *p_mat1_0, *p_mat1_1, *p_mat1_2, *p_mat1_3;
  const WORD8 *p_vec1_0;
  ae_p24x2s dp_mat1_0, dp_mat1_1, dp_mat1_2, dp_mat1_3, dp_vec1_0;
  ae_p24x2s dp_vec1_zb, dp_mat1_zb;
  ae_q56s dq_acc_0, dq_acc_1, dq_acc_2, dq_acc_3;
  ae_q56s dq_out32, dq_out;

  const WORD32 bias_buffer[1] = {0};
  const WORD32 *p_bias_load;
  WORD32 bias_address_increment = sizeof(WORD32);

  dp_mat1_zb = AE_MOVPA24(mat1_zero_bias);
  dp_vec1_zb = AE_MOVPA24(vec1_zero_bias);

  /* Check for alignment conditions */
  if (((((unsigned)p_mat1) & 1) == 0) && ((((unsigned)p_vec1) & 1) == 0) &&
      ((row_stride1 & 1) == 0) && ((cols1 & 1) == 0)) {
    /* Calculate partial zero offset adjustment outside the loop */
    WORD32 zero_offset_adjustment;

    // Constant part of total zero bias
    ae_q56s dq_zero_bias_sum =
        AE_CVTQ48A32S(vec1_zero_bias * cols1 * mat1_zero_bias);

    WORD8 *p_inp = (WORD8 *)p_vec1 - 2;
    for (i = 0; i < (cols1 >> 1); i++) {
      /* Input vector is in MSB 8 bits, matrix zero bias in LSB 8 bits */
      AE_LP8X2F_IU(dp_vec1_0, p_inp, 2);
      AE_MULAAP24S_HH_LL(dq_zero_bias_sum, dp_vec1_0, dp_mat1_zb);
    }
    /* Product is already aligned to bits 16 to 47 in QR register. */
    zero_offset_adjustment = AE_TRUNCA32Q48(dq_zero_bias_sum);

    /* If bias is not provided, use a dummy zero value from bias_buffer. */
    if (p_bias == NULL) {
      p_bias_load = bias_buffer - 1;
      bias_address_increment = 0;
    } else {
      p_bias_load = p_bias - 1;
    }

    for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
      p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1 - 2];
      p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1 - 2];
      p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1 - 2];
      p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1 - 2];
      p_vec1_0 = p_vec1 - 2;

      AE_LQ32F_XU(dq_acc_0, (ae_q32s *)p_bias_load, bias_address_increment);
      AE_LQ32F_XU(dq_acc_1, (ae_q32s *)p_bias_load, bias_address_increment);
      AE_LQ32F_XU(dq_acc_2, (ae_q32s *)p_bias_load, bias_address_increment);
      AE_LQ32F_XU(dq_acc_3, (ae_q32s *)p_bias_load, bias_address_increment);

      dq_acc_0 = AE_ADDQ56(dq_acc_0, AE_CVTQ48A32S(zero_offset_adjustment));
      dq_acc_1 = AE_ADDQ56(dq_acc_1, AE_CVTQ48A32S(zero_offset_adjustment));
      dq_acc_2 = AE_ADDQ56(dq_acc_2, AE_CVTQ48A32S(zero_offset_adjustment));
      dq_acc_3 = AE_ADDQ56(dq_acc_3, AE_CVTQ48A32S(zero_offset_adjustment));

      /* AE_LP8X2F* instruction loads in upper 8 bits of P register, so shifting
      vector right by 16 to get multiplication result in middle 32 bits of Q
      register (lower 16 bits 0) */
      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_mat1_1, p_mat1_1, 2);
        AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
        AE_LP8X2F_IU(dp_mat1_3, p_mat1_3, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);

        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);

        AE_MULAAP24S_HH_LL(dq_acc_0, dp_mat1_0, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc_1, dp_mat1_1, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc_2, dp_mat1_2, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc_3, dp_mat1_3, dp_vec1_0);
      }

      /* Pointers are aligned so can do 8X2 loads and ignore L parts of
       * registers */
      if (cols1 & 1) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_mat1_1, p_mat1_1, 2);
        AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
        AE_LP8X2F_IU(dp_mat1_3, p_mat1_3, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);

        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);

        AE_MULAP24S_HH(dq_acc_0, dp_mat1_0, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc_1, dp_mat1_1, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc_2, dp_mat1_2, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc_3, dp_mat1_3, dp_vec1_0);
      }

      dq_out32 = AE_SATQ48S(dq_acc_0);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                out_zero_bias);

      dq_out32 = AE_SATQ48S(dq_acc_1);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                out_zero_bias);

      dq_out32 = AE_SATQ48S(dq_acc_2);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                out_zero_bias);

      dq_out32 = AE_SATQ48S(dq_acc_3);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                out_zero_bias);
    }
    for (; m_itr < rows; m_itr++) {
      p_mat1_0 = &p_mat1[m_itr * row_stride1 - 2];
      p_vec1_0 = p_vec1 - 2;

      AE_LQ32F_XU(dq_acc_0, (ae_q32s *)p_bias_load, bias_address_increment);
      dq_acc_0 = AE_ADDQ56(dq_acc_0, AE_CVTQ48A32S(zero_offset_adjustment));

      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);

        AE_MULAAP24S_HH_LL(dq_acc_0, dp_mat1_0, dp_vec1_0);
      }

      dq_out32 = AE_SATQ48S(dq_acc_0);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      ADD_OUT_OFFSET_STORE_INT8(&p_out[m_itr * out_stride], dq_out,
                                out_zero_bias);
    }
  } else {
#ifndef DISABLE_NNLIB_UNALIGNED_SUPPORT
    ae_q56s dq_acc[4];

    if ((((unsigned)p_mat1) & 1) == 0) {
      for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
        p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1 - 2];
        p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1];
        p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1 - 2];
        p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1];
        p_vec1_0 = p_vec1;

        dq_acc[0] = dq_acc[1] = dq_acc[2] = dq_acc[3] = AE_ZEROQ56();

        /* Matrix elements are kept in upper 8 bits of P registers, vector
        elements are kept in lower 8 bits of P registers, typecasting to UWORD8
        is to avoid extra extui instructions since signed 8-bit load in not
        there in HiFiMini */
        for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
          AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
          dp_mat1_1 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_1[c_itr],
                                        (UWORD8)p_mat1_1[c_itr + 1]);
          AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
          dp_mat1_3 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_3[c_itr],
                                        (UWORD8)p_mat1_3[c_itr + 1]);
          dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                        (UWORD8)p_vec1_0[c_itr + 1]);
          dp_mat1_1 = AE_SLLIP24(dp_mat1_1, 8);
          dp_mat1_3 = AE_SLLIP24(dp_mat1_3, 8);
          dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
          dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);

          dp_mat1_0 = AE_SRAIP24(dp_mat1_0, 16);
          dp_mat1_0 = AE_ADDSP24S(dp_mat1_0, dp_mat1_zb);
          dp_mat1_1 = AE_SRAIP24(dp_mat1_1, 16);
          dp_mat1_1 = AE_ADDSP24S(dp_mat1_1, dp_mat1_zb);
          dp_mat1_2 = AE_SRAIP24(dp_mat1_2, 16);
          dp_mat1_2 = AE_ADDSP24S(dp_mat1_2, dp_mat1_zb);
          dp_mat1_3 = AE_SRAIP24(dp_mat1_3, 16);
          dp_mat1_3 = AE_ADDSP24S(dp_mat1_3, dp_mat1_zb);

          AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[1], dp_mat1_1, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[2], dp_mat1_2, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[3], dp_mat1_3, dp_vec1_0);
        }
        if (cols1 & 1) {
          ae_p24x2s dp_mat1_01, dp_mat1_23;
          dp_mat1_01 =
              AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[2], (UWORD8)p_mat1_1[c_itr]);
          dp_mat1_23 =
              AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[2], (UWORD8)p_mat1_3[c_itr]);
          dp_vec1_0 = AE_MOVPA24(p_vec1_0[c_itr]);
          dp_mat1_01 = AE_SLLIP24(dp_mat1_01, 8);
          dp_mat1_23 = AE_SLLIP24(dp_mat1_23, 8);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);

          dp_mat1_01 = AE_SRAIP24(dp_mat1_01, 16);
          dp_mat1_01 = AE_ADDSP24S(dp_mat1_01, dp_mat1_zb);
          dp_mat1_23 = AE_SRAIP24(dp_mat1_23, 16);
          dp_mat1_23 = AE_ADDSP24S(dp_mat1_23, dp_mat1_zb);

          AE_MULAP24S_HH(dq_acc[0], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[1], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_HH(dq_acc[2], dp_mat1_23, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[3], dp_mat1_23, dp_vec1_0);
        }

        dq_acc[0] = AE_SLLISQ56S(dq_acc[0], 16);
        dq_acc[1] = AE_SLLISQ56S(dq_acc[1], 16);
        dq_acc[2] = AE_SLLISQ56S(dq_acc[2], 16);
        dq_acc[3] = AE_SLLISQ56S(dq_acc[3], 16);

        if (p_bias != NULL) {
          for (i = 0; i < 4; i++)
            dq_acc[i] =
                AE_ADDSQ56S(dq_acc[i], *(ae_q32s *)(&p_bias[m_itr + i]));
        }

        for (i = 0; i < 4; i++) {
          dq_out32 = AE_SATQ48S(dq_acc[i]);
          MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                           out_multiplier, left_shift,
                                           right_shift);
          ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                    out_zero_bias);
        }
      }
    } else {
      for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
        p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1];
        p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1];
        p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1];
        p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1];
        p_vec1_0 = p_vec1;

        dq_acc[0] = dq_acc[1] = dq_acc[2] = dq_acc[3] = AE_ZEROQ56();

        /* Matrix elements are kept in upper 8 bits of P registers, vector
        elements are kept in lower 8 bits of P registers, typecasting to UWORD8
        is to avoid extra extui instructions since signed 8-bit load in not
        there in HiFiMini */
        for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
          dp_mat1_0 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                        (UWORD8)p_mat1_0[c_itr + 1]);
          dp_mat1_1 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_1[c_itr],
                                        (UWORD8)p_mat1_1[c_itr + 1]);
          dp_mat1_2 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[c_itr],
                                        (UWORD8)p_mat1_2[c_itr + 1]);
          dp_mat1_3 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_3[c_itr],
                                        (UWORD8)p_mat1_3[c_itr + 1]);
          dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                        (UWORD8)p_vec1_0[c_itr + 1]);
          dp_mat1_0 = AE_SLLIP24(dp_mat1_0, 8);
          dp_mat1_1 = AE_SLLIP24(dp_mat1_1, 8);
          dp_mat1_2 = AE_SLLIP24(dp_mat1_2, 8);
          dp_mat1_3 = AE_SLLIP24(dp_mat1_3, 8);
          dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
          dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);

          dp_mat1_0 = AE_SRAIP24(dp_mat1_0, 16);
          dp_mat1_0 = AE_ADDSP24S(dp_mat1_0, dp_mat1_zb);
          dp_mat1_1 = AE_SRAIP24(dp_mat1_1, 16);
          dp_mat1_1 = AE_ADDSP24S(dp_mat1_1, dp_mat1_zb);
          dp_mat1_2 = AE_SRAIP24(dp_mat1_2, 16);
          dp_mat1_2 = AE_ADDSP24S(dp_mat1_2, dp_mat1_zb);
          dp_mat1_3 = AE_SRAIP24(dp_mat1_3, 16);
          dp_mat1_3 = AE_ADDSP24S(dp_mat1_3, dp_mat1_zb);

          AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[1], dp_mat1_1, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[2], dp_mat1_2, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[3], dp_mat1_3, dp_vec1_0);
        }
        if (cols1 & 1) {
          ae_p24x2s dp_mat1_01, dp_mat1_23;
          dp_mat1_01 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                         (UWORD8)p_mat1_1[c_itr]);
          dp_mat1_23 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[c_itr],
                                         (UWORD8)p_mat1_3[c_itr]);
          dp_vec1_0 = AE_MOVPA24(p_vec1_0[c_itr]);
          dp_mat1_01 = AE_SLLIP24(dp_mat1_01, 8);
          dp_mat1_23 = AE_SLLIP24(dp_mat1_23, 8);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);

          dp_mat1_01 = AE_SRAIP24(dp_mat1_01, 16);
          dp_mat1_01 = AE_ADDSP24S(dp_mat1_01, dp_mat1_zb);
          dp_mat1_23 = AE_SRAIP24(dp_mat1_23, 16);
          dp_mat1_23 = AE_ADDSP24S(dp_mat1_23, dp_mat1_zb);

          AE_MULAP24S_HH(dq_acc[0], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[1], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_HH(dq_acc[2], dp_mat1_23, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[3], dp_mat1_23, dp_vec1_0);
        }

        dq_acc[0] = AE_SLLISQ56S(dq_acc[0], 16);
        dq_acc[1] = AE_SLLISQ56S(dq_acc[1], 16);
        dq_acc[2] = AE_SLLISQ56S(dq_acc[2], 16);
        dq_acc[3] = AE_SLLISQ56S(dq_acc[3], 16);

        if (p_bias != NULL) {
          for (i = 0; i < 4; i++)
            dq_acc[i] =
                AE_ADDSQ56S(dq_acc[i], *(ae_q32s *)(&p_bias[m_itr + i]));
        }

        for (i = 0; i < 4; i++) {
          dq_out32 = AE_SATQ48S(dq_acc[i]);
          MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                           out_multiplier, left_shift,
                                           right_shift);
          ADD_OUT_OFFSET_STORE_INT8(&p_out[(m_itr + i) * out_stride], dq_out,
                                    out_zero_bias);
        }
      }
    }
    for (; m_itr < rows; m_itr++) {
      p_mat1_0 = &p_mat1[m_itr * row_stride1];
      p_vec1_0 = p_vec1;

      dq_acc[0] = AE_ZEROQ56();

      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        dp_mat1_0 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                      (UWORD8)p_mat1_0[c_itr + 1]);
        dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                      (UWORD8)p_vec1_0[c_itr + 1]);
        dp_mat1_0 = AE_SLLIP24(dp_mat1_0, 8);
        dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);

        dp_mat1_0 = AE_SRAIP24(dp_mat1_0, 16);
        dp_mat1_0 = AE_ADDSP24S(dp_mat1_0, dp_mat1_zb);

        AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }
      if (cols1 & 1) {
        dp_mat1_0 = AE_CVTP24A16(p_mat1_0[c_itr]);
        dp_vec1_0 = AE_CVTP24A16(p_vec1_0[c_itr]);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, AE_CVTP24A16(vec1_zero_bias));

        dp_mat1_0 = AE_SRAIP24(dp_mat1_0, 16);
        dp_mat1_0 = AE_ADDSP24S(dp_mat1_0, dp_mat1_zb);

        AE_MULAP24S_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }

      dq_acc[0] = AE_SLLISQ56S(dq_acc[0], 16);

      if (p_bias != NULL)
        dq_acc[0] = AE_ADDSQ56S(dq_acc[0], *(ae_q32s *)(&p_bias[m_itr]));

      dq_out32 = AE_SATQ48S(dq_acc[0]);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      ADD_OUT_OFFSET_STORE_INT8(&p_out[m_itr * out_stride], dq_out,
                                out_zero_bias);
    }
#else
    return 1;
#endif
  }

  return 0;
}

#define STORE_INT16(ptr, data)                                         \
  {                                                                    \
    int out_i32 = AE_TRUNCA32Q48(AE_SATQ48S(data));                    \
    out_i32 = out_i32 < (int)0xffff8000L ? (int)0xffff8000L : out_i32; \
    out_i32 = out_i32 > (int)0x7fff ? (int)0x7fff : out_i32;           \
    *(ptr) = (WORD16)out_i32;                                          \
  }

WORD32 xa_nn_matXvec_out_stride_sym8sxasym8s_16(
    WORD16 *__restrict__ p_out, const WORD8 *__restrict__ p_mat1,
    const WORD8 *__restrict__ p_vec1, const WORD32 *__restrict__ p_bias,
    WORD32 rows, WORD32 cols1, WORD32 row_stride1, WORD32 out_stride,
    WORD32 vec1_zero_bias, WORD32 out_multiplier, WORD32 out_shift) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);

  /* Iterators used in for loops */
  int m_itr, c_itr, i;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  /* Shifts to match with Tensorflow */
  int left_shift, right_shift;

  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;

  const WORD8 *p_mat1_0, *p_mat1_1, *p_mat1_2, *p_mat1_3;
  const WORD8 *p_vec1_0;
  ae_p24x2s dp_mat1_0, dp_mat1_1, dp_mat1_2, dp_mat1_3, dp_vec1_0;
  ae_p24x2s dp_vec1_zb;
  ae_q56s dq_acc[4];
  ae_q56s dq_out32, dq_out;

  dp_vec1_zb = AE_MOVPA24(vec1_zero_bias);
  if (((((unsigned)p_mat1) & 1) == 0) && ((((unsigned)p_vec1) & 1) == 0) &&
      ((row_stride1 & 1) == 0)) {
    for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
      p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1 - 2];
      p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1 - 2];
      p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1 - 2];
      p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1 - 2];
      p_vec1_0 = p_vec1 - 2;

      dq_acc[0] = dq_acc[1] = dq_acc[2] = dq_acc[3] = AE_ZEROQ56();

      /* AE_LP8X2F* instruction loads in upper 8 bits of P register, so shifting
      vector right by 16 to get multiplication result in middle 32 bits of Q
      register (lower 16 bits 0) */
      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_mat1_1, p_mat1_1, 2);
        AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
        AE_LP8X2F_IU(dp_mat1_3, p_mat1_3, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc[1], dp_mat1_1, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc[2], dp_mat1_2, dp_vec1_0);
        AE_MULAAP24S_HH_LL(dq_acc[3], dp_mat1_3, dp_vec1_0);
      }
      /* Pointers are aligned so can do 8X2 loads and ignore L parts of
       * registers */
      if (cols1 & 1) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_mat1_1, p_mat1_1, 2);
        AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
        AE_LP8X2F_IU(dp_mat1_3, p_mat1_3, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAP24S_HH(dq_acc[0], dp_mat1_0, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc[1], dp_mat1_1, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc[2], dp_mat1_2, dp_vec1_0);
        AE_MULAP24S_HH(dq_acc[3], dp_mat1_3, dp_vec1_0);
      }

      if (p_bias != NULL) {
        for (i = 0; i < 4; i++)
          dq_acc[i] = AE_ADDSQ56S(dq_acc[i], *(ae_q32s *)(&p_bias[m_itr + i]));
      }

      for (i = 0; i < 4; i++) {
        dq_out32 = AE_SATQ48S(dq_acc[i]);
        MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                         out_multiplier, left_shift,
                                         right_shift);
        STORE_INT16(&p_out[(m_itr + i) * out_stride], dq_out);
      }
    }
    for (; m_itr < rows; m_itr++) {
      p_mat1_0 = &p_mat1[m_itr * row_stride1 - 2];
      p_vec1_0 = p_vec1 - 2;

      dq_acc[0] = AE_ZEROQ56();

      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }
      /* Pointers are aligned so can do 8X2 loads and ignore L parts of
       * registers */
      if (cols1 & 1) {
        AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
        AE_LP8X2F_IU(dp_vec1_0, p_vec1_0, 2);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAP24S_HH(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }

      if (p_bias != NULL)
        dq_acc[0] = AE_ADDSQ56S(dq_acc[0], *(ae_q32s *)(&p_bias[m_itr]));

      dq_out32 = AE_SATQ48S(dq_acc[0]);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      STORE_INT16(&p_out[m_itr * out_stride], dq_out);
    }
  } else {
#ifndef DISABLE_NNLIB_UNALIGNED_SUPPORT
    if ((((unsigned)p_mat1) & 1) == 0) {
      for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
        p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1 - 2];
        p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1];
        p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1 - 2];
        p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1];
        p_vec1_0 = p_vec1;

        dq_acc[0] = dq_acc[1] = dq_acc[2] = dq_acc[3] = AE_ZEROQ56();

        /* Matrix elements are kept in upper 8 bits of P registers, vector
        elements are kept in lower 8 bits of P registers, typecasting to UWORD8
        is to avoid extra extui instructions since signed 8-bit load in not
        there in HiFiMini */
        for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
          AE_LP8X2F_IU(dp_mat1_0, p_mat1_0, 2);
          dp_mat1_1 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_1[c_itr],
                                        (UWORD8)p_mat1_1[c_itr + 1]);
          AE_LP8X2F_IU(dp_mat1_2, p_mat1_2, 2);
          dp_mat1_3 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_3[c_itr],
                                        (UWORD8)p_mat1_3[c_itr + 1]);
          dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                        (UWORD8)p_vec1_0[c_itr + 1]);
          dp_mat1_1 = AE_SLLIP24(dp_mat1_1, 8);
          dp_mat1_3 = AE_SLLIP24(dp_mat1_3, 8);
          dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
          dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
          AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[1], dp_mat1_1, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[2], dp_mat1_2, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[3], dp_mat1_3, dp_vec1_0);
        }
        if (cols1 & 1) {
          ae_p24x2s dp_mat1_01, dp_mat1_23;
          dp_mat1_01 =
              AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[2], (UWORD8)p_mat1_1[c_itr]);
          dp_mat1_23 =
              AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[2], (UWORD8)p_mat1_3[c_itr]);
          dp_vec1_0 = AE_MOVPA24(p_vec1_0[c_itr]);
          dp_mat1_01 = AE_SLLIP24(dp_mat1_01, 8);
          dp_mat1_23 = AE_SLLIP24(dp_mat1_23, 8);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
          AE_MULAP24S_HH(dq_acc[0], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[1], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_HH(dq_acc[2], dp_mat1_23, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[3], dp_mat1_23, dp_vec1_0);
        }

        if (p_bias != NULL) {
          for (i = 0; i < 4; i++)
            dq_acc[i] =
                AE_ADDSQ56S(dq_acc[i], *(ae_q32s *)(&p_bias[m_itr + i]));
        }

        for (i = 0; i < 4; i++) {
          dq_out32 = AE_SATQ48S(dq_acc[i]);
          MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                           out_multiplier, left_shift,
                                           right_shift);
          STORE_INT16(&p_out[(m_itr + i) * out_stride], dq_out);
        }
      }
    } else {
      for (m_itr = 0; m_itr < (rows - 3); m_itr += 4) {
        p_mat1_0 = &p_mat1[(m_itr + 0) * row_stride1];
        p_mat1_1 = &p_mat1[(m_itr + 1) * row_stride1];
        p_mat1_2 = &p_mat1[(m_itr + 2) * row_stride1];
        p_mat1_3 = &p_mat1[(m_itr + 3) * row_stride1];
        p_vec1_0 = p_vec1;

        dq_acc[0] = dq_acc[1] = dq_acc[2] = dq_acc[3] = AE_ZEROQ56();

        /* Matrix elements are kept in upper 8 bits of P registers, vector
        elements are kept in lower 8 bits of P registers, typecasting to UWORD8
        is to avoid extra extui instructions since signed 8-bit load in not
        there in HiFiMini */
        for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
          dp_mat1_0 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                        (UWORD8)p_mat1_0[c_itr + 1]);
          dp_mat1_1 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_1[c_itr],
                                        (UWORD8)p_mat1_1[c_itr + 1]);
          dp_mat1_2 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[c_itr],
                                        (UWORD8)p_mat1_2[c_itr + 1]);
          dp_mat1_3 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_3[c_itr],
                                        (UWORD8)p_mat1_3[c_itr + 1]);
          dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                        (UWORD8)p_vec1_0[c_itr + 1]);
          dp_mat1_0 = AE_SLLIP24(dp_mat1_0, 8);
          dp_mat1_1 = AE_SLLIP24(dp_mat1_1, 8);
          dp_mat1_2 = AE_SLLIP24(dp_mat1_2, 8);
          dp_mat1_3 = AE_SLLIP24(dp_mat1_3, 8);
          dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
          dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
          AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[1], dp_mat1_1, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[2], dp_mat1_2, dp_vec1_0);
          AE_MULAAP24S_HH_LL(dq_acc[3], dp_mat1_3, dp_vec1_0);
        }
        if (cols1 & 1) {
          ae_p24x2s dp_mat1_01, dp_mat1_23;
          dp_mat1_01 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                         (UWORD8)p_mat1_1[c_itr]);
          dp_mat1_23 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_2[c_itr],
                                         (UWORD8)p_mat1_3[c_itr]);
          dp_vec1_0 = AE_MOVPA24(p_vec1_0[c_itr]);
          dp_mat1_01 = AE_SLLIP24(dp_mat1_01, 8);
          dp_mat1_23 = AE_SLLIP24(dp_mat1_23, 8);
          dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
          AE_MULAP24S_HH(dq_acc[0], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[1], dp_mat1_01, dp_vec1_0);
          AE_MULAP24S_HH(dq_acc[2], dp_mat1_23, dp_vec1_0);
          AE_MULAP24S_LL(dq_acc[3], dp_mat1_23, dp_vec1_0);
        }

        if (p_bias != NULL) {
          for (i = 0; i < 4; i++)
            dq_acc[i] =
                AE_ADDSQ56S(dq_acc[i], *(ae_q32s *)(&p_bias[m_itr + i]));
        }

        for (i = 0; i < 4; i++) {
          dq_out32 = AE_SATQ48S(dq_acc[i]);
          MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                           out_multiplier, left_shift,
                                           right_shift);
          STORE_INT16(&p_out[(m_itr + i) * out_stride], dq_out);
        }
      }
    }
    for (; m_itr < rows; m_itr++) {
      p_mat1_0 = &p_mat1[m_itr * row_stride1];
      p_vec1_0 = p_vec1;

      dq_acc[0] = AE_ZEROQ56();

      for (c_itr = 0; c_itr < (cols1 - 1); c_itr += 2) {
        dp_mat1_0 = AE_CVTP24A16X2_LL((UWORD8)p_mat1_0[c_itr],
                                      (UWORD8)p_mat1_0[c_itr + 1]);
        dp_vec1_0 = AE_CVTP24A16X2_LL((UWORD8)p_vec1_0[c_itr],
                                      (UWORD8)p_vec1_0[c_itr + 1]);
        dp_mat1_0 = AE_SLLIP24(dp_mat1_0, 8);
        dp_vec1_0 = AE_SLLIP24(dp_vec1_0, 8);
        dp_vec1_0 = AE_SRAIP24(dp_vec1_0, 16);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, dp_vec1_zb);
        AE_MULAAP24S_HH_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }
      if (cols1 & 1) {
        dp_mat1_0 = AE_CVTP24A16(p_mat1_0[c_itr]);
        dp_vec1_0 = AE_CVTP24A16(p_vec1_0[c_itr]);
        dp_vec1_0 = AE_ADDSP24S(dp_vec1_0, AE_CVTP24A16(vec1_zero_bias));
        AE_MULAP24S_LL(dq_acc[0], dp_mat1_0, dp_vec1_0);
      }

      if (p_bias != NULL)
        dq_acc[0] = AE_ADDSQ56S(dq_acc[0], *(ae_q32s *)(&p_bias[m_itr]));

      dq_out32 = AE_SATQ48S(dq_acc[0]);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      STORE_INT16(&p_out[m_itr * out_stride], dq_out);
    }
#else
    return 1;
#endif
  }

  return 0;
}
