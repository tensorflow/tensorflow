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

#ifndef __XA_NNLIB_COMMON_MACROS_H__
#define __XA_NNLIB_COMMON_MACROS_H__

#ifndef NULL
#define NULL (void *)0
#endif /* NULL */

#define ALIGNMENT 8

/* Macro for zero value */
#define ZERO64 AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0))
#define ZERO16X4 AE_MOVDA16(0)
#define ZERO16 (0)
#define ZERO32 (0)

/* Macro for 1 */
#define ONE16X4 AE_MOVDA16(1)

/* Value of ROW_UNROLL currently supported are 1,2,4,8 only */
#ifndef ROW_UNROLL
#define ROW_UNROLL 8
#endif
#define VEC_UNROLL 2

#define ACC_LSH_AFTER_FIRST_MATXVEC 0

/* Increment in bytes required for particular load
 * instructions. */
#define INCREMENT_IN_BYTES_FOR_WORD8 1
#define INCREMENT_IN_BYTES_FOR_INT16 2
#define INCREMENT_IN_BYTES_FOR_INT32 (INCREMENT_IN_BYTES_FOR_INT16 * 2)
#define INCREMENT_IN_BYTES_FOR_WORD8X4 (INCREMENT_IN_BYTES_FOR_WORD8 * 4)
#define INCREMENT_IN_BYTES_FOR_INT16X4 (INCREMENT_IN_BYTES_FOR_INT16 * 4)
#define INCREMENT_IN_BYTES_FOR_INT64 INCREMENT_IN_BYTES_FOR_INT16X4
#define INCREMENT_IN_BYTES_FOR_FLOAT32 4
#define INCREMENT_IN_BYTES_FOR_FLOAT32x2 (INCREMENT_IN_BYTES_FOR_FLOAT32 * 2)

#define HF2_AE_ADDCIRC16X4_XC(ptr, offset) \
  ptr = ptr + offset;                      \
  if (ptr >= p_end) ptr = ptr - size;

#define MULTIPLY_BY_QUANTIZED_MULTIPLIER(q_out, inp, out_multiplier, \
                                         left_shift, right_shift)    \
  {                                                                  \
    ae_q56s d1;                                                      \
    ae_p24x2s d_mul;                                                 \
    d_mul = AE_CVTP24A16X2_HL(out_multiplier, out_multiplier);       \
    d1 = AE_CVTQ48A32S(inp);                                         \
    d1 = AE_SLLAQ56(d1, left_shift);                                 \
    q_out = AE_MULFQ32SP16U_L(d1, d_mul);                            \
    q_out = AE_SRAIQ56(q_out, 16);                                   \
    AE_MULAFQ32SP16S_H(q_out, d1, d_mul);                            \
    q_out = AE_SRAAQ56(q_out, right_shift);                          \
    q_out = AE_ROUNDSQ32SYM(q_out);                                  \
  }

/* Limit effective bias_shift and acc_shift to [-63 ... 63] */
#define LIMIT_VARIABLE(_var, _left_limit, _right_limit) \
  _var = _var > _right_limit ? _right_limit             \
                             : _var < _left_limit ? _left_limit : _var;

#define LIMIT_ACC_LSH LIMIT_VARIABLE(acc_shift, -63, 63);

#define LIMIT_BIAS_LSH LIMIT_VARIABLE(bias_shift, -63, 63);

#define BW(_datatype) sizeof(_datatype)

#define ADJUST_VAR_AxB(A, B) (((8 * (4 - (BW(A) + BW(B))))))

#define ADJUST_VAR_C(C) (((64 - (8 * BW(C)))))

#define ADJUST_ACC_LSH_AxB_C(A, B, C) \
  acc_shift = acc_shift + 32;         \
  LIMIT_ACC_LSH;

#define ADJUST_BIAS_LSH_AxB(A, B) LIMIT_BIAS_LSH;

#define ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(A, B, C) \
  ADJUST_ACC_LSH_AxB_C(A, B, C);                   \
  ADJUST_BIAS_LSH_AxB(A, B);

/* ====================================================================================================
 */
#define SETUP_BIAS_f32                   \
  xtfloat _xtfloat_bias = (xtfloat)0.0f; \
  xtfloat *_xtfloat_p_bias = (xtfloat *)p_bias;

#define SETUP_BIAS_ASYM8b               \
  WORD32 _WORD32_bias;                  \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD32 *_WORD32_p_bias = (WORD32 *)p_bias;

#define SETUP_BIAS_8b                   \
  WORD8 _WORD8_bias;                    \
  UWORD32 _UWORD32_bias;                \
  ae_int64 _ae_int64_bias = ZERO64;     \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  WORD8 *_WORD8_p_bias = (WORD8 *)p_bias;

#define SETUP_BIAS_8b_BATCH                     \
  WORD8 _WORD8_bias;                            \
  WORD16 _WORD16_bias;                          \
  ae_int16 _ae_int16_bias = ZERO16;             \
  ae_int16 *_ae_int16_p_bias = &_ae_int16_bias; \
  ae_int64 _ae_int64_sat_bias = ZERO64;         \
  WORD8 *_WORD8_p_bias = (WORD8 *)p_bias;

#define SETUP_BIAS_32b                  \
  ae_int32 _ae_int32_bias = ZERO32;     \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  ae_int32 *_ae_int32_p_bias = (ae_int32 *)p_bias;

#define SETUP_BIAS_16b                  \
  ae_int16 _ae_int16_bias = ZERO16;     \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  ae_int16 *_ae_int16_p_bias = (ae_int16 *)p_bias;

#define SETUP_BIAS_64b                  \
  ae_int64 _ae_int64_bias = ZERO64;     \
  ae_int64 _ae_int64_sat_bias = ZERO64; \
  ae_int64 *_ae_int64_p_bias = (ae_int64 *)p_bias;

#define SETUP_ACC_FOR_8bx8b(idx) SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_8bx16b(idx) SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_16bx8b(idx) SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_16bx16b(idx) SETUP_ACC_64b(idx)
#define SETUP_ACC_FOR_ASYM8bxASYM8b(idx) SETUP_ACC_64b(idx)

/*------------------ time batching macros ----------------- */

#define SETUP_ACC_BATCH_ROW_FOR_16bx8b SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_8bx16b SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_8bx8b SETUP_ACC_BATCH_ROW_FOR_16bx16b
#define SETUP_ACC_BATCH_ROW_FOR_ASYM8bxASYM8b SETUP_ACC_BATCH_ROW_FOR_16bx16b

#define SETUP_ACC_BATCH_FOR_16bx8b SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_8bx16b SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_8bx8b SETUP_ACC_BATCH_FOR_16bx16b
#define SETUP_ACC_BATCH_FOR_ASYM8bxASYM8b SETUP_ACC_BATCH_FOR_16bx16b

#define SETUP_ACC_BATCH_ROW_FOR_16bx16b(idx_row) \
  SETUP_ACC_BATCH_VEC_UNROLL(idx_row);

#define SETUP_ACC_BATCH_FOR_16bx16b(idx_row, idx_vec) \
  ae_int64 _ae_int64_acc_##idx_row##_##idx_vec = ZERO64;

#define SETUP_ACC_BATCH_ROW_FOR_f32(idx_row) \
  SETUP_ACC_BATCH_VEC_UNROLL(idx_row);

#define SETUP_ACC_BATCH_FOR_f32(idx_row, idx_vec)                   \
  xtfloatx2 _xtfloatx2_acc_##idx_row##_##idx_vec = (xtfloatx2)0.0f; \
  xtfloat _xtfloat_acc_##idx_row##_##idx_vec = (xtfloat)0.0f;       \
  /*---------------------------------------------------------*/

#define SETUP_ACC_64b(idx) ae_int64 _ae_int64_acc_##idx = ZERO64;

#define SETUP_VEC1_8b                     \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  WORD8 *_WORD8_p_vec1 = (WORD8 *)p_vec1;

#define SETUP_VEC2_8b                     \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  WORD8 *_WORD8_p_vec2 = (WORD8 *)p_vec2;

#define SETUP_VEC1_16b                    \
  ae_int16x4 _ae_int16x4_vec1 = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec1 = (ae_int16x4 *)p_vec1;

#define SETUP_VEC2_16b                    \
  ae_int16x4 _ae_int16x4_vec2 = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec2 = (ae_int16x4 *)p_vec2;

#define SETUP_VEC1_ASYM8b SETUP_VEC1_8b
#define SETUP_VEC2_ASYM8b SETUP_VEC2_8b
/*------------------ time batching macros ----------------- */

#define SETUP_VEC_BATCH_8b(idx_vec)                      \
  ae_int16x4 _ae_int16x4_vec_batch_##idx_vec = ZERO16X4; \
  WORD8 *_WORD8_p_vec_batch_##idx_vec = (WORD8 *)(p_vec1[vec_itr + idx_vec]);

#define SETUP_VEC_BATCH_16b(idx_vec)                     \
  ae_int16x4 _ae_int16x4_vec_batch_##idx_vec = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec_batch_##idx_vec =        \
      (ae_int16x4 *)(p_vec1[vec_itr + idx_vec]);

#define SETUP_VEC_OFFSET_BATCH_16b(idx_vec)              \
  ae_int16x4 _ae_int16x4_vec_batch_##idx_vec = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_vec_batch_##idx_vec =        \
      (ae_int16x4 *)(p_vec1 + (vec_itr + idx_vec) * vec_offset);

#define SETUP_VEC_BATCH_f32(idx_vec)                          \
  xtfloatx2 _xtfloatx2_vec_batch_##idx_vec = (xtfloatx2)0.0f; \
  xtfloatx2 *_xtfloatx2_p_vec_batch_##idx_vec =               \
      (xtfloatx2 *)(p_vec1[vec_itr + idx_vec]);

#define SETUP_VEC_BATCH_ASYM8b SETUP_VEC_BATCH_8b
/*---------------------------------------------------------*/

#define SETUP_MAT1_8b(idx)                      \
  ae_int16x4 _ae_int16x4_mat1_##idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat1_##idx = (WORD8 *)&p_mat1[(m_itr + idx) * row_stride1];

#define SETUP_MAT2_8b(idx)                      \
  ae_int16x4 _ae_int16x4_mat2_##idx = ZERO16X4; \
  WORD8 *_WORD8_p_mat2_##idx = (WORD8 *)&p_mat2[(m_itr + idx) * row_stride2];

#define SETUP_MAT1_16b(idx)                     \
  ae_int16x4 _ae_int16x4_mat1_##idx = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_mat1_##idx =        \
      (ae_int16x4 *)&p_mat1[(m_itr + idx) * row_stride1];

#define SETUP_MAT2_16b(idx)                     \
  ae_int16x4 _ae_int16x4_mat2_##idx = ZERO16X4; \
  ae_int16x4 *_ae_int16x4_p_mat2_##idx =        \
      (ae_int16x4 *)&p_mat2[(m_itr + idx) * row_stride2];

#define SETUP_MAT1_f32(idx)                          \
  xtfloatx2 _xtfloatx2_mat1_##idx = (xtfloatx2)0.0f; \
  xtfloatx2 *_xtfloatx2_p_mat1_##idx =               \
      (xtfloatx2 *)&p_mat1[(m_itr + idx) * row_stride1];

#define SETUP_MAT1_ASYM8b SETUP_MAT1_8b
#define SETUP_MAT2_ASYM8b SETUP_MAT2_8b
/* ====================================================================== */

#define LOAD_VEC1_8b \
  AE_L8X4F_IP(_ae_int16x4_vec1, _WORD8_p_vec1, INCREMENT_IN_BYTES_FOR_WORD8X4);

#define LOAD_VEC2_8b \
  AE_L8X4F_IP(_ae_int16x4_vec2, _WORD8_p_vec2, INCREMENT_IN_BYTES_FOR_WORD8X4);

#define LOAD_VEC1_16b                               \
  AE_L16X4_IP(_ae_int16x4_vec1, _ae_int16x4_p_vec1, \
              INCREMENT_IN_BYTES_FOR_INT16X4);

#define LOAD_VEC2_16b                               \
  AE_L16X4_IP(_ae_int16x4_vec2, _ae_int16x4_p_vec2, \
              INCREMENT_IN_BYTES_FOR_INT16X4);

#define LOAD_VEC1_ASYM8b                                    \
  AE_L8X4F_IP(_ae_int16x4_vec1, _WORD8_p_vec1,              \
              INCREMENT_IN_BYTES_FOR_WORD8X4);              \
  _ae_int16x4_vec1 = AE_MOVF16X4_FROMF64(                   \
      AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec1), 8)); \
  _ae_int16x4_vec1 = AE_ADD16(_ae_int16x4_vec1, AE_MOVDA16(vec1_zero_bias));

#define LOAD_VEC2_ASYM8b                                                     \
  AE_L8X4F_IP(_ae_int16x4_vec2, _WORD8_p_vec2,                               \
              INCREMENT_IN_BYTES_FOR_WORD8X4);                               \
  _ae_int16x4_vec2 = AE_MOVF16X4_FROMF64(                                    \
      AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec2), 8));                  \
  _ae_int16x4_vec2 = AE_ADD16(_ae_int16x4_vec2, AE_MOVDA16(vec2_zero_bias)); \
/*------------------ time batching macros ----------------- */
#define LOAD_VEC_BATCH_f32(idx_vec)                                           \
  XT_LSX2IP(_xtfloatx2_vec_batch_##idx_vec, _xtfloatx2_p_vec_batch_##idx_vec, \
            INCREMENT_IN_BYTES_FOR_FLOAT32x2);

#define LOAD_VEC_BATCH_8b(idx_vec)                                           \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_##idx_vec, _WORD8_p_vec_batch_##idx_vec, \
              INCREMENT_IN_BYTES_FOR_WORD8X4);

#define LOAD_VEC_BATCH_16b(idx_vec)              \
  AE_L16X4_IP(_ae_int16x4_vec_batch_##idx_vec,   \
              _ae_int16x4_p_vec_batch_##idx_vec, \
              INCREMENT_IN_BYTES_FOR_INT16X4);

#define LOAD_VEC_BATCH_ASYM8b(idx_vec)                                       \
  AE_L8X4F_IP(_ae_int16x4_vec_batch_##idx_vec, _WORD8_p_vec_batch_##idx_vec, \
              INCREMENT_IN_BYTES_FOR_WORD8X4);                               \
  _ae_int16x4_vec_batch_##idx_vec = AE_MOVF16X4_FROMF64(                     \
      AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_vec_batch_##idx_vec), 8));   \
  _ae_int16x4_vec_batch_##idx_vec =                                          \
      AE_ADD16(_ae_int16x4_vec_batch_##idx_vec, AE_MOVDA16(vec1_zero_bias));

#define LOAD_BIAS_8b_FOR_8bx8b                  \
  _WORD8_bias = *_WORD8_p_bias++;               \
  _WORD16_bias = _WORD8_bias;                   \
  *((WORD16 *)_ae_int16_p_bias) = _WORD16_bias; \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64)_ae_int16_bias), bias_shift);

#define LOAD_BIAS_16b_FOR_8bx16b                    \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, \
                  INCREMENT_IN_BYTES_FOR_INT16);    \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64)_ae_int16_bias), bias_shift);

#define LOAD_BIAS_16b_FOR_16bx8b LOAD_BIAS_16b_FOR_8bx16b

#define LOAD_BIAS_16b_FOR_16bx16b                   \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, \
                  INCREMENT_IN_BYTES_FOR_INT16);    \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64)_ae_int16_bias), bias_shift);

#define LOAD_BIAS_f32 \
  XT_LSIP(_xtfloat_bias, _xtfloat_p_bias, INCREMENT_IN_BYTES_FOR_FLOAT32);

#define LOAD_BIAS_ASYM8b                                                \
  _WORD32_bias = *_WORD32_p_bias++;                                     \
  _ae_int64_sat_bias =                                                  \
      AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(_WORD32_bias)), 32); \
/*---------------------------------------------------------*/
#define LOAD_ROW_MAT1_8b(idx)                              \
  AE_L8X4F_IP(_ae_int16x4_mat1_##idx, _WORD8_p_mat1_##idx, \
              INCREMENT_IN_BYTES_FOR_WORD8X4);

#define LOAD_ROW_MAT2_8b(idx)                              \
  AE_L8X4F_IP(_ae_int16x4_mat2_##idx, _WORD8_p_mat2_##idx, \
              INCREMENT_IN_BYTES_FOR_WORD8X4);

#define LOAD_ROW_MAT1_16b(idx)                                  \
  AE_L16X4_IP(_ae_int16x4_mat1_##idx, _ae_int16x4_p_mat1_##idx, \
              INCREMENT_IN_BYTES_FOR_INT16X4);

#define LOAD_ROW_MAT2_16b(idx)                                  \
  AE_L16X4_IP(_ae_int16x4_mat2_##idx, _ae_int16x4_p_mat2_##idx, \
              INCREMENT_IN_BYTES_FOR_INT16X4);

#define LOAD_ROW_MAT1_f32(idx)                              \
  XT_LSX2IP(_xtfloatx2_mat1_##idx, _xtfloatx2_p_mat1_##idx, \
            INCREMENT_IN_BYTES_FOR_FLOAT32x2);

#define LOAD_ROW_MAT1_ASYM8b(idx)                                 \
  AE_L8X4F_IP(_ae_int16x4_mat1_##idx, _WORD8_p_mat1_##idx,        \
              INCREMENT_IN_BYTES_FOR_WORD8X4);                    \
  _ae_int16x4_mat1_##idx = AE_MOVF16X4_FROMF64(                   \
      AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat1_##idx), 8)); \
  _ae_int16x4_mat1_##idx =                                        \
      AE_ADD16(_ae_int16x4_mat1_##idx, AE_MOVDA16(mat1_zero_bias));

#define LOAD_ROW_MAT2_ASYM8b(idx)                                 \
  AE_L8X4F_IP(_ae_int16x4_mat2_##idx, _WORD8_p_mat2_##idx,        \
              INCREMENT_IN_BYTES_FOR_WORD8X4);                    \
  _ae_int16x4_mat2_##idx = AE_MOVF16X4_FROMF64(                   \
      AE_SRLI64(AE_MOVF64_FROMF16X4(_ae_int16x4_mat2_##idx), 8)); \
  _ae_int16x4_mat2_##idx =                                        \
      AE_ADD16(_ae_int16x4_mat2_##idx, AE_MOVDA16(mat2_zero_bias));

#define KERNEL_MAT1_VEC1_8b_8b(idx) \
  LOAD_ROW_MAT1_8b(idx);            \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec1, _ae_int16x4_mat1_##idx);

#define KERNEL_MAT2_VEC2_8b_8b(idx) \
  LOAD_ROW_MAT2_8b(idx);            \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec2, _ae_int16x4_mat2_##idx);

#define KERNEL_MAT1_VEC1_16b_8b(idx) \
  LOAD_ROW_MAT1_16b(idx);            \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec1, _ae_int16x4_mat1_##idx);

#define KERNEL_MAT2_VEC2_16b_8b(idx) \
  LOAD_ROW_MAT2_16b(idx);            \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec2, _ae_int16x4_mat2_##idx);

#define KERNEL_MAT1_VEC1_8b_16b(idx) \
  LOAD_ROW_MAT1_8b(idx);             \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec1, _ae_int16x4_mat1_##idx);

#define KERNEL_MAT2_VEC2_8b_16b(idx) \
  LOAD_ROW_MAT2_8b(idx);             \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec2, _ae_int16x4_mat2_##idx);

#define KERNEL_MAT1_VEC1_16b_16b(idx) \
  LOAD_ROW_MAT1_16b(idx);             \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec1, _ae_int16x4_mat1_##idx);

#define KERNEL_MAT2_VEC2_16b_16b(idx) \
  LOAD_ROW_MAT2_16b(idx);             \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec2, _ae_int16x4_mat2_##idx);

#define KERNEL_MAT1_VEC1_ASYM8b_ASYM8b(idx) \
  LOAD_ROW_MAT1_ASYM8b(idx);                \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec1, _ae_int16x4_mat1_##idx);

#define KERNEL_MAT2_VEC2_ASYM8b_ASYM8b(idx) \
  LOAD_ROW_MAT2_ASYM8b(idx);                \
  AE_MULAAAAQ16(_ae_int64_acc_##idx, _ae_int16x4_vec2, _ae_int16x4_mat2_##idx);

/*------------------ time batching macros ----------------- */

#define KERNEL_MAT1_VEC_BATCH_ROW_8b_8b KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_16b_8b KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_8b_16b KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ROW_ASYM8b_ASYM8b \
  KERNEL_MAT1_VEC_BATCH_ROW_16b_16b
#define KERNEL_MAT1_VEC_BATCH_8b_8b KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_16b_8b KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_8b_16b KERNEL_MAT1_VEC_BATCH_16b_16b
#define KERNEL_MAT1_VEC_BATCH_ASYM8b_ASYM8b KERNEL_MAT1_VEC_BATCH_16b_16b

#define KERNEL_MAT1_VEC_BATCH_ROW_16b_16b(idx_row) \
  KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row);

#define KERNEL_MAT1_VEC_BATCH_16b_16b(idx_row, idx_vec) \
  AE_MULAAAAQ16(_ae_int64_acc_##idx_row##_##idx_vec,    \
                _ae_int16x4_vec_batch_##idx_vec, _ae_int16x4_mat1_##idx_row);

#define KERNEL_MAT1_VEC_BATCH_ROW_f32(idx_row) \
  KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row);

#define KERNEL_MAT1_VEC_BATCH_f32(idx_row, idx_vec) \
  XT_MADD_SX2(_xtfloatx2_acc_##idx_row##_##idx_vec, \
              _xtfloatx2_vec_batch_##idx_vec, _xtfloatx2_mat1_##idx_row);

/*---------------------------------------------------------*/
#define ADD_BIAS_8b_ACC_FOR_8bx8b(idx)                                        \
  /* Load 8b bias */                                                          \
  _WORD8_bias = *_WORD8_p_bias++;                                             \
  /* Copy 8-bits to unsigned 32-bits */                                       \
  _UWORD32_bias = _WORD8_bias;                                                \
  /*Move unsigned 32 bit value to DR register*/                               \
  _ae_int64_bias = AE_MOVINT64_FROMINT32X2((AE_MOVDA32X2(_UWORD32_bias, 0))); \
  _ae_int64_bias = AE_SRAA64(_ae_int64_bias, 32);                             \
  _ae_int64_sat_bias = AE_SLAA64S(_ae_int64_bias, bias_shift);                \
  _ae_int64_acc_##idx = AE_SRAA64(_ae_int64_acc_##idx, 16);                   \
  _ae_int64_acc_##idx = AE_ADD64S(_ae_int64_acc_##idx, _ae_int64_sat_bias);

#define ADD_BIAS_32b_ACC_FOR_8bx8b(idx)                                    \
  ae_int32_loadip(_ae_int32_bias, _ae_int32_p_bias,                        \
                  INCREMENT_IN_BYTES_FOR_INT32);                           \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64)_ae_int32_bias), bias_shift); \
  _ae_int64_acc_##idx = AE_SRAA64(_ae_int64_acc_##idx, 16);                \
  _ae_int64_acc_##idx = AE_ADD64S(_ae_int64_acc_##idx, _ae_int64_sat_bias);

#define ADD_BIAS_16b_ACC_FOR_8bx16b(idx)                                   \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias,                        \
                  INCREMENT_IN_BYTES_FOR_INT16);                           \
  /* Saturate 16b bias after shift to 64b */                               \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64)_ae_int16_bias), bias_shift); \
  _ae_int64_acc_##idx = AE_SRAA64(_ae_int64_acc_##idx, 8);                 \
  _ae_int64_acc_##idx = AE_ADD64S(_ae_int64_acc_##idx, _ae_int64_sat_bias);

#define ADD_BIAS_16b_ACC_FOR_16bx8b ADD_BIAS_16b_ACC_FOR_8bx16b

#define ADD_BIAS_64b_ACC_FOR_8bx16b(idx)                                   \
  ae_int64_loadip(_ae_int64_bias, _ae_int64_p_bias,                        \
                  INCREMENT_IN_BYTES_FOR_INT64);                           \
  /* Saturate 64b bias after shift to 64b */                               \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64)_ae_int64_bias), bias_shift); \
  _ae_int64_acc_##idx = AE_SRAA64(_ae_int64_acc_##idx, 8);                 \
  _ae_int64_acc_##idx = AE_ADD64S(_ae_int64_acc_##idx, _ae_int64_sat_bias);

#define ADD_BIAS_16b_ACC_FOR_16bx16b(idx)                                  \
  ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias,                        \
                  INCREMENT_IN_BYTES_FOR_INT16);                           \
  /* Saturate 16b bias after shift to 64b */                               \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64)_ae_int16_bias), bias_shift); \
  _ae_int64_acc_##idx = AE_ADD64S(_ae_int64_acc_##idx, _ae_int64_sat_bias);

#define ADD_BIAS_64b_ACC_FOR_16bx16b(idx)                                  \
  ae_int64_loadip(_ae_int64_bias, _ae_int64_p_bias,                        \
                  INCREMENT_IN_BYTES_FOR_INT64);                           \
  /* Saturate 64b bias after shift to 64b */                               \
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64)_ae_int64_bias), bias_shift); \
  _ae_int64_acc_##idx = AE_ADD64S(_ae_int64_acc_##idx, _ae_int64_sat_bias);

#define ADD_BIAS_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx)                      \
  /* Load 32b bias */                                                   \
  _WORD32_bias = *_WORD32_p_bias++;                                     \
  _ae_int64_sat_bias =                                                  \
      AE_SRAI64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32(_WORD32_bias)), 32); \
  _ae_int64_acc_##idx = AE_ADD64S(_ae_int64_acc_##idx, _ae_int64_sat_bias);

/*------------------ time batching macros ----------------- */
#define ADD_BIAS_BATCH_ROW_8b_ACC_FOR_8bx8b(idx_row) \
  LOAD_BIAS_8b_FOR_8bx8b;                            \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_8bx16b(idx_row) \
  LOAD_BIAS_16b_FOR_8bx16b;                            \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_16bx8b(idx_row) \
  LOAD_BIAS_16b_FOR_16bx8b;                            \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);

#define ADD_BIAS_BATCH_ROW_16b_ACC_FOR_16bx16b(idx_row) \
  LOAD_BIAS_16b_FOR_16bx16b;                            \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);

#define ADD_BIAS_BATCH_ROW_ASYM8b_ACC_FOR_ASYM8bxASYM8b(idx_row) \
  LOAD_BIAS_ASYM8b ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);

#define ADD_BIAS_BATCH_8b_ACC_FOR_8bx8b(idx_row, idx_vec) \
  _ae_int64_acc_##idx_row##_##idx_vec =                   \
      AE_SRAA64(_ae_int64_acc_##idx_row##_##idx_vec, 16); \
  _ae_int64_acc_##idx_row##_##idx_vec =                   \
      AE_ADD64S(_ae_int64_acc_##idx_row##_##idx_vec, _ae_int64_sat_bias);

#define ADD_BIAS_BATCH_16b_ACC_FOR_8bx16b(idx_row, idx_vec) \
  _ae_int64_acc_##idx_row##_##idx_vec =                     \
      AE_SRAA64(_ae_int64_acc_##idx_row##_##idx_vec, 8);    \
  _ae_int64_acc_##idx_row##_##idx_vec =                     \
      AE_ADD64S(_ae_int64_acc_##idx_row##_##idx_vec, _ae_int64_sat_bias);

#define ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b(idx_row, idx_vec) \
  _ae_int64_acc_##idx_row##_##idx_vec =                      \
      AE_ADD64S(_ae_int64_acc_##idx_row##_##idx_vec, _ae_int64_sat_bias);

#define ADD_BIAS_BATCH_16b_ACC_FOR_16bx8b ADD_BIAS_BATCH_16b_ACC_FOR_8bx16b
#define ADD_BIAS_BATCH_ASYM8b_ACC_FOR_ASYM8bxASYM8b \
  ADD_BIAS_BATCH_16b_ACC_FOR_16bx16b

#define ADD_BIAS_BATCH_ROW_ACC_FOR_f32(idx_row) \
  LOAD_BIAS_f32;                                \
  ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);

#define ADD_BIAS_BATCH_ACC_FOR_f32(idx_row, idx_vec)     \
  _xtfloat_acc_##idx_row##_##idx_vec =                   \
      XT_RADD_SX2(_xtfloatx2_acc_##idx_row##_##idx_vec); \
  _xtfloat_acc_##idx_row##_##idx_vec =                   \
      XT_ADD_S(_xtfloat_acc_##idx_row##_##idx_vec, _xtfloat_bias);

#define STORE_ACC_8bx8b_AT_SCRATCH_32b(idx)  \
  (*((ae_int32 *)p_scratch + m_itr + idx)) = \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift));

#define STORE_ACC_8bx8b_AT_OUT_8b(idx)                                    \
  ae_int32 _ae_int32_tmp_var_##idx;                                       \
  ae_f32x2 _ae_f32x2_tmp_var_##idx = AE_SLAA32S(                          \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift)), 24); \
  _ae_int32_tmp_var_##idx = AE_SLAA32S(_ae_f32x2_tmp_var_##idx, -24);     \
  (*((WORD8 *)p_out + m_itr + idx)) = (*((UWORD32 *)&_ae_int32_tmp_var_##idx));

#define STORE_ACC_8bx8b_AT_OUT_16b(idx)                                   \
  ae_int32 _ae_int32_tmp_var_##idx;                                       \
  ae_f32x2 _ae_f32x2_tmp_var_##idx = AE_SLAA32S(                          \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift)), 16); \
  _ae_int32_tmp_var_##idx = AE_SLAA32S(_ae_f32x2_tmp_var_##idx, -16);     \
  (*((WORD16 *)p_out + m_itr + idx)) = (*((UWORD32 *)&_ae_int32_tmp_var_##idx));

#define STORE_ACC_8bx8b_AT_OUT_32b(idx)  \
  (*((ae_int32 *)p_out + m_itr + idx)) = \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift));

#define STORE_ACC_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx)                      \
  _ae_int32x2_acc_##idx = AE_MIN32(                                     \
      AE_MAX32(_ae_int32x2_acc_##idx, AE_MOVDA32(0)), AE_MOVDA32(255)); \
  (*((UWORD8 *)p_out + m_itr + idx)) =                                  \
      (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_##idx);

/* ====================================================================================================
 */
#define STORE_ACC_8bx16b_AT_SCRATCH_32b(idx) \
  (*((ae_int32 *)p_scratch + m_itr + idx)) = \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift));

#define STORE_ACC_8bx16b_AT_OUT_16b(idx)                                  \
  ae_int32 _ae_int32_tmp_var_##idx;                                       \
  ae_f32x2 _ae_f32x2_tmp_var_##idx = AE_SLAA32S(                          \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift)), 16); \
  _ae_int32_tmp_var_##idx = AE_SLAA32S(_ae_f32x2_tmp_var_##idx, -16);     \
  (*((WORD16 *)p_out + m_itr + idx)) = (*((UWORD32 *)&_ae_int32_tmp_var_##idx));

#define STORE_ACC_16bx8b_AT_OUT_16b STORE_ACC_8bx16b_AT_OUT_16b

#define STORE_ACC_8bx16b_AT_OUT_32b(idx) \
  (*((ae_int32 *)p_out + m_itr + idx)) = \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift));

#define STORE_ACC_8bx16b_AT_OUT_64b(idx) \
  (*((ae_int64 *)p_out + m_itr + idx)) = \
      AE_SLAA64S(_ae_int64_acc_##idx, acc_shift);

/* ====================================================================================================
 */
#define STORE_ACC_16bx16b_AT_SCRATCH_32b(idx) \
  (*((ae_int32 *)p_scratch + m_itr + idx)) =  \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift));

#define STORE_ACC_16bx16b_AT_OUT_16b(idx)                                 \
  ae_int32 _ae_int32_tmp_var_##idx;                                       \
  ae_f32x2 _ae_f32x2_tmp_var_##idx = AE_SLAA32S(                          \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift)), 16); \
  _ae_int32_tmp_var_##idx = AE_SLAA32S(_ae_f32x2_tmp_var_##idx, -16);     \
  (*((WORD16 *)p_out + m_itr + idx)) = (*((UWORD32 *)&_ae_int32_tmp_var_##idx));

#define STORE_ACC_16bx16b_AT_OUT_32b(idx) \
  (*((ae_int32 *)p_out + m_itr + idx)) =  \
      AE_ROUND32F64SSYM(AE_SLAA64S(_ae_int64_acc_##idx, acc_shift));

#define STORE_ACC_16bx16b_AT_OUT_64b(idx) \
  (*((ae_int64 *)p_out + m_itr + idx)) =  \
      AE_SLAA64S(_ae_int64_acc_##idx, acc_shift);

/*------------------ time batching macros ----------------- */
#define STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_32b(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);

#define STORE_ACC_BATCH_ROW_8bx8b_AT_OUT_8b(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);

#define STORE_ACC_BATCH_8bx8b_AT_OUT_32b(idx_row, idx_vec)      \
  (*((ae_int32 *)p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
      AE_ROUND32F64SSYM(                                        \
          AE_SLAA64S(_ae_int64_acc_##idx_row##_##idx_vec, acc_shift));

#define STORE_ACC_BATCH_8bx8b_AT_OUT_8b(idx_row, idx_vec)              \
  ae_int32 _ae_int32_tmp_var_##idx_row##_##idx_vec;                    \
  ae_f32x2 _ae_f32x2_tmp_var_##idx_row##_##idx_vec =                   \
      AE_SLAA32S(AE_ROUND32F64SSYM(AE_SLAA64S(                         \
                     _ae_int64_acc_##idx_row##_##idx_vec, acc_shift)), \
                 24);                                                  \
  _ae_int32_tmp_var_##idx_row##_##idx_vec =                            \
      AE_SLAA32S(_ae_f32x2_tmp_var_##idx_row##_##idx_vec, -24);        \
  (*((WORD8 *)p_out[vec_itr + idx_vec] + m_itr + idx_row)) =           \
      (*((UWORD32 *)&_ae_int32_tmp_var_##idx_row##_##idx_vec));

#define STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);

#define STORE_ACC_BATCH_ROW_16bx8b_AT_OUT_16b \
  STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b

#define STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_16b \
  STORE_ACC_BATCH_ROW_8bx16b_AT_OUT_64b

#define STORE_ACC_BATCH_8bx16b_AT_OUT_64b(idx_row, idx_vec)     \
  (*((ae_int64 *)p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
      AE_SLAA64S(_ae_int64_acc_##idx_row##_##idx_vec, acc_shift);

#define STORE_ACC_BATCH_8bx16b_AT_OUT_16b(idx_row, idx_vec) \
  STORE_ACC_BATCH_16bx16b_AT_OUT_16b(idx_row, idx_vec);

#define STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_64b(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);

#define STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_16b \
  STORE_ACC_BATCH_ROW_16bx16b_AT_OUT_64b

#define STORE_ACC_BATCH_16bx16b_AT_OUT_64b(idx_row, idx_vec)    \
  (*((ae_int64 *)p_out[vec_itr + idx_vec] + m_itr + idx_row)) = \
      AE_SLAA64S(_ae_int64_acc_##idx_row##_##idx_vec, acc_shift);

#define STORE_STRIDE_ACC_BATCH_16bx16b_AT_OUT_16b(idx_row, idx_vec)    \
  ae_int32 _ae_int32_tmp_var_##idx_row##_##idx_vec;                    \
  ae_f32x2 _ae_f32x2_tmp_var_##idx_row##_##idx_vec =                   \
      AE_SLAA32S(AE_ROUND32F64SSYM(AE_SLAA64S(                         \
                     _ae_int64_acc_##idx_row##_##idx_vec, acc_shift)), \
                 16);                                                  \
  _ae_int32_tmp_var_##idx_row##_##idx_vec =                            \
      AE_SLAA32S(_ae_f32x2_tmp_var_##idx_row##_##idx_vec, -16);        \
  (*((WORD16 *)p_out + (vec_itr + idx_vec) * out_offset +              \
     (m_itr + idx_row) * out_stride)) =                                \
      (*((UWORD32 *)&_ae_int32_tmp_var_##idx_row##_##idx_vec));

#define STORE_ACC_BATCH_ROW_AT_OUT_f32(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);

#define STORE_ACC_BATCH_AT_OUT_f32(idx_row, idx_vec)                \
  /*p_out value stored in a tmp pointer to make it inout for ISA */ \
  p_out_tmp = (p_out[vec_itr + idx_vec] + m_itr + idx_row);         \
  XT_SSIP(_xtfloat_acc_##idx_row##_##idx_vec, p_out_tmp, 0);

#define STORE_ACC_BATCH_ROW_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row) \
  STORE_ACC_BATCH_VEC_UNROLL(idx_row);

#define STORE_ACC_BATCH_ASYM8bxASYM8b_AT_OUT_ASYM8b(idx_row, idx_vec)          \
  _ae_int32x2_acc_##idx_row##_##idx_vec =                                      \
      AE_MIN32(AE_MAX32(_ae_int32x2_acc_##idx_row##_##idx_vec, AE_MOVDA32(0)), \
               AE_MOVDA32(255));                                               \
  (*((UWORD8 *)(p_out[vec_itr + idx_vec] + m_itr + idx_row))) =                \
      (UWORD8)AE_MOVAD32_L(_ae_int32x2_acc_##idx_row##_##idx_vec);

/*---------------------------------------------------------*/
/* Specific macros needed for extra calculations involved
  for ASYM8b */

/* This is written to match with Tensorflow */
#define ADJUST_ACC_ASYM8b(idx)                                             \
  /* Multiply accumulator with 'out_multiplier', same as Tensorflow */     \
  ae_int32x2 _ae_int32x2_acc_##idx =                                       \
      AE_SLAA32(AE_MOVINT32X2_FROMINT64(_ae_int64_acc_##idx), left_shift); \
  _ae_int32x2_acc_##idx =                                                  \
      AE_MULFP32X2RAS(_ae_int32x2_acc_##idx, AE_MOVDA32(out_multiplier));  \
  /* Shift by out_shift, same as Tensorflow */                             \
  _ae_int64_acc_##idx =                                                    \
      AE_SLAI64(AE_MOVINT64_FROMINT32X2(_ae_int32x2_acc_##idx), 32);       \
  _ae_int64_acc_##idx = AE_SRAA64(_ae_int64_acc_##idx, right_shift);       \
  _ae_int32x2_acc_##idx = AE_ROUND32F64SSYM(_ae_int64_acc_##idx);          \
  /* Add output zero point */                                              \
  (_ae_int32x2_acc_##idx) =                                                \
      AE_ADD32S(_ae_int32x2_acc_##idx, AE_MOVDA32(out_zero_bias));

/* For time batching */
#define ADJUST_ACC_BATCH_ROW_ASYM8b(idx_row) \
  ADJUST_ACC_BATCH_VEC_UNROLL(idx_row);

/* For time batching */
#define ADJUST_ACC_BATCH_ASYM8b(idx_row, idx_vec)                             \
  /* Multiply accumulator with 'out_multiplier', same as Tensorflow */        \
  ae_int32x2 _ae_int32x2_acc_##idx_row##_##idx_vec =                          \
      AE_SLAA32(AE_MOVINT32X2_FROMINT64(_ae_int64_acc_##idx_row##_##idx_vec), \
                left_shift);                                                  \
  _ae_int32x2_acc_##idx_row##_##idx_vec = AE_MULFP32X2RAS(                    \
      _ae_int32x2_acc_##idx_row##_##idx_vec, AE_MOVDA32(out_multiplier));     \
  /* Shift by out_shift, same as Tensorflow */                                \
  _ae_int64_acc_##idx_row##_##idx_vec = AE_SLAI64(                            \
      AE_MOVINT64_FROMINT32X2(_ae_int32x2_acc_##idx_row##_##idx_vec), 32);    \
  _ae_int64_acc_##idx_row##_##idx_vec =                                       \
      AE_SRAA64(_ae_int64_acc_##idx_row##_##idx_vec, right_shift);            \
  _ae_int32x2_acc_##idx_row##_##idx_vec =                                     \
      AE_ROUND32F64SSYM(_ae_int64_acc_##idx_row##_##idx_vec);                 \
  /* Add output zero point */                                                 \
  (_ae_int32x2_acc_##idx_row##_##idx_vec) = AE_ADD32S(                        \
      _ae_int32x2_acc_##idx_row##_##idx_vec, AE_MOVDA32(out_zero_bias));

/*---------------------------------------------------------*/
/* ====================================================================================================
 */
#if (ROW_UNROLL == 1)
#define SETUP_ACC UNROLL_SETUP_ACC(0)
#define SETUP_MAT1 UNROLL_SETUP_MAT1(0)
#define SETUP_MAT2 UNROLL_SETUP_MAT2(0)
#define KERNEL_MAT1_VEC1 UNROLL_KERNEL_MAT1_VEC1(0)
#define KERNEL_MAT2_VEC2 UNROLL_KERNEL_MAT2_VEC2(0)
#define ADD_BIAS_ACC UNROLL_ADD_BIAS_ACC(0)
#define ADJUST_ACC UNROLL_ADJUST_ACC(0)
#define STORE_ACC UNROLL_STORE_ACC(0)

#elif (ROW_UNROLL == 2)
#define SETUP_ACC UNROLL_SETUP_ACC(0) UNROLL_SETUP_ACC(1)
#define SETUP_MAT1 UNROLL_SETUP_MAT1(0) UNROLL_SETUP_MAT1(1)
#define SETUP_MAT2 UNROLL_SETUP_MAT2(0) UNROLL_SETUP_MAT2(1)
#define KERNEL_MAT1_VEC1 UNROLL_KERNEL_MAT1_VEC1(0) UNROLL_KERNEL_MAT1_VEC1(1)
#define KERNEL_MAT2_VEC2 UNROLL_KERNEL_MAT2_VEC2(0) UNROLL_KERNEL_MAT2_VEC2(1)
#define ADD_BIAS_ACC UNROLL_ADD_BIAS_ACC(0) UNROLL_ADD_BIAS_ACC(1)
#define ADJUST_ACC UNROLL_ADJUST_ACC(0) UNROLL_ADJUST_ACC(1)
#define STORE_ACC UNROLL_STORE_ACC(0) UNROLL_STORE_ACC(1)

#elif (ROW_UNROLL == 4)
#define SETUP_ACC     \
  UNROLL_SETUP_ACC(0) \
  UNROLL_SETUP_ACC(1) UNROLL_SETUP_ACC(2) UNROLL_SETUP_ACC(3)
#define SETUP_MAT1     \
  UNROLL_SETUP_MAT1(0) \
  UNROLL_SETUP_MAT1(1) UNROLL_SETUP_MAT1(2) UNROLL_SETUP_MAT1(3)
#define SETUP_MAT2     \
  UNROLL_SETUP_MAT2(0) \
  UNROLL_SETUP_MAT2(1) UNROLL_SETUP_MAT2(2) UNROLL_SETUP_MAT2(3)
#define KERNEL_MAT1_VEC1     \
  UNROLL_KERNEL_MAT1_VEC1(0) \
  UNROLL_KERNEL_MAT1_VEC1(1) \
  UNROLL_KERNEL_MAT1_VEC1(2) UNROLL_KERNEL_MAT1_VEC1(3)
#define KERNEL_MAT2_VEC2     \
  UNROLL_KERNEL_MAT2_VEC2(0) \
  UNROLL_KERNEL_MAT2_VEC2(1) \
  UNROLL_KERNEL_MAT2_VEC2(2) UNROLL_KERNEL_MAT2_VEC2(3)
#define ADD_BIAS_ACC     \
  UNROLL_ADD_BIAS_ACC(0) \
  UNROLL_ADD_BIAS_ACC(1) UNROLL_ADD_BIAS_ACC(2) UNROLL_ADD_BIAS_ACC(3)
#define ADJUST_ACC     \
  UNROLL_ADJUST_ACC(0) \
  UNROLL_ADJUST_ACC(1) UNROLL_ADJUST_ACC(2) UNROLL_ADJUST_ACC(3)
#define STORE_ACC     \
  UNROLL_STORE_ACC(0) \
  UNROLL_STORE_ACC(1) UNROLL_STORE_ACC(2) UNROLL_STORE_ACC(3)

#elif (ROW_UNROLL == 8)
#define SETUP_ACC     \
  UNROLL_SETUP_ACC(0) \
  UNROLL_SETUP_ACC(1) \
  UNROLL_SETUP_ACC(2) \
  UNROLL_SETUP_ACC(3) \
  UNROLL_SETUP_ACC(4) \
  UNROLL_SETUP_ACC(5) UNROLL_SETUP_ACC(6) UNROLL_SETUP_ACC(7)
#define SETUP_MAT1     \
  UNROLL_SETUP_MAT1(0) \
  UNROLL_SETUP_MAT1(1) \
  UNROLL_SETUP_MAT1(2) \
  UNROLL_SETUP_MAT1(3) \
  UNROLL_SETUP_MAT1(4) \
  UNROLL_SETUP_MAT1(5) UNROLL_SETUP_MAT1(6) UNROLL_SETUP_MAT1(7)
#define SETUP_MAT2     \
  UNROLL_SETUP_MAT2(0) \
  UNROLL_SETUP_MAT2(1) \
  UNROLL_SETUP_MAT2(2) \
  UNROLL_SETUP_MAT2(3) \
  UNROLL_SETUP_MAT2(4) \
  UNROLL_SETUP_MAT2(5) UNROLL_SETUP_MAT2(6) UNROLL_SETUP_MAT2(7)
#define KERNEL_MAT1_VEC1     \
  UNROLL_KERNEL_MAT1_VEC1(0) \
  UNROLL_KERNEL_MAT1_VEC1(1) \
  UNROLL_KERNEL_MAT1_VEC1(2) \
  UNROLL_KERNEL_MAT1_VEC1(3) \
  UNROLL_KERNEL_MAT1_VEC1(4) \
  UNROLL_KERNEL_MAT1_VEC1(5) \
  UNROLL_KERNEL_MAT1_VEC1(6) UNROLL_KERNEL_MAT1_VEC1(7)
#define KERNEL_MAT2_VEC2     \
  UNROLL_KERNEL_MAT2_VEC2(0) \
  UNROLL_KERNEL_MAT2_VEC2(1) \
  UNROLL_KERNEL_MAT2_VEC2(2) \
  UNROLL_KERNEL_MAT2_VEC2(3) \
  UNROLL_KERNEL_MAT2_VEC2(4) \
  UNROLL_KERNEL_MAT2_VEC2(5) \
  UNROLL_KERNEL_MAT2_VEC2(6) UNROLL_KERNEL_MAT2_VEC2(7)
#define ADD_BIAS_ACC     \
  UNROLL_ADD_BIAS_ACC(0) \
  UNROLL_ADD_BIAS_ACC(1) \
  UNROLL_ADD_BIAS_ACC(2) \
  UNROLL_ADD_BIAS_ACC(3) \
  UNROLL_ADD_BIAS_ACC(4) \
  UNROLL_ADD_BIAS_ACC(5) UNROLL_ADD_BIAS_ACC(6) UNROLL_ADD_BIAS_ACC(7)
#define ADJUST_ACC     \
  UNROLL_ADJUST_ACC(0) \
  UNROLL_ADJUST_ACC(1) \
  UNROLL_ADJUST_ACC(2) \
  UNROLL_ADJUST_ACC(3) \
  UNROLL_ADJUST_ACC(4) \
  UNROLL_ADJUST_ACC(5) UNROLL_ADJUST_ACC(6) UNROLL_ADJUST_ACC(7)
#define STORE_ACC     \
  UNROLL_STORE_ACC(0) \
  UNROLL_STORE_ACC(1) \
  UNROLL_STORE_ACC(2) \
  UNROLL_STORE_ACC(3) \
  UNROLL_STORE_ACC(4) \
  UNROLL_STORE_ACC(5) UNROLL_STORE_ACC(6) UNROLL_STORE_ACC(7)

#endif /* (ROW_UNROLL == 1) */

#if (ROW_UNROLL == 4 && VEC_UNROLL == 2)

#define SETUP_VEC_BATCH UNROLL_SETUP_VEC_BATCH(0) UNROLL_SETUP_VEC_BATCH(1)

#define SETUP_ACC_BATCH         \
  UNROLL_ROW_SETUP_ACC_BATCH(0) \
  UNROLL_ROW_SETUP_ACC_BATCH(1) \
  UNROLL_ROW_SETUP_ACC_BATCH(2) UNROLL_ROW_SETUP_ACC_BATCH(3)
#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_SETUP_ACC_BATCH(idx_row, 0) UNROLL_SETUP_ACC_BATCH(idx_row, 1)
#define SETUP_ACC_BATCH_TAIL   \
  UNROLL_SETUP_ACC_BATCH(0, 0) \
  UNROLL_SETUP_ACC_BATCH(1, 0) \
  UNROLL_SETUP_ACC_BATCH(2, 0) UNROLL_SETUP_ACC_BATCH(3, 0)

#define LOAD_VEC_BATCH UNROLL_LOAD_VEC_BATCH(0) UNROLL_LOAD_VEC_BATCH(1)
#define LOAD_MAT1         \
  UNROLL_LOAD_ROW_MAT1(0) \
  UNROLL_LOAD_ROW_MAT1(1) UNROLL_LOAD_ROW_MAT1(2) UNROLL_LOAD_ROW_MAT1(3)

#define KERNEL_MAT1_VEC_BATCH         \
  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0) \
  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(1) \
  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(2) UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(3)
#define KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row, 0)        \
  UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row, 1)
#define KERNEL_MAT1_VEC_BATCH_TAIL   \
  UNROLL_KERNEL_MAT1_VEC_BATCH(0, 0) \
  UNROLL_KERNEL_MAT1_VEC_BATCH(1, 0) \
  UNROLL_KERNEL_MAT1_VEC_BATCH(2, 0) UNROLL_KERNEL_MAT1_VEC_BATCH(3, 0)

#define ADD_BIAS_ACC_BATCH   \
  UNROLL_ROW_ADD_BIAS_ACC(0) \
  UNROLL_ROW_ADD_BIAS_ACC(1) \
  UNROLL_ROW_ADD_BIAS_ACC(2) UNROLL_ROW_ADD_BIAS_ACC(3)
#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row) \
  UNROLL_ADD_BIAS_ACC_BATCH(idx_row, 0) UNROLL_ADD_BIAS_ACC_BATCH(idx_row, 1)
#define ADD_BIAS_ACC_BATCH_TAIL                     \
  LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(0, 0)         \
      LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(1, 0)     \
          LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(2, 0) \
              LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(3, 0)

#define STORE_ACC_BATCH   \
  UNROLL_ROW_STORE_ACC(0) \
  UNROLL_ROW_STORE_ACC(1) UNROLL_ROW_STORE_ACC(2) UNROLL_ROW_STORE_ACC(3)
#define STORE_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_STORE_ACC_BATCH(idx_row, 0) UNROLL_STORE_ACC_BATCH(idx_row, 1)
#define STORE_ACC_BATCH_TAIL   \
  UNROLL_STORE_ACC_BATCH(0, 0) \
  UNROLL_STORE_ACC_BATCH(1, 0) \
  UNROLL_STORE_ACC_BATCH(2, 0) UNROLL_STORE_ACC_BATCH(3, 0)

#define ADJUST_ACC_BATCH_TAIL   \
  UNROLL_ADJUST_ACC_BATCH(0, 0) \
  UNROLL_ADJUST_ACC_BATCH(1, 0) \
  UNROLL_ADJUST_ACC_BATCH(2, 0) UNROLL_ADJUST_ACC_BATCH(3, 0)
#define ADJUST_ACC_BATCH   \
  UNROLL_ROW_ADJUST_ACC(0) \
  UNROLL_ROW_ADJUST_ACC(1) UNROLL_ROW_ADJUST_ACC(2) UNROLL_ROW_ADJUST_ACC(3)
#define ADJUST_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_ADJUST_ACC_BATCH(idx_row, 0) UNROLL_ADJUST_ACC_BATCH(idx_row, 1)

#endif /* (ROW_UNROLL == 4 && VEC_UNROLL == 2)*/

#endif /* __XA_NNLIB_COMMON_MACROS_H__ */
