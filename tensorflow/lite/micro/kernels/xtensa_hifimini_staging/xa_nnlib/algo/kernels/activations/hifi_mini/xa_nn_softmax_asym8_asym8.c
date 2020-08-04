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

#define ALIGNMENT 8 /* 8 bytes alignment */
#define ALIGNED_SIZE(x, bytes) (((x) + (bytes - 1)) & (~(bytes - 1)))
#define ALIGN_PTR(x, bytes) ((((unsigned)(x)) + (bytes - 1)) & (~(bytes - 1)))

#ifndef AE_LP8X2F_IU
#define AE_LP8X2F_IU(p_x, p_in, x)                           \
  AE_LP16F_IU(p_x, (ae_p16s *)p_in, x);                      \
  ae_p24x2s p_tmp1 = AE_SLLIP24(p_x, 8);                     \
  ae_p24x2s p_tmp2 = AE_ANDP48(p_x, AE_MOVPA24(0xFFFF0000)); \
  p_x = AE_SELP24_LL(p_tmp2, p_tmp1);

#endif

#define NSA64_T(y, x)               \
  {                                 \
    ae_q56s q_tmp = *(ae_q56s *)&x; \
    y = AE_NSAQ56S(q_tmp) + 8;      \
  }

#define MULFP32X2RAS_T(result, a, b)             \
  {                                              \
    ae_q56s q_a = AE_CVTQ48A32S(a);              \
    ae_p24x2s p_b = AE_CVTP24A16X2_HL(b, b);     \
    ae_q56s q_out = AE_MULFQ32SP16U_L(q_a, p_b); \
    q_out = AE_SRAIQ56(q_out, 16);               \
    AE_MULAFQ32SP16S_H(q_out, q_a, p_b);         \
    q_out = AE_ROUNDSQ32ASYM(q_out);             \
    *(ae_q32s *)&result = q_out;                 \
  }

#define MULFP32X2RS_T(result, a, b)              \
  {                                              \
    ae_q56s q_a = AE_CVTQ48A32S(a);              \
    ae_p24x2s p_b = AE_CVTP24A16X2_HL(b, b);     \
    ae_q56s q_out = AE_MULFQ32SP16U_L(q_a, p_b); \
    q_out = AE_SRAIQ56(q_out, 16);               \
    AE_MULAFQ32SP16S_H(q_out, q_a, p_b);         \
    q_out = AE_ROUNDSQ32SYM(q_out);              \
    *(ae_q32s *)&result = q_out;                 \
  }
#define ADD32S_T(result, a, b)             \
  {                                        \
    ae_q56s q_a = AE_CVTQ48A32S(a);        \
    ae_q56s q_b = AE_CVTQ48A32S(b);        \
    ae_q56s q_out = AE_ADDSQ56S(q_a, q_b); \
    q_out = AE_SATQ48S(q_out);             \
    *(ae_q32s *)&result = q_out;           \
  }

#define SUB32S_T(result, a, b)             \
  {                                        \
    ae_q56s q_a = AE_CVTQ48A32S(a);        \
    ae_q56s q_b = AE_CVTQ48A32S(b);        \
    ae_q56s q_out = AE_SUBSQ56S(q_a, q_b); \
    q_out = AE_SATQ48S(q_out);             \
    *(ae_q32s *)&result = q_out;           \
  }

#define SLAI32S_T(result, a, b)         \
  {                                     \
    ae_q56s q_a = AE_CVTQ48A32S(a);     \
    ae_q56s q_out = AE_SLLIQ56(q_a, b); \
    q_out = AE_SATQ48S(q_out);          \
    *(ae_q32s *)&result = q_out;        \
  }

#define SRAA32RS_T(result, a, b)             \
  {                                          \
    ae_q56s q_a = AE_CVTQ48A32S(a);          \
    ae_q56s q_out = AE_SLAASQ56S(q_a, (-b)); \
    q_out = AE_ROUNDSQ32ASYM(q_out);         \
    *(ae_q32s *)&result = q_out;             \
  }

#define SRAI32R_T(result, a, b)         \
  {                                     \
    ae_q56s q_a = AE_CVTQ48A32S(a);     \
    ae_q56s q_out = AE_SRAIQ56(q_a, b); \
    q_out = AE_ROUNDSQ32ASYM(q_out);    \
    *(ae_q32s *)&result = q_out;        \
  }

static const int CONSTANT_TERM = (0x70f5a894);
static const int CONSTANT_1_OVER_3 = (0x2aaaaaab);
static const int CONSTANT_1_OVER_8 = (0x10000000);
static const int ONE_QUATER_Q26 = (0x1000000);  // Q6.26
static const int MASK = (0xffffff);
static const int Q31 = 0x7fffffff;
static const int constant_48_over_17 = 1515870810;
static const int constant_neg_32_over_17 = -1010580540;  // Q29
static const int F2_ONE = 0x20000000;

static const int constant_neg_32_over_17_Q21 = -3947580;  // Q21
static const int constant_48_over_17_Q21 = 5921370;       // Q21

static ae_p24x2s GetReciprocal(ae_q56s q_x, int x_integerbits, int *lsh) {
  int headroom_plus_one;
  ae_p24x2s p_x;
  ae_q56s q_tmp;
  ae_p24x2s p_half_den;
  int i;

  headroom_plus_one = AE_NSAQ56S(q_x) + 8;
  headroom_plus_one = headroom_plus_one - 31;
  *lsh = x_integerbits - headroom_plus_one;

  q_x = (q_x << (headroom_plus_one + 15));
  p_half_den = AE_ROUNDSP24Q48SYM(q_x);

  q_tmp = AE_CVTQ48A32S(constant_48_over_17);
  AE_MULAFP24S_LL(q_tmp, p_half_den, AE_MOVPA24(constant_neg_32_over_17_Q21));
  p_x = AE_ROUNDSP24Q48SYM(q_tmp);

  for (i = 0; i < 3; i++) {
    q_tmp = AE_CVTQ48A32S(F2_ONE);
    AE_MULSFP24S_LL(q_tmp, p_x, p_half_den);
    ae_p24x2s p_one_minus_half_denominator_times_x = AE_ROUNDSP24Q48SYM(q_tmp);

    q_tmp = AE_MULFP24S_LL(p_x, p_one_minus_half_denominator_times_x);
    ae_p24x2s p_m = AE_ROUNDSP24Q48SYM(q_tmp);
    p_m = AE_SLLISP24S(p_m, 2);
    p_x = AE_ADDSP24S(p_x, p_m);
  }

  p_x = AE_SLLISP24S(p_x, 1);

  return p_x;
}

static const int MASK_16BITS = (0xffff);
static const int ONE_QUATER_Q18 = (0x10000);          // Q18
static const int CONSTANT_1_OVER_8_Q23 = (0x100000);  // Q23
static const int CONSTANT_1_OVER_3_Q23 = (0x2aaaaa);  // Q23
static const int CONSTANT_TERM_Q23 = (0x70f5a8);      // Q23
static const int Q23 = 0x7fffff;

#define GEMMLOWP_EXP_BARREL_SHIFTER_OPT_II(p_in_out, exponent,                \
                                           FixedPointMultiplier, p_remainder) \
  {                                                                           \
    ae_p24x2s p_out;                                                          \
                                                                              \
    ae_p24x2s p_zero = AE_ZEROP48();                                          \
                                                                              \
    ae_p24x2s p_scale = AE_MOVPA24(1 << (18 + exponent));                     \
    ae_p24x2s p_mask = p_remainder & p_scale;                                 \
                                                                              \
    ae_p24x2s p_FixedPointMultiplier = AE_MOVPA24(FixedPointMultiplier >> 8); \
                                                                              \
    ae_q56s q_tmp1 = AE_MULFP24S_HH(p_in_out, p_FixedPointMultiplier);        \
    ae_q56s q_tmp2 = AE_MULFP24S_LL(p_in_out, p_FixedPointMultiplier);        \
    ae_p24x2s p_t1 = AE_ROUNDSP24Q48SYM(q_tmp1);                              \
    ae_p24x2s p_t2 = AE_ROUNDSP24Q48SYM(q_tmp2);                              \
    p_out = AE_SELP24_LL(p_t1, p_t2);                                         \
                                                                              \
    xtbool2 flag_le = AE_LTP24S(p_zero, p_mask);                              \
    AE_MOVTP24X2(p_in_out, p_out, flag_le);                                   \
  }

#define EXP_Q26_II(p_exp_y, p_inp_t)                                        \
  {                                                                         \
    ae_p24x2s p_x1_in, p_x2, p_x3, p_x4, p_x4_by_4, p_y1, p_y2, p_y3, p_y4, \
        p_y5, p_y6, p_y;                                                    \
                                                                            \
    p_x2 = p_inp_t & AE_MOVPA24(MASK_16BITS);                               \
    ae_p24x2s p_a_mod_quater_minus_q_1_by_4 =                               \
        p_x2 - AE_MOVPA24(ONE_QUATER_Q18);                                  \
    ae_p24x2s p_x_in = p_a_mod_quater_minus_q_1_by_4 << 5;                  \
    ae_p24x2s p_remainder = p_a_mod_quater_minus_q_1_by_4 - p_inp_t;        \
                                                                            \
    p_x1_in = AE_ADDSP24S(p_x_in, AE_MOVPA24(CONSTANT_1_OVER_8_Q23));       \
                                                                            \
    ae_q56s q_tmp1 = AE_MULFP24S_HH(p_x1_in, p_x1_in);                      \
    ae_q56s q_tmp2 = AE_MULFP24S_LL(p_x1_in, p_x1_in);                      \
    ae_p24x2s p_t1 = AE_ROUNDSP24Q48SYM(q_tmp1);                            \
    ae_p24x2s p_t2 = AE_ROUNDSP24Q48SYM(q_tmp2);                            \
    p_x2 = AE_SELP24_LL(p_t1, p_t2);                                        \
                                                                            \
    q_tmp1 = AE_MULFP24S_HH(p_t1, p_x1_in);                                 \
    q_tmp2 = AE_MULFP24S_LL(p_t2, p_x1_in);                                 \
    p_t1 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
    p_t2 = AE_ROUNDSP24Q48SYM(q_tmp2);                                      \
    p_x3 = AE_SELP24_LL(p_t1, p_t2);                                        \
                                                                            \
    q_tmp1 = AE_MULFP24S_HH(p_x2, p_x2);                                    \
    q_tmp2 = AE_MULFP24S_LL(p_x2, p_x2);                                    \
    p_t1 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
    p_t2 = AE_ROUNDSP24Q48SYM(q_tmp2);                                      \
    p_x4 = AE_SELP24_LL(p_t1, p_t2);                                        \
    p_x4_by_4 = p_x4 >> 2;                                                  \
                                                                            \
    p_y1 = AE_ADDSP24S(p_x4_by_4, p_x3);                                    \
                                                                            \
    ae_p24x2s p_const = AE_MOVPA24(CONSTANT_1_OVER_3_Q23);                  \
    q_tmp1 = AE_MULFP24S_HH(p_y1, p_const);                                 \
    q_tmp2 = AE_MULFP24S_LL(p_y1, p_const);                                 \
    p_t1 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
    p_t2 = AE_ROUNDSP24Q48SYM(q_tmp2);                                      \
    p_y2 = AE_SELP24_LL(p_t1, p_t2);                                        \
                                                                            \
    p_y3 = AE_ADDSP24S(p_y2, p_x2);                                         \
    p_y4 = p_y3 >> 1;                                                       \
                                                                            \
    p_y5 = AE_ADDSP24S(p_x1_in, p_y4); /* ADD32S_T(y5, x1_in, y4);  */      \
                                                                            \
    p_const = AE_MOVPA24(CONSTANT_TERM_Q23);                                \
    q_tmp1 = AE_MULFP24S_HH(p_y5, p_const);                                 \
    q_tmp2 = AE_MULFP24S_LL(p_y5, p_const);                                 \
    p_t1 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
    p_t2 = AE_ROUNDSP24Q48SYM(q_tmp2);                                      \
    p_y6 = AE_SELP24_LL(p_t1, p_t2);                                        \
    p_y = AE_ADDSP24S(p_y6, p_const);                                       \
                                                                            \
    {                                                                       \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_II(p_y, -2, 1672461947, p_remainder); \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_II(p_y, -1, 1302514674, p_remainder); \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_II(p_y, 0, 790015084, p_remainder);   \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_II(p_y, 1, 290630308, p_remainder);   \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_II(p_y, 2, 39332535, p_remainder);    \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_II(p_y, 3, 720401, p_remainder);      \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_II(p_y, 4, 242, p_remainder);         \
    }                                                                       \
    p_exp_y = p_y;                                                          \
    p_const = AE_MOVPA24(Q23);                                              \
    xtbool2 flag_eq = AE_EQP24(p_inp_t, AE_ZEROP48());                      \
    AE_MOVTP24X2(p_exp_y, p_const, flag_eq);                                \
  }

#define GEMMLOWP_EXP_BARREL_SHIFTER_OPT_I(p_in_out, exponent,                 \
                                          FixedPointMultiplier, p_remainder)  \
  {                                                                           \
    ae_p24x2s p_out;                                                          \
                                                                              \
    ae_p24x2s p_zero = AE_ZEROP48();                                          \
                                                                              \
    ae_p24x2s p_scale = AE_MOVPA24(1 << (18 + exponent));                     \
    ae_p24x2s p_mask = p_remainder & p_scale;                                 \
                                                                              \
    ae_p24x2s p_FixedPointMultiplier = AE_MOVPA24(FixedPointMultiplier >> 8); \
                                                                              \
    ae_q56s q_tmp1 = AE_MULFP24S_HH(p_in_out, p_FixedPointMultiplier);        \
    p_out = AE_ROUNDSP24Q48SYM(q_tmp1);                                       \
                                                                              \
    xtbool2 flag_le = AE_LTP24S(p_zero, p_mask);                              \
    AE_MOVTP24X2(p_in_out, p_out, flag_le);                                   \
  }

#define EXP_Q26_I(p_exp_y, p_inp_t)                                         \
  {                                                                         \
    ae_p24x2s p_x1_in, p_x2, p_x3, p_x4, p_x4_by_4, p_y1, p_y2, p_y3, p_y4, \
        p_y5, p_y6, p_y;                                                    \
                                                                            \
    p_x2 = p_inp_t & AE_MOVPA24(MASK_16BITS);                               \
    ae_p24x2s p_a_mod_quater_minus_q_1_by_4 =                               \
        p_x2 - AE_MOVPA24(ONE_QUATER_Q18);                                  \
    ae_p24x2s p_x_in = p_a_mod_quater_minus_q_1_by_4 << 5;                  \
    ae_p24x2s p_remainder = p_a_mod_quater_minus_q_1_by_4 - p_inp_t;        \
                                                                            \
    p_x1_in = AE_ADDSP24S(p_x_in, AE_MOVPA24(CONSTANT_1_OVER_8_Q23));       \
                                                                            \
    ae_q56s q_tmp1 = AE_MULFP24S_HH(p_x1_in, p_x1_in);                      \
    p_x2 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
                                                                            \
    q_tmp1 = AE_MULFP24S_HH(p_x2, p_x1_in);                                 \
    p_x3 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
                                                                            \
    q_tmp1 = AE_MULFP24S_HH(p_x2, p_x2);                                    \
    p_x4 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
    p_x4_by_4 = p_x4 >> 2;                                                  \
                                                                            \
    p_y1 = AE_ADDSP24S(p_x4_by_4, p_x3);                                    \
                                                                            \
    ae_p24x2s p_const = AE_MOVPA24(CONSTANT_1_OVER_3_Q23);                  \
    q_tmp1 = AE_MULFP24S_HH(p_y1, p_const);                                 \
    p_y2 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
                                                                            \
    p_y3 = AE_ADDSP24S(p_y2, p_x2);                                         \
    p_y4 = p_y3 >> 1;                                                       \
                                                                            \
    p_y5 = AE_ADDSP24S(p_x1_in, p_y4); /* ADD32S_T(y5, x1_in, y4);  */      \
                                                                            \
    p_const = AE_MOVPA24(CONSTANT_TERM_Q23);                                \
    q_tmp1 = AE_MULFP24S_HH(p_y5, p_const);                                 \
    p_y6 = AE_ROUNDSP24Q48SYM(q_tmp1);                                      \
    p_y = AE_ADDSP24S(p_y6, p_const);                                       \
                                                                            \
    {                                                                       \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_I(p_y, -2, 1672461947, p_remainder);  \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_I(p_y, -1, 1302514674, p_remainder);  \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_I(p_y, 0, 790015084, p_remainder);    \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_I(p_y, 1, 290630308, p_remainder);    \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_I(p_y, 2, 39332535, p_remainder);     \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_I(p_y, 3, 720401, p_remainder);       \
      GEMMLOWP_EXP_BARREL_SHIFTER_OPT_I(p_y, 4, 242, p_remainder);          \
    }                                                                       \
    p_exp_y = p_y;                                                          \
    p_const = AE_MOVPA24(Q23);                                              \
    xtbool2 flag_eq = AE_EQP24(p_inp_t, AE_ZEROP48());                      \
    AE_MOVTP24X2(p_exp_y, p_const, flag_eq);                                \
  }

WORD32 xa_nn_vec_softmax_asym8u_8(UWORD8 *__restrict__ pOut,
                                  const UWORD8 *__restrict__ pVec,
                                  WORD32 diffmin, WORD32 input_beta_left_shift,
                                  WORD32 input_beta_multiplier,
                                  WORD32 vec_length, pVOID pScratch) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(pOut, -1);
  XA_NNLIB_ARG_CHK_PTR(pVec, -1);
  XA_NNLIB_ARG_CHK_PTR(pScratch, -1);
  /* Pointer alignment checks */
  /* No alignment (1-byte) needed for any pointer */
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(
      ((input_beta_left_shift < -31) || (input_beta_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

  int i;
  int shift_bits_reciprocal;
  UWORD8 *p_in;
  WORD32 *__restrict pExp = (WORD32 *)ALIGN_PTR(pScratch, ALIGNMENT);
  ae_p24f *__restrict pTmpScratch = (ae_p24f *)pExp;
  int max;
  ae_p24x2s p_x;
  ae_p24x2s p_max = AE_MOVPA24(0xFF800000);
  ae_p24x2s p_recip_sum_exp;
  int pre_loop_count;
  int main_loop_count;
  int post_loop_count;

  if (vec_length > 1) {
    pre_loop_count = (int)pVec & 0x1;
    main_loop_count = vec_length - pre_loop_count;
    post_loop_count = (main_loop_count & 1);
    main_loop_count = main_loop_count >> 1;
  } else {
    pre_loop_count = 0;
    main_loop_count = 0;
    post_loop_count = vec_length;
  }

  /* Calculating Max */
  {
    p_in = (UWORD8 *)pVec;

    if (pre_loop_count) {
      p_x = AE_MOVPA24(*p_in++);
      p_max = AE_MAXP24S(p_max, p_x);
    }

    p_in -= 2;
    for (i = 0; i < main_loop_count; i++) {
      AE_LP8X2F_IU(p_x, p_in, 2 * sizeof(WORD8));
      p_x = AE_SRLIP24(p_x, 16);
      p_max = AE_MAXP24S(p_max, p_x);
    }

    if (post_loop_count) {
      p_in += 2;
      p_x = AE_MOVPA24(*p_in);
      p_max = AE_MAXP24S(p_max, p_x);
    }
    p_max = AE_MAXP24S(p_max, AE_SELP24_LH(p_max, p_max));
    max = AE_MOVAP24S_L(p_max);
  }

  /* Calculate exponents */
  {
    ae_q56s q_sum_exp = AE_ZEROQ56();
    ae_p24x2s p_rem_x, p_y, p_exp_y;
    ae_p24x2s p_zero = AE_ZEROP48();
    ae_p24x2s p_input_beta_multiplier =
        AE_MOVPA24((input_beta_multiplier >> 8));
    ae_p24x2s p_diffmin = AE_MOVPA24(diffmin);
    int input_beta_left_shift_for_24bit = input_beta_left_shift - 8;

    p_in = (UWORD8 *)pVec;
    WUR_AE_SAR(input_beta_left_shift_for_24bit);

    if (pre_loop_count) {
      p_x = AE_MOVPA24(*p_in++);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);

      EXP_Q26_I(p_exp_y, p_dequantized_y1)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      *pTmpScratch++ = p_exp_y;

      p_exp_y = p_exp_y >> 4;

      AE_MULAP24S_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }

    p_in -= 2;
    for (i = 0; i < main_loop_count; i++) {
      AE_LP8X2F_IU(p_x, p_in, 2 * sizeof(WORD8));
      p_x = AE_SRLIP24(p_x, 16);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_HH(p_y, p_input_beta_multiplier);
      ae_q56s q_dequantized_y2 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);
      ae_p24x2s p_dequantized_y2 = AE_ROUNDSP24Q48ASYM(q_dequantized_y2);

      ae_p24x2s p_dequantized =
          AE_SELP24_LL(p_dequantized_y1, p_dequantized_y2);

      EXP_Q26_II(p_exp_y, p_dequantized)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      *pTmpScratch++ = AE_SELP24_HH(p_exp_y, p_exp_y);
      *pTmpScratch++ = p_exp_y; /* store lower element */

      p_exp_y = p_exp_y >> 4;

      AE_MULAAP24S_HH_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }
    if (post_loop_count) {
      p_in += 2;

      p_x = AE_MOVPA24(*p_in);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);

      EXP_Q26_I(p_exp_y, p_dequantized_y1)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      *pTmpScratch = p_exp_y;

      p_exp_y = p_exp_y >> 4;

      AE_MULAP24S_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }
    p_recip_sum_exp = GetReciprocal(q_sum_exp, 12, &shift_bits_reciprocal);
  }

  /* Calculate output */
  {
    ae_p24x2s p_exp;

    int shift_val = -(shift_bits_reciprocal + 31 - 8 - 8);

    ae_p24x2s p_min = AE_ZEROP48();
    ae_p24x2s p_max = AE_MOVPA24(255);

    for (i = 0; i<vec_length >> 1; i++) {
      int out;

      p_exp = *(ae_p24x2f *)&pExp[2 * i];

      ae_q56s q_tmp1 = AE_MULFP24S_HH(p_exp, p_recip_sum_exp);
      ae_q56s q_tmp2 = AE_MULFP24S_LL(p_exp, p_recip_sum_exp);

      q_tmp1 = AE_SLAASQ56S(q_tmp1, shift_val);
      q_tmp2 = AE_SLAASQ56S(q_tmp2, shift_val);

      ae_p24x2s p_out1 = AE_ROUNDSP24Q48ASYM(q_tmp1);
      ae_p24x2s p_out2 = AE_ROUNDSP24Q48ASYM(q_tmp2);

      ae_p24x2s p_out = AE_SELP24_LL(p_out1, p_out2);

      p_out = AE_MAXP24S(p_out, p_min);
      p_out = AE_MINP24S(p_out, p_max);

      out = AE_MOVAP24S_H(p_out);
      *pOut++ = (UWORD8)out;

      out = AE_MOVAP24S_L(p_out);
      *pOut++ = (UWORD8)out;
    }

    if (vec_length & 0x1) {
      int out;

      p_exp = *(ae_p24f *)&pExp[vec_length - 1];

      ae_q56s q_tmp1 = AE_MULFP24S_LL(p_exp, p_recip_sum_exp);

      q_tmp1 = AE_SLAASQ56S(q_tmp1, shift_val);

      ae_p24x2s p_out = AE_ROUNDSP24Q48ASYM(q_tmp1);

      p_out = AE_MAXP24S(p_out, p_min);
      p_out = AE_MINP24S(p_out, p_max);

      out = AE_MOVAP24S_L(p_out);
      *pOut++ = (UWORD8)out;
    }
  }

  return 0;
}

WORD32 xa_nn_vec_softmax_asym8s_8(WORD8 *__restrict__ pOut,
                                  const WORD8 *__restrict__ pVec,
                                  WORD32 diffmin, WORD32 input_beta_left_shift,
                                  WORD32 input_beta_multiplier,
                                  WORD32 vec_length, pVOID pScratch) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(pOut, -1);
  XA_NNLIB_ARG_CHK_PTR(pVec, -1);
  XA_NNLIB_ARG_CHK_PTR(pScratch, -1);
  /* Pointer alignment checks */
  /* No alignment (1-byte) needed for any pointer */
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(
      ((input_beta_left_shift < -31) || (input_beta_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

  int i;
  int shift_bits_reciprocal;
  WORD8 *p_in;
  WORD32 *__restrict pExp = (WORD32 *)ALIGN_PTR(pScratch, ALIGNMENT);
  ae_p24x2s p_recip_sum_exp;
  ae_p24x2s p_x;
  ae_p24x2s p_max = AE_MOVPA24(0xFF800000);

  int pre_loop_count;
  int main_loop_count;
  int post_loop_count;

  if (vec_length > 1) {
    pre_loop_count = (int)pVec & 0x1;
    main_loop_count = vec_length - pre_loop_count;
    post_loop_count = (main_loop_count & 1);
    main_loop_count = main_loop_count >> 1;
  } else {
    pre_loop_count = 0;
    main_loop_count = 0;
    post_loop_count = vec_length;
  }

  /* Calculating Max */
  {
    p_in = (WORD8 *)pVec;

    if (pre_loop_count) {
      p_x = AE_MOVPA24(*p_in++);
      p_max = AE_MAXP24S(p_max, p_x);
    }

    p_in -= 2;
    for (i = 0; i < main_loop_count; i++) {
      AE_LP8X2F_IU(p_x, p_in, 2 * sizeof(WORD8));
      p_max = AE_MAXP24S(p_max, p_x);
    }
    p_max = AE_SRAIP24(p_max, 16);

    if (post_loop_count) {
      p_in += 2;
      p_x = AE_MOVPA24(*p_in);
      p_max = AE_MAXP24S(p_max, p_x);
    }
    p_max = AE_MAXP24S(p_max, AE_SELP24_LH(p_max, p_max));
  }

  /* Calculate exponents */
  {
    ae_q56s q_sum_exp = AE_ZEROQ56();
    ae_p24x2s p_rem_x, p_y, p_exp_y;
    ae_p24x2s p_zero = AE_ZEROP48();
    ae_p24x2s p_input_beta_multiplier =
        AE_MOVPA24((input_beta_multiplier >> 8));
    ae_p24x2s p_diffmin = AE_MOVPA24(diffmin);
    int input_beta_left_shift_for_24bit = input_beta_left_shift - 8;

    p_in = (WORD8 *)pVec;
    WUR_AE_SAR(input_beta_left_shift_for_24bit);

    if (pre_loop_count) {
      p_x = AE_MOVPA24(*p_in++);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);

      EXP_Q26_I(p_exp_y, p_dequantized_y1)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      *(ae_p24f *)&pExp[0] = p_exp_y;

      p_exp_y = p_exp_y >> 4;

      AE_MULAP24S_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }

    p_in -= 2;
    for (i = 0; i < main_loop_count; i++) {
      AE_LP8X2F_IU(p_x, p_in, 2 * sizeof(WORD8));
      p_x = AE_SRAIP24(p_x, 16);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_HH(p_y, p_input_beta_multiplier);
      ae_q56s q_dequantized_y2 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);
      ae_p24x2s p_dequantized_y2 = AE_ROUNDSP24Q48ASYM(q_dequantized_y2);

      ae_p24x2s p_dequantized =
          AE_SELP24_LL(p_dequantized_y1, p_dequantized_y2);

      EXP_Q26_II(p_exp_y, p_dequantized)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      //*(ae_p24x2f *)&pExp[pre_loop_count + 2*i] = p_exp_y;
      *(ae_p24f *)&pExp[pre_loop_count + 2 * i] =
          AE_SELP24_HH(p_exp_y, p_exp_y);
      *(ae_p24f *)&pExp[pre_loop_count + 2 * i + 1] =
          AE_SELP24_LL(p_exp_y, p_exp_y);
      //*(ae_p24f *)&pExp[0] = p_exp_y;

      p_exp_y = p_exp_y >> 4;

      AE_MULAAP24S_HH_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }

    if (post_loop_count) {
      p_in += 2;

      p_x = AE_MOVPA24(*p_in);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);

      EXP_Q26_I(p_exp_y, p_dequantized_y1)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      *(ae_p24f *)&pExp[vec_length - 1] = p_exp_y;

      p_exp_y = p_exp_y >> 4;

      AE_MULAP24S_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }

    p_recip_sum_exp = GetReciprocal(q_sum_exp, 12, &shift_bits_reciprocal);
  }

  /* Calculate output */
  pExp = (WORD32 *)ALIGN_PTR(pScratch, ALIGNMENT);
  {
    ae_p24x2s p_exp;

    int shift_val = -(shift_bits_reciprocal + 31 - 8 - 8);

    ae_p24x2s p_min = AE_MOVPA24(-128);
    ae_p24x2s p_max = AE_MOVPA24(127);

    for (i = 0; i<vec_length >> 1; i++) {
      int out;

      p_exp = *(ae_p24x2f *)&pExp[2 * i];

      ae_q56s q_tmp1 = AE_MULFP24S_HH(p_exp, p_recip_sum_exp);
      ae_q56s q_tmp2 = AE_MULFP24S_LL(p_exp, p_recip_sum_exp);

      q_tmp1 = AE_SLAASQ56S(q_tmp1, shift_val);
      q_tmp2 = AE_SLAASQ56S(q_tmp2, shift_val);

      ae_p24x2s p_out1 = AE_ROUNDSP24Q48ASYM(q_tmp1);
      ae_p24x2s p_out2 = AE_ROUNDSP24Q48ASYM(q_tmp2);

      ae_p24x2s p_out = AE_SELP24_LL(p_out1, p_out2);

      p_out = AE_SUBSP24S(p_out, AE_MOVPA24(128));
      p_out = AE_MAXP24S(p_out, p_min);
      p_out = AE_MINP24S(p_out, p_max);

      out = AE_MOVAP24S_H(p_out);
      *pOut++ = (WORD8)out;

      out = AE_MOVAP24S_L(p_out);
      *pOut++ = (WORD8)out;
    }

    if (vec_length & 0x1) {
      int out;

      p_exp = *(ae_p24f *)&pExp[vec_length - 1];

      ae_q56s q_tmp1 = AE_MULFP24S_LL(p_exp, p_recip_sum_exp);

      q_tmp1 = AE_SLAASQ56S(q_tmp1, shift_val);

      ae_p24x2s p_out = AE_ROUNDSP24Q48ASYM(q_tmp1);

      p_out = AE_SUBSP24S(p_out, AE_MOVPA24(128));
      p_out = AE_MAXP24S(p_out, p_min);
      p_out = AE_MINP24S(p_out, p_max);

      out = AE_MOVAP24S_L(p_out);
      *pOut++ = (WORD8)out;
    }
  }

  return 0;
}

WORD32 xa_nn_vec_softmax_asym8s_16(WORD16 *__restrict__ pOut,
                                   const WORD8 *__restrict__ pVec,
                                   WORD32 diffmin, WORD32 input_beta_left_shift,
                                   WORD32 input_beta_multiplier,
                                   WORD32 vec_length, pVOID pScratch) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(pOut, -1);
  XA_NNLIB_ARG_CHK_PTR(pVec, -1);
  XA_NNLIB_ARG_CHK_PTR(pScratch, -1);
  /* Pointer alignment checks */
  /* No alignment (1-byte) needed for any pointer */
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(
      ((input_beta_left_shift < -31) || (input_beta_left_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((input_beta_multiplier < 0), -1);

  int i;
  int shift_bits_reciprocal;
  WORD8 *p_in;
  WORD32 *__restrict pExp = (WORD32 *)ALIGN_PTR(pScratch, ALIGNMENT);
  ae_p24x2s p_recip_sum_exp;
  ae_p24x2s p_x;
  ae_p24x2s p_max = AE_MOVPA24(0xFF800000);

  int pre_loop_count;
  int main_loop_count;
  int post_loop_count;

  if (vec_length > 1) {
    pre_loop_count = (int)pVec & 0x1;
    main_loop_count = vec_length - pre_loop_count;
    post_loop_count = (main_loop_count & 1);
    main_loop_count = main_loop_count >> 1;
  } else {
    pre_loop_count = 0;
    main_loop_count = 0;
    post_loop_count = vec_length;
  }

  /* Calculating Max */
  {
    p_in = (WORD8 *)pVec;

    if (pre_loop_count) {
      p_x = AE_MOVPA24(*p_in++);
      p_max = AE_MAXP24S(p_max, p_x);
    }

    p_in -= 2;
    for (i = 0; i < main_loop_count; i++) {
      AE_LP8X2F_IU(p_x, p_in, 2 * sizeof(WORD8));
      p_max = AE_MAXP24S(p_max, p_x);
    }
    p_max = AE_SRAIP24(p_max, 16);

    if (post_loop_count) {
      p_in += 2;
      p_x = AE_MOVPA24(*p_in);
      p_max = AE_MAXP24S(p_max, p_x);
    }
    p_max = AE_MAXP24S(p_max, AE_SELP24_LH(p_max, p_max));
  }

  /* Calculate exponents */
  {
    ae_q56s q_sum_exp = AE_ZEROQ56();
    ae_p24x2s p_rem_x, p_y, p_exp_y;
    ae_p24x2s p_zero = AE_ZEROP48();
    ae_p24x2s p_input_beta_multiplier =
        AE_MOVPA24((input_beta_multiplier >> 8));
    ae_p24x2s p_diffmin = AE_MOVPA24(diffmin);
    int input_beta_left_shift_for_24bit = input_beta_left_shift - 8;

    p_in = (WORD8 *)pVec;
    WUR_AE_SAR(input_beta_left_shift_for_24bit);

    if (pre_loop_count) {
      p_x = AE_MOVPA24(*p_in++);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);

      EXP_Q26_I(p_exp_y, p_dequantized_y1)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      *(ae_p24f *)&pExp[0] = p_exp_y;

      p_exp_y = p_exp_y >> 4;

      AE_MULAP24S_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }

    p_in -= 2;
    for (i = 0; i < main_loop_count; i++) {
      AE_LP8X2F_IU(p_x, p_in, 2 * sizeof(WORD8));
      p_x = AE_SRAIP24(p_x, 16);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_HH(p_y, p_input_beta_multiplier);
      ae_q56s q_dequantized_y2 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);
      ae_p24x2s p_dequantized_y2 = AE_ROUNDSP24Q48ASYM(q_dequantized_y2);

      ae_p24x2s p_dequantized =
          AE_SELP24_LL(p_dequantized_y1, p_dequantized_y2);

      EXP_Q26_II(p_exp_y, p_dequantized)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      *(ae_p24f *)&pExp[pre_loop_count + 2 * i] =
          AE_SELP24_HH(p_exp_y, p_exp_y);
      *(ae_p24f *)&pExp[pre_loop_count + 2 * i + 1] =
          AE_SELP24_LL(p_exp_y, p_exp_y);

      p_exp_y = p_exp_y >> 4;

      AE_MULAAP24S_HH_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }

    if (post_loop_count) {
      p_in += 2;

      p_x = AE_MOVPA24(*p_in);
      p_rem_x = p_x - p_max;
      p_y = AE_SLLSSP24S(p_rem_x);

      ae_q56s q_dequantized_y1 = AE_MULFP24S_LL(p_y, p_input_beta_multiplier);

      ae_p24x2s p_dequantized_y1 = AE_ROUNDSP24Q48ASYM(q_dequantized_y1);

      EXP_Q26_I(p_exp_y, p_dequantized_y1)

      xtbool2 flag_cmp = AE_LTP24S(p_rem_x, p_diffmin);
      AE_MOVTP24X2(p_exp_y, p_zero, flag_cmp);

      *(ae_p24f *)&pExp[vec_length - 1] = p_exp_y;

      p_exp_y = p_exp_y >> 4;

      AE_MULAP24S_LL(q_sum_exp, p_exp_y, AE_MOVPA24(1));
    }

    p_recip_sum_exp = GetReciprocal(q_sum_exp, 12, &shift_bits_reciprocal);
  }

  /* Calculate output */
  pExp = (WORD32 *)ALIGN_PTR(pScratch, ALIGNMENT);
  {
    ae_p24x2s p_exp;

    int shift_val = -(shift_bits_reciprocal + 31 - 8 - 16);

    ae_p24x2s p_min = AE_MOVPA24(-32768);
    ae_p24x2s p_max = AE_MOVPA24(32767);

    for (i = 0; i<vec_length >> 1; i++) {
      int out;

      p_exp = *(ae_p24x2f *)&pExp[2 * i];

      ae_q56s q_tmp1 = AE_MULFP24S_HH(p_exp, p_recip_sum_exp);
      ae_q56s q_tmp2 = AE_MULFP24S_LL(p_exp, p_recip_sum_exp);

      q_tmp1 = AE_SLAASQ56S(q_tmp1, shift_val);
      q_tmp2 = AE_SLAASQ56S(q_tmp2, shift_val);

      ae_p24x2s p_out1 = AE_ROUNDSP24Q48ASYM(q_tmp1);
      ae_p24x2s p_out2 = AE_ROUNDSP24Q48ASYM(q_tmp2);

      ae_p24x2s p_out = AE_SELP24_LL(p_out1, p_out2);

      p_out = AE_SUBSP24S(p_out, AE_MOVPA24(32768));
      p_out = AE_MAXP24S(p_out, p_min);
      p_out = AE_MINP24S(p_out, p_max);

      out = AE_MOVAP24S_H(p_out);
      *pOut++ = (WORD16)out;

      out = AE_MOVAP24S_L(p_out);
      *pOut++ = (WORD16)out;
    }

    if (vec_length & 0x1) {
      int out;

      p_exp = *(ae_p24f *)&pExp[vec_length - 1];

      ae_q56s q_tmp1 = AE_MULFP24S_LL(p_exp, p_recip_sum_exp);

      q_tmp1 = AE_SLAASQ56S(q_tmp1, shift_val);

      ae_p24x2s p_out = AE_ROUNDSP24Q48ASYM(q_tmp1);

      p_out = AE_SUBSP24S(p_out, AE_MOVPA24(32768));
      p_out = AE_MAXP24S(p_out, p_min);
      p_out = AE_MINP24S(p_out, p_max);

      out = AE_MOVAP24S_L(p_out);
      *pOut++ = (WORD16)out;
    }
  }

  return 0;
}

int xa_nn_get_softmax_scratch_size(int inp_precision, int out_precision,
                                   int length) {
  int size_of_one_elm_in_bytes, total_bytes;
  (void)out_precision;

  /* This function returns scratch size required by softmax implementation in
     bytes scratch memory is needed to save exponents of inputs computed in the
     function, every exponent is computed as 32 bit (4 bytes) number currently*/
  switch (inp_precision) {
    case PREC_ASYM8U:
      size_of_one_elm_in_bytes = 4;
      break;
    case PREC_SYM8S:
      size_of_one_elm_in_bytes = 4;
      break;
    default:
      size_of_one_elm_in_bytes = 4;
      break;
  }

  total_bytes = size_of_one_elm_in_bytes * length;
  total_bytes = ALIGNED_SIZE(total_bytes, ALIGNMENT);

  return total_bytes;
}
