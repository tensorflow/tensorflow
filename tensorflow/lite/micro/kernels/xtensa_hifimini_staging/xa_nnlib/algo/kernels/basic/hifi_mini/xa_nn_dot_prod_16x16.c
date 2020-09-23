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

/*----------------------------Main function---------------------------------*/
WORD32 xa_nn_dot_prod_16x16_asym8s(
    WORD8 *__restrict__ p_out,               /* pointer to output */
    const WORD16 *__restrict__ p_inp1_start, /* pointer to input1 */
    const WORD16 *__restrict__ p_inp2_start, /* pointer to input2 */
    const WORD32 *bias_ptr, WORD32 vec_length, WORD32 out_multiplier,
    WORD32 out_shift, WORD32 out_zero_bias, WORD32 vec_count) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_start, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_start, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_start, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_start, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  int left_shift, right_shift;
  int loopcnt;
  const WORD32 bias_buffer[2] = {0, 0};
  const WORD32 *p_bias_load;
  WORD32 bias_address_increment = sizeof(WORD32);

  if (bias_ptr == NULL) {
    p_bias_load = bias_buffer - 1;
    bias_address_increment = 0;
  } else {
    p_bias_load = bias_ptr - 1;
  }

  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;
  /* inp1 4-bytes aligned, inp2 4-bytes aligned and vec_length is multple of 2
   */
  if (((((unsigned)p_inp1_start) & 0x3) == 0) &&
      ((((unsigned)p_inp2_start) & 0x3) == 0) && ((vec_length & 0x1) == 0)) {
    const ae_p16x2s *pt_inp1, *pt_inp2;
    pt_inp1 = (const ae_p16x2s *)&p_inp1_start[-2];
    pt_inp2 = (const ae_p16x2s *)&p_inp2_start[-2];

    ae_q56s output_int8_max_56 = AE_CVTQ48A32S(127);
    ae_q56s output_int8_min_56 = AE_CVTQ48A32S(-128);
    for (loopcnt = 0; loopcnt < vec_count; loopcnt++) {
      ae_p24x2s dp_inp1, dp_inp2;
      ae_q32s dq_out32;
      ae_q56s dq_out;
      int i;

      AE_LQ32F_XU(dq_out, (ae_q32s *)p_bias_load, bias_address_increment);

      for (i = 0; i < (vec_length >> 1); i++) {
        AE_LP16X2F_IU(dp_inp1, pt_inp1, 4);
        AE_LP16X2F_IU(dp_inp2, pt_inp2, 4);
        AE_MULAAP24S_HH_LL(dq_out, dp_inp1, dp_inp2);
      }

      dq_out32 = AE_SATQ48S(dq_out);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      dq_out = AE_ADDSQ56S(dq_out, AE_CVTQ48A32S(out_zero_bias));

      dq_out = AE_MAXQ56S(dq_out, output_int8_min_56);
      dq_out = AE_MINQ56S(dq_out, output_int8_max_56);
      *p_out++ = (WORD8)AE_TRUNCA32Q48(dq_out);
    }
  } else {
#ifndef DISABLE_NNLIB_UNALIGNED_SUPPORT
    for (loopcnt = 0; loopcnt < vec_count; loopcnt++) {
      ae_p24x2s dp_inp1, dp_inp2;
      ae_q32s dq_out32;
      ae_q56s dq_out;
      int i;
      const WORD16 *p_inp1 = (WORD16 *)&p_inp1_start[loopcnt * vec_length];
      const WORD16 *p_inp2 = (WORD16 *)&p_inp2_start[loopcnt * vec_length];

      AE_LQ32F_XU(dq_out, (ae_q32s *)p_bias_load, bias_address_increment);

      if (((((unsigned)p_inp1) & 3) != 0 && (((unsigned)p_inp2) & 3) != 0) ||
          ((((unsigned)p_inp1) & 3) == 0 && (((unsigned)p_inp2) & 3) == 0)) {
        int pre_loop_count = ((int)(((unsigned)p_inp1) & 3)) >> 1;
        if (pre_loop_count != 0) {
          dp_inp1 = AE_CVTP24A16X2_LL(*p_inp1++, *p_inp2++);
          AE_MULAP24S_HL(dq_out, dp_inp1, dp_inp1);
        }
        const ae_p16x2s *pt_inp1, *pt_inp2;
        pt_inp1 = (const ae_p16x2s *)(p_inp1 - 2);
        pt_inp2 = (const ae_p16x2s *)(p_inp2 - 2);
        for (i = 0; i < (vec_length - pre_loop_count - 1); i += 2) {
          AE_LP16X2F_IU(dp_inp1, pt_inp1, 4);
          AE_LP16X2F_IU(dp_inp2, pt_inp2, 4);
          AE_MULAAP24S_HH_LL(dq_out, dp_inp1, dp_inp2);
        }
        if ((vec_length - pre_loop_count) & 1) {
          dp_inp1 = AE_CVTP24A16X2_LL(p_inp1[i], p_inp2[i]);
          AE_MULAP24S_HL(dq_out, dp_inp1, dp_inp1);
        }
      } else {
        /* One of the pointers in not aligned to 4 bytes, if it is p_inp1, swap
         * them */
        if ((((unsigned)p_inp1) & 3) != 0) {
          const WORD16 *p_tmp;
          p_tmp = p_inp1;
          p_inp1 = p_inp2;
          p_inp2 = p_tmp;
        }
        const ae_p16x2s *pt_inp1 = (const ae_p16x2s *)(p_inp1 - 2);
        const ae_p16s *pt_inp2 = (const ae_p16s *)(p_inp2 - 1);
        for (i = 0; i < (vec_length - 1); i += 2) {
          ae_p24x2s dp_t0, dp_t1;
          AE_LP16X2F_IU(dp_inp1, pt_inp1, 4);
          AE_LP16F_IU(dp_t0, pt_inp2, 2);
          AE_LP16F_IU(dp_t1, pt_inp2, 2);
          dp_inp2 = AE_SELP24_LL(dp_t0, dp_t1);
          AE_MULAAP24S_HH_LL(dq_out, dp_inp1, dp_inp2);
        }
        if (vec_length & 1) {
          dp_inp1 = AE_CVTP24A16X2_LL(p_inp1[i], p_inp2[i]);
          AE_MULAP24S_HL(dq_out, dp_inp1, dp_inp1);
        }
      }
      dq_out32 = AE_SATQ48S(dq_out);
      MULTIPLY_BY_QUANTIZED_MULTIPLIER(dq_out, AE_TRUNCA32Q48(dq_out32),
                                       out_multiplier, left_shift, right_shift);
      dq_out = AE_ADDSQ56S(dq_out, AE_CVTQ48A32S(out_zero_bias));
      WORD32 out_i32 = AE_TRUNCA32Q48(AE_SATQ48S(dq_out));
      out_i32 = out_i32 < -128 ? -128 : out_i32;
      out_i32 = out_i32 > 127 ? 127 : out_i32;
      *p_out++ = (WORD8)out_i32;
    }
#else
    return 1;
#endif
  }
  return 0;
}
