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

#define ALIGN_PTR(x, bytes) ((((unsigned)(x)) + (bytes - 1)) & (~(bytes - 1)))

#define LIMIT(out, inp, min, max) \
  {                               \
    out = min;                    \
    out = AE_MAXP24S(inp, min);   \
    out = AE_MINP24S(out, max);   \
  }

#define STORE_8X2_FROM_24X2(out_ptr, val) \
  {                                       \
    int o1, o2;                           \
    o1 = AE_MOVAP24S_H(val);              \
    o2 = AE_MOVAP24S_L(val);              \
    *out_ptr++ = (WORD8)o1;               \
    *out_ptr++ = (WORD8)o2;               \
  }

/*
 * inp: p_vec: 4 byte aligned input pointer
 * out: p_out: no alignment needed for output pointer*/
WORD32 xa_nn_vec_activation_min_max_asym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_vec,
    int activation_min, int activation_max, WORD32 vec_length) {
  int i;
  ae_p24x2s x, y, min, max;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

  WORD8 *p_o = p_out;
  WORD8 *p_v = (WORD8 *)p_vec;

  min = AE_SRAIP24(AE_CVTP24A16(activation_min), 8);
  max = AE_SRAIP24(AE_CVTP24A16(activation_max), 8);

  int pre_loop_count = 0;
  // pre loop, active when input ptr is not 4 byte aligned
  pre_loop_count = (int)((unsigned)ALIGN_PTR(p_v, 4) - (unsigned)p_v);
  pre_loop_count = (pre_loop_count < vec_length) ? pre_loop_count : vec_length;

  vec_length = vec_length - pre_loop_count;
  vec_length = (vec_length < 0) ? 0 : vec_length;

  for (i = 0; i < pre_loop_count; i++) {
    int i1;
    i1 = ((WORD8)*p_v++);
    x = AE_MOVPA24(i1);
    LIMIT(y, x, min, max)
    i1 = AE_MOVAP24S_H(y);
    *p_o++ = (WORD8)i1;
  }

  if ((activation_max >= (int)127) && (activation_min <= (int)-128)) {
    p_v = p_v - 2;
    for (i = 0; i < (vec_length >> 1); i++) {
      AE_LP8X2F_IU(x, (WORD8 *)p_v, 2 * sizeof(WORD8));
      y = AE_SRAIP24(x, 16);

      STORE_8X2_FROM_24X2(p_o, y)
    }
    if (vec_length & 1) {
      p_v = p_v + 2;
      int i1;
      i1 = (WORD8)p_v[0];
      *p_o++ = (WORD8)i1;
    }
  } else if ((activation_max < (int)127) && (activation_min <= (int)-128)) {
    p_v = p_v - 2;
    for (i = 0; i < (vec_length >> 1); i++) {
      AE_LP8X2F_IU(x, (WORD8 *)p_v, 2 * sizeof(WORD8));
      y = AE_SRAIP24(x, 16);

      y = AE_MINP24S(y, max);

      STORE_8X2_FROM_24X2(p_o, y)
    }
    if (vec_length & 1) {
      p_v = p_v + 2;
      int i1;
      i1 = (WORD8)p_v[0];
      y = AE_MOVPA24(i1);

      y = AE_MINP24S(y, max);

      i1 = AE_MOVAP24S_H(y);
      *p_o++ = (WORD8)i1;
    }
  } else if ((activation_max >= (int)127) && (activation_min > (int)-128)) {
    p_v = p_v - 2;
    for (i = 0; i < (vec_length >> 1); i++) {
      AE_LP8X2F_IU(x, (WORD8 *)p_v, 2 * sizeof(WORD8));
      y = AE_SRAIP24(x, 16);

      y = AE_MAXP24S(y, min);

      STORE_8X2_FROM_24X2(p_o, y)
    }
    if (vec_length & 1) {
      p_v = p_v + 2;
      int i1;
      i1 = (WORD8)p_v[0];
      y = AE_MOVPA24(i1);

      y = AE_MAXP24S(y, min);

      i1 = AE_MOVAP24S_H(y);
      *p_o++ = (WORD8)i1;
    }
  } else {
    p_v = p_v - 2;
    for (i = 0; i < (vec_length >> 1); i++) {
      AE_LP8X2F_IU(x, (WORD8 *)p_v, 2 * sizeof(WORD8));
      x = AE_SRAIP24(x, 16);
      LIMIT(y, x, min, max)
      STORE_8X2_FROM_24X2(p_o, y)
    }
    if (vec_length & 1) {
      p_v = p_v + 2;
      int i1;
      i1 = (WORD8)p_v[0];
      x = AE_MOVPA24(i1);
      LIMIT(y, x, min, max)
      i1 = AE_MOVAP24S_H(y);
      *p_o++ = (WORD8)i1;
    }
  }
  return 0;
}
