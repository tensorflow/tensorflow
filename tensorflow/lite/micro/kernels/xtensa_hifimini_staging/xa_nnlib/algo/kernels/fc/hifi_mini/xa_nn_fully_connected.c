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

#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_type_def.h"

WORD32 xa_nn_fully_connected_asym8uxasym8u_asym8u(
    UWORD8 *__restrict__ p_out, const UWORD8 *__restrict__ p_weight,
    const UWORD8 *__restrict__ p_inp, const WORD32 *__restrict__ p_bias,
    WORD32 weight_depth, WORD32 out_depth, WORD32 input_zero_bias,
    WORD32 weight_zero_bias, WORD32 out_multiplier, WORD32 out_shift,
    WORD32 out_zero_bias) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -255 || input_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((weight_zero_bias < -255 || weight_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < 0 || out_zero_bias > 255), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_out_stride_asym8uxasym8u_asym8u(
      p_out, p_weight, p_inp, p_bias, out_depth /* rows */
      ,
      weight_depth /* cols */
      ,
      weight_depth /* row_stride */
      ,
      1 /* out_stride */
      ,
      weight_zero_bias, input_zero_bias, out_multiplier, out_shift,
      out_zero_bias);
  return ret;
}

WORD32 xa_nn_fully_connected_sym8sxasym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_weight,
    const WORD8 *__restrict__ p_inp, const WORD32 *__restrict__ p_bias,
    WORD32 weight_depth, WORD32 out_depth, WORD32 input_zero_bias,
    WORD32 out_multiplier, WORD32 out_shift, WORD32 out_zero_bias) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_out_stride_sym8sxasym8s_asym8s(
      p_out, p_weight, p_inp, p_bias, out_depth /* rows */
      ,
      weight_depth /* cols */
      ,
      weight_depth /* row_stride */
      ,
      1 /* out_stride */
      ,
      input_zero_bias, out_multiplier, out_shift, out_zero_bias);
  return ret;
}

WORD32 xa_nn_fully_connected_asym8sxasym8s_asym8s(
    WORD8 *__restrict__ p_out, const WORD8 *__restrict__ p_weight,
    const WORD8 *__restrict__ p_inp, const WORD32 *__restrict__ p_bias,
    WORD32 weight_depth, WORD32 out_depth, WORD32 weight_zero_bias,
    WORD32 input_zero_bias, WORD32 out_multiplier, WORD32 out_shift,
    WORD32 out_zero_bias) {
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((weight_zero_bias < -127 || weight_zero_bias > 128),
                        -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_out_stride_asym8sxasym8s_asym8s(
      p_out, p_weight, p_inp, p_bias, out_depth /* rows */
      ,
      weight_depth /* cols */
      ,
      weight_depth /* row_stride */
      ,
      1 /* out_stride */
      ,
      weight_zero_bias, input_zero_bias, out_multiplier, out_shift,
      out_zero_bias);
  return ret;
}
