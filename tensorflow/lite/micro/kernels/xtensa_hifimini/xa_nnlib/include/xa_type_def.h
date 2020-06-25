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

#ifndef __XA_TYPE_DEF_H__
#define __XA_TYPE_DEF_H__

#include <stdint.h>

/****************************************************************************/
/*     types               type define    prefix        examples      bytes */
/************************  ***********    ******    ****************  ***** */
typedef signed char WORD8;      /* b       WORD8    b_name     1   */
typedef signed char* pWORD8;    /* pb      pWORD8   pb_nmae    1   */
typedef unsigned char UWORD8;   /* ub      UWORD8   ub_count   1   */
typedef unsigned char* pUWORD8; /* pub     pUWORD8  pub_count  1   */

typedef int16_t WORD16;     /* s       WORD16   s_count    2   */
typedef int16_t* pWORD16;   /* ps      pWORD16  ps_count   2   */
typedef uint16_t UWORD16;   /* us      UWORD16  us_count   2   */
typedef uint16_t* pUWORD16; /* pus     pUWORD16 pus_count  2   */

typedef signed int WORD24;      /* k       WORD24   k_count    3   */
typedef signed int* pWORD24;    /* pk      pWORD24  pk_count   3   */
typedef unsigned int UWORD24;   /* uk      UWORD24  uk_count   3   */
typedef unsigned int* pUWORD24; /* puk     pUWORD24 puk_count  3   */

typedef signed int WORD32;      /* i       WORD32   i_count    4   */
typedef signed int* pWORD32;    /* pi      pWORD32  pi_count   4   */
typedef unsigned int UWORD32;   /* ui      UWORD32  ui_count   4   */
typedef unsigned int* pUWORD32; /* pui     pUWORD32 pui_count  4   */

typedef int64_t WORD40;     /* m       WORD40   m_count    5   */
typedef int64_t* pWORD40;   /* pm      pWORD40  pm_count   5   */
typedef uint64_t UWORD40;   /* um      UWORD40  um_count   5   */
typedef uint64_t* pUWORD40; /* pum     pUWORD40 pum_count  5   */

typedef int64_t WORD64;      /* h       WORD64   h_count    8   */
typedef int64_t* pWORD64;    /* ph      pWORD64  ph_count   8   */
typedef uint64_t UWORD64;    /* uh      UWORD64  uh_count   8   */
typedef uint64_t* pUWORD64;  /* puh     pUWORD64 puh_count  8   */

typedef float FLOAT32;    /* f       FLOAT32  f_count    4   */
typedef float* pFLOAT32;  /* pf      pFLOAT32 pf_count   4   */
typedef double FLOAT64;   /* d       UFLOAT64 d_count    8   */
typedef double* pFlOAT64; /* pd      pFLOAT64 pd_count   8   */

typedef void VOID;   /* v       VOID     v_flag     4   */
typedef void* pVOID; /* pv      pVOID    pv_flag    4   */

/* variable size types: platform optimized implementation */
typedef signed int BOOL;       /* bool    BOOL     bool_true      */
typedef unsigned int UBOOL;    /* ubool   BOOL     ubool_true     */
typedef signed int FLAG;       /* flag    FLAG     flag_false     */
typedef unsigned int UFLAG;    /* uflag   FLAG     uflag_false    */
typedef signed int LOOPIDX;    /* lp      LOOPIDX  lp_index       */
typedef unsigned int ULOOPIDX; /* ulp     SLOOPIDX ulp_index      */
typedef signed int WORD;       /* lp      LOOPIDX  lp_index       */
typedef unsigned int UWORD;    /* ulp     SLOOPIDX ulp_index      */

typedef LOOPIDX LOOPINDEX;   /* lp    LOOPIDX  lp_index       */
typedef ULOOPIDX ULOOPINDEX; /* ulp   SLOOPIDX ulp_index      */

#define PLATFORM_INLINE __inline

typedef struct xa_codec_opaque {
  WORD32 _;
} * xa_codec_handle_t;

typedef int XA_ERRORCODE;

typedef XA_ERRORCODE xa_codec_func_t(xa_codec_handle_t p_xa_module_obj,
                                     WORD32 i_cmd, WORD32 i_idx,
                                     pVOID pv_value);

#endif /* __XA_TYPE_DEF_H__ */
