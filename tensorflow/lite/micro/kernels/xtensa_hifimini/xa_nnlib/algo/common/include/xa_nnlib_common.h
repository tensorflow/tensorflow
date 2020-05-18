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
#ifndef __XA_NNLIB_COMMON_H__
#define __XA_NNLIB_COMMON_H__

#include <stddef.h>
#include <inttypes.h>

#include <xtensa/config/core-isa.h>
#include <xtensa/tie/xt_core.h>
#include <xtensa/tie/xt_misc.h>
#include <xtensa/tie/xt_hifi2.h>
#if XCHAL_HAVE_HIFI4_VFPU
  #include <xtensa/tie/xt_FP.h>
#endif

#include "xa_type_def.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_standards.h"
#include "xa_nnlib_kernels_api.h"

#endif /* __XA_NNLIB_COMMON_H__ */
