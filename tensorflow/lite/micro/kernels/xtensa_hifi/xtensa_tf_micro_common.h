/******************************************************************************
 * Copyright (C) 2019 Cadence Design Systems, Inc.
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

#ifndef __XTENSA_TF_MICRO_COMMON__
#define __XTENSA_TF_MICRO_COMMON__

#include "xa_nnlib_api.h"
#include "xa_nnlib_standards.h"

#define CHECK_ERR_HIFI_NNLIB_KER(ret, err_msg) \
  if (ret != 0) {                              \
    TF_LITE_KERNEL_LOG(context, err_msg);      \
    return kTfLiteError;                       \
  }

#ifndef XTENSA_NNLIB_MAX_SCRATCH_SIZE
#define XTENSA_NNLIB_MAX_SCRATCH_SIZE (70 * 1024)
#endif

#define ALLOCATE_XTENSA_NNLIB_SCRATCH_MEM \
  uint8_t xtensa_nnlib_scratch_buf[XTENSA_NNLIB_MAX_SCRATCH_SIZE];

#define MIN(a, b) (a) < (b) ? (a) : (b);
#define MAX(a, b) (a) > (b) ? (a) : (b);

#define ACTIVATION_MIN_MAX(data_type, out, inp, min, max) \
  {                                                       \
    data_type temp = MAX(inp, min);                       \
    out = MIN(temp, max);                                 \
  }

#define ACTIVATION_MIN_MAX_F32(out, inp, min, max) \
  {                                                \
    float temp = MAX(inp, min);                    \
    out = MIN(temp, max);                          \
  }

#define ACTIVATION_MIN_MAX_ASYM8(out, inp, min, max) \
  {                                                  \
    int32_t temp = MAX((int32_t)inp, min);           \
    out = (uint8_t)MIN(temp, max);                   \
  }

#define ALIGNED_SIZE(x, bytes) (((x) + (bytes - 1)) & (~(bytes - 1)))
#define ALIGN_PTR(x, bytes) ((((unsigned)(x)) + (bytes - 1)) & (~(bytes - 1)))

#endif /* __XTENSA_TF_MICRO_COMMON__ */
