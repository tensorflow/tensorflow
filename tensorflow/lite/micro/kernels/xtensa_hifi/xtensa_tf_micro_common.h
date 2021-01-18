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
