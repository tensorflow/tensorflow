/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_H_

#if defined(HIFIMINI)
#include <xtensa/tie/xt_hifi2.h>
#elif defined(FUSION_F1)
#include "include/nnlib/xa_nnlib_api.h"
#include "include/nnlib/xa_nnlib_standards.h"

#define ALIGNED_SIZE(x, bytes) (((x) + (bytes - 1)) & (~(bytes - 1)))
#define ALIGN_PTR(x, bytes) ((((unsigned)(x)) + (bytes - 1)) & (~(bytes - 1)))
#endif

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_H_
