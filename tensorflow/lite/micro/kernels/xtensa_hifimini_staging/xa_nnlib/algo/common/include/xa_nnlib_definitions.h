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

#ifndef __XA_OPUS_CODEC_DEFINITIONS_H__
#define __XA_OPUS_CODEC_DEFINITIONS_H__

#include "tensorflow/lite/micro/kernels/xtensa_hifimini/xa_nnlib/algo/common/include/xa_api_defs.h"

/* Identification Strings */
#define LIBNAME "HiFi Mini Neural Network Library"
#define LIBVERSION "0.6.0"

#define LIB_APIVERSION_MAJOR 1
#define LIB_APIVERSION_MINOR 0

#if LIB_APIVERSION_MAJOR != XA_APIVERSION_MAJOR || \
    LIB_APIVERSION_MINOR != XA_APIVERSION_MINOR
// #error "Version Mismatch"
#endif

#define LIB_APIVERSION \
  XA_MAKE_VERSION_STR(LIB_APIVERSION_MAJOR, LIB_APIVERSION_MINOR)

#endif /* __XA_OPUS_CODEC_DEFINITIONS_H__ */
