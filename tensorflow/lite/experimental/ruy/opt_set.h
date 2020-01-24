/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_OPT_SET_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_OPT_SET_H_

// RUY_OPT_SET is a compile-time API that Ruy provides for enabling/disabling
// certain optimizations. It should be used by defining that macro on the
// compiler command line.
//
// Each bit in RUY_OPT_SET controls a particular optimization done in Ruy.
#define RUY_OPT_INTRINSICS 0x1
#define RUY_OPT_ASM 0x2
#define RUY_OPT_TUNING 0x4
#define RUY_OPT_FAT_KERNEL 0x8
#define RUY_OPT_NATIVE_ROUNDING 0x10
#define RUY_OPT_FRACTAL 0x20
#define RUY_OPT_FRACTAL_U 0x40
#define RUY_OPT_AVOID_ALIASING 0x80
#define RUY_OPT_MAX_STREAMING 0x100
#define RUY_OPT_PACK_AHEAD 0x200
#define RUY_OPT_PREFETCH_LOAD 0x400

#if !defined(RUY_OPT_SET)
#ifdef RUY_OPTIMIZE_FOR_MATMUL_BENCHMARK
#define RUY_OPT_SET (~RUY_OPT_PREFETCH_LOAD)
#else
// Default to all optimizations.
#define RUY_OPT_SET (~0)
#endif
#endif

#define RUY_OPT_ENABLED(ruy_opt) ((RUY_OPT_SET & ruy_opt) != 0)

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_OPT_SET_H_
