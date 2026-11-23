/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_TRANSPOSE_KERNELS_INTERNAL_H_
#define XLA_PJRT_TRANSPOSE_KERNELS_INTERNAL_H_

namespace xla {

// Makes sure Highway has already detected the CPU before we look up a function
// pointer with `HWY_DYNAMIC_POINTER_T`. Otherwise `HWY_DYNAMIC_POINTER_T`
// returns Highway's setup wrapper, which would re-check CPU features on every
// call when saved inside a `TransposePlan`.
void EnsureHighwayTargetInitialized();

}  // namespace xla

// Highway targets disabled in `transpose_kernels_interleave.cc` (SVE/SVE2 is
// slower than NEON for deinterleave, and NEON_BF16 / AVX3_SPR only add
// bfloat16/FP16 over NEON / AVX3_DL). Defined as a macro without including
// `detect_targets.h` so `transpose_kernels_interleave.cc` can set
// `HWY_DISABLED_TARGETS` before any Highway header is included.
#define XLA_INTERLEAVE_DISABLED_TARGETS \
  (HWY_ALL_SVE | HWY_NEON_BF16 | HWY_AVX3_SPR)

// Stops the compiler from keeping a separate base pointer per row across the
// outer `count` loop. Without this, x86-64 runs out of integer registers and
// spills loop variables to the stack. We avoid `hwy::PreventElision` here
// because its memory barrier forces the compiler to recompute row offsets
// inside the inner loop.
#if defined(__GNUC__) || defined(__clang__)
#define XLA_HIDE_POINTER_FROM_OPTIMIZER(ptr) asm("" : "+r"(ptr))
#else
#define XLA_HIDE_POINTER_FROM_OPTIMIZER(ptr) ((void)(ptr))
#endif

#endif  // XLA_PJRT_TRANSPOSE_KERNELS_INTERNAL_H_
