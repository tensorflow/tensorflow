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

// # What is "packing"?
//
// Before feeding data to the gemm kernels (the parts of Ruy that do lots
// of multiply-add operations), Ruy first performs a data transformation (which
// we call "packing") on the input matrices. This transformation has two main
// goals:
// - rearrange data into blocks that are a convenient size/layout for the gemm
// kernels to consume. This helps make the memory access pattern of the gemm
// kernel simpler and more contiguous, and puts the data in a layout most
// convenient for specific arithmetic instructions in the gemm kernel.
// - compute row/column sums needed for handling quantization with non-symmetric
// zero points.
//
// # Simplified algorithmic analysis of packing
//
// Packing is a relatively simple transformation which does a small constant
// amount of work on each element of an input matrix, and hence for an NxM
// matrix performs O(N*M) work. If N and M are of the same order, then this is
// O(N^2) work.
//
// A NxKxM matrix multiplication requires N*K*M multiply-accumulate operations.
// Note that if N, K, and M are all the same order, then the number of
// multiply-accumulate operations is O(N^3).
//
// Thus, the O(N^2) cost of packing is small compared to the O(N^3) work, in the
// case of all dimensions being roughly the same order.
//
// # Packing cost can be significant
//
// When matrix * matrix multiplications begin to look more like matrix * vector
// multiplications, packing cost can become significant. We sometimes call these
// cases "gemv-like".
//
// Continuing the algorithmic analysis above, if we consider a case where an
// NxKxM matrix multiplication has either N = O(1) or M = O(1), then the
// situation is different. In this case, the multiply-accumulate work is only
// quadratic, so the quadratic cost of packing can be come significant.
//
// Another way to say this is that the cost of packing an input matrix (either
// the LHS or RHS) is amortized across the non-depth dimension of the opposite
// input matrix. Thus, when the LHS has very few rows or the RHS has very few
// columns, the cost of packing the opposite input matrix can become
// significant.
//
// As a rough rule of thumb, the cost of packing starts to become significant
// when either N or M is below 32 (and other dimensions are hundreds), with very
// significant packing costs at 8 or below. This varies by data type, Path, and
// tuning, so these numbers are only rough guides.
//
// One practical use case that is affected by this is inference of
// fully connected neural network layers with a low batch size. The weight
// matrix (which is a constant for inference) is the one affected by significant
// packing cost.
//
// Ruy provides an API in ruy_advanced.h for advanced users to pre-pack
// input matrices that are affected by significant packing costs.
//
// # Implementation notes
//
// Ruy's packing routines always operate on a range of columns and can be
// applied to either the LHS or RHS. This is possible because Ruy internally
// implements a TrMul, so the accumulation along depth is done along columns of
// both the LHS and RHS (whereas for a normal Mul the accumulation along depth
// for the LHS is along rows). As another example, we are always computing
// column sums for quantization (and never row sums, since the LHS is
// transposed).

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_

#include "tensorflow/lite/experimental/ruy/platform.h"

// IWYU pragma: begin_exports
#if RUY_PLATFORM(NEON)
#include "tensorflow/lite/experimental/ruy/pack_arm.h"
#elif RUY_PLATFORM(X86)
#include "tensorflow/lite/experimental/ruy/pack_x86.h"
#else
#include "tensorflow/lite/experimental/ruy/pack_common.h"
#endif
// IWYU pragma: end_exports

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PACK_H_
