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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRMUL_PARAMS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRMUL_PARAMS_H_

#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

using RunKernelFn = void(Tuning, const SidePair<PMatrix>&, void*,
                         const SidePair<int>&, const SidePair<int>&, DMatrix*);

using RunPackFn = void(Tuning, const DMatrix&, PMatrix*, int, int);

// Type-erased data needed for implementing TrMul.
struct TrMulParams {
  TrMulParams() : run_pack{nullptr, nullptr}, is_prepacked{false, false} {}
  // Helper functions for invoking the function pointers.
  void RunPack(Side side, Tuning tuning, int start, int end) {
    run_pack[side](tuning, src[side], &packed[side], start, end);
  }
  void RunKernel(Tuning tuning, const SidePair<int>& start,
                 const SidePair<int>& end) {
    run_kernel(tuning, packed, spec, start, end, &dst);
  }

  // path id, can be useful info for some fine-tuning, e.g. to guess reasonable
  // cache sizes when not runtime-detectable.
  Path path;

  // optional overriding by the spec for testing purposes
  int cache_friendly_traversal_threshold = 0;

  // Function pointers to type-erased entry points for kernels and packers.
  SidePair<RunPackFn*> run_pack;
  RunKernelFn* run_kernel = nullptr;

  // Matrices and packed matrices.
  SidePair<DMatrix> src;
  DMatrix dst;
  SidePair<PMatrix> packed;
  SidePair<bool> is_prepacked;

  // Type-erased Spec.
  void* spec = nullptr;
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TRMUL_PARAMS_H_
