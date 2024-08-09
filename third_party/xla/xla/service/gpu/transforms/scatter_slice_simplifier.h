/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_SCATTER_SLICE_SIMPLIFIER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_SCATTER_SLICE_SIMPLIFIER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Replaces scatters followed by truncation slices with a new scatter using
// a different output shape, and the slices are eliminated.
//
// (a) Single output     (b) Multiple outputs    (c) Elementwise users
//
//   T[N+1] scatter       (T1, T2) scatter         T scatter  T constant
//          v                v       v                  v     v
//     T[N] slice          T1 gte    T2 gte            T maximum
//                           v       v                     v
//                         T1 slice  T2 slice           T slice
//
// This pattern is used when the last element of the scatter output is intended
// to accumulate updates from the input elements that should be ignored.
// This is slow if there are many indices mapped to the last output index and
// the scatter is implemented using atomics, so everything collides on that one
// memory location.
// As OOB scatter indices are dropped by the GPU implementation, we can remove
// the slice step entirely and avoid the memory congestion in the scatter step.

class ScatterSliceSimplifier : public HloModulePass {
 public:
  absl::string_view name() const override { return "scatter-slice-simplifier"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_SCATTER_SLICE_SIMPLIFIER_H_
