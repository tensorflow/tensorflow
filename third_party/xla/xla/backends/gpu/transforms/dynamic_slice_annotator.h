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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_ANNOTATOR_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_ANNOTATOR_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Annotates dynamic-slice and dynamic-update-slice instructions with a
// DynamicSliceConfig backend config (offset + stride * iteration). Handles
// both loop-dependent offsets (linear function of a while loop's induction
// variable, stride != 0) and fully static offsets (all constants, stride = 0).
class DynamicSliceAnnotator : public HloModulePass {
 public:
  absl::string_view name() const override { return "dynamic-slice-annotator"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_ANNOTATOR_H_
