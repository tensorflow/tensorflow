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

#ifndef XLA_SERVICE_CPU_SMALL_REGION_HOISTING_PASS_H_
#define XLA_SERVICE_CPU_SMALL_REGION_HOISTING_PASS_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::cpu {

// Hoists maximal schedule-contiguous runs of "region-eligible" instructions
// into separate functions, tagged `xla_cpu_small_call`, so the thunk emitter
// can compile each run into a single kernel instead of a series of per-op
// thunks. This generalizes SmallWhileLoopHoistingPass: a small while loop is
// just one region shape. It targets the small-model dispatch-floor regression
// (jax #26145, #33666, #37465, #26021) where per-thunk dispatch dominates the
// useful compute.
//
// A maximal region is split at "unavailable" instructions (custom-call,
// infeed/outfeed, scatter, sort, fft, partition/replica-id, custom fusions,
// collectives). A region is hoisted when its aggregate `bytes_accessed` is
// below `small_buffer_access_size` AND it either contains at least
// `min_region_size` instructions or contains a control-flow op (while /
// conditional), whose dispatch cost scales with trip count regardless of
// static instruction count.
class SmallRegionHoistingPass final : public HloModulePass {
 public:
  explicit SmallRegionHoistingPass(int64_t small_buffer_access_size,
                                   int64_t min_region_size = 4,
                                   int64_t max_instruction_count = 2000,
                                   int64_t max_regions_limit = 20);

  absl::string_view name() const final { return "small-region-hoisting"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) final;

 private:
  int64_t small_buffer_access_size_;
  int64_t min_region_size_;
  int64_t max_instruction_count_;
  int64_t max_regions_limit_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_SMALL_REGION_HOISTING_PASS_H_
