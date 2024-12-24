/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_CANONICALIZE_ALL_GATHER_FOR_CSE_H_
#define XLA_SERVICE_SPMD_CANONICALIZE_ALL_GATHER_FOR_CSE_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Performs canonicalizations on AllGather for CSE.
class CanonicalizeAllGatherForCSE : public HloModulePass {
 public:
  CanonicalizeAllGatherForCSE() : next_channel_id_(0) {}

  ~CanonicalizeAllGatherForCSE() override = default;
  absl::string_view name() const override { return "canon-all-gather-for-cse"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RunOnComputation(HloComputation* comp);
  int64_t NextChannelId() { return next_channel_id_++; }

  int64_t next_channel_id_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SPMD_CANONICALIZE_ALL_GATHER_FOR_CSE_H_
