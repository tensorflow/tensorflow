/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SPMD_PREPARE_H_
#define XLA_SERVICE_SPMD_SPMD_PREPARE_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace spmd {

// Performs preparation steps for better SPMD partitioning of ops.
// This is organized as a separate pass so it can be interleaved with other
// optimizations over sharded ops or shardings.
class SpmdPrepare : public HloModulePass {
 public:
  explicit SpmdPrepare() = default;

  ~SpmdPrepare() override = default;
  absl::string_view name() const override { return "spmd-prepare"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace spmd
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SPMD_PREPARE_H_
