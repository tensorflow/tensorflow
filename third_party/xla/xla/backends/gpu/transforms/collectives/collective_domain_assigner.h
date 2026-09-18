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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_DOMAIN_ASSIGNER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_DOMAIN_ASSIGNER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class GpuTopology;

namespace gpu {

// Assigns the SCALE_UP_FABRIC communication domain to collectives whose every
// participant group is contained in a contiguous range of global devices. The
// range size is the fast-interconnect partition size from GpuTopology.
class CollectiveDomainAssigner : public HloModulePass {
 public:
  explicit CollectiveDomainAssigner(const GpuTopology& gpu_topology);

  absl::string_view name() const override {
    return "collective-domain-assigner";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const GpuTopology& gpu_topology_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_DOMAIN_ASSIGNER_H_
