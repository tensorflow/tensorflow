/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_CONV_CANONICALIZATION_H_
#define XLA_SERVICE_CPU_CONV_CANONICALIZATION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace cpu {

// An HLO pass that canonicalizes the dimension numbers of all top-level
// convolutions in the given module.
//
// In order to hit the fast path of using Eigen's convolution implementation, a
// convolution's dimension numbers need to satisfy certain constraints (so
// called canonical convolutions). This pass expands non-canonical convolutions
// into reshapes and canonical convolutions, so that these non-canonical
// convolutions can run faster.
class ConvCanonicalization : public HloModulePass {
 public:
  explicit ConvCanonicalization(
      const TargetMachineFeatures* target_machine_features)
      : target_machine_features_(*target_machine_features) {}

  ~ConvCanonicalization() override {}
  absl::string_view name() const override {
    return "convolution-canonicalization";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const TargetMachineFeatures& target_machine_features_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CONV_CANONICALIZATION_H_
