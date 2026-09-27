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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_INSTRUCTION_HOISTER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_INSTRUCTION_HOISTER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// HLO pass that hoists parameters, constants, bitcasts and GetTupleElement
// operations to increase opportunities for prefetching.
class InstructionHoister : public HloModulePass {
 public:
  explicit InstructionHoister(bool hoist_parameters = true,
                              bool host_constants = true,
                              bool hoist_bitcasts = false,
                              bool hoist_gtes = false)
      : hoist_parameters_(hoist_parameters),
        host_constants_(host_constants),
        hoist_bitcasts_(hoist_bitcasts),
        hoist_gtes_(hoist_gtes) {}

  ~InstructionHoister() override = default;

  absl::string_view name() const override { return "instruction-hoister"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool hoist_parameters_;
  bool host_constants_;
  bool hoist_bitcasts_;
  bool hoist_gtes_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_INSTRUCTION_HOISTER_H_
