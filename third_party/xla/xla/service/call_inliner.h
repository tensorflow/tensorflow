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

#ifndef XLA_SERVICE_CALL_INLINER_H_
#define XLA_SERVICE_CALL_INLINER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// For every kCall operation in the main computation, we inline the body of the
// called function, and proceed recursively.
class CallInliner : public HloModulePass {
 public:
  using InlinedInstructionMap =
      absl::flat_hash_map<HloInstruction*, HloInstruction*>;

  // Inlines one call instruction.  Returns a mapping from the original
  // instructions to their inlined versions.
  static absl::StatusOr<InlinedInstructionMap> Inline(HloInstruction* call);

  // If single_call_site is true, only functions with a single call site will be
  // inlined.
  // If update_domain is true, the exit domains could be updated for calls which
  // are being inlined if necessary.
  explicit CallInliner(bool single_call_site = false,
                       bool update_domain = false)
      : single_call_site_(single_call_site), update_domain_(update_domain) {}
  ~CallInliner() override = default;
  absl::string_view name() const override { return "CallInliner"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Returns true if the instruction is a kCall operation and is eligible for
  // inlining.
  virtual bool IsInlineableCallOp(HloInstruction* instruction) const;

 private:
  bool single_call_site_;
  bool update_domain_;
};

}  // namespace xla

#endif  // XLA_SERVICE_CALL_INLINER_H_
