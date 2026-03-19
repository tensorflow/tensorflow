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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_graph.h"

namespace xla {

// For every kCall operation in the main computation, we inline the body of the
// called function, and proceed recursively.
class CallInliner : public HloModulePass {
 public:
  enum class InlineOverridePolicy {
    kAllowInline,                    // Allow inlining as normal.
    kProhibitInline,                 // Prohibit inlining of this callsite.
    kAllowIgnoreFrontendAttributes,  // Allow even with the 'inlineable'
                                     // frontend attribute is set != 'true'.
  };

  using InlinedInstructionMap =
      absl::flat_hash_map<HloInstruction*, HloInstruction*>;

  // Inlines one call instruction.  Returns a mapping from the original
  // instructions to their inlined versions.
  static absl::StatusOr<InlinedInstructionMap> Inline(HloInstruction* call);

  // If single_call_site is true, only functions with a single call site will be
  // inlined.
  // If update_domain is true, the exit domains could be updated for calls which
  // are being inlined if necessary.
  // If the callback `override_policy` is provided, callsites will be inlined
  // according to the policy returned.
  explicit CallInliner(
      bool single_call_site = false, bool update_domain = false,
      absl::flat_hash_set<std::string> composites_to_preserve = {},
      std::optional<std::function<InlineOverridePolicy(const CallGraph&,
                                                       const HloInstruction*)>>
          override_policy = std::nullopt)
      : single_call_site_(single_call_site),
        update_domain_(update_domain),
        composites_to_preserve_(std::move(composites_to_preserve)),
        override_policy_(std::move(override_policy)) {}
  ~CallInliner() override = default;
  absl::string_view name() const override { return "call-inliner"; }

  absl::StatusOr<bool> RunWithInlineMap(
      HloModule* module, std::optional<InlinedInstructionMap*> inline_map,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Returns true if the instruction is a kCall operation and is eligible for
  // inlining.
  virtual bool IsInlineableCallOp(HloInstruction* instruction) const;

  // Maximum length of an op_name that can be formed during inlining.
  static constexpr int kMaxOpNameSize = 1024;

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> InlineAndLegalize(
      const CallGraph& call_graph, HloComputation* computation,
      absl::Span<HloInstruction* const> instruction_sequence,
      std::optional<InlinedInstructionMap*> inline_map);

  bool ShouldInline(const CallGraph& call_graph,
                    HloInstruction* instruction) const;

  bool single_call_site_;
  bool update_domain_;
  absl::flat_hash_set<std::string> composites_to_preserve_;
  std::optional<std::function<InlineOverridePolicy(const CallGraph& call_graph,
                                                   const HloInstruction*)>>
      override_policy_;
};

// Returns true if the computation has instructions that are inlinable.
bool IsInlineableComputation(HloComputation* computation);

struct InlinedModule {
  std::unique_ptr<HloModule> module;
  std::unique_ptr<HloCloneContext> clone_context;
  CallInliner::InlinedInstructionMap clone_inlined_map;
  const HloInstruction* get_inlined_inst(const HloInstruction* inst);
};

// Given a module, this function first clones the module, then inlines the
// module, and returns the inlined module, clone context and inlined map in
// InlinedModule struct.
absl::StatusOr<InlinedModule> GetInlinedModule(const HloModule* module);

}  // namespace xla

#endif  // XLA_SERVICE_CALL_INLINER_H_
