/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_DOT_LIBRARY_REWRITER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_DOT_LIBRARY_REWRITER_H_

#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/transforms/library_matcher.h"
#include "xla/backends/cpu/transforms/onednn_matcher.h"
#include "xla/backends/cpu/transforms/xnn_matcher.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "tsl/platform/protobuf.h"

namespace xla::cpu {

enum class FusionDirection {
  kUp,    // Traverse up (to parents).
  kDown,  // Traverse down (to children).
  kBoth,  // Traverse both up and down.
};

struct DotLibraryRewriterOptions {
  bool use_onednn = false;
  bool use_xnnpack = false;
  const tsl::protobuf::RepeatedField<int>* onednn_fusion_types = nullptr;
  const tsl::protobuf::RepeatedField<int>* xnn_fusion_types = nullptr;
};

// Rewrites suitable Dot operations into library fusions.
class DotLibraryRewriter : public HloModulePass {
 public:
  explicit DotLibraryRewriter(
      const TargetMachineFeatures* target_machine_features,
      const DotLibraryRewriterOptions& options)
      : target_machine_features_(target_machine_features),
        options_(std::move(options)) {
    // Initialize library matchers.
    if (options_.use_onednn && options_.onednn_fusion_types != nullptr &&
        !options_.onednn_fusion_types->empty()) {
      libs_.push_back(std::make_unique<OneDnnMatcher>(
          target_machine_features_, options_.onednn_fusion_types));
    }
    if (options_.use_xnnpack && options_.xnn_fusion_types != nullptr &&
        !options_.xnn_fusion_types->empty()) {
      libs_.push_back(std::make_unique<XnnMatcher>(target_machine_features_,
                                                   options_.xnn_fusion_types));
    }
    for (std::unique_ptr<LibraryMatcher>& lib : libs_) {
      supported_ops_.merge(lib->SupportedOps());
    }

    // Check if any library supports each of the fusion types.
    fuse_dot_ =
        absl::c_any_of(libs_, [](const auto& lib) { return lib->fuse_dot(); });
    fuse_eltwise_ = absl::c_any_of(
        libs_, [](const auto& lib) { return lib->fuse_eltwise(); });
  }
  ~DotLibraryRewriter() override = default;

  // Returns the first library matcher that supports the given instruction.
  absl::StatusOr<LibraryMatcher*> ChooseLibrary(HloInstruction* instr);

  // Adds all immediate neighbors (parents and children) of `instr` that are
  // eligible for fusion to `queue`.
  void AddFusionCandidates(
      HloInstruction* fusion, HloInstruction* instr, FusionDirection dir,
      std::queue<std::pair<HloInstruction*, FusionDirection>>& queue);

  // Merges two fusions `main` and `neighbor` together. `main` is the current
  // fusion instruction we are growing. `neighbor` is a neighboring fusion node
  // found through BFS from `main`.
  absl::Status MergeFusionInstructions(HloFusionInstruction* main,
                                       HloFusionInstruction* neighbor,
                                       FusionDirection dir);

  // Fuses `to_fuse` into the fusion `fusion` based on the specified direction.
  // Returns the pointer to the new `to_fuse` node in the fusion region.
  absl::StatusOr<HloInstruction*> GrowFusion(HloFusionInstruction* fusion,
                                             HloInstruction* to_fuse,
                                             FusionDirection dir);

  // Fuses as many neighbors around `fusion` as possible
  absl::Status FuseNeighbors(HloFusionInstruction* fusion, LibraryMatcher* lib);

  // Finds and creates fusions in the given computation.
  absl::StatusOr<bool> ProcessComputation(HloComputation* computation);

  // Runs the pass.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  absl::string_view name() const override { return "dot-library-rewriter"; }

 private:
  const TargetMachineFeatures* target_machine_features_;
  const DotLibraryRewriterOptions options_;
  std::vector<std::unique_ptr<LibraryMatcher>> libs_;
  absl::flat_hash_set<HloOpcode> supported_ops_;
  absl::flat_hash_set<HloInstruction*> fused_;
  bool fuse_dot_ = false;
  bool fuse_eltwise_ = false;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_DOT_LIBRARY_REWRITER_H_
