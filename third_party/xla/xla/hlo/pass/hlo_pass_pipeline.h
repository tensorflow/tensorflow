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

#ifndef XLA_HLO_PASS_HLO_PASS_PIPELINE_H_
#define XLA_HLO_PASS_HLO_PASS_PIPELINE_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/compilation_stats.h"
#include "xla/types.h"

namespace xla {

class PhaseOrderPipeline;

// Pipeline of HLO passes.
class HloPassPipeline : public HloPassInterface {
 public:
  explicit HloPassPipeline(const std::string& name,
                           CompilationStats* compilation_stats = nullptr)
      : name_(name), compilation_stats_(compilation_stats) {
    if (compilation_stats == nullptr) {
      empty_compilation_stats_ = CompilationStats::MakeNoopStats();
      compilation_stats_ = empty_compilation_stats_.get();
    }
  }
  absl::string_view name() const override { return name_; }

  // Add a pass to the pipeline. It should be called with the arguments for the
  // pass constructor:
  //
  //   pipeline.AddPass<FooPass>(constructor_arg1, constructor_arg2);
  //
  // Returns a reference to the added pass.
  template <typename T, typename... Args>
  T& AddPass(Args&&... args) {
    CHECK(!run_called_) << "AddPass cannot be called after Run";
    auto pass = new T(std::forward<Args>(args)...);
    passes_.push_back(std::unique_ptr<T>(pass));
    return *pass;
  }

  // Add an invariant-checking pass to the pipeline. It will be run before and
  // after each HLO pass. The invariant checking pass must not mutate the graph
  // (it is required to always return "false" from its Run() method).
  template <typename T, typename... Args>
  T& AddInvariantChecker(Args&&... args) {
    CHECK(!run_called_) << "AddInvariantChecker cannot be called after Run";
    auto pass = new T(std::forward<Args>(args)...);
    invariant_checkers_.push_back(std::unique_ptr<T>(pass));
    return *pass;
  }

  // Add an invariant-checking pass to the pipeline on debug builds only.
  template <typename T, typename... Args>
  void AddInvariantCheckerDebug(Args&&... args) {
#ifndef NDEBUG
    AddInvariantChecker<T>(std::forward<Args>(args)...);
#endif  // NDEBUG
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
  using HloPassInterface::RunOnModuleGroup;
  absl::StatusOr<bool> RunOnModuleGroup(
      HloModuleGroup* module_group,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  bool IsPassPipeline() const override { return true; }

  // Return size of passes_.
  int PassesSize() { return passes_.size(); }
  // Return reference to pass specified by index.
  HloPassInterface& GetPass(int index) { return *passes_[index]; }

 private:
  // Returns the set of passes which are enabled. DebugOptions can selectively
  // disable passes via --xla_disable_hlo_passes flag.
  std::vector<HloPassInterface*> GetEnabledPasses(
      const DebugOptions& debug_options);

  // Maybe dumps the given module or module group depending on flag values
  // contained in DebugOptions of module config. If it is dumped, saves the
  // filenames of the dumps into module metadata.
  void MaybeDumpHloAndSaveFilenames(HloModuleGroup& module_group,
                                    absl::string_view after_pass_name,
                                    absl::string_view before_pass_name);
  void MaybeDumpHloAndSaveFilenames(HloModule& module,
                                    absl::string_view after_pass_name,
                                    absl::string_view before_pass_name);

  // Runs the invariant checker on the given HLO for specified
  // `execution_threads`. Empty `execution_threads` means all execution threads
  // are included. HloT can be either HloModule or HloModuleGroup.
  template <typename HloT>
  absl::Status RunInvariantCheckers(HloT* hlo,
                                    absl::string_view after_pass_name) {
    return RunInvariantCheckers(hlo, after_pass_name, /*execution_threads=*/{});
  }
  template <typename HloT>
  absl::Status RunInvariantCheckers(
      HloT* hlo, absl::string_view after_pass_name,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Helper which runs the given pass on the given HLO. HloT can be either
  // HloModule or HloModuleGroup.
  template <typename HloT>
  absl::StatusOr<bool> RunPassesInternal(
      HloT* hlo, const DebugOptions& debug_options,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Helpers which run the given passes on the given HLO construct. Only
  // computations with specified `execution_threads` are considered by the pass,
  // empty thread list means all `execution_threads` are considered. These
  // helpers enable templating of the core of the pipeline logic by providing
  // HloModule and HloModuleGroup specific methods with the same name.
  static absl::StatusOr<bool> RunHelper(
      HloPassInterface* pass, HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {
    TF_ASSIGN_OR_RETURN(bool changed, pass->Run(module, execution_threads));
    module->Cleanup();
    return changed;
  }
  static absl::StatusOr<bool> RunHelper(
      HloPassInterface* pass, HloModuleGroup* module_group,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {
    TF_ASSIGN_OR_RETURN(
        bool changed, pass->RunOnModuleGroup(module_group, execution_threads));
    module_group->Cleanup();
    return changed;
  }

  const std::string name_;
  std::vector<std::unique_ptr<HloPassInterface>> passes_;
  std::vector<std::unique_ptr<HloPassInterface>> invariant_checkers_;
  bool run_called_ = false;

  CompilationStats* compilation_stats_;
  // Default stats instance for when one is not passed in the constructor.
  // Use via compilation_stats_, not directly.
  std::unique_ptr<CompilationStats> empty_compilation_stats_;

  // Allow PhaseOrderPipeline to modify private passes_ member in order to
  // perform PhaseOrdering.
  friend class ::xla::PhaseOrderPipeline;
};

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PASS_PIPELINE_H_
