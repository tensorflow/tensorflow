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

#ifndef XLA_HLO_TOOLS_COMPARISON_COMPARISON_MANAGER_H_
#define XLA_HLO_TOOLS_COMPARISON_COMPARISON_MANAGER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/comparison_manager.pb.h"
#include "xla/hlo/tools/comparison/comparison_options.pb.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"

namespace xla::numerics::comparison {

// Converts a ComparisonVariant enum value to its string representation.
inline absl::string_view ComparisonVariantToString(
    ComparisonOptions::ComparisonVariant variant) {
  switch (variant) {
    case ComparisonOptions::COMPARISON_VARIANT_BASELINE:
      return "baseline";
    case ComparisonOptions::COMPARISON_VARIANT_TARGET:
      return "target";
    default:
      return "baseline";
  }
}

// Abstract base class for the core comparison logic manager.
// This class manages the state of HLO module comparisons, caches incoming
// tensor summaries from baseline and target runs, performs comparisons when
// corresponding tensors are available, and invokes callbacks for results and
// module completion. Concrete implementations must define how comparison
// results and module completion events are handled.
class ComparisonManager {
 public:
  explicit ComparisonManager(bool log_samples,
                             absl::string_view hlo_module_dump_dir)
      : log_samples_(log_samples), hlo_module_dump_dir_(hlo_module_dump_dir) {}
  virtual ~ComparisonManager() = default;

  // Registers an HLO module for comparison, associating it with a specific
  // variant. This should be called for both the baseline and target variants
  // before adding tensors for the respective module. If both variants have been
  // added, the module is marked as "ready".
  //
  // Args:
  //   `variant`: Indicates whether this module is the baseline or target.
  //   `module`: Pointer to the HLO module structure. The manager does not take
  //           ownership.
  virtual absl::Status RegisterHloModule(
      ComparisonOptions::ComparisonVariant variant, const HloModule& module);

  // Blocks the calling thread until both the baseline and target HLO modules
  // with the given name have been registered via `RegisterHloModule`.
  //
  // Args:
  //   `hlo_module_name`: The name of the HLO module to wait for.
  virtual absl::Status WaitUntilHloModuleIsReady(
      absl::string_view hlo_module_name);

  // Registers an HLO module run for comparison, associating it with a specific
  // variant. This should be called for both the baseline and target variants
  // before adding tensors for the respective module.
  //
  // Args:
  //   `variant`: Indicates whether this module is the baseline or target.
  //   `logical_device_id`: The logical device id of the HLO module run.
  //   `run_id`: The run id of the HLO module.
  //   `hlo_module_name`: The name of the HLO module.
  virtual absl::Status RegisterHloModuleRun(
      ComparisonOptions::ComparisonVariant variant, int32_t logical_device_id,
      uint64_t run_id, absl::string_view hlo_module_name);

  // Blocks the calling thread until any HLO module has been added via
  // `RegisterHloModule`.
  virtual absl::Status WaitUntilAnyHloModuleIsRunning();

  // Records a tensor summary from a specific run variant for the given HLO
  // module. If the corresponding tensor summary from the other variant has
  // already been added, this triggers a comparison, and `OnComparisonResult` is
  // called.
  //
  // Args:
  //   `hlo_module_name`: The name of the HLO module this tensor belongs to.
  //   `variant`: Indicates whether this tensor is from the baseline or target
  //      run.
  //   `summary`: The TensorSummary proto containing metadata and potentially
  //      data hashes for the tensor.
  virtual absl::Status RecordTensor(
      ComparisonOptions::ComparisonVariant variant,
      absl::string_view hlo_module_name, const TensorSummary& summary);

  // Marks the run for a specific variant of an HLO module as finished.
  // If both baseline and target variants have been marked as finished,
  // `OnHloModuleFinished` is called.
  //
  // Args:
  //   `variant`: The variant (baseline or target) whose run has finished.
  //   `logical_device_id`: The logical device id of the HLO module run.
  //   `run_id`: The run id of the HLO module.
  //   `module_stats`: The statistics of the module run.
  virtual absl::Status FinishHloModuleRun(
      ComparisonOptions::ComparisonVariant variant, int32_t logical_device_id,
      uint64_t run_id, const ModuleStats& module_stats);

  // Blocks the calling thread until all HLO modules have been marked as
  // finished via `FinishHloModule`.
  virtual absl::Status WaitUntilAllModulesAreFinished();

 protected:
  // Pure virtual callback method invoked when a comparison between a baseline
  // and target tensor is performed. Subclasses must implement this to handle
  // the comparison outcome (e.g., log it, store it, report mismatches).
  //
  // Note on thread safety: This method is called by the comparison manager
  // under a lock, so subclasses don't need to protect against concurrent
  // access.
  //
  // Args:
  //   `hlo_module_name`: The name of the HLO module the compared tensors belong
  //     to.
  //   `result`: A proto containing details about the comparison outcome.
  virtual absl::Status OnComparisonResult(
      absl::string_view hlo_module_name,
      const TensorComparisonResult& result) = 0;

  // Pure virtual callback method invoked when both the baseline and target runs
  // for a specific HLO module have been marked as finished via
  // `FinishHloModule`. Subclasses can implement this to perform cleanup or
  // final reporting for the completed module comparison.
  //
  // Note on thread safety: This method is called by the comparison manager
  // under a lock, so subclasses don't need to protect against concurrent
  // access.
  //
  // Args:
  //   `hlo_module_name`: The name of the HLO module whose comparison is
  //     finished.
  //   `stats`: A proto containing statistics about the comparison.
  virtual absl::Status OnHloModuleFinished(absl::string_view hlo_module_name,
                                           const ComparisonStats& stats) = 0;

 private:
  // Helper struct representing the position of a tensor in the original HLO
  // module. This is used to uniquely identify a tensor across potentially
  // different graph structures. Semantically it's the same as
  // TensorPosition, but we define a separate struct here so that it can
  // be used as a key in an absl::flat_hash_map with gtl::Extend.
  struct TensorPositionStruct {
    TensorPositionStruct(absl::string_view instruction_name,
                         absl::Span<const int64_t> shape_index,
                         int64_t iteration_index)
        : instruction_name(instruction_name),
          shape_index(shape_index.begin(), shape_index.end()),
          iteration_index(iteration_index) {}
    std::string instruction_name;
    absl::InlinedVector<int64_t, 1> shape_index;
    int64_t iteration_index;

    bool operator==(const TensorPositionStruct& other) const {
      return instruction_name == other.instruction_name &&
             shape_index == other.shape_index &&
             iteration_index == other.iteration_index;
    }
    bool operator!=(const TensorPositionStruct& other) const {
      return !(*this == other);
    }
    template <typename H>
    friend H AbslHashValue(H h, const TensorPositionStruct& pos) {
      return H::combine(std::move(h), pos.instruction_name, pos.shape_index,
                        pos.iteration_index);
    }
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const TensorPositionStruct& pos) {
      absl::Format(&sink, "%s@%s#%d", pos.instruction_name,
                   absl::StrJoin(pos.shape_index, ","), pos.iteration_index);
    }
    friend std::ostream& operator<<(std::ostream& os,
                                    const TensorPositionStruct& pos) {
      return os << absl::StrCat(pos);
    }
  };

  // Helper struct representing the key of a tensor summary. This is used to
  // uniquely identify a tensor summary across different runs.
  struct TensorComparisonKey {
    TensorComparisonKey(
        ComparisonOptions::ComparisonVariant variant,
        absl::InlinedVector<TensorPositionStruct, 2>&& positions)
        : variant(variant), positions(std::move(positions)) {}

    TensorComparisonKey(
        ComparisonOptions::ComparisonVariant variant,
        absl::Span<const TensorPositionStruct* const> position_ptrs)
        : variant(variant) {
      positions.reserve(position_ptrs.size());
      for (const TensorPositionStruct* pos_ptr : position_ptrs) {
        positions.push_back(*pos_ptr);
      }
    }

    TensorComparisonKey(ComparisonOptions::ComparisonVariant variant,
                        absl::Span<const TensorPosition* const> position_ptrs)
        : variant(variant) {
      positions.reserve(position_ptrs.size());
      for (const TensorPosition* pos_ptr : position_ptrs) {
        positions.push_back(TensorPositionStruct(pos_ptr->instruction_name(),
                                                 pos_ptr->shape_index(),
                                                 pos_ptr->iteration_index()));
      }
    }

    ComparisonOptions::ComparisonVariant variant;
    absl::InlinedVector<TensorPositionStruct, 2> positions;

    bool operator==(const TensorComparisonKey& other) const {
      return variant == other.variant && positions == other.positions;
    }
    bool operator!=(const TensorComparisonKey& other) const {
      return !(*this == other);
    }
    template <typename H>
    friend H AbslHashValue(H h, const TensorComparisonKey& key) {
      return H::combine(std::move(h), key.variant, key.positions);
    }
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const TensorComparisonKey& key) {
      absl::Format(&sink, "variant=%d positions=[%s]",
                   static_cast<int>(key.variant),
                   absl::StrJoin(key.positions, ", "));
    }
    friend std::ostream& operator<<(std::ostream& os,
                                    const TensorComparisonKey& key) {
      return os << absl::StrCat(key);
    }
  };

  // Helper struct associating an HLO module pointer with its comparison
  // variant.
  struct HloModuleAndVariant {
    const HloModule* module;
    ComparisonOptions::ComparisonVariant variant;

    HloModuleAndVariant() = default;
    HloModuleAndVariant(const HloModule* module,
                        ComparisonOptions::ComparisonVariant variant)
        : module(module), variant(variant) {}
  };

  // Helper struct identifying an HLO module run.
  struct HloRun {
    int32_t logical_device_id;
    uint64_t run_id;
    ComparisonOptions::ComparisonVariant variant;

    bool operator==(const HloRun& other) const {
      return logical_device_id == other.logical_device_id &&
             run_id == other.run_id && variant == other.variant;
    }
    bool operator!=(const HloRun& other) const { return !(*this == other); }
    template <typename H>
    friend H AbslHashValue(H h, const HloRun& run) {
      return H::combine(std::move(h), run.logical_device_id, run.run_id,
                        run.variant);
    }
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const HloRun& run) {
      absl::Format(&sink, "HloRun(logical_device_id=%d, run_id=%d, variant=%d)",
                   run.logical_device_id, run.run_id,
                   static_cast<int>(run.variant));
    }
    friend std::ostream& operator<<(std::ostream& os, const HloRun& run) {
      return os << absl::StrCat(run);
    }
  };

  // Internal struct holding the comparison state for a single HLO module.
  struct HloModuleComparisonState {
    // Maps target instruction name to baseline instruction name.
    absl::flat_hash_map<std::string, std::string> instruction_diff_map_left;
    // Maps baseline instruction name to target instruction name.
    absl::flat_hash_map<std::string, std::string> instruction_diff_map_right;

    // Cache of received tensor summaries, keyed by their TensorComparisonKey.
    absl::flat_hash_map<TensorComparisonKey, TensorSummary> tensor_summaries;

    // Set indicating which variants (baseline, target) have been
    // registered at compile time.
    absl::flat_hash_set<ComparisonOptions::ComparisonVariant> variants;

    // Set indicating which runs have been registered at run
    // time.
    absl::flat_hash_set<HloRun> hlo_runs;

    // Notification triggered when both variants of the HLO module are added.
    std::unique_ptr<absl::Notification> ready_notification =
        std::make_unique<absl::Notification>();

    // Notification triggered when both variants of the HLO module are
    // finished.
    std::unique_ptr<absl::Notification> finish_notification =
        std::make_unique<absl::Notification>();

    ComparisonStats stats;
  };

  // Performs the actual comparison between two tensor summaries.
  TensorComparisonResult CompareTensor(const TensorSummary& baseline,
                                       const TensorSummary& target) const;

  // Given a key for one variant, attempts to find the corresponding key for
  // the other variant using the instruction_diff_bimap. Returns std::nullopt
  // if no corresponding instruction mapping exists.
  std::optional<TensorComparisonKey> FlipKey(
      absl::string_view hlo_module_name, const TensorComparisonKey& key) const;

  // Mutex protecting access to shared state (maps and comparison states).
  absl::Mutex mutex_;

  // Map from HLO module name to its HloModule pointer and variant.
  absl::flat_hash_map<std::string, HloModuleAndVariant> hlo_module_map_
      ABSL_GUARDED_BY(mutex_);
  // Map from HLO module name to its comparison state.
  absl::flat_hash_map<std::string, HloModuleComparisonState> comparison_states_
      ABSL_GUARDED_BY(mutex_);
  // Map from hlo run to HLO module name.
  absl::flat_hash_map<HloRun, std::string> hlo_module_name_by_hlo_run_
      ABSL_GUARDED_BY(mutex_);

  // Notification triggered when all HLO modules have been marked as finished.
  absl::Notification all_modules_finished_notification_;
  // Notification triggered when the first HLO module has been marked as
  // running.
  absl::Notification any_modules_running_notification_;

  // Whether to log samples in the tensor summaries. If false, the samples
  // field will be cleared before storing the tensor summary in the comparison
  // results.
  bool log_samples_;

  // Directory to dump HLO modules to used by the comparison tool.
  std::string hlo_module_dump_dir_;
};

}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_COMPARISON_MANAGER_H_
