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

#ifndef XLA_HLO_IR_HLO_MODULE_H_
#define XLA_HLO_IR_HLO_MODULE_H_

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dynamic_parameter_binding.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/iterator_util.h"
#include "xla/printer.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/name_uniquer.h"
#include "xla/xla.pb.h"
#include "tsl/lib/gtl/iterator_range.h"
#include "tsl/platform/logging.h"

namespace xla {

using LayoutCanonicalizationCallback =
    std::function<absl::StatusOr<std::pair<std::vector<Shape>, Shape>>(
        const HloModule& module)>;

// Helper class to maintain a copy-on-write storage of an object of the
// specified type. Logically Variant<MutableOwned, ImmutableShared>.
template <typename T>
class CopyOnWrite {
 public:
  static_assert(!std::is_const_v<T>);
  explicit CopyOnWrite(
      std::variant<std::unique_ptr<T>, std::shared_ptr<const T>> ptr)
      : ownership_(std::move(ptr)), ptr_([&]() -> decltype(ptr_) {
          if (auto* owned = std::get_if<std::unique_ptr<T>>(&ownership_)) {
            return owned->get();
          }
          return std::get<std::shared_ptr<const T>>(ownership_).get();
        }()) {}

  // Obtains a const reference to the read-only copy of the object, could be
  // sharing the storage with other CopyOnWrite<T> instances.
  const T& get() const { return *ptr_; }

  // Obtains a mutable reference to an exclusively owned copy of the object. If
  // the object was sharing storage with other CopyOnWrite<T> instances, make a
  // deep copy inline and transform into exclusively owned copy.
  T& get_mutable() {
    if (auto* owned = std::get_if<std::unique_ptr<T>>(&ownership_)) {
      return **owned;
    }
    auto& shared = std::get<std::shared_ptr<const T>>(ownership_);
    DeepCopyToNewUnique(T(*shared));
    return const_cast<T&>(*ptr_);
  }
  // Deep copies the provided value into an exclusively owned copy of the
  // object.
  void set(T&& value) {
    if (auto* owned = std::get_if<std::unique_ptr<T>>(&ownership_)) {
      **owned = std::forward<T>(value);
    } else {
      DeepCopyToNewUnique(std::forward<T>(value));
    }
  }
  // If the instance is in MutableOwned state, move the storage into
  // ImmutableShared state.
  // If the instance is in ImmutableShared state, returns the shared storage.
  const std::shared_ptr<const T>& FreezeAndShare() const {
    if (auto* owned = std::get_if<std::unique_ptr<T>>(&ownership_)) {
      ownership_ = std::shared_ptr<const T>(std::move(*owned));
    }
    return std::get<std::shared_ptr<const T>>(ownership_);
  }

 private:
  void DeepCopyToNewUnique(T&& value) {
    auto owned = std::make_unique<T>(std::forward<T>(value));
    ptr_ = owned.get();
    ownership_ = std::move(owned);
  }
  mutable std::variant<std::unique_ptr<T>, std::shared_ptr<const T>> ownership_;
  const T* ptr_;
};

// Describes a compilation unit at the HLO level.
//
// HloModule is the top-level unit in the HLO IR.  It corresponds to a whole
// "program".  Running a module, from beginning to end, is the only way to run
// an XLA program.
//
// A module contains one "entry computation"; this HloComputation is like main()
// in a C program.  The result of running the module is the result of running
// this computation.
//
// A module also contains some number of "nested computations".  Each nested
// computation is attached to an HloInstruction within some other computation.
// The meaning of the nested computation depends on the instruction it's
// attached to.
class HloModule {
 public:
  // Constructor.
  HloModule(const std::string& name, HloModuleConfig config);
  // REQUIRED:
  // - comp_envs must not be null.
  HloModule(const std::string& name, HloModuleConfig config,
            std::unique_ptr<CompilationEnvironments> comp_envs);
  HloModule(const std::string& name,
            std::variant<std::unique_ptr<HloModuleConfig>,
                         std::shared_ptr<const HloModuleConfig>>
                config,
            std::unique_ptr<CompilationEnvironments> comp_envs);
  virtual ~HloModule() = default;

  // Adds an entry computation to the module. A module can only have one entry
  // computation. Returns a pointer to the newly added computation.
  HloComputation* AddEntryComputation(
      std::unique_ptr<HloComputation> computation);

  // Same as the AddEntryComputation function above but the module's
  // entry_computation_layout is updated to match the layout of the new entry
  // computation.
  HloComputation* AddEntryComputationWithLayouts(
      std::unique_ptr<HloComputation> computation);

  // Replaces the current entry computation with another computation.
  // The new entry computation must be a computation that is already in the
  // module.
  void ReplaceEntryComputation(HloComputation* entry_computation);

  // Adds an embedded computation to the module.
  HloComputation* AddEmbeddedComputation(
      std::unique_ptr<HloComputation> computation);

  // Removes an embedded computation.
  Status RemoveEmbeddedComputation(HloComputation* to_remove);

  // Removes unused computations.
  Status RemoveUnusedComputations();

  // Marks duplicate fusions with the same name to be able to group them for
  // analysis purposes (e.g. through Xprof).
  void MarkFusionDuplications(
      const absl::flat_hash_map<HloComputation*, HloComputation*>&
          replacements);

  // Replaces all uses of computations that are keys of 'replacements' with
  // the corresponding values in 'replacements'. Replaces the entry computation,
  // if applicable.
  //
  // This function iterates over all instructions in the module to find
  // computations to replace. We could speed it up by keeping track of users of
  // computations.
  //
  // N.B.: This function does not update the computations_ field of the
  // HloModule with the newly added compututations. Therefore, along with
  // invoking this function, if a replacement computation is not already present
  // in module, it should be separately added into the module using
  // `AddEmbeddedComputation`.
  void ReplaceComputations(
      const absl::flat_hash_map<HloComputation*, HloComputation*>&
          replacements);

  const std::string& name() const { return name_; }
  void set_name(std::string name) { name_ = std::move(name); }

  // Move computations from the input module to this one, while ensuring that
  // the names of instructions within the computations are unchanged.
  void MoveComputationsFrom(HloModule* module, bool make_names_unique = false);

  // Returns a deep copy of this module including all computations.
  std::unique_ptr<HloModule> Clone(const std::string& suffix = "clone") const;
  std::unique_ptr<HloModule> Clone(const HloModuleConfig& config,
                                   const std::string& suffix = "clone") const;
  std::unique_ptr<HloModule> Clone(
      std::shared_ptr<const HloModuleConfig> config,
      const std::string& suffix = "clone") const;

  // Performs a deep clone of the computation, by recursively cloning all
  // the called computations as well. If the clone context is specified, it
  // will be populated with the cloned object mappings.
  HloComputation* DeepCloneComputation(HloComputation* computation,
                                       HloCloneContext* context = nullptr);

  // Return a pointer to the entry computation of the module.
  HloComputation* entry_computation() const {
    CHECK_NE(nullptr, entry_computation_);
    return entry_computation_;
  }

  bool has_entry_computation() const { return entry_computation_ != nullptr; }

  // Returns the root instruction shape of entry computation.
  //
  // Precondition: entry_computation_ is not nullptr.
  const Shape& result_shape() const {
    CHECK_NE(nullptr, entry_computation_);
    return entry_computation()->root_instruction()->shape();
  }

  // Creates the ComputationLayout which describes the current status of the HLO
  // module entry computation.
  ComputationLayout compute_computation_layout() const {
    return ComputationLayout(entry_computation()->ComputeProgramShape(),
                             /*ignore_layouts=*/false);
  }

  ComputationLayout* mutable_entry_computation_layout() {
    return config_.get_mutable().mutable_entry_computation_layout();
  }

  const ComputationLayout& entry_computation_layout() const {
    return config_.get().entry_computation_layout();
  }

  void set_frontend_attributes(FrontendAttributes frontend_attributes) {
    frontend_attributes_ = std::move(frontend_attributes);
  }

  void add_frontend_attributes(FrontendAttributes frontend_attributes) {
    frontend_attributes_.mutable_map()->insert(
        frontend_attributes.map().begin(), frontend_attributes.map().end());
  }

  const FrontendAttributes& frontend_attributes() const {
    return frontend_attributes_;
  }

  void set_use_auto_spmd_partitioning(bool use) {
    use_auto_spmd_partitioning_ = use;
  }

  bool use_auto_spmd_partitioning() const {
    return use_auto_spmd_partitioning_;
  }

  // Based on module's entry_computation sharded shapes,
  // layout_canonicalization_callback_ computes and
  // returns <argument_layouts, result_layout> for module's entry computation.
  // argument_layouts is std::vector<Shape> and results_layout is Shape.
  // layout_canonicalization_callback_ is used only when
  // use_auto_spmd_partitioning_ = true.
  void set_layout_canonicalization_callback(
      LayoutCanonicalizationCallback callback) {
    layout_canonicalization_callback_ = std::move(callback);
  }

  LayoutCanonicalizationCallback layout_canonicalization_callback() const {
    return layout_canonicalization_callback_;
  }

  // Generates a hash value of an HLO module. Hash considers
  // information on opcode, shape, operands, and typically a root instruction.
  // This function returns the same hash value for equivalent HLO modules,
  // with respect to HloInstruction::Identical() method.
  template <typename H>
  friend H AbslHashValue(H h, const HloModule& module) {
    h = H::combine(std::move(h), module.entry_computation_layout());
    // Use MakeComputationSorted() instead of MakeComputationPostOrder()
    // because naming may affect the order of MakeComputationPostOrder() but not
    // MakeComputationSorted().
    auto computations = module.MakeComputationSorted();
    for (auto* computation : computations) {
      h = H::combine(std::move(h), *computation);
    }
    return H::combine(std::move(h), computations.size());
  }

  // Gets the computations in this module.
  //
  // Returns a view of HloComputation*s, so you can iterate over this in the
  // natural way:
  //
  //   for (HloComputation* c : module->computations()) { ... }
  //
  tsl::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::const_iterator>>
  computations() const {
    return {MakeUnwrappingIterator(computations_.begin()),
            MakeUnwrappingIterator(computations_.end())};
  }
  tsl::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::iterator>>
  computations() {
    return {MakeUnwrappingIterator(computations_.begin()),
            MakeUnwrappingIterator(computations_.end())};
  }

  // Similar as above, but return a filtered view of computations for specified
  // `execution_threads`. Empty `execution_threads` list means all execution
  // threads are included.
  tsl::gtl::iterator_range<FilteringUnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::const_iterator,
      std::function<bool(const HloComputation*)>>>
  computations(
      const absl::flat_hash_set<absl::string_view>& execution_threads) const {
    // Pass execution_threads by value to the predicate to ensure it lives
    // beyond this function.
    std::function<bool(const HloComputation*)> pred =
        [execution_threads](const HloComputation* computation) {
          if (execution_threads.empty()) {
            return true;
          }
          return execution_threads.contains(computation->execution_thread());
        };
    return MakeFilteringUnwrappingIteratorRange(computations_.begin(),
                                                computations_.end(), pred);
  }

  // Returns the computation in this module that has the name `name`.  Returns
  // null if there is no such computation.
  HloComputation* GetComputationWithName(absl::string_view name);

  // Gets the number of computations in this module.
  int64_t computation_count() const { return computations_.size(); }

  // Returns the mutable computation for the given index.
  HloComputation* mutable_computation(int64_t idx) {
    CHECK(idx >= 0 && idx < computations_.size());
    return computations_[idx].get();
  }

  // Gets the number of instructions in this module.
  int64_t instruction_count() const;

  // Deallocate removed instructions in each computation.
  void Cleanup() {
    for (auto& comp : computations_) {
      comp->Cleanup();
    }
  }

  // Compute and return a post order of all computations in the module. The sort
  // is defined like so: if computation A has an instruction which calls
  // computation B, then A will appear after B in the sort.
  std::vector<HloComputation*> MakeComputationPostOrder() const {
    return MakeComputationPostOrder({});
  }
  // Similar as above but only returns computations with specified
  // `execution_threads`. Empty `execution_threads` list means all execution
  // threads are included.
  std::vector<HloComputation*> MakeComputationPostOrder(
      const absl::flat_hash_set<absl::string_view>& execution_threads) const;
  // Same as MakeComputationPostOrder() but only returns the computations that
  // are on specified `execution_threads` and are also found in the passed in
  // allowList. Empty `execution_threads` list means all execution threads are
  // included.
  std::vector<HloComputation*> MakeComputationPostOrder(
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      const absl::flat_hash_set<HloComputation*>& allow_list) const;

  // If config().content_aware_computation_sorting() is true, sorts computations
  // by their contents, otherwise returns MakeComputationPostOrder().
  std::vector<HloComputation*> MakeComputationSorted() const {
    return MakeComputationSorted({});
  }
  // Same as above but only for specified `execution_threads`. Empty
  // `execution_threads` list means all execution threads are included.
  std::vector<HloComputation*> MakeComputationSorted(
      const absl::flat_hash_set<absl::string_view>& execution_threads) const;

  // Gets the computations in this module which aren't for fusion nodes.
  //
  // Postcondition: All computations in the returned list have
  // !IsFusionComputation().
  //
  // Note: Callers can and do rely on the return value here being a *snapshot*
  // of the module's non-fusion computations -- that is, it's OK to add or
  // remove computations from a module while iterating over
  // MakeNonfusionComputations().
  std::vector<HloComputation*> MakeNonfusionComputations() const {
    return MakeNonfusionComputations({});
  }
  // Same as above but only for specified `execution_threads`. Empty
  // `execution_threads` list means all execution threads are included.
  std::vector<HloComputation*> MakeNonfusionComputations(
      const absl::flat_hash_set<absl::string_view>& execution_threads) const;

  // Same as MakeNonfusionComputations() but sorting computations by content.
  std::vector<HloComputation*> MakeNonfusionComputationsSorted() const {
    return MakeNonfusionComputationsSorted({});
  }
  // Same as above but only for specified `execution_threads`. Empty
  // `execution_threads` list means all execution threads are included.
  std::vector<HloComputation*> MakeNonfusionComputationsSorted(
      const absl::flat_hash_set<absl::string_view>& execution_threads) const;

  HloModuleConfig& mutable_config() { return config_.get_mutable(); }
  const HloModuleConfig& config() const { return config_.get(); }
  void set_config(HloModuleConfig config) { config_.set(std::move(config)); }

  const std::shared_ptr<const HloModuleConfig>& shared_config() const {
    return config_.FreezeAndShare();
  }

  bool is_dynamic() const { return is_dynamic_; }
  void set_is_dynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }

  // Prints a string representation of the module.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  void Print(Printer* printer) const {
    return Print(printer, HloPrintOptions::Default());
  }
  void Print(Printer* printer, const HloPrintOptions& options) const;

  // Return a string representation of the module.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  std::string ToString() const { return ToString(HloPrintOptions::Default()); }
  std::string ToString(const HloPrintOptions& options) const;

  // Returns a Cord representation of the module.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  absl::Cord ToCord() const { return ToCord(HloPrintOptions::Default()); }
  absl::Cord ToCord(const HloPrintOptions& options) const;

  // Convert an HloModule to or from a proto.
  HloModuleProto ToProto() const;
  static absl::StatusOr<std::unique_ptr<HloModule>> CreateFromProto(
      const HloModuleProto& proto, const HloModuleConfig& module_config,
      bool prohibit_empty_literal = true);

  // Convert an HloModule to or from a proto that includes module configuration
  absl::StatusOr<HloModuleProtoWithConfig> ToProtoWithConfig() const;
  static absl::StatusOr<std::unique_ptr<HloModule>> CreateFromProtoWithConfig(
      const HloModuleProtoWithConfig& proto,
      bool prohibit_empty_literal = true);

  // Creates and returns an HloModuleConfig with an appropriate program shape
  // for the HLO module in the given proto.
  static absl::StatusOr<HloModuleConfig> CreateModuleConfigFromProto(
      const HloModuleProto& module, const DebugOptions& debug_options,
      const ExecutionOptions* execution_options = nullptr);

  // Creates and returns an HloModuleConfig with an appropriate program shape
  // for the HLO module in the given proto.
  static absl::StatusOr<HloModuleConfig> CreateModuleConfigFromShape(
      const ProgramShape& program_shape, const DebugOptions& debug_options,
      const ExecutionOptions* execution_options = nullptr);

  // Outlines the given expression from the given computation.
  // instructions_to_outline contains the instructions that form the expression.
  //
  // Precondition: instructions in instructions_to_outline are in topological
  // order (root of outlined instructions last). TODO(jingyue): takes a set of
  // instructions and topologically sorts them.
  HloInstruction* OutlineExpressionFromComputation(
      absl::Span<HloInstruction* const> instructions_to_outline,
      const std::string& outlined_computation_name,
      HloComputation* computation);

  // Returns a randomly generated uint64_t.
  uint64_t RandomNew64() const;

  // Returns the NameUniquer for uniquing instruction names in this module.
  NameUniquer& instruction_name_uniquer() { return instruction_name_uniquer_; }

  // Assign a new unique dense id for an instruction
  int NewUniqueInstructionId() {
    int result = next_unique_id_;
    next_unique_id_++;
    return result;
  }

  // input_output_alias_config indicates the list of aliased buffers that are
  // expected from the module.
  HloInputOutputAliasConfig& input_output_alias_config() {
    return input_output_alias_config_;
  }
  const HloInputOutputAliasConfig& input_output_alias_config() const {
    return input_output_alias_config_;
  }
  void set_input_output_alias_config(HloInputOutputAliasConfig config) {
    input_output_alias_config_ = std::move(config);
  }

  // buffer_donor_config_ indicates the set of input buffer donors that are
  // expected from the module.
  HloBufferDonorConfig& buffer_donor_config() { return buffer_donor_config_; }
  const HloBufferDonorConfig& buffer_donor_config() const {
    return buffer_donor_config_;
  }
  void set_buffer_donor_config(HloBufferDonorConfig config) {
    buffer_donor_config_ = std::move(config);
  }

  // Returns an id that is unique to this module across all modules created over
  // the lifetime of this process.
  int unique_id() const { return unique_id_; }

  // Sets the schedule of the module to the given schedule.
  Status set_schedule(HloSchedule schedule);

  // Clears the schedule of the module.
  void clear_schedule() { schedule_.reset(); }

  // Returns true if the module has a schedule set.
  bool has_schedule() const { return schedule_.has_value(); }

  // Returns the schedule of the module. CHECK fails if no schedule is set.
  const HloSchedule& schedule() const { return *schedule_; }
  HloSchedule& schedule() { return *schedule_; }

  HloComputation* AddComputationAndUnifyNamesAndIds(
      std::unique_ptr<HloComputation> computation, bool is_entry) {
    computation->ClearUniqueIdInternal();
    for (auto* instruction : computation->instructions()) {
      instruction->ClearUniqueIdInternal();
    }
    return AddComputationInternal(std::move(computation), is_entry,
                                  /*uniquify_identifiers=*/true,
                                  /*preserve_entry_layouts=*/true);
  }

  void SetAndUniquifyInstrName(HloInstruction* instr, absl::string_view name) {
    instr->SetAndSanitizeName(name);
    instr->UniquifyName(&instruction_name_uniquer_);
  }

  void SetAndUniquifyComputationName(HloComputation* computation,
                                     absl::string_view name) {
    computation->SetAndSanitizeName(name);
    computation->UniquifyName(&computation_name_uniquer_);
  }

  Status CheckUniqueNamesAndIdsForComputationsAndInstructions() const;

  // Checks if this config has a list of entry parameters' HLO shardings for
  // SPMD.
  bool has_spmd_parameters_shardings() const {
    return spmd_parameters_shardings_.has_value();
  }

  // Getter and setter for the list of entry parameters' HLO shardings for SPMD.
  const std::vector<HloSharding>& spmd_parameters_shardings() const {
    CHECK(spmd_parameters_shardings_.has_value());
    return *spmd_parameters_shardings_;
  }
  void set_spmd_parameters_shardings(
      const std::vector<HloSharding>& shardings) {
    spmd_parameters_shardings_ = shardings;
  }

  // Checks if this config has the entry computation output's HLO sharding for
  // SPMD.
  bool has_spmd_output_sharding() const {
    return spmd_output_sharding_.has_value();
  }

  // Getter and setter for the entry computation output's HLO shardings for
  // SPMD.
  const HloSharding& spmd_output_sharding() const {
    CHECK(spmd_output_sharding_.has_value());
    return *spmd_output_sharding_;
  }
  void set_spmd_output_sharding(const HloSharding& sharding) {
    spmd_output_sharding_ = sharding;
  }

  // Describes a buffer to be used for cross program prefetching.
  struct CrossProgramPrefetchInfo {
    // The parameter to prefetch.
    int64_t parameter;
    // Index of the buffer within a tuple-typed parameter.
    ShapeIndex index;
    // Offset into alt memory where the cross program pretched buffer will be
    // stored.
    std::optional<int64_t> alt_memory_offset;
  };

  // Add a program argument to be prefetched across programs.
  void AddCrossProgramPrefetch(
      int64_t parameter, const ShapeIndex& index,
      std::optional<int64_t> alt_memory_offset = std::nullopt) {
    cross_program_prefetches_.emplace_back(
        CrossProgramPrefetchInfo{parameter, index, alt_memory_offset});
  }

  Status SetCrossProgramPrefetchOffset(int64_t prefetch_index, int64_t offset) {
    TF_RET_CHECK(prefetch_index < cross_program_prefetches_.size());
    auto& [parameter, index, optional_offset] =
        cross_program_prefetches_[prefetch_index];
    TF_RET_CHECK(!optional_offset.has_value());
    optional_offset = offset;
    return OkStatus();
  }

  // Get the list of program arguments to be prefetch across programs.
  absl::Span<const CrossProgramPrefetchInfo> CrossProgramPrefetches() const {
    return cross_program_prefetches_;
  }

  const HloModuleMetadata& metadata() const { return metadata_; }
  HloModuleMetadata* metadata() { return &metadata_; }

  // Moves (not copies) metadata from this HloModule to `module`. To be used
  // in cases like HloModuleGroup::ReplaceModule when metadata should be
  // transferred out of a module before it's destroyed.
  void MoveMetadataToModule(HloModule* module) {
    module->metadata_ = std::move(metadata_);
  }

  int64_t profile_version() const { return profile_version_; }

  void set_profile_version(int64_t profile_version) {
    profile_version_ = profile_version;
  }

  void add_profile_info(const HloModuleProto::ProfileInfo& profile_info) {
    profile_info_list_.push_back(profile_info);
  }

  void set_profile_info(
      const std::vector<HloModuleProto::ProfileInfo>& profile_info) {
    profile_info_list_ = profile_info;
  }

  const std::vector<HloModuleProto::ProfileInfo>& profile_info() const {
    return profile_info_list_;
  }

  void set_autofdo_profile_key(HloModuleProto::ProfileType profile_type,
                               absl::string_view profile_key) {
    autofdo_profile_keys_[profile_type] = std::string(profile_key);
  }

  void set_autofdo_profile_keys(
      const absl::flat_hash_map<HloModuleProto::ProfileType, std::string>&
          profile_keys) {
    for (const auto& [profile_type, profile_key] : profile_keys) {
      autofdo_profile_keys_[profile_type] = profile_key;
    }
  }

  const absl::flat_hash_map<HloModuleProto::ProfileType, std::string>&
  autofdo_profile_keys() const {
    return autofdo_profile_keys_;
  }

  bool has_module_autofdo_profiles() const {
    return !autofdo_profile_keys_.empty();
  }

  void set_relative_speedup(double relative_speedup) {
    relative_speedup_ = relative_speedup;
  }

  // Sets the **unoptimized** fingerprint for the module. This fingerprint is
  // prior to any optimizations.
  void set_autofdo_fingerprint(absl::string_view fingerprint) {
    autofdo_fingerprint_ = std::string(fingerprint);
  }

  std::string autofdo_fingerprint() const { return autofdo_fingerprint_; }

  CompilationEnvironments& comp_envs() const { return *comp_envs_; }

  // Get 128-bit fingerprint of the module by printing it using the given print
  // options.
  std::string GetFingerprint128(const HloPrintOptions& options =
                                    HloPrintOptions::ModuleFingerprint()) const;

  // Describes a stack frame.
  struct StackFrame {
    std::string_view file_name;
    std::string_view function_name;
    int line = 0;
    int column = 0;

    // 1-based index of the parent frame.
    // 0 value indicates that the current frame is the root.
    int parent_frame_id = 0;

    bool empty() const {
      return line == 0 && column == 0 && file_name.empty() &&
             function_name.empty();
    }
  };

  // Getter for the specific stack frame. Argument is a 1-based index.
  StackFrame get_stack_frame(int id) const;

 private:
  HloComputation* AddComputationInternal(
      std::unique_ptr<HloComputation> computation, bool is_entry,
      bool uniquify_identifiers, bool preserve_entry_layouts);

  std::string name_;
  CopyOnWrite<HloModuleConfig> config_;
  HloComputation* entry_computation_ = nullptr;
  std::vector<std::unique_ptr<HloComputation>> computations_;

  // Random number generator engine to use when generating random numbers per
  // HloModule compilation.
  // TODO(b/25995601): Replace with better seed setting or dev/random for
  // where we don't need deterministic execution.
  mutable std::mt19937_64 rng_{42};
  mutable absl::Mutex rng_mutex_;

  // Unique name generator for computation and instruction names, which are
  // unique per module.
  NameUniquer computation_name_uniquer_{/*separator=*/"."};
  NameUniquer instruction_name_uniquer_{/*separator=*/"."};
  int next_unique_id_ = 0;

  // Used to keep track of the next unique module id that should be assigned.
  static std::atomic<int> next_unique_module_id_;
  // A unique id to label modules with.
  const int unique_id_;

  // The HloSchedule of the module. The schedule if it exists contains a
  // sequential order of instructions for each non-fusion computation in the
  // module.
  std::optional<HloSchedule> schedule_;

  // alias_config indicates the alias information of input/output buffers that
  // are expected from the module.
  HloInputOutputAliasConfig input_output_alias_config_;

  // buffer_donor_config_ indicates the donor information of input buffers that
  // are expected from the module.
  HloBufferDonorConfig buffer_donor_config_;

  // Attributes passed from the frontend to give hints to the backend about
  // how to compile this HLO.
  FrontendAttributes frontend_attributes_;

  // The HLO shardings of the entry computation's parameters for
  // SPMD-partitioned programs.
  std::optional<std::vector<HloSharding>> spmd_parameters_shardings_;

  // The HLO sharding of the entry computation's output (root) for
  // SPMD-partitioned programs.
  std::optional<HloSharding> spmd_output_sharding_;

  // Arguments to be prefetched across programs.
  std::vector<CrossProgramPrefetchInfo> cross_program_prefetches_;

  // Metadata for this module, such as its canonical id and the HLO passes run.
  HloModuleMetadata metadata_;

  // True if the module contains dynamic computation.
  bool is_dynamic_ = false;

  // Optional compilation profile handle.
  int64_t profile_version_ = 0;

  // An array of ProfileInfo specifying what optimization profiles this module
  // contains, along with the relative speedups.
  std::vector<HloModuleProto::ProfileInfo> profile_info_list_;

  // Relative speedup of best config compared to default config.
  double relative_speedup_;

  // The unoptimized module fingerprint.
  std::string autofdo_fingerprint_;

  // The keys used to retrieve the optimization profiles this module is compiled
  // with, per profile type.
  absl::flat_hash_map<HloModuleProto::ProfileType, std::string>
      autofdo_profile_keys_;

  bool use_auto_spmd_partitioning_ = false;

  // Layout canonicalization callback, used only when
  // use_auto_spmd_partitioning_ = true.
  LayoutCanonicalizationCallback layout_canonicalization_callback_;

  // Compilation environments (protos that carry command line flags and
  // environment variables).
  std::unique_ptr<CompilationEnvironments> comp_envs_ =
      std::make_unique<CompilationEnvironments>();

  // Stack frame indexes flat representation.
  std::optional<StackFrameIndexProto> stack_frame_index_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_MODULE_H_
