/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_

#include <list>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {

// Describes a compilation unit at the HLO level.
//
// A HLO module contains one or more HLO computations. The module contains one
// "entry" computation which produces the result. The module also includes any
// embedded computations used by instructions such as "map" and "reduce". All
// computations are owned by the module.
class HloModule {
 public:
  HloModule(const string& name,
            const VersionedComputationHandle& entry_computation_handle,
            const HloModuleConfig& config);

  // Constructor without a versioned computation handle. This constructor should
  // only be used for HloModules used outside of the XLA service (eg
  // tests). The versioned handle is used by the service in the compilation
  // cache. A default configuration is created for this module.
  explicit HloModule(const string& name);
  explicit HloModule(const string& name, const HloModuleConfig& config);

  // Adds an entry computation to the module. A module can only have one entry
  // computation. Returns a pointer to the newly added computation.
  HloComputation* AddEntryComputation(
      std::unique_ptr<HloComputation> computation);

  // Adds an embedded computation to the module.
  HloComputation* AddEmbeddedComputation(
      std::unique_ptr<HloComputation> computation);

  // Removes an embedded computation.
  Status RemoveEmbeddedComputation(HloComputation* to_remove);

  // Replaces all uses of computations that are keys of 'replacements' with
  // the corresponding values in 'replacements'. Replaces the entry computation,
  // if applicable.
  //
  // This function iterates over all instructions in the module to find
  // computations to replace. We could speed it up by keeping track of users of
  // computations.
  void ReplaceComputations(
      const std::unordered_map<HloComputation*, HloComputation*>& replacements);

  const string& name() const { return name_; }

  // Returns a deep copy of this module including all computations.
  std::unique_ptr<HloModule> Clone(const string& suffix = "clone") const;

  // Return a pointer to the entry computation of the module..
  HloComputation* entry_computation() const {
    CHECK_NE(nullptr, entry_computation_);
    return entry_computation_;
  }

  ComputationLayout* mutable_entry_computation_layout() {
    return config_.mutable_entry_computation_layout();
  }

  const VersionedComputationHandle& entry_computation_handle() const {
    return entry_computation_handle_;
  }

  // Gets the computations in this module.
  //
  // Returns a view of HloComputation*s, so you can iterate over this in the
  // natural way:
  //
  //   for (HloComputation* c : module->computations()) { ... }
  //
  tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::const_iterator>>
  computations() const {
    return {MakeUnwrappingIterator(computations_.begin()),
            MakeUnwrappingIterator(computations_.end())};
  }
  tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<HloComputation>>::iterator>>
  computations() {
    return {MakeUnwrappingIterator(computations_.begin()),
            MakeUnwrappingIterator(computations_.end())};
  }

  // Gets the number of computations in this module.
  int64 computation_count() const { return computations_.size(); }

  // Compute and return a post order of all computations in the module. The sort
  // is defined like so: if computation A has an instruction which calls
  // computation B, then A will appear after B in the sort.
  std::list<HloComputation*> MakeComputationPostOrder() const;

  // Gets the computations in this module which aren't for fusion nodes.
  //
  // Postcondition: All computations in the returned list have
  // !IsFusionComputation().
  //
  // Note: Callers can and do rely on the return value here being a *snapshot*
  // of the module's non-fusion computations -- that is, it's OK to add or
  // remove computations from a module while iterating over
  // MakeNonfusionComputations().
  std::vector<HloComputation*> MakeNonfusionComputations() const;

  const HloModuleConfig& config() const { return config_; }

  string ToString(bool include_large_constants = false) const;

  // Convert an HloModule to or from a proto.
  HloModuleProto ToProto() const;
  static StatusOr<std::unique_ptr<HloModule>> CreateFromProto(
      const HloModuleProto& proto, const HloModuleConfig& module_config,
      const VersionedComputationHandle& entry_computation_handle =
          VersionedComputationHandle());

  // Creates and returns an HloModuleConfig with an appropriate program shape
  // for the HLO module in the given proto.
  static StatusOr<HloModuleConfig> CreateModuleConfigFromProto(
      const HloModuleProto& module);

  // Outlines the given expression from the given computation.
  // instructions_to_outline contains the instructions that form the expression.
  //
  // Precondition: instructions in instructions_to_outline are in topological
  // order (root of outlined instructions last). TODO(jingyue): takes a set of
  // instructions and topologically sorts them.
  HloInstruction* OutlineExpressionFromComputation(
      tensorflow::gtl::ArraySlice<HloInstruction*> instructions_to_outline,
      const string& outlined_computation_name, HloComputation* computation);

  // Returns a randomly generated uint64.
  uint64 RandomNew64() const;

  // Returns the unique name for a computation in this module.
  string GetUniqueCompuationName(const string& prefix) {
    return computation_name_uniquer_.GetUniqueName(prefix);
  }

  // Returns the NameUniquer for uniquing instruction names in this module.
  NameUniquer& instruction_name_uniquer() { return instruction_name_uniquer_; }

  // Assign a new unique dense id for an instruction
  int NewUniqueInstructionId() {
    int result = next_unique_id_;
    next_unique_id_++;
    return result;
  }

  // Returns the number of unique intruction ids given out.  All ids up to
  // this point are guaranteed to be in the range [0..NumUniqueInstructionIds())
  int NumUniqueInstructionIds() const { return next_unique_id_; }

 private:
  HloComputation* AddComputationInternal(
      std::unique_ptr<HloComputation> computation, bool is_entry,
      bool uniquify_names);

  const string name_;
  HloModuleConfig config_;
  HloComputation* entry_computation_ = nullptr;
  std::vector<std::unique_ptr<HloComputation>> computations_;

  // Random number generator engine to use when generating random numbers per
  // HloModule compilation.
  // TODO(b/25995601): Replace with better seed setting or dev/random for
  // where we don't need deterministic execution.
  mutable std::mt19937_64 rng_{42};
  mutable tensorflow::mutex rng_mutex_;

  // Versioned handle of the entry computation of the module.
  bool has_entry_computation_handle_ = false;
  VersionedComputationHandle entry_computation_handle_;

  // Unique name generator for computation and instruction names, which are
  // unique per module.
  NameUniquer computation_name_uniquer_{/*separator=*/"."};
  NameUniquer instruction_name_uniquer_{/*separator=*/"."};
  int next_unique_id_ = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
