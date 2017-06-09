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

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
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

  // Adds an entry computation to the module. A module can only have one entry
  // computation. Returns a pointer to the newly added computation.
  HloComputation* AddEntryComputation(
      std::unique_ptr<HloComputation> computation);

  // Adds an embedded computation to the module.
  HloComputation* AddEmbeddedComputation(
      std::unique_ptr<HloComputation> computation);

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

  const std::vector<std::unique_ptr<HloComputation>>& computations() const {
    return computations_;
  }

  // Compute and return a post order of all computations in the module. The sort
  // is defined like so: if computation A has an instruction which calls
  // computation B, then A will appear after B in the sort.
  std::list<HloComputation*> MakeComputationPostOrder() const;

  const HloModuleConfig& config() const { return config_; }

  string ToString() const;
  HloModuleProto ToProto() const;

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

 private:
  HloComputation* AddComputationInternal(
      std::unique_ptr<HloComputation> computation);

  const string name_;
  HloModuleConfig config_;
  HloComputation* entry_computation_;
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

  // Unique name generator for computation names, which are unique per module.
  NameUniquer computation_name_uniquer_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_H_
