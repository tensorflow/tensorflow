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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_TRACKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_TRACKER_H_

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service/user_computation.h"
#include "tensorflow/compiler/xla/service/versioned_computation_handle.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Tracks computations for the XLA service; computations can be registered
// with a UserComputation instance and can be resolved from a handle for later
// use.
//
// This class is also capable of serializing/deserializing computations that it
// tracks (and to serialize properly you need to serialize all referred-to
// computations as well).
class ComputationTracker {
 public:
  ComputationTracker();

  // Creates a new UserComputation object and returns the corresponding
  // ComputationHandle for it.
  //
  // Precondition: user_computation is not already present in the map.
  ComputationHandle NewComputation(const string& computation_name);

  // Restores session data for a computation that has been serialized, and
  // allocates a new computation handle for it.
  StatusOr<ComputationHandle> LoadSessionModule(
      const SessionModule& session_module);

  // Snapshots a computation (referenced by the provided handle) at its latest
  // version, returning a module where it is the entry, and any referred-to
  // computations are entrained as "embedded" (non-entry) computations.
  StatusOr<std::unique_ptr<SessionModule>> SnapshotComputation(
      const ComputationHandle& computation);

  // Resolves a ComputationHandle to a UserComputation that is present in the
  // map.
  StatusOr<UserComputation*> Resolve(
      const ComputationHandle& computation) const;

  // Builds an HLO module using the specified computation as the entry. The
  // module will include the entry computation as well as all computations which
  // are called directly or indirectly from the entry computation via operations
  // like "map". config is the HLO module configuration to use for the
  // constructed module.
  // If include_unreachable_instructions is true, then instructions
  // which are not reachable from the root are lowered into HloInstructions
  // including unreachable parameters. This ensures the entry HloComputation has
  // the same program shape (ProgramShape) as the entry UserComputation.
  StatusOr<std::unique_ptr<HloModule>> BuildHloModule(
      const VersionedComputationHandle& entry_handle,
      const HloModuleConfig& config,
      bool include_unreachable_instructions = true) const;

  string ToString() const;

 private:
  // Bumps the next_computation_ number and returns the allocated number wrapped
  // in a ComputationHandle.
  ComputationHandle AllocateHandle()
      EXCLUSIVE_LOCKS_REQUIRED(computation_mutex_);

  // Loads a session computation into a UserComputation, registers it, and
  // returns the computation handle of the registered computation. If old_to_new
  // is provided, it is used for remapping references to computations present in
  // session_computation.
  //
  // old_to_new will be updated with the mapping from session_computation's old
  // handle to the returned handle value, and may not be null.
  StatusOr<ComputationHandle> LoadSessionComputation(
      const SessionComputation& session_computation,
      std::map<int64, ComputationHandle>* old_to_new)
      EXCLUSIVE_LOCKS_REQUIRED(computation_mutex_);

  // Internal implementation of Resolve method which requires, but does not
  // acquire the mutex.
  StatusOr<UserComputation*> ResolveInternal(
      const ComputationHandle& computation) const
      EXCLUSIVE_LOCKS_REQUIRED(computation_mutex_);

  // Builds a post order sort of a computation ("entry") and all of its embedded
  // computations including all transitively embedded computations. An embedded
  // computation (the callee) will always appear in the sort before the
  // computation which calls the embedded computation (the caller). Necessarily,
  // the entry computation is the last element in the sort. visited and
  // post_order should be empty when calling. post_order contains the post order
  // sort when the function return.
  void ComputeComputationPostOrder(
      const VersionedComputationHandle& versioned_handle,
      std::set<VersionedComputationHandle>* visited,
      std::list<VersionedComputationHandle>* post_order) const
      EXCLUSIVE_LOCKS_REQUIRED(computation_mutex_);

  string ToStringInternal() const EXCLUSIVE_LOCKS_REQUIRED(computation_mutex_);

  // Guards the computation mapping. Marked mutable so that the Resolve method
  // can remain const; Resolve does't really modify the tracker in any way, but
  // it has to lock the mutex for safety.
  mutable tensorflow::mutex computation_mutex_;

  // The next sequence number to assign to a computation, guarded by the same
  // mutex as the mapping as they'll be mutated at the same time.
  int64 next_computation_ GUARDED_BY(computation_mutex_);

  // Mapping from ComputationHandle value to the corresponding registered
  // UserComputation object.
  std::map<int64, std::unique_ptr<UserComputation>> opaque_to_computation_
      GUARDED_BY(computation_mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(ComputationTracker);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_TRACKER_H_
