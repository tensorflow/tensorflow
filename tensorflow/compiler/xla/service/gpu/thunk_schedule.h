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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_SCHEDULE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_SCHEDULE_H_

#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

// Encapsulates in which order and on which streams the thunks are executed. A
// schedule contains
//
// 1. A stream assignment indicating which stream each thunk is executed on.
//
// 2. A total order of all thunks. If A is ordered before B and they are
// assigned to the same stream, then A completes before B starts. If A is
// ordered before B and they are on different streams, their actual execution
// order is not determined.
//
// 3. A set of dependency edges. If A and B are scheduled on different streams
// and A has to complete before B starts (e.g. A produces an input of B), then B
// "depends" on A.
class ThunkSchedule {
 public:
  // `thunk_to_hlo` is an one-to-one map. Every thunk in this container maps to
  // an HLO, but not every HLO ever exists produces a Thunk.
  //
  // thunk_to_hlo.keys() == set(thunks).
  ThunkSchedule(
      std::unique_ptr<ThunkSequence> thunks,
      std::unique_ptr<StreamAssignment> stream_assignment,
      absl::flat_hash_map<const Thunk*, const HloInstruction*> thunk_to_hlo);

  // Single stream, trivial schedule in the ThunkSequence order.
  explicit ThunkSchedule(std::unique_ptr<ThunkSequence> thunks);

  // Returns the total order of executing all the thunks.
  const ThunkSequence& TotalOrder() const { return *thunks_; }

  // Thunks that `thunk` depends on.
  const std::list<const Thunk*>& DependsOn(const Thunk* thunk) const;
  // Whether `thunk` is depended by another thunk.
  bool Depended(const Thunk* thunk) const {
    return depended_by_.contains(thunk);
  }

  // Delegates to StreamAssignment.
  int StreamCount() const {
    if (stream_assignment_) {
      return stream_assignment_->StreamCount();
    }
    return 1;
  }
  int StreamNumberForThunk(const Thunk* thunk) const {
    if (stream_assignment_) {
      return stream_assignment_->StreamNumberForHlo(*thunk_to_hlo_.at(thunk));
    }
    return 0;
  }

  std::string ToString() const;

 private:
  void RemoveRedundantDependencyEdges();

  // Adds `operand` and its transitive operands to the dependency list of
  // `thunk`.
  //
  // Precondition: `operand` is a non-trivial (i.e. excluding
  // thunk.hlo_instruction_ itself) transitive operand of
  // thunk.hlo_instruction_.
  void AddDependenciesOnTransitiveOperands(
      const Thunk& thunk, const HloInstruction& operand,
      const absl::flat_hash_map<const HloInstruction*, Thunk*>& hlo_to_thunk);

  std::unique_ptr<ThunkSequence> thunks_;

  absl::flat_hash_map<const Thunk*, std::list<const Thunk*>> depends_on_;
  absl::flat_hash_set<const Thunk*> depended_by_;
  std::list<const Thunk*> empty_thunk_list_;

  std::unique_ptr<StreamAssignment> stream_assignment_;

  absl::flat_hash_map<const Thunk*, const HloInstruction*> thunk_to_hlo_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_THUNK_SCHEDULE_H_
