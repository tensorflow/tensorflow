/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULING_H_

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// A memory scheduler computes an execution sequence for the HLO instructions in
// 'computation' that minimizes peak memory, given a points-to analysis result
// that describes buffer aliasing, together with a target-specific size function
// that maps a tensor's logical size to its padded size.
typedef std::function<StatusOr<std::vector<const HloInstruction*>>(
    const HloComputation&, const TuplePointsToAnalysis&,
    const LogicalBuffer::SizeFunction&,
    const tensorflow::gtl::FlatMap<const HloComputation*, int64>&)>
    MemorySchedulerAlgorithm;

// List scheduler
StatusOr<std::vector<const HloInstruction*>> ListMemoryScheduler(
    const HloComputation& computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const tensorflow::gtl::FlatMap<const HloComputation*, int64>&
        memory_by_computation);

// DFS-order scheduler
StatusOr<std::vector<const HloInstruction*>> DFSMemoryScheduler(
    const HloComputation& computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const tensorflow::gtl::FlatMap<const HloComputation*, int64>&
        memory_by_computation);

// Naive Post Order scheduler
StatusOr<std::vector<const HloInstruction*>> PostOrderMemoryScheduler(
    const HloComputation& computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const tensorflow::gtl::FlatMap<const HloComputation*, int64>&
        memory_by_computation);

// The default scheduling algorithm. Runs both the list scheduler
// and the DFS scheduler, and chooses whichever returns a lower min-memory,
// not accounting for fragmentation.
StatusOr<std::vector<const HloInstruction*>> DefaultMemoryScheduler(
    const HloComputation& computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const tensorflow::gtl::FlatMap<const HloComputation*, int64>&
        memory_by_computation);

// Returns an HloModuleSequence which seeks to minimize the memory required for
// the computation. size_function is the function returning the number of bytes
// required for a LogicalBuffer.
StatusOr<SequentialHloOrdering::HloModuleSequence> ScheduleComputationsInModule(
    const HloModule& module, const LogicalBuffer::SizeFunction& size_function,
    const MemorySchedulerAlgorithm& algorithm = {});

// Computes the schedule for a single computation.
// Currently only used by the GPU backend.
StatusOr<std::vector<const HloInstruction*>> ScheduleOneComputation(
    const HloComputation& computation,
    const LogicalBuffer::SizeFunction& size_function);

// Transforms the given schedule such that it is (again) a valid schedule for
// the module. This is used to update a schedule after the HLO module has been
// transformed in some way. In general, the only transformations to the module
// for which a schedule can be updated is the addition or removal of
// instructions to/from the module. Updating the schedule after new dependencies
// between existing instructions in the module is not supported and may result
// in an error status returned.
//
// Instructions in the module which also exist in the given schedule will remain
// in the same order in the updated schedule. Instructions which exist in the
// module but not in the given schedule will be placed as early as possible in
// the updated schedule.
//
// 'id_sequence' is a mirror of the given schedule 'sequence' but with
// HloInstruction ids rather than HloInstruction pointers. This should be
// constructed using ComputeIdSchedule below after the schedule is constructed
// but before the HLO module is transformed.
Status UpdateSchedule(
    const HloModule& module,
    const tensorflow::gtl::FlatMap<const HloComputation*, std::vector<int>>&
        id_sequence,
    SequentialHloOrdering::HloModuleSequence* sequence);

// Constructs a copy of the given schedule but with HloInstruction unique ids
// rather than HloInstruction pointers. This is necessary for updating a
// schedule as HloInstruction points in the schedule may become invalid if
// instructions are removed from the module. Used by UpdateSchedule above..
// TODO(b/113175018): Remove this function when HLO schedule is its own class.
tensorflow::gtl::FlatMap<const HloComputation*, std::vector<int>>
ComputeIdSchedule(const SequentialHloOrdering::HloModuleSequence& sequence);

// Verifies that the given schedule is valid for the given module. Specifically,
// the schedule contains exactly the instructions in the module and every
// dependency in the module is satisfied in the schedule.
Status VerifySchedule(const HloModule& module,
                      const SequentialHloOrdering::HloModuleSequence& sequence);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULING_H_
