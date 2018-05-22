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

// Returns the minimum memory required to compute the given module sequence,
// assuming no fragmentation.
StatusOr<int64> MinimumMemoryForSequence(
    const SequentialHloOrdering::HloModuleSequence& module_sequence,
    const LogicalBuffer::SizeFunction& size_function);

// Returns the minimum memory required to compute the given computation,
// assuming no fragmentation.
StatusOr<int64> MinimumMemoryForComputation(
    const HloComputation& computation,
    const std::vector<const HloInstruction*>& sequence,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function);

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
StatusOr<SequentialHloOrdering::HloModuleSequence>
CreateMemoryMinimizingSequence(const HloModule& module,
                               const LogicalBuffer::SizeFunction& size_function,
                               const MemorySchedulerAlgorithm& algorithm = {});

// Overload of above that computes the sequence for a single computation.
// Currently only used by the GPU backend.
StatusOr<std::vector<const HloInstruction*>> CreateMemoryMinimizingSequence(
    const HloComputation& computation,
    const LogicalBuffer::SizeFunction& size_function);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULING_H_
