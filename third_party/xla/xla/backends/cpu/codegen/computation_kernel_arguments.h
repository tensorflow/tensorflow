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

#ifndef XLA_BACKENDS_CPU_CODEGEN_COMPUTATION_KERNEL_ARGUMENTS_H_
#define XLA_BACKENDS_CPU_CODEGEN_COMPUTATION_KERNEL_ARGUMENTS_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Deterministic parameter regions for a computation emitted as a single kernel.
//
// Each parameter is a disjoint byte region of one allocation. It covers a
// maximal group of overlapping non empty leaf array slices from the computation
// and its nested non fusion computations. Thread local slices and tuple index
// tables are skipped. Read only regions (`arguments`) come before writable
// regions (`results`), each ordered by first visit. Parameter shapes are
// `u8[size]`.
//
// The fingerprint does not use emitters::GetComputationFingerprint, which only
// describes kernel operands and outputs, not the buffers of intermediate
// instructions.
class ComputationKernelArguments {
 public:
  // Returns an error if a visited instruction or one of its tuple elements has
  // no unique slice in `assignment`.
  static absl::StatusOr<ComputationKernelArguments> Create(
      const HloComputation& computation, const BufferAssignment& assignment);

  // Merges overlapping slices of the same allocation into covering slices,
  // sorted by allocation index and offset. Empty slices are ignored; adjacent
  // non overlapping slices remain separate.
  static std::vector<BufferAllocation::Slice> MergeOverlapping(
      absl::Span<const BufferAllocation::Slice> slices);

  const KernelSpec::Buffers& arguments() const { return arguments_; }
  const KernelSpec::Buffers& results() const { return results_; }

  // Equal fingerprints mean a kernel emitted for one computation is valid for
  // the other's arguments and results.
  const std::string& fingerprint() const { return fingerprint_; }

 private:
  ComputationKernelArguments(KernelSpec::Buffers arguments,
                             KernelSpec::Buffers results,
                             std::string fingerprint)
      : arguments_(std::move(arguments)),
        results_(std::move(results)),
        fingerprint_(std::move(fingerprint)) {}

  KernelSpec::Buffers arguments_;
  KernelSpec::Buffers results_;
  std::string fingerprint_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_COMPUTATION_KERNEL_ARGUMENTS_H_
