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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Deterministic argument table for a computation emitted as a single kernel.
//
// Operand slices of all instructions become arguments and slices of all
// non-parameter instructions become results, including nested tuple elements
// and instructions of nested non-fusion computations. Thread-local slices are
// skipped. Entries are deduplicated by (shape, slice) in first-visit order.
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

  const KernelSpec::Buffers& arguments() const { return arguments_; }
  const KernelSpec::Buffers& results() const { return results_; }

  // Index of each slice in the buffer table of arguments followed by results.
  // A slice listed more than once maps to its last position.
  const absl::flat_hash_map<BufferAllocation::Slice, int64_t>&
  buffer_table_index() const {
    return buffer_table_index_;
  }

  // Equal fingerprints mean a kernel emitted by ComputationKernelEmitter for
  // one computation is valid for the other's arguments and results. Other
  // emitters must check what they bake into the code. Unset if the kernel must
  // not be reused.
  const std::optional<std::string>& fingerprint() const { return fingerprint_; }

 private:
  ComputationKernelArguments(
      KernelSpec::Buffers arguments, KernelSpec::Buffers results,
      absl::flat_hash_map<BufferAllocation::Slice, int64_t> buffer_table_index,
      std::optional<std::string> fingerprint)
      : arguments_(std::move(arguments)),
        results_(std::move(results)),
        buffer_table_index_(std::move(buffer_table_index)),
        fingerprint_(std::move(fingerprint)) {}

  KernelSpec::Buffers arguments_;
  KernelSpec::Buffers results_;
  absl::flat_hash_map<BufferAllocation::Slice, int64_t> buffer_table_index_;
  std::optional<std::string> fingerprint_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_COMPUTATION_KERNEL_ARGUMENTS_H_
