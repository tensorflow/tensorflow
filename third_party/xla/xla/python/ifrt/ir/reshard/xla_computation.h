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

#ifndef XLA_PYTHON_IFRT_IR_RESHARD_XLA_COMPUTATION_H_
#define XLA_PYTHON_IFRT_IR_RESHARD_XLA_COMPUTATION_H_

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/memory.h"
#include "xla/shape.h"

namespace xla {
namespace ifrt {
namespace reshard {

class XlaComputationBuilder {
 public:
  explicit XlaComputationBuilder(absl::string_view device_kind)
      : device_kind_(device_kind) {}

  // Builds an XLA computation that reshards inputs in an SPMD manner.
  absl::StatusOr<std::unique_ptr<xla::ifrt::HloProgram>>
  BuildXlaReshardComputation(
      const xla::Shape& xla_shape, const xla::HloSharding& old_hlo_sharding,
      const xla::HloSharding& new_hlo_sharding,
      absl::Span<const xla::ifrt::MemoryKind> memory_kinds,
      bool pre_resharding);

  // Builds a zeros XLA computation that emits zeros buffers.
  absl::StatusOr<std::unique_ptr<xla::ifrt::HloProgram>>
  BuildXlaZerosComputation(
      const xla::Shape& xla_shape, const xla::HloSharding& new_hlo_sharding,
      absl::Span<const xla::ifrt::MemoryKind> memory_kinds);

  // Builds an XLA computation that applies all-reduce and reshards inputs in an
  // SPMD manner. It first takes a replica dimension from the first dimension
  // and reduces it, where only the one row has real data and other rows have
  // zeros. This reduction effectively replicates the real data to all replica
  // devices. Then, it makes a sharding change (same as pre-resharding) to
  // potentially reshard the data.
  absl::StatusOr<std::unique_ptr<xla::ifrt::HloProgram>>
  BuildXlaReduceComputation(
      const xla::Shape& old_xla_shape, const xla::HloSharding& old_hlo_sharding,
      const xla::Shape& new_xla_shape, const xla::HloSharding& new_hlo_sharding,
      absl::Span<const xla::ifrt::MemoryKind> memory_kinds);

 private:
  std::string device_kind_;
};

// Dumps an `xla::ifrt::HloProgram` into a string.
std::string DumpHloProgram(xla::ifrt::HloProgram& program);

}  // namespace reshard
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_RESHARD_XLA_COMPUTATION_H_
