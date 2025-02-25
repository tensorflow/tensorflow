/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/hlo_runner_interface.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"

namespace xla {

namespace {
template <class T>
std::vector<T*> MakePointerVector(absl::Span<T> input_vec) {
  std::vector<T*> output_pointers;
  output_pointers.reserve(input_vec.size());
  for (auto& input : input_vec) {
    output_pointers.push_back(&input);
  }
  return output_pointers;
}
}  // namespace

absl::StatusOr<Literal> HloRunnerInterface::Execute(
    std::unique_ptr<HloModule> module, absl::Span<const Literal> arguments,
    bool run_hlo_passes, ExecutionProfile* profile) {
  // Construct a vector of plain pointers for the arguments.
  auto argument_pointers = MakePointerVector<const Literal>(arguments);
  return Execute(
      /*module=*/std::move(module),
      /*arguments=*/argument_pointers,
      /*run_hlo_passes=*/run_hlo_passes,
      /*profile=*/profile);
}

absl::StatusOr<Literal> HloRunnerInterface::ExecuteWithBufferAssignment(
    std::unique_ptr<HloModule> module,
    const BufferAssignmentProto* buffer_assignment_proto,
    absl::Span<const Literal> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  // Construct a vector of plain pointers for the arguments.
  auto argument_pointers = MakePointerVector<const Literal>(arguments);
  return ExecuteWithBufferAssignment(
      /*module=*/std::move(module),
      /*buffer_assignment_proto=*/buffer_assignment_proto,
      /*arguments=*/argument_pointers,
      /*run_hlo_passes=*/run_hlo_passes,
      /*profile=*/profile);
}

absl::StatusOr<Literal> HloRunnerInterface::ExecuteWithExecutable(
    OpaqueExecutable* executable, absl::Span<const Literal> arguments,
    ExecutionProfile* profile) {
  // Construct a vector of plain pointers for the arguments.
  auto argument_pointers = MakePointerVector<const Literal>(arguments);
  return ExecuteWithExecutable(executable, argument_pointers, profile);
}

}  // namespace xla
