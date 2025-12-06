/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/autotuning/redzone_buffers.h"

#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

absl::StatusOr<RedzoneBuffers> RedzoneBuffers::FromInstruction(
    const HloInstruction& instruction, se::DeviceAddressAllocator* allocator,
    se::Stream* stream, BuffersToCreate buffers_to_create,
    bool should_init_buffers, bool should_check_correctness,
    int redzone_padding_bytes) {
  ProgramShape program_shape;
  for (const HloInstruction* operand : instruction.operands()) {
    program_shape.AddParameter(operand->shape(), std::string(operand->name()));
  }
  *program_shape.mutable_result() = instruction.shape();
  return FromProgramShape(program_shape, buffers_to_create, should_init_buffers,
                          should_check_correctness, redzone_padding_bytes,
                          allocator, stream);
}

absl::StatusOr<RedzoneBuffers> RedzoneBuffers::FromComputation(
    const HloComputation& computation, se::DeviceAddressAllocator* allocator,
    se::Stream* stream, BuffersToCreate buffers_to_create,
    bool should_init_buffers, bool should_check_correctness,
    int redzone_padding_bytes) {
  return FromProgramShape(computation.ComputeProgramShape(), buffers_to_create,
                          should_init_buffers, should_check_correctness,
                          redzone_padding_bytes, allocator, stream);
}

absl::StatusOr<RedzoneBuffers> RedzoneBuffers::FromProgramShape(
    const ProgramShape& program_shape, BuffersToCreate buffers_to_create,
    bool should_init_buffers, bool should_check_correctness,
    int redzone_padding_bytes, se::DeviceAddressAllocator* allocator,
    se::Stream* stream) {
  tsl::profiler::TraceMe traceme("create redzone buffers");
  RedzoneBuffers buffers;
  buffers.redzone_allocator_ = std::make_unique<se::RedzoneAllocator>(
      stream, allocator,
      /*memory_limit=*/std::numeric_limits<int64_t>::max(),
      /*redzone_size=*/should_check_correctness ? redzone_padding_bytes : 0);

  int64_t rng_state = 0;

  TF_RETURN_IF_ERROR(buffers.CreateInputs(program_shape.parameters(),
                                          should_init_buffers, rng_state));

  if (buffers_to_create == BuffersToCreate::kAllInputsAllOutputs ||
      buffers_to_create == BuffersToCreate::kAllInputsOutputsNoScratch) {
    TF_RETURN_IF_ERROR(buffers.CreateOutputs(program_shape.result(),
                                             buffers_to_create,
                                             should_init_buffers, rng_state));
  }
  return buffers;
}

absl::Status RedzoneBuffers::CreateInputs(absl::Span<const Shape> input_shapes,
                                          bool should_init_buffers,
                                          int64_t& rng_state) {
  tsl::profiler::TraceMe traceme("create inputs");
  for (const auto& input_shape : input_shapes) {
    TF_ASSIGN_OR_RETURN(se::DeviceAddressBase buf,
                        redzone_allocator_->CreateBuffer(
                            input_shape, should_init_buffers, rng_state));
    input_buffers_.push_back(buf);
    input_shapes_.push_back(input_shape);
  }
  return absl::OkStatus();
}

absl::Status RedzoneBuffers::CreateOutputs(const Shape& output_shape,
                                           BuffersToCreate buffers_to_create,
                                           bool should_init_buffers,
                                           int64_t& rng_state) {
  tsl::profiler::TraceMe traceme("create outputs");
  if (!output_shape.IsTuple()) {
    TF_ASSIGN_OR_RETURN(se::DeviceAddressBase buf,
                        redzone_allocator_->CreateBuffer(
                            output_shape, should_init_buffers, rng_state));
    output_buffers_.push_back(buf);
    output_shape_ = output_shape;
    return absl::OkStatus();
  }

  // The output is a tuple.

  auto current_shape_it = output_shape.tuple_shapes().begin();
  auto end = output_shape.tuple_shapes().end();
  end -= buffers_to_create == kAllInputsAllOutputs ? 0 : 1;

  output_shape_ = std::distance(current_shape_it, end) == 1
                      ? output_shape_ = *current_shape_it
                      : ShapeUtil::MakeTupleShape(
                            std::vector<Shape>{current_shape_it, end});

  for (; current_shape_it < end; current_shape_it++) {
    if (current_shape_it->IsTuple()) {
      return Unimplemented("Nested tuples are unsupported by RedzoneBuffers.");
    }
    TF_ASSIGN_OR_RETURN(se::DeviceAddressBase buf,
                        redzone_allocator_->CreateBuffer(
                            *current_shape_it, should_init_buffers, rng_state));
    output_buffers_.push_back(buf);
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
