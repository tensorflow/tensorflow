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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_REDZONE_BUFFERS_H_
#define XLA_SERVICE_GPU_AUTOTUNING_REDZONE_BUFFERS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// A RedZone allocator and a collection of buffers that store the inputs and
// outputs of an HloInstruction. These are used when running the instruction
// for autotuning.
class RedzoneBuffers {
 public:
  enum BuffersToCreate {
    // Create a buffer for all of the instruction's operands. The result shape
    // is ignored.
    kAllInputs = 0,
    // Create a buffer for all of the instruction's operands and the entire
    // result shape. If the result shape is a tuple, a separate buffer is
    // created for each subshape.
    kAllInputsAllOutputs = 1,
    // Create a buffer for all of the instruction's operands and all of the
    // subshapes of the result tuple, except for the last one. The last subshape
    // is considered a scratch buffer and is assumed to be allocated elsewhere.
    // If the result shape is not a tuple, this will create a buffer
    // corresponding to the entire shape - equivalent to `kAllInputsAllOutputs`.
    kAllInputsOutputsNoScratch = 2,
  };

  static absl::StatusOr<RedzoneBuffers> FromInstruction(
      const HloInstruction& instruction, se::DeviceMemoryAllocator* allocator,
      se::Stream* stream, BuffersToCreate buffers_to_create,
      bool should_init_buffers, bool should_check_correctness,
      int redzone_padding_bytes);

  static absl::StatusOr<RedzoneBuffers> FromComputation(
      const HloComputation& computation, se::DeviceMemoryAllocator* allocator,
      se::Stream* stream, BuffersToCreate buffers_to_create,
      bool should_init_buffers, bool should_check_correctness,
      int redzone_padding_bytes);

  const std::vector<se::DeviceMemoryBase>& input_buffers() const {
    return input_buffers_;
  }

  const std::vector<Shape>& input_shapes() const { return input_shapes_; }

  const std::vector<se::DeviceMemoryBase>& output_buffers() const {
    return output_buffers_;
  }

  const Shape& output_shape() const { return output_shape_; }
  se::RedzoneAllocator& RedzoneAllocator() const { return *redzone_allocator_; }

 private:
  absl::Status CreateInputs(
      const HloInstruction::InstructionVector& instructions,
      bool should_init_buffers, int64_t& rng_state);

  absl::Status CreateOutputs(const HloInstruction& instruction,
                             BuffersToCreate buffers_to_create,
                             bool should_init_buffers, int64_t& rng_state);

  std::unique_ptr<se::RedzoneAllocator> redzone_allocator_;
  std::vector<se::DeviceMemoryBase> input_buffers_;
  std::vector<Shape> input_shapes_;
  std::vector<se::DeviceMemoryBase> output_buffers_;
  Shape output_shape_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNING_REDZONE_BUFFERS_H_
