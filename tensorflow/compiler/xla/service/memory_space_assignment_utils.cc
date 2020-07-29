/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/memory_space_assignment_utils.h"

namespace xla {

bool MemorySpaceAssignmentUtils::IsIntervalAllowedInAlternateMemory(
    const GlobalDecreasingSizeBestFitHeap::BufferInterval& interval) {
  // If the buffer is a tuple, don't use this algorithm for now. The buffers
  // that are pointed to by the tuple will still use this algorithm.  Because
  // tuples are cheap to place in the alternate memory (they are just pointers)
  // we don't need to use prefetch/evict logic.
  if (interval.buffer->shape().IsTuple()) {
    VLOG(4) << "Keeping value " << interval.buffer->ToShortString()
            << " in default mem because it is a tuple.";
    return false;
  }

  // Don't place scalars in the alternate memory.
  if (ShapeUtil::IsEffectiveScalar(interval.buffer->shape())) {
    VLOG(4) << "Keeping value " << interval.buffer->ToShortString()
            << " in default mem because it is a scalar.";
    return false;
  }

  // The semantics of TupleSelect are weird: TupleSelect doesn't define a
  // buffer, but just forwards the buffers in the either left or right side.
  // This means the two different inputs to TupleSelect must not alias, yet they
  // should be allocated in the same memory space, and both buffers must be kept
  // alive for the entire live range of TupleSelect. Instead, just don't
  // allocate TupleSelect in the alternate memory space.
  // TODO(berkin): Not allocating add-dependencies either since they need to be
  // treated specially. We should revisit this later.
  for (const HloPosition& position : interval.buffer->positions()) {
    if (position.instruction->opcode() == HloOpcode::kTupleSelect ||
        position.instruction->opcode() == HloOpcode::kAddDependency) {
      VLOG(4) << "Keeping value " << interval.buffer->ToShortString()
              << " in default mem because it has a tuple-select or "
              << "add-dependency position.";
      return false;
    }
  }

  // Send and Recv HLOs return a request identifier. These should not be
  // allocated in the alternate memory.
  for (const HloPosition& position : interval.buffer->positions()) {
    if ((position.instruction->opcode() == HloOpcode::kSend ||
         position.instruction->opcode() == HloOpcode::kRecv)) {
      // TODO(berkin): Send/recv buffers need a stable buffer allocation
      // throughout sending/receiving. Disable memory space allocation for these
      // for now.
      if (position.index == ShapeIndex({0})) {
        VLOG(4) << "Keeping value " << interval.buffer->ToShortString()
                << " in default mem because it is a send/recv buffer.";
        return false;
      } else if (position.index == ShapeIndex({1})) {
        VLOG(4) << "Keeping value " << interval.buffer->ToShortString()
                << " in default mem because it is a request identifier for "
                   "send/recv.";
        return false;
      }
    }

    if ((position.instruction->opcode() == HloOpcode::kCollectivePermuteStart ||
         position.instruction->opcode() == HloOpcode::kCollectivePermuteDone)) {
      // Disable memory space allocation for these for now.
      if (position.index == ShapeIndex({0})) {
        VLOG(4) << "Keeping value " << interval.buffer->ToShortString()
                << " in default mem because it is a collective-permute buffer.";
        return false;
      } else if (position.index == ShapeIndex({1})) {
        VLOG(4) << "Keeping value " << interval.buffer->ToShortString()
                << " in default mem because it is a collective-permute buffer.";
        return false;
      }
    }
  }

  return true;
}

}  // namespace xla
