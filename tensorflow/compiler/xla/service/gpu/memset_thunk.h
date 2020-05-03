/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MEMSET_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MEMSET_THUNK_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/stream_executor/stream_executor.h"

// This file contains thunks that set a buffer's elements to a particular value.
// This can be faster than emitting a kernel to set the elements.

namespace xla {
namespace gpu {

// Thunk that zeroes out a given chunk of memory.
class MemzeroThunk : public Thunk {
 public:
  explicit MemzeroThunk(const BufferAllocation::Slice& dest,
                        const HloInstruction* hlo)
      : Thunk(Kind::kMemzero, hlo), dest_(dest) {}

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const BufferAllocation::Slice dest_;
};

// Thunk that sets a given chunk of memory to a particular 32-bit value.  The
// destination chunk must have size divisible by 32 bits.
class Memset32BitValueThunk : public Thunk {
 public:
  explicit Memset32BitValueThunk(uint32 value,
                                 const BufferAllocation::Slice& dest,
                                 const HloInstruction* hlo)
      : Thunk(Kind::kMemset32BitValue, hlo), value_(value), dest_(dest) {}

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  uint32 value_;
  const BufferAllocation::Slice dest_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MEMSET_THUNK_H_
