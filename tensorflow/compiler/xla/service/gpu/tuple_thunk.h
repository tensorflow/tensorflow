/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TUPLE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TUPLE_THUNK_H_

#include <vector>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// A thunk that copies the addresses of tuple elements to the buffer of the
// tuple. This avoids emitting kernels that may suffer from the parameter space
// issue (b/31336476).
class TupleThunk : public Thunk {
 public:
  TupleThunk(tensorflow::gtl::ArraySlice<BufferAllocation::Slice>
                 tuple_element_buffers,
             const BufferAllocation::Slice& dest_buffer,
             const HloInstruction* hlo_instruction)
      : Thunk(Kind::kTuple, hlo_instruction),
        tuple_element_buffers_(tuple_element_buffers.begin(),
                               tuple_element_buffers.end()),
        dest_buffer_(dest_buffer) {}

  TupleThunk(const TupleThunk&) = delete;
  TupleThunk& operator=(const TupleThunk&) = delete;

  Status ExecuteOnStream(const BufferAllocations& buffer_allocations,
                         se::Stream* stream) override;

 private:
  const std::vector<BufferAllocation::Slice> tuple_element_buffers_;
  const BufferAllocation::Slice dest_buffer_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TUPLE_THUNK_H_
