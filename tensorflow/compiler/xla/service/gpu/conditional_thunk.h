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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONDITIONAL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONDITIONAL_THUNK_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// ConditionalThunk implements the conditional instruction on GPU by reading the
// predicate of the conditional and executing the true or the false computation
// depending on the value of the predicate.
//
// ConditionalThunk assumes that the buffers of the conditional result and the
// result of the true and false computations share the same allocation. Also,
// the buffers of the true operand of the conditional and that of the parameter
// instruction of the true computation share the same allocation. Similarly, the
// buffers of the false operand and that of the parameter instruction of the
// false computation share the same allocation.
class ConditionalThunk : public Thunk {
 public:
  ConditionalThunk(
      const BufferAllocation::Slice& branch_index_buffer_index,
      absl::Span<const BufferAllocation::Slice> branch_operand_buffer_indexes,
      std::vector<ThunkSequence> branch_thunk_sequences,
      const HloInstruction* hlo);

  ConditionalThunk(const ConditionalThunk&) = delete;
  ConditionalThunk& operator=(const ConditionalThunk&) = delete;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const bool branch_index_is_bool_;
  BufferAllocation::Slice branch_index_buffer_index_;
  std::vector<BufferAllocation::Slice> branch_operand_buffer_indexes_;
  std::vector<std::unique_ptr<SequentialThunk>> branch_thunks_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CONDITIONAL_THUNK_H_
