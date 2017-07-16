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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FOR_THUNK_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FOR_THUNK_H_

#include <vector>

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// ForThunk executes 'loop_limit' invocations of 'body_thunk_sequence'.
class ForThunk : public Thunk {
 public:
  ForThunk(const int64 loop_limit,
           std::unique_ptr<ThunkSequence> body_thunk_sequence,
           const HloInstruction* hlo);
  ForThunk(const ForThunk&) = delete;
  ForThunk& operator=(const ForThunk&) = delete;

  tensorflow::Status Initialize(const GpuExecutable& executable) override;
  tensorflow::Status ExecuteOnStream(
      const BufferAllocations& buffer_allocations,
      perftools::gputools::Stream* stream) override;

 private:
  const int64 loop_limit_;
  std::unique_ptr<SequentialThunk> body_thunk_sequence_;
};

}  // namespace gpu
}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FOR_THUNK_H_
