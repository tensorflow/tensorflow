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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SEQUENTIAL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SEQUENTIAL_THUNK_H_

#include <vector>

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// A thunk that wraps a list of sub-thunks. Executing this thunk executes all
// the sub-thunks sequentially. This is useful to implement instructions that
// require multiple kernel launches or library calls.
class SequentialThunk : public Thunk {
 public:
  SequentialThunk(std::vector<std::unique_ptr<Thunk>>&& thunks,
                  const HloInstruction* hlo);
  SequentialThunk(const SequentialThunk&) = delete;
  SequentialThunk& operator=(const SequentialThunk&) = delete;

  const std::vector<std::unique_ptr<Thunk>>& thunks() const { return thunks_; }

  tensorflow::Status Initialize(const GpuExecutable& executable) override;
  tensorflow::Status ExecuteOnStream(
      const BufferAllocations& buffer_allocations, se::Stream* stream) override;

 private:
  // The list of sub-thunks.
  std::vector<std::unique_ptr<Thunk>> thunks_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SEQUENTIAL_THUNK_H_
