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

#ifndef XLA_SERVICE_GPU_FOR_THUNK_H_
#define XLA_SERVICE_GPU_FOR_THUNK_H_

#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/sequential_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// ForThunk executes 'loop_limit' invocations of 'body_thunk_sequence'.
class ForThunk : public Thunk {
 public:
  ForThunk(ThunkInfo thunk_info, const int64_t loop_limit,
           std::unique_ptr<ThunkSequence> body_thunk_sequence);
  ForThunk(const ForThunk&) = delete;
  ForThunk& operator=(const ForThunk&) = delete;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

  SequentialThunk* body_thunk_sequence() { return body_thunk_sequence_.get(); }

 private:
  const int64_t loop_limit_;
  std::unique_ptr<SequentialThunk> body_thunk_sequence_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FOR_THUNK_H_
