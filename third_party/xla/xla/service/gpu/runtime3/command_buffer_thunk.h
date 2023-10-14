/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_THUNK_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/gpu/runtime3/command_buffer_cmd.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/stream_executor_pimpl.h"

namespace xla::gpu {

class CommandBufferThunk : public Thunk {
 public:
  explicit CommandBufferThunk(CommandBufferCmdSequence commands,
                              ThunkInfo thunk_info);

  Status Initialize(se::StreamExecutor*, ExecutableSource) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  // se::CommandBuffer is not thread safe, and we guard it with a mutex to
  // guarantee that we do not mutate it concurrently.
  struct State {
    explicit State(se::CommandBuffer command_buffer);

    absl::Mutex mutex;
    se::CommandBuffer command_buffer ABSL_GUARDED_BY(mutex);
  };
  using OwnedCommandBuffer = std::unique_ptr<State>;

  // Returns a command buffer instantiated for `executor` or creates new one.
  StatusOr<State*> GetOrCreateCommandBuffer(se::StreamExecutor* executor);

  // Command sequence that initializes command buffers on each executor.
  CommandBufferCmdSequence commands_;

  // Command buffer sequence instantiates command buffers on all executors.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, OwnedCommandBuffer> command_buffers_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_THUNK_H_
