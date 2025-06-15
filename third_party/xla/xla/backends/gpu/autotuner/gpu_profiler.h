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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_GPU_PROFILER_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_GPU_PROFILER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class GpuProfiler : public Profiler {
 public:
  static std::unique_ptr<GpuProfiler> Create(
      stream_executor::StreamExecutor* stream_executor, ProfileOptions options);

  absl::StatusOr<std::vector<ProfileResult>> ProfileWithSharedBuffers(
      std::vector<std::unique_ptr<Executable>> executables) override;

 private:
  explicit GpuProfiler(
      stream_executor::StreamExecutor* stream_executor,
      std::unique_ptr<stream_executor::DeviceMemoryAllocator> allocator,
      std::unique_ptr<stream_executor::Stream> stream, ProfileOptions options)
      : stream_executor_(stream_executor),
        allocator_(std::move(allocator)),
        stream_(std::move(stream)),
        options_(options) {}

  absl::StatusOr<ExecutionOutput> Execute(Executable* executable,
                                          std::vector<ExecutionInput> inputs,
                                          ExecutionProfile* profile);

  absl::StatusOr<RedzoneBuffers> CreateInputBuffers(
      const Executable* executable);

  absl::StatusOr<ProfileResult> ProfileInternal(Executable* executable,
                                                RedzoneBuffers& buffers);

  stream_executor::StreamExecutor* stream_executor_;
  std::unique_ptr<stream_executor::DeviceMemoryAllocator> allocator_;
  std::unique_ptr<stream_executor::Stream> stream_;
  ProfileOptions options_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_GPU_PROFILER_H_
