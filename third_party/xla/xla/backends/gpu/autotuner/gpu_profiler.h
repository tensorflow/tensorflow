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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct GpuInputBuffers : public InputBuffers {
  // We only use the input buffers from the redzone buffers.
  RedzoneBuffers redzone_buffers;
};

class GpuProfiler : public Profiler {
 public:
  static std::unique_ptr<GpuProfiler> Create(
      stream_executor::StreamExecutor* stream_executor, ProfileOptions options,
      se::DeviceAddressAllocator* external_allocator = nullptr);

  ~GpuProfiler() override;

  // The input buffers shapes are taken from the attatched HloModule to the
  // executable.
  // TODO(b/407494793): Add a better way to get the input buffer shapes.
  absl::StatusOr<std::unique_ptr<InputBuffers>> CreateInputBuffers(
      const Executable* executable,
      const HloInstruction* instr = nullptr) override;

  absl::StatusOr<ProfileResult> Profile(Executable* executable,
                                        const InputBuffers& buffers) override;

  absl::Status CheckInputBuffers(InputBuffers& buffers) override;

  absl::Status CheckOutputBuffer(ScopedShapedBuffer& output,
                                 ScopedShapedBuffer& reference,
                                 float rtol) override;

 private:
  explicit GpuProfiler(
      se::StreamExecutor* stream_executor,
      se::DeviceAddressAllocator* allocator,
      std::unique_ptr<se::DeviceAddressAllocator> owned_allocator,
      se::Stream* stream, ProfileOptions options, int64_t cache_flush_bytes)
      : stream_executor_(stream_executor),
        allocator_(allocator),
        owned_allocator_(std::move(owned_allocator)),
        stream_(stream),
        options_(options),
        cache_flush_bytes_(cache_flush_bytes) {}

  absl::StatusOr<ExecutionOutput> Execute(
      Executable* executable, std::vector<ExecutionInput> inputs,
      ExecutionProfile* profile, se::DeviceAddressAllocator* allocator);

  // Evicts the inputs of the previous run from the L2 cache by overwriting a
  // buffer that is larger than the cache. No-op if cache flushing is disabled.
  absl::Status FlushCaches();

  se::StreamExecutor* stream_executor_;
  se::DeviceAddressAllocator* allocator_;
  std::unique_ptr<se::DeviceAddressAllocator> owned_allocator_;
  se::Stream* stream_;
  ProfileOptions options_;
  int64_t cache_flush_bytes_ = 0;
  bool cache_flush_alloc_failed_ = false;
  bool logged_cache_flush_alloc_failure_ = false;
  // Reused across timed runs only when
  // `!allocator_->AllowsAsynchronousDeallocation()`.
  se::ScopedDeviceAddress<uint8_t> cache_flush_buffer_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_GPU_PROFILER_H_
