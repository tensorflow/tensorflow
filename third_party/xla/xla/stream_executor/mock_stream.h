/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_MOCK_STREAM_H_
#define XLA_STREAM_EXECUTOR_MOCK_STREAM_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/test.h"

namespace stream_executor {

// Implements the Stream interface for testing.
class MockStream : public Stream {
 public:
  MockStream() = default;
  MOCK_METHOD(PlatformSpecificHandle, platform_specific_handle, (),
              (const, override));
  MOCK_METHOD(bool, ok, (), (const, override));
  MOCK_METHOD(absl::Status, RefreshStatus, (), (override));
  MOCK_METHOD(absl::StatusOr<Stream *>, GetOrCreateSubStream, (), (override));
  MOCK_METHOD(void, ReturnSubStream, (Stream * sub_stream), (override));
  MOCK_METHOD(absl::Status, WaitFor, (Stream * other), (override));
  MOCK_METHOD(absl::Status, WaitFor, (Event * event), (override));
  MOCK_METHOD(absl::Status, RecordEvent, (Event * event), (override));
  MOCK_METHOD(absl::Status, Memcpy,
              (void *host_dst, const DeviceMemoryBase &gpu_src, uint64_t size),
              (override));
  MOCK_METHOD(absl::Status, Memcpy,
              (DeviceMemoryBase * gpu_dst, const void *host_src, uint64_t size),
              (override));
  MOCK_METHOD(absl::Status, Memcpy,
              (DeviceMemoryBase * gpu_dst, const DeviceMemoryBase &gpu_src,
               uint64_t size),
              (override));
  MOCK_METHOD(absl::Status, MemZero,
              (DeviceMemoryBase * location, uint64_t size), (override));
  MOCK_METHOD(absl::Status, Memset32,
              (DeviceMemoryBase * location, uint32_t pattern, uint64_t size),
              (override));
  MOCK_METHOD(absl::Status, BlockHostUntilDone, (), (override));
  MOCK_METHOD(absl::Status, DoHostCallbackWithStatus,
              (absl::AnyInvocable<absl::Status() &&> callback), (override));
  MOCK_METHOD(StreamExecutor *, parent, (), (const, override));
  MOCK_METHOD(CudaComputeCapability, GetCudaComputeCapability, (),
              (const, override));
  MOCK_METHOD(RocmComputeCapability, GetRocmComputeCapability, (),
              (const, override));
  MOCK_METHOD((std::variant<StreamPriority, int>), priority, (),
              (const, override));
  MOCK_METHOD(absl::Status, Launch,
              (const ThreadDim &thread_dims, const BlockDim &block_dims,
               const std::optional<ClusterDim> &cluster_dims, const Kernel &k,
               const KernelArgs &args),
              (override));
  MOCK_METHOD(const std::string &, GetName, (), (const, override));
  MOCK_METHOD(void, SetName, (std::string name), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<EventBasedTimer>>,
              CreateEventBasedTimer, (bool use_delay_kernel), (override));
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MOCK_STREAM_H_
