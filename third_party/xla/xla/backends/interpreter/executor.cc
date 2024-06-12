/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/interpreter/executor.h"

#include <cstring>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "xla/status_macros.h"

namespace stream_executor {
namespace interpreter {

host::HostStream *AsExecutorStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream *>(stream);
}

DeviceMemoryBase XlaInterpreterExecutor::Allocate(uint64_t size,
                                                  int64_t memory_space) {
  return DeviceMemoryBase(new char[size], size);
}

void XlaInterpreterExecutor::Deallocate(DeviceMemoryBase *mem) {
  delete[] static_cast<char *>(mem->opaque());
}

absl::Status XlaInterpreterExecutor::Memcpy(Stream *stream, void *host_dst,
                                            const DeviceMemoryBase &dev_src,
                                            uint64_t size) {
  AsExecutorStream(stream)->EnqueueTask([this, host_dst, dev_src, size]() {
    // Ignore errors.
    absl::Status ok = SynchronousMemcpy(host_dst, dev_src, size);
  });
  return AsExecutorStream(stream)->BlockUntilDone();
}

absl::Status XlaInterpreterExecutor::Memcpy(Stream *stream,
                                            DeviceMemoryBase *dev_dst,
                                            const void *host_src,
                                            uint64_t size) {
  AsExecutorStream(stream)->EnqueueTask([this, dev_dst, host_src, size]() {
    // Ignore errors.
    absl::Status ok = SynchronousMemcpy(dev_dst, host_src, size);
  });
  return AsExecutorStream(stream)->BlockUntilDone();
}

absl::Status XlaInterpreterExecutor::SynchronousMemcpy(
    DeviceMemoryBase *dev_dst, const void *host_src, uint64_t size) {
  memcpy(dev_dst->opaque(), host_src, size);
  return absl::OkStatus();
}

absl::Status XlaInterpreterExecutor::SynchronousMemcpy(
    void *host_dst, const DeviceMemoryBase &dev_src, uint64_t size) {
  memcpy(host_dst, dev_src.opaque(), size);
  return absl::OkStatus();
}

bool XlaInterpreterExecutor::HostCallback(
    Stream *stream, absl::AnyInvocable<absl::Status() &&> callback) {
  AsExecutorStream(stream)->EnqueueTaskWithStatus(std::move(callback));
  return true;
}

absl::Status XlaInterpreterExecutor::BlockHostUntilDone(Stream *stream) {
  return AsExecutorStream(stream)->BlockUntilDone();
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
XlaInterpreterExecutor::CreateDeviceDescription(int device_ordinal) {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  builder.set_name("Interpreter");
  builder.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);
  builder.set_clock_rate_ghz(static_cast<float>(CLOCKS_PER_SEC) / 1e9);

  return builder.Build();
}

}  // namespace interpreter
}  // namespace stream_executor
