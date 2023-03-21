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

#include "tensorflow/compiler/xla/backends/interpreter/executor.h"

#include <cstring>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace stream_executor {
namespace interpreter {

host::HostStream *AsExecutorStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream *>(stream->implementation());
}

XlaInterpreterExecutor::XlaInterpreterExecutor(
    const PluginConfig &plugin_config)
    : plugin_config_(plugin_config) {}

XlaInterpreterExecutor::~XlaInterpreterExecutor() {}

DeviceMemoryBase XlaInterpreterExecutor::Allocate(uint64_t size,
                                                  int64_t memory_space) {
  return DeviceMemoryBase(new char[size], size);
}

void *XlaInterpreterExecutor::GetSubBuffer(DeviceMemoryBase *parent,
                                           uint64_t offset_bytes,
                                           uint64_t /*size_bytes*/) {
  return parent + offset_bytes;
}

void XlaInterpreterExecutor::Deallocate(DeviceMemoryBase *mem) {
  delete[] static_cast<char *>(mem->opaque());
}

bool XlaInterpreterExecutor::Memcpy(Stream *stream, void *host_dst,
                                    const DeviceMemoryBase &dev_src,
                                    uint64_t size) {
  AsExecutorStream(stream)->EnqueueTask([this, host_dst, dev_src, size]() {
    // Ignore errors.
    tsl::Status ok = SynchronousMemcpy(host_dst, dev_src, size);
  });
  tsl::Status status = AsExecutorStream(stream)->BlockUntilDone();
  if (status.ok()) {
    return true;
  }

  // TODO(b/199316985): Return 'tsl::Status' instead of 'bool', so we don't need
  // to throw away error information here.
  LOG(WARNING) << "Memcpy: error on stream: " << status;
  return false;
}

bool XlaInterpreterExecutor::Memcpy(Stream *stream, DeviceMemoryBase *dev_dst,
                                    const void *host_src, uint64_t size) {
  AsExecutorStream(stream)->EnqueueTask([this, dev_dst, host_src, size]() {
    // Ignore errors.
    tsl::Status ok = SynchronousMemcpy(dev_dst, host_src, size);
  });
  tsl::Status status = AsExecutorStream(stream)->BlockUntilDone();
  if (status.ok()) {
    return true;
  }

  // TODO(b/199316985): Return 'tsl::Status' instead of 'bool', so we don't need
  // to throw away error information here.
  LOG(WARNING) << "Memcpy: error on stream: " << status;
  return false;
}

tsl::Status XlaInterpreterExecutor::SynchronousMemcpy(DeviceMemoryBase *dev_dst,
                                                      const void *host_src,
                                                      uint64_t size) {
  memcpy(dev_dst->opaque(), host_src, size);
  return ::tsl::OkStatus();
}

tsl::Status XlaInterpreterExecutor::SynchronousMemcpy(
    void *host_dst, const DeviceMemoryBase &dev_src, uint64_t size) {
  memcpy(host_dst, dev_src.opaque(), size);
  return ::tsl::OkStatus();
}

bool XlaInterpreterExecutor::HostCallback(
    Stream *stream, absl::AnyInvocable<tsl::Status() &&> callback) {
  AsExecutorStream(stream)->EnqueueTaskWithStatus(std::move(callback));
  return true;
}

bool XlaInterpreterExecutor::CreateStreamDependency(Stream *dependent,
                                                    Stream *other) {
  AsExecutorStream(dependent)->EnqueueTaskWithStatus(
      [other]() { return other->BlockHostUntilDone(); });
  tsl::Status status = AsExecutorStream(dependent)->BlockUntilDone();
  if (status.ok()) {
    return true;
  }

  // TODO(b/199316985): Return 'tsl::Status' instead of 'bool', so we don't need
  // to throw away error information here.
  LOG(WARNING) << "CreateStreamDependency: error on stream: " << status;
  return false;
}

bool XlaInterpreterExecutor::StartTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Start(stream);
  return true;
}

bool XlaInterpreterExecutor::StopTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Stop(stream);
  return true;
}

tsl::Status XlaInterpreterExecutor::BlockHostUntilDone(Stream *stream) {
  return AsExecutorStream(stream)->BlockUntilDone();
}

tsl::StatusOr<std::unique_ptr<DeviceDescription>>
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
