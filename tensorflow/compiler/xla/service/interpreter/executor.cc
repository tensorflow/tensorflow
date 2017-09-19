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

#include "tensorflow/compiler/xla/service/interpreter/executor.h"

#include <cstring>

#include "tensorflow/compiler/xla/status_macros.h"

namespace perftools {
namespace gputools {
namespace interpreter {

host::HostStream *AsExecutorStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream *>(stream->implementation());
}

InterpreterExecutor::InterpreterExecutor(const PluginConfig &plugin_config)
    : plugin_config_(plugin_config) {}

InterpreterExecutor::~InterpreterExecutor() {}

void *InterpreterExecutor::Allocate(uint64 size) { return new char[size]; }

void *InterpreterExecutor::AllocateSubBuffer(DeviceMemoryBase *parent,
                                             uint64 offset_bytes,
                                             uint64 /*size_bytes*/) {
  return parent + offset_bytes;
}

void InterpreterExecutor::Deallocate(DeviceMemoryBase *mem) {
  if (!mem->is_sub_buffer()) {
    delete[] static_cast<char *>(mem->opaque());
  }
}

bool InterpreterExecutor::Memcpy(Stream *stream, void *host_dst,
                                 const DeviceMemoryBase &dev_src, uint64 size) {
  AsExecutorStream(stream)->EnqueueTask([this, host_dst, dev_src, size]() {
    port::Status ok = SynchronousMemcpy(host_dst, dev_src, size);
  });
  return true;
}

bool InterpreterExecutor::Memcpy(Stream *stream, DeviceMemoryBase *dev_dst,
                                 const void *host_src, uint64 size) {
  AsExecutorStream(stream)->EnqueueTask([this, dev_dst, host_src, size]() {
    port::Status ok = SynchronousMemcpy(dev_dst, host_src, size);
  });
  return true;
}

port::Status InterpreterExecutor::SynchronousMemcpy(DeviceMemoryBase *dev_dst,
                                                    const void *host_src,
                                                    uint64 size) {
  memcpy(dev_dst->opaque(), host_src, size);
  return port::Status::OK();
}

port::Status InterpreterExecutor::SynchronousMemcpy(
    void *host_dst, const DeviceMemoryBase &dev_src, uint64 size) {
  memcpy(host_dst, dev_src.opaque(), size);
  return port::Status::OK();
}

bool InterpreterExecutor::HostCallback(Stream *stream,
                                       std::function<void()> callback) {
  AsExecutorStream(stream)->EnqueueTask(callback);
  return true;
}

bool InterpreterExecutor::CreateStreamDependency(Stream *dependent,
                                                 Stream *other) {
  AsExecutorStream(dependent)->EnqueueTask(
      [other]() { other->BlockHostUntilDone(); });
  AsExecutorStream(dependent)->BlockUntilDone();
  return true;
}

bool InterpreterExecutor::StartTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Start(stream);
  return true;
}

bool InterpreterExecutor::StopTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Stop(stream);
  return true;
}

bool InterpreterExecutor::BlockHostUntilDone(Stream *stream) {
  AsExecutorStream(stream)->BlockUntilDone();
  return true;
}

DeviceDescription *InterpreterExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  builder.set_name("Interpreter");
  builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);
  builder.set_clock_rate_ghz(static_cast<float>(CLOCKS_PER_SEC) / 1e9);

  return builder.Build().release();
}

}  // namespace interpreter
}  // namespace gputools
}  // namespace perftools
