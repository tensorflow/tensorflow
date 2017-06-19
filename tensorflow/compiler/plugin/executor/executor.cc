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

#include "tensorflow/compiler/plugin/executor/executor.h"
#include "tensorflow/compiler/plugin/executor/platform_id.h"

#include "tensorflow/compiler/xla/status_macros.h"

#include <stdlib.h>
#include <string.h>

namespace se = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace executorplugin {

host::HostStream *AsExecutorStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream *>(stream->implementation());
}

ExecutorExecutor::ExecutorExecutor(const PluginConfig &plugin_config)
    : plugin_config_(plugin_config) {}

ExecutorExecutor::~ExecutorExecutor() {}

void *ExecutorExecutor::Allocate(uint64 size) {
  void *buf = new char[size];
  return buf;
}

void *ExecutorExecutor::AllocateSubBuffer(DeviceMemoryBase *parent,
                                         uint64 offset_bytes,
                                         uint64 size_bytes) {
  return parent + offset_bytes;
}

void ExecutorExecutor::Deallocate(DeviceMemoryBase *mem) {
  if (!mem->is_sub_buffer()) {
    delete[] static_cast<char *>(mem->opaque());
  }
}

bool ExecutorExecutor::Memcpy(Stream *stream, void *host_dst,
                             const DeviceMemoryBase &dev_src, uint64 size) {
  AsExecutorStream(stream)->EnqueueTask([this, host_dst, dev_src, size]() {
    port::Status ok = SynchronousMemcpy(host_dst, dev_src, size);
  });
  return true;
}

bool ExecutorExecutor::Memcpy(Stream *stream, DeviceMemoryBase *dev_dst,
                             const void *host_src, uint64 size) {
  AsExecutorStream(stream)->EnqueueTask([this, dev_dst, host_src, size]() {
    port::Status ok = SynchronousMemcpy(dev_dst, host_src, size);
  });
  return true;
}

port::Status ExecutorExecutor::SynchronousMemcpy(DeviceMemoryBase *dev_dst,
                                                const void *host_src,
                                                uint64 size) {
  memcpy(dev_dst->opaque(), host_src, size);
  return port::Status::OK();
}

port::Status ExecutorExecutor::SynchronousMemcpy(void *host_dst,
                                                const DeviceMemoryBase &dev_src,
                                                uint64 size) {
  memcpy(host_dst, dev_src.opaque(), size);
  return port::Status::OK();
}

bool ExecutorExecutor::HostCallback(Stream *stream,
                                   std::function<void()> callback) {
  AsExecutorStream(stream)->EnqueueTask(callback);
  return true;
}

bool ExecutorExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  AsExecutorStream(dependent)->EnqueueTask(
      [other]() { other->BlockHostUntilDone(); });
  AsExecutorStream(dependent)->BlockUntilDone();
  return true;
}

bool ExecutorExecutor::StartTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Start(stream);
  return true;
}

bool ExecutorExecutor::StopTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Stop(stream);
  return true;
}

bool ExecutorExecutor::BlockHostUntilDone(Stream *stream) {
  AsExecutorStream(stream)->BlockUntilDone();
  return true;
}

DeviceDescription *ExecutorExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  builder.set_name("Executor");
  builder.set_device_vendor("VectorName");
  builder.set_platform_version("1.0");
  builder.set_driver_version("1.0");
  builder.set_runtime_version("1.0");
  builder.set_pci_bus_id("1");
  builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);
  builder.set_clock_rate_ghz(static_cast<float>(CLOCKS_PER_SEC) / 1e9);

  auto built = builder.Build();
  return built.release();
}

}  // namespace executorplugin
}  // namespace gputools
}  // namespace perftools
