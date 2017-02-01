/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/poplar/stream_executor/poplar_executor.h"

#include <string.h>

#include "tensorflow/stream_executor/poplar/poplar_platform_id.h"
#include "tensorflow/compiler/poplar/stream_executor/poplar_stream.h"
#include "tensorflow/compiler/poplar/stream_executor/poplar_timer.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace gpu = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace poplar {

PoplarStream *AsPoplarStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<PoplarStream *>(stream->implementation());
}

PoplarExecutor::PoplarExecutor(const PluginConfig &plugin_config)
    : plugin_config_(plugin_config) {}

PoplarExecutor::~PoplarExecutor() {}

void *PoplarExecutor::Allocate(uint64 size) { return new char[size]; }

void *PoplarExecutor::AllocateSubBuffer(DeviceMemoryBase *parent,
                                      uint64 offset_bytes, uint64 size_bytes) {
  return reinterpret_cast<char *>(parent->opaque()) + offset_bytes;
}

void PoplarExecutor::Deallocate(DeviceMemoryBase *mem) {
  if (!mem->is_sub_buffer()) {
    delete[] static_cast<char *>(mem->opaque());
  }
}

bool PoplarExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
  memset(location->opaque(), 0, size);
  return true;
}

bool PoplarExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                     uint64 size) {
  memset(location->opaque(), value, size);
  return true;
}

bool PoplarExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &gpu_src, uint64 size) {
  void *src_mem = const_cast<void *>(gpu_src.opaque());
  AsPoplarStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool PoplarExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                          const void *host_src, uint64 size) {
  void *dst_mem = gpu_dst->opaque();
  AsPoplarStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool PoplarExecutor::MemcpyDeviceToDevice(Stream *stream,
                                        DeviceMemoryBase *gpu_dst,
                                        const DeviceMemoryBase &gpu_src,
                                        uint64 size) {
  void *dst_mem = gpu_dst->opaque();
  void *src_mem = const_cast<void *>(gpu_src.opaque());
  AsPoplarStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(src_mem, dst_mem, size); });
  return true;
}

bool PoplarExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                           uint64 size) {
  void *gpu_mem = location->opaque();
  AsPoplarStream(stream)->EnqueueTask(
      [gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return true;
}

bool PoplarExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                          uint8 pattern, uint64 size) {
  void *gpu_mem = location->opaque();
  AsPoplarStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return true;
}

bool PoplarExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                            uint32 pattern, uint64 size) {
  void *gpu_mem = location->opaque();
  AsPoplarStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return true;
}

port::Status PoplarExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                             const void *host_src,
                                             uint64 size) {
  memcpy(gpu_dst->opaque(), host_src, size);
  return port::Status::OK();
}

port::Status PoplarExecutor::SynchronousMemcpy(void *host_dst,
                                             const DeviceMemoryBase &gpu_src,
                                             uint64 size) {
  memcpy(host_dst, gpu_src.opaque(), size);
  return port::Status::OK();
}

port::Status PoplarExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src, uint64 size) {
  memcpy(gpu_dst->opaque(), gpu_src.opaque(), size);
  return port::Status::OK();
}

bool PoplarExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  AsPoplarStream(stream)->EnqueueTask(callback);
  return true;
}

bool PoplarExecutor::AllocateStream(Stream *stream) { return true; }

void PoplarExecutor::DeallocateStream(Stream *stream) {}

bool PoplarExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  AsPoplarStream(dependent)->EnqueueTask(
      [other]() { other->BlockHostUntilDone(); });
  AsPoplarStream(dependent)->BlockUntilDone();
  return true;
}

bool PoplarExecutor::StartTimer(Stream *stream, Timer *timer) {
  dynamic_cast<PoplarTimer *>(timer->implementation())->Start(stream);
  return true;
}

bool PoplarExecutor::StopTimer(Stream *stream, Timer *timer) {
  dynamic_cast<PoplarTimer *>(timer->implementation())->Stop(stream);
  return true;
}

bool PoplarExecutor::BlockHostUntilDone(Stream *stream) {
  AsPoplarStream(stream)->BlockUntilDone();
  return true;
}

DeviceDescription *PoplarExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  // TODO(rspringer): How to report a value that's based in reality but that
  // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
  builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);

  builder.set_clock_rate_ghz(static_cast<float>(CLOCKS_PER_SEC) / 1e9);

  auto built = builder.Build();
  return built.release();
}

}  // namespace poplar
}  // namespace gputools
}  // namespace perftools
