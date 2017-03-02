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

// Implementation of HostExecutor class [of those methods not defined in the
// class declaration].
#include "tensorflow/stream_executor/host/host_gpu_executor.h"

#include <string.h>

#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/plugin_registry.h"

namespace gpu = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace host {

HostStream *AsHostStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<HostStream *>(stream->implementation());
}

HostExecutor::HostExecutor(const PluginConfig &plugin_config)
    : plugin_config_(plugin_config) {}

HostExecutor::~HostExecutor() {}

void *HostExecutor::Allocate(uint64 size) { return new char[size]; }

void *HostExecutor::AllocateSubBuffer(DeviceMemoryBase *parent,
                                      uint64 offset_bytes, uint64 size_bytes) {
  return reinterpret_cast<char *>(parent->opaque()) + offset_bytes;
}

void HostExecutor::Deallocate(DeviceMemoryBase *mem) {
  if (!mem->is_sub_buffer()) {
    delete[] static_cast<char *>(mem->opaque());
  }
}

bool HostExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
  memset(location->opaque(), 0, size);
  return true;
}

bool HostExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                     uint64 size) {
  memset(location->opaque(), value, size);
  return true;
}

bool HostExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &gpu_src, uint64 size) {
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  void *src_mem = const_cast<void *>(gpu_src.opaque());
  AsHostStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool HostExecutor::Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst,
                          const void *host_src, uint64 size) {
  void *dst_mem = gpu_dst->opaque();
  // Enqueue the [asynchronous] memcpy on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool HostExecutor::MemcpyDeviceToDevice(Stream *stream,
                                        DeviceMemoryBase *gpu_dst,
                                        const DeviceMemoryBase &gpu_src,
                                        uint64 size) {
  void *dst_mem = gpu_dst->opaque();
  void *src_mem = const_cast<void *>(gpu_src.opaque());
  // Enqueue this [asynchronous] "device-to-device" (i.e., host-to-host, given
  // the nature of the HostExecutor) memcpy  on the stream (HostStream)
  // associated with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(src_mem, dst_mem, size); });
  return true;
}

bool HostExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                           uint64 size) {
  void *gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size]() { memset(gpu_mem, 0, size); });
  return true;
}

bool HostExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                          uint8 pattern, uint64 size) {
  void *gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return true;
}

bool HostExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                            uint32 pattern, uint64 size) {
  void *gpu_mem = location->opaque();
  // Enqueue the [asynchronous] memzero on the stream (HostStream) associated
  // with the HostExecutor.
  AsHostStream(stream)->EnqueueTask(
      [gpu_mem, size, pattern]() { memset(gpu_mem, pattern, size); });
  return true;
}

port::Status HostExecutor::SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                             const void *host_src,
                                             uint64 size) {
  memcpy(gpu_dst->opaque(), host_src, size);
  return port::Status::OK();
}

port::Status HostExecutor::SynchronousMemcpy(void *host_dst,
                                             const DeviceMemoryBase &gpu_src,
                                             uint64 size) {
  memcpy(host_dst, gpu_src.opaque(), size);
  return port::Status::OK();
}

port::Status HostExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src, uint64 size) {
  memcpy(gpu_dst->opaque(), gpu_src.opaque(), size);
  return port::Status::OK();
}

bool HostExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  AsHostStream(stream)->EnqueueTask(callback);
  return true;
}

bool HostExecutor::AllocateStream(Stream *stream) { return true; }

void HostExecutor::DeallocateStream(Stream *stream) {}

bool HostExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  AsHostStream(dependent)->EnqueueTask(
      [other]() { other->BlockHostUntilDone(); });
  AsHostStream(dependent)->BlockUntilDone();
  return true;
}

bool HostExecutor::StartTimer(Stream *stream, Timer *timer) {
  dynamic_cast<HostTimer *>(timer->implementation())->Start(stream);
  return true;
}

bool HostExecutor::StopTimer(Stream *stream, Timer *timer) {
  dynamic_cast<HostTimer *>(timer->implementation())->Stop(stream);
  return true;
}

bool HostExecutor::BlockHostUntilDone(Stream *stream) {
  AsHostStream(stream)->BlockUntilDone();
  return true;
}

DeviceDescription *HostExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  // TODO(rspringer): How to report a value that's based in reality but that
  // doesn't result in thrashing or other badness? 4GiB chosen arbitrarily.
  builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);

  builder.set_clock_rate_ghz(
      static_cast<float>(
          tensorflow::profile_utils::CpuUtils::GetCycleCounterFrequency()) /
      1e9);

  auto built = builder.Build();
  return built.release();
}

bool HostExecutor::SupportsBlas() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::BlasFactory>(kHostPlatformId,
                                                plugin_config_.blas())
      .ok();
}

blas::BlasSupport *HostExecutor::CreateBlas() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::BlasFactory> status =
      registry->GetFactory<PluginRegistry::BlasFactory>(kHostPlatformId,
                                                        plugin_config_.blas());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve BLAS factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

bool HostExecutor::SupportsFft() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::FftFactory>(kHostPlatformId,
                                               plugin_config_.fft())
      .ok();
}

fft::FftSupport *HostExecutor::CreateFft() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::FftFactory> status =
      registry->GetFactory<PluginRegistry::FftFactory>(kHostPlatformId,
                                                       plugin_config_.fft());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve FFT factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

bool HostExecutor::SupportsRng() const {
  return PluginRegistry::Instance()
      ->GetFactory<PluginRegistry::RngFactory>(kHostPlatformId,
                                               plugin_config_.rng())
      .ok();
}

rng::RngSupport *HostExecutor::CreateRng() {
  PluginRegistry *registry = PluginRegistry::Instance();
  port::StatusOr<PluginRegistry::RngFactory> status =
      registry->GetFactory<PluginRegistry::RngFactory>(kHostPlatformId,
                                                       plugin_config_.rng());
  if (!status.ok()) {
    LOG(ERROR) << "Unable to retrieve RNG factory: "
               << status.status().error_message();
    return nullptr;
  }

  return status.ValueOrDie()(this);
}

}  // namespace host
}  // namespace gputools
}  // namespace perftools
