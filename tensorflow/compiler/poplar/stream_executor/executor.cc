/* Copyright 2017 Graphcore Ltd
 */

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

#include "tensorflow/compiler/poplar/stream_executor/executor.h"

#include <string.h>
#include <dlfcn.h>

#include "tensorflow/stream_executor/poplar/poplar_platform_id.h"

#include <poplar/Tensor.hpp>

namespace se = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace poplarplugin {

struct TensorControl {
  bool on_device = false;
  poplar::Tensor device_tensor;
  char data[0];
};

std::string GetPathToGraphProgFile() {
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of( '/' ) + 1);
    path = path + "tf.gp";
    return path;
  }
  return "";
}

PoplarStream *AsPoplarStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<PoplarStream *>(stream->implementation());
}

PoplarExecutor::PoplarExecutor(const PluginConfig &plugin_config)
    : plugin_config_(plugin_config) {
}

PoplarExecutor::~PoplarExecutor() {}

void *PoplarExecutor::Allocate(uint64 size) {
  void* raw_buf = new char[size + sizeof(TensorControl)];
  TensorControl* allocated = reinterpret_cast<TensorControl*>(raw_buf);
  allocated->on_device = false;
  return allocated;
}

void *PoplarExecutor::AllocateSubBuffer(DeviceMemoryBase *parent,
                                      uint64 offset_bytes, uint64 size_bytes) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(parent->opaque());
  return tc->data + offset_bytes;
}

void PoplarExecutor::Deallocate(DeviceMemoryBase *mem) {
  if (!mem->is_sub_buffer()) {
    delete[] static_cast<char *>(mem->opaque());
  }
}

bool
PoplarExecutor::SynchronousMemZero(DeviceMemoryBase *location, uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(location->opaque());
  memset(tc->data, 0, size);
  return true;
}

bool PoplarExecutor::SynchronousMemSet(DeviceMemoryBase *location, int value,
                                     uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(location->opaque());
  memset(tc->data, value, size);
  return true;
}

bool PoplarExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &pop_src, uint64 size) {
  const TensorControl* tc =
          reinterpret_cast<const TensorControl*>(pop_src.opaque());
  const void *src_mem = tc->data;
  AsPoplarStream(stream)->EnqueueTask(
      [host_dst, src_mem, size]() { memcpy(host_dst, src_mem, size); });
  return true;
}

bool PoplarExecutor::Memcpy(Stream *stream, DeviceMemoryBase *pop_dst,
                          const void *host_src, uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(pop_dst->opaque());
  void *dst_mem = tc->data;
  AsPoplarStream(stream)->EnqueueTask(
      [dst_mem, host_src, size]() { memcpy(dst_mem, host_src, size); });
  return true;
}

bool PoplarExecutor::MemcpyDeviceToDevice(Stream *stream,
                                        DeviceMemoryBase *pop_dst,
                                        const DeviceMemoryBase &pop_src,
                                        uint64 size) {
  const TensorControl* tc_src =
          reinterpret_cast<const TensorControl*>(pop_src.opaque());
  TensorControl* tc_dst = reinterpret_cast<TensorControl*>(pop_dst->opaque());
  void *dst_mem = tc_dst->data;
  const void *src_mem = tc_src->data;
  AsPoplarStream(stream)->EnqueueTask(
      [src_mem, dst_mem, size]() { memcpy(dst_mem, src_mem, size); });
  return true;
}

bool PoplarExecutor::MemZero(Stream *stream, DeviceMemoryBase *location,
                           uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(location->opaque());
  void *pop_mem = tc->data;
  AsPoplarStream(stream)->EnqueueTask(
      [pop_mem, size]() { memset(pop_mem, 0, size); });
  return true;
}

bool PoplarExecutor::Memset(Stream *stream, DeviceMemoryBase *location,
                          uint8 pattern, uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(location->opaque());
  void *pop_mem = tc->data;
  AsPoplarStream(stream)->EnqueueTask(
      [pop_mem, size, pattern]() { memset(pop_mem, pattern, size); });
  return true;
}

bool PoplarExecutor::Memset32(Stream *stream, DeviceMemoryBase *location,
                            uint32 pattern, uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(location->opaque());
  void *pop_mem = tc->data;
  AsPoplarStream(stream)->EnqueueTask(
      [pop_mem, size, pattern]() { memset(pop_mem, pattern, size); });
  return true;
}

port::Status PoplarExecutor::SynchronousMemcpy(DeviceMemoryBase *pop_dst,
                                             const void *host_src,
                                             uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(pop_dst->opaque());

  memcpy(tc->data, host_src, size);
  return port::Status::OK();
}

port::Status PoplarExecutor::SynchronousMemcpy(void *host_dst,
                                             const DeviceMemoryBase &pop_src,
                                             uint64 size) {
  const TensorControl* tc =
          reinterpret_cast<const TensorControl*>(pop_src.opaque());
  memcpy(host_dst, tc->data, size);
  return port::Status::OK();
}

port::Status PoplarExecutor::SynchronousMemcpyDeviceToDevice(
    DeviceMemoryBase *pop_dst, const DeviceMemoryBase &pop_src, uint64 size) {
  TensorControl* tc_dst = reinterpret_cast<TensorControl*>(pop_dst->opaque());
  const TensorControl* tc_src =
          reinterpret_cast<const TensorControl*>(pop_src.opaque());

  memcpy(tc_dst->data, tc_src->data, size);
  return port::Status::OK();
}

bool PoplarExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  AsPoplarStream(stream)->EnqueueTask(callback);
  return true;
}

bool PoplarExecutor::AllocateStream(Stream *stream) {
  return true;
}

void PoplarExecutor::DeallocateStream(Stream *stream) {
}

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

  // TODO populate dynamic values correctly
  builder.set_name("Poplar");
  builder.set_device_vendor("Graphcore");
  builder.set_platform_version("1.0");
  builder.set_driver_version("1.0");
  builder.set_runtime_version("1.0");
  builder.set_pci_bus_id("1");
  builder.set_device_memory_size(static_cast<uint64>(4) * 1024 * 1024 * 1024);
  builder.set_clock_rate_ghz(static_cast<float>(CLOCKS_PER_SEC) / 1e9);

  auto built = builder.Build();
  return built.release();
}

port::StatusOr<se::DeviceMemoryBase>
PoplarExecutor::AllocateOutputBuffer(const xla::Shape& shape) {

  if (shape.element_type() != xla::TUPLE) {
    uint64 size(xla::ShapeUtil::ByteSizeOf(shape));
    void* buf(Allocate(size));
    TensorControl* tc = reinterpret_cast<TensorControl*>(buf);
    tc->on_device = true;
    return se::DeviceMemoryBase(buf, size);
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*)));
    TensorControl* tc = reinterpret_cast<TensorControl*>(Allocate(size));

    void** buf = reinterpret_cast<void**>(tc->data);
    for (const auto& s : shape.tuple_shapes()) {
      TensorControl* tc =
              reinterpret_cast<TensorControl*>(
                      Allocate(xla::ShapeUtil::ByteSizeOf(s)));
      tc->on_device = true;
      *buf++ = tc;
    }

    return se::DeviceMemoryBase(tc, size);
  }
}

void
PoplarExecutor::CopyDataToPoplar(DeviceMemoryBase* mem, void* buf) const {
  const TensorControl* tc =
          reinterpret_cast<const TensorControl*>(mem->opaque());
  memcpy(buf, tc->data, mem->size());
}

void
PoplarExecutor::CopyDataFromPoplar(const xla::Shape& shape,
                                   const std::vector<char*>& bufs,
                                   DeviceMemoryBase* mem) const {
  TensorControl* tc = reinterpret_cast<TensorControl*>(mem->opaque());
  if (xla::ShapeUtil::IsTuple(shape)) {
    TensorControl** subs(reinterpret_cast<TensorControl**>(tc->data));

    for (int64 i=0; i<xla::ShapeUtil::TupleElementCount(shape); i++) {
      const xla::Shape& sub(xla::ShapeUtil::GetTupleElementShape(shape, i));
      memcpy(subs[i]->data, bufs[i], xla::ShapeUtil::ByteSizeOf(sub));
    }
  } else {
    memcpy(tc->data, bufs[0], xla::ShapeUtil::ByteSizeOf(shape));
  }
}

}  // namespace poplarplugin
}  // namespace gputools
}  // namespace perftools
