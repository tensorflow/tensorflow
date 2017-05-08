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

#include "tensorflow/compiler/plugin/example/executor.h"
#include "tensorflow/compiler/plugin/example/platform_id.h"

#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/core/lib/strings/stringprintf.h"

#include <fstream>

#include <string.h>

namespace se = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace exampleplugin {

host::HostStream *AsExampleStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream *>(stream->implementation());
}

ExampleExecutor::ExampleExecutor(const PluginConfig &plugin_config)
    : plugin_config_(plugin_config) {
}

ExampleExecutor::~ExampleExecutor() {}

void *ExampleExecutor::Allocate(uint64 size) {
  void* buf = new char[size];
  return buf;
}

void *ExampleExecutor::AllocateSubBuffer(DeviceMemoryBase *parent,
                                      uint64 offset_bytes, uint64 size_bytes) {
  return parent + offset_bytes;
}

void ExampleExecutor::Deallocate(DeviceMemoryBase *mem) {
  if (!mem->is_sub_buffer()) {
    delete[] static_cast<char *>(mem->opaque());
  }
}

bool ExampleExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &dev_src, uint64 size) {
  AsExampleStream(stream)->EnqueueTask(
      [this, host_dst, dev_src, size]() {
        port::Status ok = SynchronousMemcpy(host_dst, dev_src, size); });
  return true;
}

bool ExampleExecutor::Memcpy(Stream *stream, DeviceMemoryBase *dev_dst,
                          const void *host_src, uint64 size) {
  AsExampleStream(stream)->EnqueueTask(
      [this, dev_dst, host_src, size]() {
        port::Status ok = SynchronousMemcpy(dev_dst, host_src, size); });
  return true;
}

port::Status ExampleExecutor::SynchronousMemcpy(DeviceMemoryBase *dev_dst,
                                                const void *host_src,
                                                uint64 size) {
  // Copy to device
  return port::Status::OK();
}

port::Status ExampleExecutor::SynchronousMemcpy(void *host_dst,
                                                const DeviceMemoryBase &dev_src,
                                                uint64 size) {
  // Copy from device
  return port::Status::OK();
}

bool ExampleExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  AsExampleStream(stream)->EnqueueTask(callback);
  return true;
}

bool ExampleExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  AsExampleStream(dependent)->EnqueueTask(
      [other]() { other->BlockHostUntilDone(); });
  AsExampleStream(dependent)->BlockUntilDone();
  return true;
}

bool ExampleExecutor::StartTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Start(stream);
  return true;
}

bool ExampleExecutor::StopTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Stop(stream);
  return true;
}

bool ExampleExecutor::BlockHostUntilDone(Stream *stream) {
  AsExampleStream(stream)->BlockUntilDone();
  return true;
}

DeviceDescription *ExampleExecutor::PopulateDeviceDescription() const {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  builder.set_name("Example");
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

DeviceMemoryBase
ExampleExecutor::AllocateSingleOutput(const xla::Shape& shape) {
  int64 size(xla::ShapeUtil::ByteSizeOf(shape));
  void* buf = Allocate(size);
  memset(buf, 0, size);
  return se::DeviceMemoryBase(buf, size);
}

port::StatusOr<DeviceMemoryBase>
ExampleExecutor::AllocateOutputBuffer(const xla::Shape& shape) {

  if (shape.element_type() != xla::TUPLE) {
    return AllocateSingleOutput(shape);
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*)));
    void** buf = reinterpret_cast<void**>(Allocate(size));
    for (int64 n=0; n<xla::ShapeUtil::TupleElementCount(shape); n++) {
      se::DeviceMemoryBase out(AllocateSingleOutput(shape.tuple_shapes(n)));
      *buf++ = out.opaque();
    }

    return DeviceMemoryBase(buf, size);
  }
}

port::StatusOr<DeviceMemoryBase>
ExampleExecutor::ExecuteGraph(const xla::Shape& shape,
                              Args args) {

  // Execute the graph here
  DeviceMemoryBase ret;
  TF_ASSIGN_OR_RETURN(ret, AllocateOutputBuffer(shape));
  return ret;
}

}  // namespace exampleplugin
}  // namespace gputools
}  // namespace perftools
