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

#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/conversions.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"

#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/core/lib/strings/stringprintf.h"

#include <fstream>

#include <string.h>

#include <poplar/IPUModel.hpp>
#include <poplar/Tensor.hpp>

/*
 * TensorControl is a structure that maintains state about the location
 * of a tensor - either on the device or cached on the host.
 *
 * Tensorflow/XLA assumes that a tensor is on the device when the device
 * allocator is called (PoplarExecutor::Allocate).  However, Poplar cannot
 * allocate tensors independently of the compiled Engine.  The TensorControl
 * structure tracks where the tensors are.
 *
 * TensorControl has three pieces of interacting state:
 *   on_device: This says whether the data is on the device (in one of the
 *              tensors belonging to the currently loaded engine).  When this
 *              is false, it means the data is being held in the host side
 *              buffer.
 *
 *   input_handle: If the tensor is on_device, and this is not -1, then it
 *                 indicates which of the input tensors of the current engine
 *                 contains the data.
 *
 *   output_handle: If the tensor is on_device, and this is not empty, then it
 *                  indicates which of the output tensors of the current
 *                  engine contains the data.
 *
 *   The states are:
 *     on_device=false :
 *       The data is in the host buffer.  If this buffer is passed as an
 *       argument when an engine is executed then it must be copied to the
 *       device.
 *
 *     on_device=true, input_handle not empty, output_handle is empty :
 *       During the previous engine execution, the data was copied to the
 *       device as one of the arguments.  On the next execution, if the engine
 *       does not change, and the argument index is the same, then the data
 *       does not need to be recopied to the device.  I suspect that this case
 *       is rare.
 *
 *     on_device=true, input_handle is empty, output_handle not empty :
 *       During the last execution, the buffer was allocated to represent one
 *       of the outputs of the engine.  If the host wants to read the data back
 *       then it will have to be retrieved from the device.  If the next
 *       execution changes the engine, then the data will have to be read back.
 *
 *     on_device=true, input_handle not empty, output_handle not empty :
 *       During the last execution, the buffer was an argument to the execution
 *       and was also one of the output parameters.  This typically indicates
 *       that it is a variable (weights/biases) that has been updated in place.
 *       If the next execution doesn't change the engine, and the data is not
 *       read back to the host in between executions, and the data remains as
 *       an argument to the same input number, then the data does not need to be
 *       copied back to the host.  This is the ideal situation when executing an
 *       engine repeatedly with the same set of weights/biases.
 *
 */
namespace se = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace poplarplugin {

std::string
GetInputCopyHandle(int64 parameter, int64 index) {
  return tensorflow::strings::Printf("%lld.%lld", parameter, index);
}

std::string
GetOutputCopyHandle(int64 index) {
  return tensorflow::strings::Printf("%lld", index);
}

host::HostStream *AsPoplarStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream *>(stream->implementation());
}

PoplarExecutor::PoplarExecutor() :
    profile_enabled_(false) {}

PoplarExecutor::~PoplarExecutor() {}

void *PoplarExecutor::Allocate(uint64 size) {
  void* raw_buf = new char[size + sizeof(TensorControl)];
  TensorControl* allocated = new (raw_buf) TensorControl();
  allocated->size = size;
  allocated->on_device = false;
  allocated->input_handle.clear();
  allocated->output_handle.clear();
  allocated->output_convertor = nullptr;
  {
    std::lock_guard<std::recursive_mutex> g(mutex_);
    allocations_.push_back(allocated);
  }
  return allocated;
}

void *PoplarExecutor::AllocateSubBuffer(DeviceMemoryBase *parent,
                                      uint64 offset_bytes, uint64 size_bytes) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(parent->opaque());
  return tc->data + offset_bytes;
}

void PoplarExecutor::Deallocate(DeviceMemoryBase *mem) {
  if (!mem->is_sub_buffer()) {
    TensorControl* tc = reinterpret_cast<TensorControl*>(mem->opaque());
    {
      std::lock_guard <std::recursive_mutex> g(mutex_);
      allocations_.remove(tc);
    }
    tc->~TensorControl();
    delete[] static_cast<char *>(mem->opaque());
  }
}

bool PoplarExecutor::Memcpy(Stream *stream, void *host_dst,
                          const DeviceMemoryBase &pop_src, uint64 size) {
  AsPoplarStream(stream)->EnqueueTask(
      [this, host_dst, pop_src, size]() {
        port::Status ok = SynchronousMemcpy(host_dst, pop_src, size); });
  return true;
}

bool PoplarExecutor::Memcpy(Stream *stream, DeviceMemoryBase *pop_dst,
                          const void *host_src, uint64 size) {
  DeviceMemoryBase dst = *pop_dst;
  AsPoplarStream(stream)->EnqueueTask(
      [this, dst, host_src, size]() mutable {
        port::Status ok = SynchronousMemcpy(&dst, host_src, size); });
  return true;
}

port::Status PoplarExecutor::SynchronousMemcpy(DeviceMemoryBase *pop_dst,
                                               const void *host_src,
                                               uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(pop_dst->opaque());
  memcpy(tc->data, host_src, size);
  {
    std::lock_guard <std::recursive_mutex> g(mutex_);
    tc->on_device = false;
    tc->input_handle.clear();
  }
  return port::Status::OK();
}

port::Status PoplarExecutor::SynchronousMemcpy(void *host_dst,
                                               const DeviceMemoryBase &pop_src,
                                               uint64 size) {
  const TensorControl* tc =
          reinterpret_cast<const TensorControl*>(pop_src.opaque());
  {
    std::lock_guard <std::recursive_mutex> g(mutex_);
    if (tc->on_device == true && !tc->output_handle.empty()) {
      TF_RETURN_IF_ERROR(MoveDeviceToHost(const_cast<TensorControl*>(tc)));
    }
  }
  memcpy(host_dst, tc->data, size);
  return port::Status::OK();
}

bool PoplarExecutor::HostCallback(Stream *stream,
                                std::function<void()> callback) {
  AsPoplarStream(stream)->EnqueueTask(callback);
  return true;
}

bool PoplarExecutor::CreateStreamDependency(Stream *dependent, Stream *other) {
  AsPoplarStream(dependent)->EnqueueTask(
      [other]() { auto ok = other->BlockHostUntilDone(); });
  AsPoplarStream(dependent)->BlockUntilDone();
  return true;
}

bool PoplarExecutor::StartTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Start(stream);
  return true;
}

bool PoplarExecutor::StopTimer(Stream *stream, Timer *timer) {
  dynamic_cast<host::HostTimer *>(timer->implementation())->Stop(stream);
  return true;
}

port::Status PoplarExecutor::BlockHostUntilDone(Stream *stream)  {
  AsPoplarStream(stream)->BlockUntilDone();
  return port::Status::OK();
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

port::Status PoplarExecutor::InitializePoplarDevice(
    int ordinal,
    const tensorflow::IPUOptions::DeviceConfig& cfg) {

  // TODO - if there is a previously configured device then close it

  tensorflow::IPUOptions::DeviceConfig::Type type = cfg.type();

  if (type == tensorflow::IPUOptions::DeviceConfig::DEFAULT) {
    type = tensorflow::IPUOptions::DeviceConfig::CPU;
  }

  switch (type) {
    case tensorflow::IPUOptions::DeviceConfig::IPU:
      return port::Status{
          port::error::INTERNAL,
          tensorflow::strings::Printf(
              "IPU device type not supported on ordinal %d", ordinal)};
    case tensorflow::IPUOptions::DeviceConfig::IPU_MODEL:
    {
      poplar::IPUModel model;
      model.IPUExchangeType =
          poplar::IPUModel::ExchangeType::AGGRESSIVE_MULTICAST;
      poplar_device_ = model.createDevice();
      profile_enabled_ = cfg.enable_profile();
      break;
    }
    case tensorflow::IPUOptions::DeviceConfig::CPU:
      poplar_device_ = poplar::Device::createCPUDevice();
      break;
    default:
      return port::Status{
          port::error::INTERNAL,
          tensorflow::strings::Printf(
              "unrecognized poplar device type for ordinal %d: %d", ordinal,
              type)};
  }
  return port::Status::OK();
}

port::Status PoplarExecutor::GetCompilerReports(std::string& out) {
  std::lock_guard <std::recursive_mutex> g(mutex_);
  while (reports_.size() > 0) {
    out += reports_.front();
    reports_.pop_front();
  }
  return port::Status::OK();
}

void
PoplarExecutor::FlattenedDeviceMemoryList(InputPairList& list,
                                          const xla::Shape& shape,
                                          void* base) {
  TensorControl *tc = static_cast<TensorControl *>(base);
  if (xla::ShapeUtil::IsTuple(shape)) {
    void** ptrs = reinterpret_cast<void**>(tc->data);
    for (unsigned int t=0; t<xla::ShapeUtil::TupleElementCount(shape); t++) {
      void* ptr = ptrs[t];
      FlattenedDeviceMemoryList(list,
                                xla::ShapeUtil::GetTupleElementShape(shape, t),
                                ptr);
    }
  } else {
    list.push_back(std::make_pair(tc, GetInputConversionFunction(shape)));
  }
}

void
PoplarExecutor::CreateArgsHandleMap(ArgsHandleMap& arg_map, const Args& args,
                                    const std::vector<xla::Shape>& shapes) {
  for (unsigned int a=0; a<args.size(); a++) {
    InputPairList bufs;
    FlattenedDeviceMemoryList(bufs, shapes[a],
                              const_cast<void*>(args[a].opaque()));
    for (unsigned i=0; i<bufs.size(); i++) {
      arg_map[GetInputCopyHandle(a, i)] = bufs[i];
    }
  }
}

std::tuple<se::DeviceMemoryBase,int64>
PoplarExecutor::AllocateSingleOutput(xla::DeviceMemoryAllocator* allocator,
                                     const xla::Shape& shape,
                                     const int64 n,
                                     const OutputMap& map,
                                     const Args& args) {
  int64 size(xla::ShapeUtil::ByteSizeOf(shape));
  auto it(map.find(n));
  if (it != map.end()) {
    // The output is an in-place update of one of the inputs
    se::DeviceMemoryBase buf(args[it->second]);
    TensorControl* tc = reinterpret_cast<TensorControl*>(buf.opaque());
    tc->size = size;
    tc->on_device = true;
    tc->output_handle = GetOutputCopyHandle(n);
    tc->output_convertor = GetOutputConversionFunction(shape);
    return std::make_tuple(buf, n+1);
  } else {
    // The output is not one of the inputs
    se::DeviceMemoryBase allocated =
            allocator->Allocate(0, size, false).ConsumeValueOrDie();
    TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());
    tc->size = size;
    tc->on_device = true;
    tc->output_handle = GetOutputCopyHandle(n);
    tc->output_convertor = GetOutputConversionFunction(shape);
    return std::make_tuple(allocated, n+1);
  }
}

std::tuple<se::DeviceMemoryBase,int64>
PoplarExecutor::AllocateOutputBuffer(xla::DeviceMemoryAllocator* allocator,
                                     const xla::Shape& shape,
                                     const int64 n,
                                     const OutputMap& map,
                                     const Args& args) {
  // This needs to allocate buffers of the form that can be fetched by
  // PoplarTransferManager::TransferLiteralFromDevice
  if (shape.element_type() != xla::TUPLE) {
    return AllocateSingleOutput(allocator, shape, n, map, args);
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*)));
    se::DeviceMemoryBase allocated =
            allocator->Allocate(0, size, false).ConsumeValueOrDie();
    TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());

    void** buf = reinterpret_cast<void**>(tc->data);
    int64 new_n = n;
    for (int64 i=0; i<xla::ShapeUtil::TupleElementCount(shape); i++) {
      se::DeviceMemoryBase out;
      std::tie(out, new_n) = AllocateOutputBuffer(allocator,
                                                  shape.tuple_shapes(i),
                                                  new_n,
                                                  map,
                                                  args);
      *buf++ = out.opaque();
    }

    return std::make_tuple(se::DeviceMemoryBase(tc, size), new_n);
  }
}

std::tuple<se::DeviceMemoryBase,int64>
PoplarExecutor::RemapArgs(const xla::Shape& shape,
                          const int64 n,
                          const OutputMap& map,
                          const Args& args) {
  if (shape.element_type() != xla::TUPLE) {
    return std::make_tuple(args[map.at(n)], n+1);
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void *)));
    TensorControl *tc = reinterpret_cast<TensorControl *>(Allocate(size));

    void **buf = reinterpret_cast<void **>(tc->data);
    int64 new_n = n;
    for (int64 i = 0; i < xla::ShapeUtil::TupleElementCount(shape); i++) {
      se::DeviceMemoryBase out;
      std::tie(out, new_n) = RemapArgs(shape.tuple_shapes(i), new_n, map, args);
      *buf++ = out.opaque();
    }

    return std::make_tuple(se::DeviceMemoryBase(tc, size), new_n);
  }
}

port::Status
PoplarExecutor::MoveDeviceToHost(TensorControl* tc) const {
  void* buf(static_cast<void*>(tc->data));
  if (tc->output_convertor) {
    current_engine_->readTensor(tc->output_handle, buf);
    std::vector<char> converted = tc->output_convertor(buf, 0, tc->size);
    memcpy(buf, converted.data(), converted.size());
  } else {
    current_engine_->readTensor(tc->output_handle, buf);
  }
  tc->on_device = false;
  tc->output_handle.clear();
  tc->input_handle.clear();
  return port::Status::OK();
}

port::StatusOr<se::DeviceMemoryBase>
PoplarExecutor::GetTupleBufferByIndex(const se::DeviceMemoryBase& base,
                                      int64 value) {
  const TensorControl* tc =
          reinterpret_cast<const TensorControl*>(base.opaque());
  void** bufs = (void**)tc->data;
  int64 size = reinterpret_cast<const TensorControl*>(bufs[value])->size;

  return se::DeviceMemoryBase(bufs[value], size);
}

port::StatusOr<se::DeviceMemoryBase>
PoplarExecutor::ExecuteEngine(perftools::gputools::StreamExecutor* executor,
                              const std::shared_ptr<poplar::Engine>& engine,
                              xla::DeviceMemoryAllocator* allocator,
                              const xla::Shape& output_shape,
                              const Args& args,
                              const OutputMap& output_map,
                              const std::vector<xla::Shape>& parameter_shapes,
                              bool dump_report_) {

  perftools::gputools::DeviceMemoryBase retbuf;
  int64 tensor_count;

  bool engine_changed(current_engine_ != engine);
  {
    std::lock_guard <std::recursive_mutex> g(mutex_);

    if (engine == NULL) {
      // An empty engine is a graph that just passes its inputs through
      // to its outputs.  A variable reading graph is such a thing.
      std::tie(retbuf, tensor_count) =
              RemapArgs(output_shape, 0, output_map, args);
    } else {
      ArgsHandleMap arg_map;
      CreateArgsHandleMap(arg_map, args, parameter_shapes);

      // Pull previous execution output back from device if:
      // a) the engine is changing
      // b) output buffer isn't an input to the current engine
      // c) output buffer isn't currently in the right place for the new input
      for (const auto &tc : allocations_) {
        if (tc->on_device == true) {
          if (!tc->output_handle.empty()) {
            if (engine_changed) {
              TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
            } else if (tc->input_handle.empty()) {
              TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
            } else if (arg_map.count(tc->input_handle) > 0 &&
                       tc != arg_map[tc->input_handle].first) {
              TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
            }
          } else {
            if (arg_map.count(tc->input_handle) > 0 &&
                tc != arg_map[tc->input_handle].first) {
              // Mark any old inputs as invalid
              tc->input_handle.clear();
              tc->on_device = false;
            }
          }
        }
      }

      current_engine_ = engine;

      // Put data on the device if:
      // a) the engine has changed
      // b) it is not on the device
      // c) it is on the device, but in the wrong place
      for (auto mem : arg_map) {
        TensorControl *tc = mem.second.first;
        if (tc->on_device == false ||
            tc->input_handle != mem.first ||
            engine_changed) {
          void *buf(static_cast<void *>(tc->data));
          ConversionFn fn = mem.second.second;
          if (fn != nullptr) {
            std::vector<char> converted = fn(buf, tc->size, 0);
            current_engine_->writeTensor(mem.first, converted.data());
          } else {
            current_engine_->writeTensor(mem.first, buf);
          }
          tc->on_device = true;
          tc->input_handle = mem.first;
        }
      }

      std::tie(retbuf, tensor_count) = AllocateOutputBuffer(allocator,
                                                            output_shape, 0,
                                                            output_map, args);

      engine->run(0);

      try {
        if (profile_enabled_ && dump_report_) {

          poplar::Engine::ReportOptions opts;
          opts.doLayerWiseProfile = true;
          // TODO enable this using flags? opts.showVariableStorage = true;

          std::stringstream stream;
          engine->report(stream, opts);
          reports_.push_back(stream.str());
        }
      } catch (std::logic_error e) {
        VLOG(2) << "Error producing execution report: " << e.what();
      }
    }
  }

  return retbuf;
}

}  // namespace poplarplugin
}  // namespace gputools
}  // namespace perftools
