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
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"

#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/core/lib/strings/stringprintf.h"

#include <fstream>

#include <string.h>

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
 *   output_handle: If the tensor is on_device, and this is not -1, then it
 *                  indicates which of the output tensors of the current
 *                  engine contains the data.
 *
 *   The states are:
 *     on_device=false :
 *       The data is in the host buffer.  If this buffer is passed as an
 *       argument when an engine is executed then it must be copied to the
 *       device.
 *
 *     on_device=true, input_handle!=-1, output_handle==-1 :
 *       During the previous engine execution, the data was copied to the
 *       device as one of the arguments.  On the next execution, if the engine
 *       does not change, and the argument index is the same, then the data
 *       does not need to be recopied to the device.  I suspect that this case
 *       is rare.
 *
 *     on_device=true, input_handle==-1, output_handle!=-1 :
 *       During the last execution, the buffer was allocated to represent one
 *       of the outputs of the engine.  If the host wants to read the data back
 *       then it will have to be retreived from the device.  If the next
 *       execution changes the engine, then the data will have to be read back.
 *
 *     on_device=true, input_handle!=-1, output_handle!=-1 :
 *       During the last execution, the buffer was an argument to the execution
 *       and was also one of the output parameters.  This typically indicates
 *       that it is a variable (weights/biases) that has been updated in place.
 *       If the next execution doesn't change the engine, and the data is not
 *       read back to the host inbetween, then the data does not need to be
 *       copied back to the host.  This is the ideal situation when executing
 *       an engine repeatedly with the same set of weights/biases.
 *
 */
namespace se = ::perftools::gputools;

namespace perftools {
namespace gputools {
namespace poplarplugin {

std::string
GetCopyHandle(int64 i) {
  static const std::string handles[10] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  if (i < 10) return handles[i];
  return tensorflow::strings::Printf("%lld", i);
}

host::HostStream *AsPoplarStream(Stream *stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream *>(stream->implementation());
}

PoplarExecutor::PoplarExecutor(const PluginConfig &plugin_config)
    : plugin_config_(plugin_config),
      report_counter(0) {
}

PoplarExecutor::~PoplarExecutor() {}

void *PoplarExecutor::Allocate(uint64 size) {
  void* raw_buf = new char[size + sizeof(TensorControl)];
  TensorControl* allocated = reinterpret_cast<TensorControl*>(raw_buf);
  allocated->size = size;
  allocated->on_device = false;
  allocated->input_handle = -1;
  allocated->output_handle = -1;
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
  AsPoplarStream(stream)->EnqueueTask(
      [this, pop_dst, host_src, size]() {
        port::Status ok = SynchronousMemcpy(pop_dst, host_src, size); });
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
    tc->input_handle = -1;
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
    if (tc->on_device) {
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
      [other]() { other->BlockHostUntilDone(); });
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

se::DeviceMemoryBase
PoplarExecutor::AllocateSingleOutput(const xla::Shape& shape,
                                     int64 n,
                                     ConversionFn convertor_fn,
                                     const OutputMap& map,
                                     const Args& args) {
  auto it(map.find(n));
  if (it != map.end()) {
    se::DeviceMemoryBase buf(args[it->second]);
    TensorControl* tc = reinterpret_cast<TensorControl*>(buf.opaque());
    tc->on_device = true;
    tc->output_handle = n;
    tc->output_convertor = convertor_fn;
    return buf;
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape));
    TensorControl* tc =
            reinterpret_cast<TensorControl*>(Allocate(size));
    tc->on_device = true;
    tc->output_handle = n;
    tc->output_convertor = convertor_fn;
    return se::DeviceMemoryBase(tc, size);
  }
}

port::StatusOr<se::DeviceMemoryBase>
PoplarExecutor::AllocateOutputBuffer(const xla::Shape& shape,
                                     const OutputMap& map,
                                     const ConversionList& output_convertors,
                                     const Args& args) {

  if (shape.element_type() != xla::TUPLE) {
    return AllocateSingleOutput(shape, 0, output_convertors[0], map, args);
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*)));
    TensorControl* tc = reinterpret_cast<TensorControl*>(Allocate(size));

    void** buf = reinterpret_cast<void**>(tc->data);
    for (int64 n=0; n<xla::ShapeUtil::TupleElementCount(shape); n++) {
      se::DeviceMemoryBase out(AllocateSingleOutput(shape.tuple_shapes(n),
                                                    n,
                                                    output_convertors[n],
                                                    map,
                                                    args));
      *buf++ = out.opaque();
    }

    return se::DeviceMemoryBase(tc, size);
  }
}

port::StatusOr<DeviceMemoryBase>
PoplarExecutor::RemapArgs(const xla::Shape& shape,
                          const OutputMap& output_map,
                          const Args& args) {
  if (shape.element_type() != xla::TUPLE) {
    return args[0];
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void *)));
    TensorControl *tc = reinterpret_cast<TensorControl *>(Allocate(size));

    void **buf = reinterpret_cast<void **>(tc->data);
    for (int64 n = 0; n < xla::ShapeUtil::TupleElementCount(shape); n++) {
      se::DeviceMemoryBase out(args[n]);
      *buf++ = out.opaque();
    }

    return se::DeviceMemoryBase(tc, size);

  }
}

port::Status
PoplarExecutor::MoveDeviceToHost(TensorControl* tc) const {
  if (tc->on_device == true && tc->output_handle != -1) {
    void* buf(static_cast<void*>(tc->data));
    if (tc->output_convertor) {
      current_engine_->readTensor(GetCopyHandle(tc->output_handle), buf);
      std::vector<char> converted = tc->output_convertor(buf, 0, tc->size);
      memcpy(buf, converted.data(), converted.size());
    } else {
      current_engine_->readTensor(GetCopyHandle(tc->output_handle), buf);
    }
    tc->on_device = false;
    tc->output_handle = -1;
    return port::Status::OK();
  } else {
    return tensorflow::errors::Internal("Tensor not on device");
  }
}

port::StatusOr<se::DeviceMemoryBase>
PoplarExecutor::ExecuteEngine(const std::shared_ptr<poplar::Engine>& engine,
                              const xla::Shape& shape,
                              const Args& args,
                              const OutputMap& output_map,
                              const ConversionList& input_convertors,
                              const ConversionList& output_convertors) {

  perftools::gputools::DeviceMemoryBase retbuf;

  bool engine_changed(current_engine_ != engine);
  {
    std::lock_guard <std::recursive_mutex> g(mutex_);

    if (engine == NULL) {
      // An empty engine is a graph that just passes its inputs through
      // to its outputs.  A variable reading graph is such a thing.
      TF_ASSIGN_OR_RETURN(retbuf,
                          RemapArgs(shape, output_map, args));

    } else {
      // Pull previous execution output back from device if:
      // a) the engine is changing
      // b) output buffer isn't an input to the current engine
      // c) output buffer isn't currently in the right place for the new input
      for (const auto &tc : allocations_) {
        if (tc->on_device == true && tc->output_handle != -1) {
          if (engine_changed) {
            TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
          } else if (tc->input_handle == -1) {
            TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
          } else if ((void *) tc != args[tc->input_handle].opaque()) {
            TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
          }
        }
      }

      current_engine_ = engine;

      // Put data on the device if:
      // a) the engine has changed
      // b) it is not on the device
      // c) it is on the device, but in the wrong place
      for (size_t a = 0; a < args.size(); a++) {
        auto mem = args[a];
        TensorControl *tc = reinterpret_cast<TensorControl *>(mem.opaque());
        if (tc->on_device == false || tc->input_handle != (int64)a || engine_changed) {
          void *buf(static_cast<void *>(tc->data));
          if (input_convertors[a]) {
            std::vector<char> converted = input_convertors[a](buf, tc->size, 0);
            current_engine_->writeTensor(GetCopyHandle(a), converted.data());
          } else {
            current_engine_->writeTensor(GetCopyHandle(a), buf);
          }
          tc->on_device = true;
          tc->input_handle = a;
        }
      }

      TF_ASSIGN_OR_RETURN(retbuf,
                          AllocateOutputBuffer(shape,
                                               output_map,
                                               output_convertors,
                                               args));

      engine->run(0);

      try {
        const char *report_prefix = getenv("TF_POPLAR_REPORT_FILENAME");
        if (report_prefix != NULL) {
          std::string report_filename(report_prefix);
          report_filename.append(
                  tensorflow::strings::Printf("_%d", report_counter));

          std::ofstream stream;
          stream.open(report_filename);

          poplar::Engine::ReportOptions opts;
          opts.doLayerWiseProfile = true;
          engine->report(stream, opts);

          report_counter++;
        }
      } catch (std::logic_error e) {
        LOG(WARNING) << "Error producing execution report: " << e.what();
      }
    }
  }

  return retbuf;
}

}  // namespace poplarplugin
}  // namespace gputools
}  // namespace perftools
