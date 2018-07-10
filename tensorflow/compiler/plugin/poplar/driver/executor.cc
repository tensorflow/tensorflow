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
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/hlo_hash.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"

#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/core/lib/strings/stringprintf.h"

#include <fstream>

#include <string.h>

#include <poplar/DeviceSet.hpp>
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
namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

std::string GetInputCopyHandle(int64 parameter, int64 index) {
  return tensorflow::strings::Printf("%lld.%lld", parameter, index);
}

std::string GetOutputCopyHandle(int64 index) {
  return tensorflow::strings::Printf("%lld", index);
}

se::host::HostStream* AsPoplarStream(se::Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<se::host::HostStream*>(stream->implementation());
}

PoplarExecutor::PoplarExecutor()
    : ordinal_(0),
      poplar_device_(poplar::Device::createCPUDevice()),
      poplar_device_hash_(0),
      cache_directory_(std::string()),
      active_xla_device_(nullptr),
      profile_compilation_(false),
      profile_poplar_text_(false),
      profile_execution_(false),
      profile_io_(false) {}

PoplarExecutor::~PoplarExecutor() {}

void* PoplarExecutor::Allocate(uint64 size) {
  void* raw_buf = new char[size + sizeof(TensorControl)];
  TensorControl* allocated = new (raw_buf) TensorControl();
  allocated->size = size;
  allocated->ref_count = 1;
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

void* PoplarExecutor::AllocateSubBuffer(se::DeviceMemoryBase* parent,
                                        uint64 offset_bytes,
                                        uint64 size_bytes) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(parent->opaque());
  return tc->data + offset_bytes;
}

void PoplarExecutor::Deallocate(se::DeviceMemoryBase* mem) {
  if (!mem->is_sub_buffer()) {
    bool free = false;
    TensorControl* tc = reinterpret_cast<TensorControl*>(mem->opaque());
    {
      std::lock_guard<std::recursive_mutex> g(mutex_);
      if (--tc->ref_count == 0) {
        allocations_.remove(tc);
        free = true;
      }
    }
    if (free) {
      tc->~TensorControl();
      delete[] static_cast<char*>(mem->opaque());
    }
  }
}

bool PoplarExecutor::Memcpy(se::Stream* stream, void* host_dst,
                            const se::DeviceMemoryBase& pop_src, uint64 size) {
  AsPoplarStream(stream)->EnqueueTask([this, host_dst, pop_src, size]() {
    Status ok = SynchronousMemcpy(host_dst, pop_src, size);
  });
  return true;
}

bool PoplarExecutor::Memcpy(se::Stream* stream, se::DeviceMemoryBase* pop_dst,
                            const void* host_src, uint64 size) {
  se::DeviceMemoryBase dst = *pop_dst;
  AsPoplarStream(stream)->EnqueueTask([this, dst, host_src, size]() mutable {
    Status ok = SynchronousMemcpy(&dst, host_src, size);
  });
  return true;
}

Status PoplarExecutor::SynchronousMemcpy(se::DeviceMemoryBase* pop_dst,
                                         const void* host_src, uint64 size) {
  TensorControl* tc = reinterpret_cast<TensorControl*>(pop_dst->opaque());
  memcpy(tc->data, host_src, size);
  {
    std::lock_guard<std::recursive_mutex> g(mutex_);
    tc->on_device = false;
    tc->input_handle.clear();
  }
  return Status::OK();
}

Status PoplarExecutor::SynchronousMemcpy(void* host_dst,
                                         const se::DeviceMemoryBase& pop_src,
                                         uint64 size) {
  const TensorControl* tc =
      reinterpret_cast<const TensorControl*>(pop_src.opaque());
  {
    std::lock_guard<std::recursive_mutex> g(mutex_);
    if (tc->on_device == true && !tc->output_handle.empty()) {
      TF_RETURN_IF_ERROR(MoveDeviceToHost(const_cast<TensorControl*>(tc)));
    }
  }
  memcpy(host_dst, tc->data, size);
  return Status::OK();
}

bool PoplarExecutor::HostCallback(se::Stream* stream,
                                  std::function<void()> callback) {
  AsPoplarStream(stream)->EnqueueTask(callback);
  return true;
}

bool PoplarExecutor::CreateStreamDependency(se::Stream* dependent,
                                            se::Stream* other) {
  AsPoplarStream(dependent)->EnqueueTask(
      [other]() { auto ok = other->BlockHostUntilDone(); });
  AsPoplarStream(dependent)->BlockUntilDone();
  return true;
}

bool PoplarExecutor::StartTimer(se::Stream* stream, se::Timer* timer) {
  dynamic_cast<se::host::HostTimer*>(timer->implementation())->Start(stream);
  return true;
}

bool PoplarExecutor::StopTimer(se::Stream* stream, se::Timer* timer) {
  dynamic_cast<se::host::HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

Status PoplarExecutor::BlockHostUntilDone(se::Stream* stream) {
  AsPoplarStream(stream)->BlockUntilDone();
  return Status::OK();
}

bool PoplarExecutor::SynchronizeAllActivity() {
  // TODO actually ensure that all execution has finished
  return true;
}

se::DeviceDescription* PoplarExecutor::PopulateDeviceDescription() const {
  se::internal::DeviceDescriptionBuilder builder;

  builder.set_name("Poplar");
  builder.set_platform_version("1.0");

  auto built = builder.Build();
  return built.release();
}

Status PoplarExecutor::InitializePoplarDevice(
    void* device, const tensorflow::IPUOptions::DeviceConfig& cfg) {
  TF_RETURN_IF_ERROR(ClosePoplarDevice(device));

  tensorflow::IPUOptions::DeviceConfig::Type type = cfg.type();

  poplar::DeviceSet device_set = poplar::DeviceSet::getDeviceSet();

  int num_ipus = cfg.ipu_model_config().num_ipus();
  int tiles_per_ipu = cfg.ipu_model_config().tiles_per_ipu();

  if (num_ipus == 0) {
    num_ipus = 1;
  }

  if (type == tensorflow::IPUOptions::DeviceConfig::DEFAULT) {
    if (device_set.getDevices(poplar::TargetType::IPU, num_ipus).size() > 0) {
      type = tensorflow::IPUOptions::DeviceConfig::IPU;
    } else {
      type = tensorflow::IPUOptions::DeviceConfig::CPU;
    }
  }

  bool opened = false;
  switch (type) {
    case tensorflow::IPUOptions::DeviceConfig::IPU: {
      auto devices = device_set.getDevices(poplar::TargetType::IPU, num_ipus);
      for (auto& d : devices) {
        if (d.attach()) {
          poplar_device_ = d;

          // Temporary fix to prevent many long compilation times
          if (tiles_per_ipu == 0) {
            tiles_per_ipu = 4;
          }

          if (tiles_per_ipu > 0) {
            poplar_device_ = poplar_device_.createVirtualDevice(tiles_per_ipu);
          }

          profile_compilation_ = false;
          profile_poplar_text_ = false;
          profile_execution_ = false;
          profile_io_ = false;
          opened = true;
          break;
        }
      }
      break;
    }
    case tensorflow::IPUOptions::DeviceConfig::IPU_MODEL: {
      poplar::IPUModel model;
      if (num_ipus != 0) {
        model.numIPUs = num_ipus;
      }
      if (tiles_per_ipu != 0) {
        model.tilesPerIPU = tiles_per_ipu;
      }
      poplar_device_ = model.createDevice();
      if (poplar_device_.attach()) {
        profile_compilation_ = cfg.profiling().enable_compilation_trace();
        profile_poplar_text_ = cfg.profiling().enable_poplar_reports_text();
        profile_execution_ = cfg.profiling().enable_execution_trace();
        profile_io_ = cfg.profiling().enable_io_trace();
        opened = true;
      }
      break;
    }
    case tensorflow::IPUOptions::DeviceConfig::CPU: {
      poplar_device_ = poplar::Device::createCPUDevice();
      if (poplar_device_.attach()) {
        profile_compilation_ = false;
        profile_poplar_text_ = false;
        profile_execution_ = false;
        profile_io_ = false;
        opened = true;
      }
      break;
    }
    default:
      return xla::InternalError(
          "Unrecognized poplar device type for ordinal %d: %d", ordinal_, type);
  }

  if (!opened) {
    return xla::ResourceExhausted(
        "Unable to acquire poplar device type for ordinal %d", ordinal_);
  }

  random_type_ = cfg.random_type();

  option_flags_ = poplar::OptionFlags();
  option_flags_.set("target.textSectionSizeInBytes", "0xa000");
  option_flags_.set("target.workerStackSizeInBytes", "0x400");
  for (const auto& opt : cfg.compilation_options()) {
    option_flags_.set(opt.option(), opt.value());
  }

  active_xla_device_ = device;

  cache_directory_ = cfg.engine_cache_directory();

  std::vector<int64> poplar_target;
  const auto& target = poplar_device_.getTarget();
  poplar_target.push_back(target.getNumTiles());
  poplar_target.push_back(target.getDataPathWidth());
  poplar_target.push_back(target.getBytesPerTile());
  poplar_target.push_back(target.getNumWorkerContexts());
  poplar_target.push_back(target.getTilesPerIPU());
  poplar_target.push_back(target.getNumIPUs());
  poplar_target.push_back((unsigned)target.getTargetType());

  for (int64 h : poplar_target) {
    poplar_device_hash_ = tensorflow::Hash64Combine(poplar_device_hash_, h);
  }

  return Status::OK();
}

Status PoplarExecutor::ClosePoplarDevice(void* device) {
  if (device == active_xla_device_) {
    poplar_device_.detach();
    active_xla_device_ = nullptr;
  }
  return Status::OK();
}

bool PoplarExecutor::HaveExecutableCache() const {
  return !cache_directory_.empty();
}

std::string PoplarExecutor::CachedExecutableFilename(
    const HloModule& module) const {
  HloHash module_hash(&module);
  uint64 hash = module_hash.GetHash();
  hash = tensorflow::Hash64Combine(hash, poplar_device_hash_);

  std::string filename = tensorflow::strings::Printf("%0llx.xla_engine", hash);

  return tensorflow::io::JoinPath(cache_directory_, filename);
}

bool PoplarExecutor::HaveCachedExecutable(const std::string& filename) const {
  return false;
}

void PoplarExecutor::AddEventRecord(tensorflow::IpuTraceEvent::Type type,
                                    const std::string& module_name,
                                    const std::string& content, int value) {
  uint64 now = tensorflow::Env::Default()->NowMicros();
  tensorflow::IpuTraceEvent evt;
  evt.set_timestamp(static_cast<double>(now) / 1000000.0);
  evt.set_type(type);
  evt.set_ordinal(ordinal_);
  evt.set_module_name(std::move(module_name));
  evt.set_data_str(std::move(content));
  evt.set_data_int(value);
  reports_.push_back(evt);
}

const poprand::RandomGenMode PoplarExecutor::GetRandomGenMode() const {
  switch (random_type_) {
    case tensorflow::IPUOptions::DeviceConfig::NOT_REPEATABLE:
      return poprand::NOT_REPEATABLE;
    case tensorflow::IPUOptions::DeviceConfig::SYSTEM_REPEATABLE:
      return poprand::SYSTEM_REPEATABLE;
    case tensorflow::IPUOptions::DeviceConfig::ALWAYS_REPEATABLE:
      return poprand::ALWAYS_REPEATABLE;
    default:
      return poprand::NOT_REPEATABLE;
  }
}

Status PoplarExecutor::GetCompilerEvents(
    std::list<tensorflow::IpuTraceEvent>& out) {
  std::lock_guard<std::recursive_mutex> g(mutex_);
  out.splice(out.end(), std::move(reports_));
  reports_.clear();
  return Status::OK();
}

void PoplarExecutor::FlattenedDeviceMemoryList(InputPairList& list,
                                               const xla::Shape& shape,
                                               void* base, bool streamed) {
  TensorControl* tc = static_cast<TensorControl*>(base);
  if (xla::ShapeUtil::IsTuple(shape)) {
    void** ptrs = reinterpret_cast<void**>(tc->data);
    for (unsigned int t = 0; t < xla::ShapeUtil::TupleElementCount(shape);
         t++) {
      void* ptr = ptrs[t];
      FlattenedDeviceMemoryList(
          list, xla::ShapeUtil::GetTupleElementShape(shape, t), ptr, streamed);
    }
  } else {
    list.push_back(InputDef(tc, GetInputConversionFunction(shape), streamed));
  }
}

void PoplarExecutor::CreateArgsHandleMap(
    ArgsHandleMap& arg_map, const Args& args,
    const xla::poplarplugin::PoplarExecutable& executable) {
  const auto* comp = executable.module().entry_computation();
  std::vector<xla::Shape> shapes(comp->num_parameters());
  for (const auto& inst : comp->parameter_instructions()) {
    shapes[inst->parameter_number()] = inst->shape();
  }
  const auto& streamed = executable.ParameterStreamed();
  for (unsigned int a = 0; a < args.size(); a++) {
    InputPairList bufs;
    FlattenedDeviceMemoryList(bufs, shapes[a],
                              const_cast<void*>(args[a].opaque()), streamed[a]);
    for (unsigned i = 0; i < bufs.size(); i++) {
      arg_map[GetInputCopyHandle(a, i)] = bufs[i];
    }
  }
}

std::tuple<se::DeviceMemoryBase, int64> PoplarExecutor::AllocateSingleOutput(
    xla::DeviceMemoryAllocator* allocator, const xla::Shape& shape,
    const int64 n, const OutputMap& map, const Args& args,
    const std::vector<bool>& streamed) {
  int64 size(xla::ShapeUtil::ByteSizeOf(shape));
  auto it(map.find(n));
  if (it != map.end() && args.size() > n) {
    // The output is an in-place update of one of the inputs
    // TODO: is this a multi-threading bug?
    se::DeviceMemoryBase buf(args[it->second]);
    TensorControl* tc = reinterpret_cast<TensorControl*>(buf.opaque());
    tc->size = size;
    tc->on_device = streamed[n] ? false : true;
    tc->ref_count++;
    tc->output_handle = GetOutputCopyHandle(n);
    tc->output_convertor = GetOutputConversionFunction(shape);
    return std::make_tuple(buf, n + 1);
  } else {
    // The output is not one of the inputs
    se::DeviceMemoryBase allocated =
        allocator->Allocate(0, size, false).ConsumeValueOrDie().Forget();
    TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());
    tc->size = size;
    tc->on_device = streamed[n] ? false : true;
    tc->output_handle = GetOutputCopyHandle(n);
    tc->output_convertor = GetOutputConversionFunction(shape);
    return std::make_tuple(allocated, n + 1);
  }
}

std::tuple<se::DeviceMemoryBase, int64> PoplarExecutor::AllocateOutputBuffer(
    xla::DeviceMemoryAllocator* allocator, const xla::Shape& shape,
    const int64 n, const OutputMap& map, const Args& args,
    const std::vector<bool>& streamed) {
  // This needs to allocate buffers of the form that can be fetched by
  // PoplarTransferManager::TransferLiteralFromDevice
  if (shape.element_type() != xla::TUPLE) {
    return AllocateSingleOutput(allocator, shape, n, map, args, streamed);
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*)));
    se::DeviceMemoryBase allocated =
        allocator->Allocate(0, size, false).ConsumeValueOrDie().Forget();
    TensorControl* tc = reinterpret_cast<TensorControl*>(allocated.opaque());

    void** buf = reinterpret_cast<void**>(tc->data);
    int64 new_n = n;
    for (int64 i = 0; i < xla::ShapeUtil::TupleElementCount(shape); i++) {
      se::DeviceMemoryBase out;
      std::tie(out, new_n) = AllocateOutputBuffer(
          allocator, shape.tuple_shapes(i), new_n, map, args, streamed);
      *buf++ = out.opaque();
    }

    return std::make_tuple(se::DeviceMemoryBase(tc, size), new_n);
  }
}

std::tuple<se::DeviceMemoryBase, int64> PoplarExecutor::RemapArgs(
    const xla::Shape& shape, const int64 n, const OutputMap& map,
    const Args& args) {
  if (shape.element_type() != xla::TUPLE) {
    se::DeviceMemoryBase buf = args[map.at(n)];
    TensorControl* tc = reinterpret_cast<TensorControl*>(buf.opaque());
    tc->ref_count++;
    return std::make_tuple(buf, n + 1);
  } else {
    int64 size(xla::ShapeUtil::ByteSizeOf(shape, sizeof(void*)));
    TensorControl* tc = reinterpret_cast<TensorControl*>(Allocate(size));

    void** buf = reinterpret_cast<void**>(tc->data);
    int64 new_n = n;
    for (int64 i = 0; i < xla::ShapeUtil::TupleElementCount(shape); i++) {
      se::DeviceMemoryBase out;
      std::tie(out, new_n) = RemapArgs(shape.tuple_shapes(i), new_n, map, args);
      *buf++ = out.opaque();
    }

    return std::make_tuple(se::DeviceMemoryBase(tc, size), new_n);
  }
}

Status PoplarExecutor::MoveDeviceToHost(TensorControl* tc) {
  void* buf(static_cast<void*>(tc->data));
  try {
    if (tc->output_convertor) {
      current_engine_->readTensor(tc->output_handle, buf);
      std::vector<char> converted = tc->output_convertor(buf, 0, tc->size);
      memcpy(buf, converted.data(), converted.size());
    } else {
      current_engine_->readTensor(tc->output_handle, buf);
    }
    if (profile_io_) {
      AddEventRecord(tensorflow::IpuTraceEvent::DEVICE_TO_HOST_TRANSFER, "",
                     tc->output_handle, 0);
    }
    tc->on_device = false;
    tc->output_handle.clear();
    tc->input_handle.clear();
    return Status::OK();
  } catch (std::logic_error& e) {
    return tensorflow::errors::Internal("Poplar host read error: ", e.what());
  }
}

StatusOr<se::DeviceMemoryBase> PoplarExecutor::GetTupleBufferByIndex(
    const se::DeviceMemoryBase& base, int64 value) {
  const TensorControl* tc =
      reinterpret_cast<const TensorControl*>(base.opaque());
  void** bufs = (void**)tc->data;
  int64 size = reinterpret_cast<const TensorControl*>(bufs[value])->size;

  return se::DeviceMemoryBase(bufs[value], size);
}

void PoplarExecutor::FlattenedOutputDeviceMemoryList(std::vector<void*>& list,
                                                     const xla::Shape& shape,
                                                     void* base) {
  TensorControl* tc = static_cast<TensorControl*>(base);
  if (xla::ShapeUtil::IsTuple(shape)) {
    void** ptrs = reinterpret_cast<void**>(tc->data);
    for (unsigned int t = 0; t < xla::ShapeUtil::TupleElementCount(shape);
         t++) {
      void* ptr = ptrs[t];
      FlattenedOutputDeviceMemoryList(
          list, xla::ShapeUtil::GetTupleElementShape(shape, t), ptr);
    }
  } else {
    list.push_back(tc->data);
  }
}

StatusOr<se::DeviceMemoryBase> PoplarExecutor::ExecuteEngine(
    perftools::gputools::StreamExecutor* executor,
    const xla::poplarplugin::PoplarExecutable& executable,
    xla::DeviceMemoryAllocator* allocator, const Args& args) {
  const auto& output_map = executable.OutputMapping();
  const auto& output_shape = executable.result_shape();
  const auto& engine = executable.Engine();

  perftools::gputools::DeviceMemoryBase retbuf;
  int64 tensor_count;

  bool engine_changed(current_engine_ != engine);
  {
    std::lock_guard<std::recursive_mutex> g(mutex_);

    if (engine == NULL) {
      // An empty engine is a graph that just passes its inputs through
      // to its outputs.  A variable reading graph is such a thing.
      std::tie(retbuf, tensor_count) =
          RemapArgs(output_shape, 0, output_map, args);
    } else {
      if (!executable.has_module()) {
        return tensorflow::errors::InvalidArgument(
            "Executable must have an HloModule");
      }

      ArgsHandleMap arg_map;
      CreateArgsHandleMap(arg_map, args, executable);

      // Pull previous execution output back from device if:
      // a) it is on the device _and_
      // b)   the engine is changing _or_
      // c)   output buffer isn't an input to the current engine _or_
      // d)   output buffer isn't currently in the right place for the new input
      for (const auto& tc : allocations_) {
        if (tc->on_device == true) {
          if (!tc->output_handle.empty()) {
            if (engine_changed) {
              TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
            } else if (tc->input_handle.empty()) {
              TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
            } else if (arg_map.count(tc->input_handle) > 0 &&
                       tc != arg_map.at(tc->input_handle).tc) {
              TF_RETURN_IF_ERROR(MoveDeviceToHost(tc));
            }
          } else {
            if (arg_map.count(tc->input_handle) > 0 &&
                tc != arg_map.at(tc->input_handle).tc) {
              // Mark any old inputs as invalid
              tc->input_handle.clear();
              tc->on_device = false;
            }
          }
        }
      }

      if (engine_changed) {
        // TODO Load new engine

        if (profile_io_) {
          AddEventRecord(tensorflow::IpuTraceEvent::LOAD_ENGINE,
                         executable.module().name(), "", 0);
        }
      }

      current_engine_ = engine;

      // Put data on the device if:
      // a) the engine has changed
      // b) it is not on the device
      // c) it is on the device, but in the wrong place
      try {
        for (auto mem : arg_map) {
          TensorControl* tc = mem.second.tc;
          void* buf(static_cast<void*>(tc->data));
          if (mem.second.streamed) {
            current_engine_->connectStream(mem.first, buf);
          } else {
            if (tc->on_device == false || tc->input_handle != mem.first ||
                engine_changed) {
              ConversionFn fn = mem.second.fn;
              if (fn != nullptr) {
                std::vector<char> converted = fn(buf, tc->size, 0);
                current_engine_->writeTensor(mem.first, converted.data());
              } else {
                current_engine_->writeTensor(mem.first, buf);
              }
              tc->on_device = true;
              tc->input_handle = mem.first;
              if (profile_io_) {
                AddEventRecord(tensorflow::IpuTraceEvent::HOST_TO_DEVICE_TRANSFER,
                               "", mem.first, 0);
              }
            }
          }
        }
      } catch (std::logic_error e) {
        return tensorflow::errors::Internal(
          "Poplar host write error ", e.what());
      }

      std::tie(retbuf, tensor_count) =
          AllocateOutputBuffer(allocator, output_shape, 0, output_map, args,
                               executable.OutputStreamed());

      try {
        const auto& streamed = executable.OutputStreamed();
        std::vector<void*> bufs;
        FlattenedOutputDeviceMemoryList(bufs, output_shape, retbuf.opaque());
        for (int o = 0; o < streamed.size(); o++) {
          if (streamed[o]) {
            current_engine_->connectStream(GetOutputCopyHandle(o), bufs[o]);
          }
        }

        current_engine_->run(0);
      }
      catch (std::logic_error e) {
        return tensorflow::errors::Internal(
            "Poplar execution error ", e.what());
      }


      try {
        if (profile_execution_) {
          poplar::OptionFlags opts;
          opts.set("doLayerWiseBreakdown", "true");
          // opts.set("doLayerWisePerIPUBreakdown", "true");
          // opts.set("doLayerWisePerTileBreakdown", "true");

          std::stringstream stream;
          if (executable.DumpReport()) {
            auto rep = current_engine_->getExecutionReport(opts);
            rep.printSummary(stream);

            current_engine_->reportIntervals(stream);
          }

          AddEventRecord(tensorflow::IpuTraceEvent::EXECUTE, "", stream.str(),
                         0);
        }
      } catch (std::logic_error e) {
        VLOG(2) << "Error producing execution report: " << e.what();
      }
    }
  }

  return retbuf;
}

}  // namespace poplarplugin
}  // namespace xla
