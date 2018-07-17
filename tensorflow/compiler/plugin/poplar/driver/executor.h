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

// Declares the PoplarExecutor class, which is a CPU-only implementation of
// the StreamExecutor interface. For now, this is used for testing and to
// examine the performance of host-based StreamExecutor code.
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXECUTOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_EXECUTOR_H_

#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/host/host_timer.h"

#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/protobuf/config.pb.h"

#include <list>
#include <mutex>

#include <poplar/Device.hpp>
#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>

#include <poprand/RandomGen.hpp>

namespace se = stream_executor;

namespace xla {

class HloModule;

namespace poplarplugin {

class PoplarExecutable;

std::string GetInputCopyHandle(int64 parameter, int64 index);
std::string GetOutputCopyHandle(int64 index);

typedef std::vector<char> (*ConversionFn)(const void *, int64, int64);

using Args = tensorflow::gtl::ArraySlice<se::DeviceMemoryBase>;

// This maps outputs to inputs where the tensor is the same
using OutputMap = std::map<int64, int64>;

using ConversionList = std::vector<ConversionFn>;

class PoplarExecutor : public se::internal::StreamExecutorInterface {
 public:
  explicit PoplarExecutor();
  ~PoplarExecutor() override;

  Status Init(int device_ordinal, se::DeviceOptions) override {
    ordinal_ = device_ordinal;
    return Status::OK();
  }

  bool GetKernel(const se::MultiKernelLoaderSpec &spec,
                 se::KernelBase *kernel) override {
    return false;
  }
  bool Launch(se::Stream *stream, const se::ThreadDim &thread_dims,
              const se::BlockDim &block_dims, const se::KernelBase &kernel,
              const se::KernelArgsArrayBase &args) override {
    return false;
  }

  void *Allocate(uint64 size) override;
  void *AllocateSubBuffer(se::DeviceMemoryBase *mem, uint64 offset_bytes,
                          uint64 size_bytes) override;
  void Deallocate(se::DeviceMemoryBase *mem) override;

  void *HostMemoryAllocate(uint64 size) override { return new char[size]; }
  void HostMemoryDeallocate(void *mem) override {
    delete[] static_cast<char *>(mem);
  }
  bool HostMemoryRegister(void *mem, uint64 size) override { return true; }
  bool HostMemoryUnregister(void *mem) override { return true; }

  bool Memcpy(se::Stream *stream, void *host_dst, const se::DeviceMemoryBase &,
              uint64 size) override;
  bool Memcpy(se::Stream *stream, se::DeviceMemoryBase *, const void *,
              uint64 size) override;
  bool MemcpyDeviceToDevice(se::Stream *stream, se::DeviceMemoryBase *pop_dst,
                            const se::DeviceMemoryBase &host_src,
                            uint64 size) override {
    return false;
  }

  bool MemZero(se::Stream *stream, se::DeviceMemoryBase *location,
               uint64 size) override {
    return false;
  }
  bool Memset(se::Stream *stream, se::DeviceMemoryBase *, uint8,
              uint64 size) override {
    return false;
  }
  bool Memset32(se::Stream *stream, se::DeviceMemoryBase *, uint32,
                uint64 size) override {
    return false;
  }

  bool SynchronizeAllActivity() override;
  bool SynchronousMemZero(se::DeviceMemoryBase *location, uint64) override {
    return false;
  }

  bool SynchronousMemSet(se::DeviceMemoryBase *location, int value,
                         uint64 size) override {
    return false;
  }

  Status SynchronousMemcpy(se::DeviceMemoryBase *pop_dst, const void *host_src,
                           uint64 size) override;
  Status SynchronousMemcpy(void *host_dst, const se::DeviceMemoryBase &pop_src,
                           uint64 size) override;
  Status SynchronousMemcpyDeviceToDevice(se::DeviceMemoryBase *,
                                         const se::DeviceMemoryBase &,
                                         uint64 size) override {
    return xla::Unimplemented("Not implemented");
  }

  bool HostCallback(se::Stream *stream,
                    std::function<void()> callback) override;

  Status AllocateEvent(se::Event *event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status DeallocateEvent(se::Event *event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status RecordEvent(se::Stream *stream, se::Event *event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status WaitForEvent(se::Stream *stream, se::Event *event) override {
    return xla::Unimplemented("Not implemented");
  }

  se::Event::Status PollForEventStatus(se::Event *event) override {
    return se::Event::Status::kError;
  }

  bool AllocateStream(se::Stream *stream) override { return true; }
  void DeallocateStream(se::Stream *stream) override {}
  bool CreateStreamDependency(se::Stream *, se::Stream *) override;

  bool AllocateTimer(se::Timer *timer) override { return true; }
  void DeallocateTimer(se::Timer *timer) override {}
  bool StartTimer(se::Stream *stream, se::Timer *timer) override;
  bool StopTimer(se::Stream *stream, se::Timer *timer) override;

  Status BlockHostUntilDone(se::Stream *stream) override;

  int PlatformDeviceCount() override { return 1; }

  bool DeviceMemoryUsage(int64 *free, int64 *total) const override {
    return false;
  }

  se::DeviceDescription *PopulateDeviceDescription() const override;

  Status EnablePeerAccessTo(StreamExecutorInterface *other) override {
    return Status::OK();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override {
    return true;
  }

  se::SharedMemoryConfig GetDeviceSharedMemoryConfig() override {
    return se::SharedMemoryConfig::kDefault;
  }

  Status SetDeviceSharedMemoryConfig(se::SharedMemoryConfig config) override {
    return xla::Unimplemented("Shared memory not supported");
  }

  std::unique_ptr<se::internal::EventInterface> CreateEventImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<se::internal::KernelInterface> CreateKernelImplementation()
      override {
    return nullptr;
  }

  std::unique_ptr<se::internal::StreamInterface> GetStreamImplementation()
      override {
    return std::unique_ptr<se::internal::StreamInterface>(
        new se::host::HostStream());
  }

  std::unique_ptr<se::internal::TimerInterface> GetTimerImplementation()
      override {
    return std::unique_ptr<se::internal::TimerInterface>(
        new se::host::HostTimer());
  }

  // Poplar Interface

  std::string GetDeviceTargetName() const;

  Status ConfigurePoplarDevice(const tensorflow::IPUOptions::DeviceConfig &);

  const poplar::Device &GetPoplarDevice() const { return poplar_device_; }

  const poprand::RandomGenMode GetRandomGenMode() const;

  const poplar::OptionFlags &GetOptionsFlags() const { return option_flags_; }

  bool CompilerReportingEnabled() const {
    return current_config_.profiling().enable_compilation_trace();
  }

  bool CompilerReportingTextFormat() const {
    return current_config_.profiling().enable_poplar_reports_text();
  }

  void AddEventRecord(tensorflow::IpuTraceEvent::Type type,
                      const std::string &module_name,
                      const std::string &content, int value);

  Status GetCompilerEvents(std::list<tensorflow::IpuTraceEvent> &out);

  StatusOr<se::DeviceMemoryBase> ExecuteEngine(
      se::StreamExecutor *executor, const xla::poplarplugin::PoplarExecutable &,
      xla::DeviceMemoryAllocator *allocator, const Args &);

  StatusOr<se::DeviceMemoryBase> GetTupleBufferByIndex(
      const se::DeviceMemoryBase &base, int64 value);

  bool HaveExecutableCache() const;

  std::string CachedExecutableFilename(const HloModule &module) const;

  bool HaveCachedExecutable(const std::string &filename) const;

 private:
  struct TensorControl {
    size_t size = 0;
    unsigned int ref_count = 0;
    bool on_device = false;
    std::string input_handle;
    std::string output_handle;
    ConversionFn output_convertor;
    char data[0];
  };
  struct InputDef {
    TensorControl *tc;
    ConversionFn fn;
    bool streamed;

    InputDef() {}
    InputDef(TensorControl *tc, ConversionFn fn, bool streamed)
        : tc(tc), fn(fn), streamed(streamed) {}
    InputDef(const InputDef &other)
        : tc(other.tc), fn(other.fn), streamed(other.streamed) {}
  };
  using InputPairList = std::vector<InputDef>;
  using ArgsHandleMap = std::map<std::string, InputDef>;

  static void FlattenedDeviceMemoryList(InputPairList &, const xla::Shape &,
                                        void *, bool);

  static void FlattenedOutputDeviceMemoryList(std::vector<void *> &list,
                                              const xla::Shape &shape,
                                              void *base);

  static void CreateArgsHandleMap(ArgsHandleMap &, const Args &,
                                  const xla::poplarplugin::PoplarExecutable &);

  std::tuple<se::DeviceMemoryBase, int64> AllocateSingleOutput(
      xla::DeviceMemoryAllocator *allocator, const xla::Shape &shape,
      const int64 n, const OutputMap &map, const Args &args,
      const std::vector<bool> &streamed);

  std::tuple<se::DeviceMemoryBase, int64> AllocateOutputBuffer(
      xla::DeviceMemoryAllocator *allocator, const xla::Shape &shape,
      const int64 n, const OutputMap &map, const Args &args,
      const std::vector<bool> &streamed);

  std::tuple<se::DeviceMemoryBase, int64> RemapArgs(const xla::Shape &,
                                                    const int64,
                                                    const OutputMap &,
                                                    const Args &);

  std::tuple<se::DeviceMemoryBase, int64> ConstantOutput(
      xla::DeviceMemoryAllocator* allocator, const xla::Shape& shape,
      const int64 n, const std::vector<std::unique_ptr<Literal>>& constant);

  Status MoveDeviceToHost(TensorControl *tc);

  int ordinal_;

  std::recursive_mutex mutex_;

  std::shared_ptr<poplar::Engine> current_engine_;

  bool device_open_;

  poplar::Device poplar_device_;

  int64 poplar_device_hash_;

  poplar::OptionFlags option_flags_;

  std::list<TensorControl *> allocations_;

  tensorflow::IPUOptions::DeviceConfig current_config_;

  std::list<tensorflow::IpuTraceEvent> reports_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
