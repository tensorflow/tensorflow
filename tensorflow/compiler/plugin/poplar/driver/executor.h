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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/input_output_aliasing_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/transfer_manager.h"

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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/io/path.h"

#include <list>
#include <mutex>

#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>

#include <poprand/RandomGen.hpp>

namespace se = stream_executor;

namespace xla {

class HloModule;

namespace poplarplugin {

enum PoplarProgramType {
  HOST_TO_DEVICE,
  MAIN_SEQUENCE,
  DEVICE_TO_HOST,
};

class PoplarExecutable;

std::string GetInfeedCopyHandle(const std::string& name, int64 shape_index);
std::string GetOutfeedCopyHandle(const std::string& name, int64 shape_index);
std::string GetInputCopyHandle(int64 parameter, int64 index);
std::string GetOutputCopyHandle(int64 output_index, int64 flat_tensor_index);

xla::poplarplugin::PoplarXfeedManager* GetXfeedManager(int device_ordinal);

void ResetXfeedManager(int device_ordinal);

typedef std::vector<char> (*ConversionFn)(const void*, int64, int64);

using Args = tensorflow::gtl::ArraySlice<se::DeviceMemoryBase>;

using ConversionList = std::vector<ConversionFn>;

class PoplarExecutor : public se::internal::StreamExecutorInterface {
 public:
  explicit PoplarExecutor();
  ~PoplarExecutor() override;

  Status Init(int device_ordinal, se::DeviceOptions) override {
    ordinal_ = device_ordinal;
    return Status::OK();
  }

  bool GetKernel(const se::MultiKernelLoaderSpec& spec,
                 se::KernelBase* kernel) override {
    return false;
  }
  bool Launch(se::Stream* stream, const se::ThreadDim& thread_dims,
              const se::BlockDim& block_dims, const se::KernelBase& kernel,
              const se::KernelArgsArrayBase& args) override {
    return false;
  }

  void* Allocate(uint64 size) override;
  void* AllocateSubBuffer(se::DeviceMemoryBase* mem, uint64 offset_bytes,
                          uint64 size_bytes) override;
  void Deallocate(se::DeviceMemoryBase* mem) override;

  void* HostMemoryAllocate(uint64 size) override { return new char[size]; }
  void HostMemoryDeallocate(void* mem) override {
    delete[] static_cast<char*>(mem);
  }
  bool HostMemoryRegister(void* mem, uint64 size) override { return true; }
  bool HostMemoryUnregister(void* mem) override { return true; }

  bool Memcpy(se::Stream* stream, void* host_dst, const se::DeviceMemoryBase&,
              uint64 size) override;
  bool Memcpy(se::Stream* stream, se::DeviceMemoryBase*, const void*,
              uint64 size) override;
  bool MemcpyDeviceToDevice(se::Stream* stream, se::DeviceMemoryBase* pop_dst,
                            const se::DeviceMemoryBase& host_src,
                            uint64 size) override;

  bool MemZero(se::Stream* stream, se::DeviceMemoryBase* location,
               uint64 size) override {
    return false;
  }
  bool Memset(se::Stream* stream, se::DeviceMemoryBase*, uint8,
              uint64 size) override {
    return false;
  }
  bool Memset32(se::Stream* stream, se::DeviceMemoryBase*, uint32,
                uint64 size) override {
    return false;
  }

  bool SynchronizeAllActivity() override;
  bool SynchronousMemZero(se::DeviceMemoryBase* location, uint64) override {
    return false;
  }

  bool SynchronousMemSet(se::DeviceMemoryBase* location, int value,
                         uint64 size) override {
    return false;
  }

  Status SynchronousMemcpy(se::DeviceMemoryBase* pop_dst, const void* host_src,
                           uint64 size) override;
  Status SynchronousMemcpy(void* host_dst, const se::DeviceMemoryBase& pop_src,
                           uint64 size) override;
  Status SynchronousMemcpyDeviceToDevice(se::DeviceMemoryBase*,
                                         const se::DeviceMemoryBase&,
                                         uint64 size) override;

  bool HostCallback(se::Stream* stream,
                    std::function<void()> callback) override;
  bool HostCallback(se::Stream* stream,
                    std::function<Status()> callback) override;

  Status AllocateEvent(se::Event* event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status DeallocateEvent(se::Event* event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status RecordEvent(se::Stream* stream, se::Event* event) override {
    return xla::Unimplemented("Not implemented");
  }

  Status WaitForEvent(se::Stream* stream, se::Event* event) override {
    return xla::Unimplemented("Not implemented");
  }

  se::Event::Status PollForEventStatus(se::Event* event) override {
    return se::Event::Status::kError;
  }

  bool AllocateStream(se::Stream* stream) override { return true; }
  void DeallocateStream(se::Stream* stream) override {}
  bool CreateStreamDependency(se::Stream*, se::Stream*) override;

  bool AllocateTimer(se::Timer* timer) override { return true; }
  void DeallocateTimer(se::Timer* timer) override {}
  bool StartTimer(se::Stream* stream, se::Timer* timer) override;
  bool StopTimer(se::Stream* stream, se::Timer* timer) override;

  Status BlockHostUntilDone(se::Stream* stream) override;

  int PlatformDeviceCount() override { return 1; }

  bool DeviceMemoryUsage(int64* free, int64* total) const override {
    return false;
  }

  se::DeviceDescription* PopulateDeviceDescription() const override;

  Status EnablePeerAccessTo(StreamExecutorInterface* other) override {
    return Status::OK();
  }

  bool CanEnablePeerAccessTo(StreamExecutorInterface* other) override {
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
  static se::host::HostStream* AsPoplarStream(se::Stream* stream);

  std::string GetDeviceTargetName() const;

  Status ConfigurePoplarDevice(const IpuOptions&);

  const poplar::Device& GetPoplarDevice() const { return poplar_device_; }

  const poprand::RandomGenMode GetRandomGenMode() const;

  const poplar::OptionFlags& GetOptionsFlags() const { return option_flags_; }

  const poplar::OptionFlags& GetReportFlags() const { return report_options_; }

  bool IpuTraceEventsEnabled() const {
    return current_config_.profiling().enable_ipu_trace_events();
  }

  bool CompilerReportingEnabled() const {
    return current_config_.profiling().enable_compilation_trace();
  }

  int64 ReportEventNthExecution() const {
    return current_config_.profiling().report_every_nth_execution();
  }

  bool CompilerReportingTextFormat() const {
    return current_config_.profiling().enable_poplar_reports_text();
  }

  bool AlwaysRearrangeCopiesOnTheHost() const {
    return current_config_.speed_size_config()
        .always_rearrange_copies_on_the_host();
  }

  bool DisableGraphConvCaching() const {
    return current_config_.speed_size_config()
        .disable_graph_convolution_caching();
  }

  bool NormInputRecomputationEnabled() const {
    // Re-computation of non linearities is enabled by default unless the user
    // has specifically told us not to do it.
    return current_config_.speed_size_config().has_recompute_norm_inputs()
               ? current_config_.speed_size_config().recompute_norm_inputs()
               : false;
  }

  poplar::OptionFlags GetConvolutionOptions() const { return conv_options_; }

  poplar::OptionFlags GetPoolingOptions() const { return pooling_options_; }

  bool RetainControlDependencies() const {
    return current_config_.retain_control_dependencies();
  }

  int64 GetNumberOfReplicas() const {
    if (current_config_.device_config_size() > ordinal_) {
      return current_config_.device_config(ordinal_).num_replicas();
    } else {
      return 0;
    }
  }

  void AddCompileBeginEventRecord(const std::string& module_name,
                                  const std::string& xla_graph);

  void AddCompileEndEventRecord(const std::string& module_name,
                                const std::string& compilation_report,
                                const std::string& tensor_map_json,
                                int64 duration);

  void AddHostToDeviceEventRecord(const std::string& transfer_json);

  void AddDeviceToHostEventRecord(const std::string& transfer_json);

  void AddLoadEngineEventRecord(const std::string& module_name);

  void AddExecuteEventRecord(const std::string& module_name,
                             const std::string& report);

  Status GetCompilerEvents(std::list<tensorflow::IpuTraceEvent>& out);

  StatusOr<se::DeviceMemoryBase> ExecuteEngine(
      se::StreamExecutor* executor, xla::poplarplugin::PoplarExecutable&,
      xla::DeviceMemoryAllocator* allocator, const Args&);

  StatusOr<se::DeviceMemoryBase> GetTupleBufferByIndex(
      const se::DeviceMemoryBase& base, int64 value);

  bool HaveExecutableCache() const;

  std::string CachedExecutableFilename(const HloModule& module) const;

  bool HaveCachedExecutable(const std::string& filename) const;

  void AboutToFreeEngine(poplar::Engine* engine);

  const int device_ordinal() const;

  static poplar::DeviceManager& GetDeviceManager();

  void CreateInfeedDatasetIterator(
      const std::string&, std::unique_ptr<tensorflow::data::IteratorBase>,
      std::unique_ptr<tensorflow::data::IteratorContext>,
      const std::vector<xla::Shape>&);

  void setFlagIfNotPresent(poplar::OptionFlags& opts, const std::string& key,
                           const std::string& value);

 private:
  struct TensorControl {
    size_t size = 0;
    unsigned int ref_count = 0;
    bool on_device = false;
    std::string input_handle;
    std::string output_handle;
    ConversionFn output_convertor;
    std::vector<char> converted_data;
    char* data;

    TensorControl(size_t size_);
    ~TensorControl();
  };

  struct InputDef {
    TensorControl* tc;
    ConversionFn fn;
    bool streamed;

    InputDef() {}
    InputDef(TensorControl* tc, ConversionFn fn, bool streamed)
        : tc(tc), fn(fn), streamed(streamed) {}
    InputDef(const InputDef& other)
        : tc(other.tc), fn(other.fn), streamed(other.streamed) {}
  };
  using InputPairList = std::vector<InputDef>;
  using ArgsHandleMap = std::map<std::string, InputDef>;

  struct OutputDef {
    TensorControl* tc;
    bool streamed;

    OutputDef() {}
    OutputDef(TensorControl* tc, bool streamed) : tc(tc), streamed(streamed) {}
    OutputDef(const OutputDef& other)
        : tc(other.tc), streamed(other.streamed) {}
  };

  using OutputPairList = std::vector<OutputDef>;
  using OutputsHandleMap = std::map<std::string, OutputDef>;

  static void FlattenedDeviceMemoryList(
      InputPairList&, const xla::Shape&, void*,
      const InputOutputAliasingMap::InputInfo&);

  static void FlattenedOutputDeviceMemoryList(
      OutputPairList&, const xla::Shape&, void*,
      const InputOutputAliasingMap::OutputInfo&);

  void UpdateArgsHandleMap(const Args&,
                           const xla::poplarplugin::PoplarExecutable&);

  void UpdateOutputsHandleMap(
      const xla::poplarplugin::PoplarExecutable& executable,
      const xla::Shape& shape, se::DeviceMemoryBase retbuf);

  // These classes are used to pass around information for specific output
  // allocation type
  class OutputAllocation {
   public:
    virtual se::DeviceMemoryBase GetAllocation(
        xla::DeviceMemoryAllocator*, const xla::Shape&, const int64, int64&,
        const Args&, const InputOutputAliasingMap::OutputInfo&,
        const ArgsHandleMap&, const int) const = 0;

   protected:
    OutputAllocation(){};
  };

  class ConstantOutputAllocation : public OutputAllocation {
   public:
    ConstantOutputAllocation(const std::vector<std::vector<Literal>>& constants)
        : constants_(constants) {}

    se::DeviceMemoryBase GetAllocation(
        xla::DeviceMemoryAllocator*, const xla::Shape&, const int64, int64&,
        const Args&, const InputOutputAliasingMap::OutputInfo&,
        const ArgsHandleMap&, const int) const override;

   private:
    const std::vector<std::vector<Literal>>& constants_;
  };

  class RemapOutputAllocation : public OutputAllocation {
   public:
    RemapOutputAllocation(PoplarExecutor* executor,
                          const std::vector<uint64>& remap_map,
                          const InputOutputAliasingMap& io_map)
        : executor_(executor),
          remap_map_(remap_map),
          input_output_aliasing_map_(io_map) {}

    se::DeviceMemoryBase GetAllocation(
        xla::DeviceMemoryAllocator*, const xla::Shape&, const int64, int64&,
        const Args&, const InputOutputAliasingMap::OutputInfo&,
        const ArgsHandleMap&, const int) const override;

   private:
    PoplarExecutor* executor_;
    const std::vector<uint64>& remap_map_;
    const InputOutputAliasingMap& input_output_aliasing_map_;
  };

  class BufferOutputAllocation : public OutputAllocation {
   public:
    BufferOutputAllocation(){};

    se::DeviceMemoryBase GetAllocation(
        xla::DeviceMemoryAllocator*, const xla::Shape&, const int64, int64&,
        const Args&, const InputOutputAliasingMap::OutputInfo&,
        const ArgsHandleMap&, const int) const override;
  };

  se::DeviceMemoryBase HandleOutputBuffer(
      xla::DeviceMemoryAllocator* allocator,
      const OutputAllocation& allocation_info, const xla::Shape& shape,
      const int64 output_index, int64& flat_tensor_index, const Args& args,
      const InputOutputAliasingMap::OutputInfo& output_info);

  se::DeviceMemoryBase GetOutputBuffer(
      const xla::poplarplugin::PoplarExecutable& executable,
      xla::DeviceMemoryAllocator* allocator,
      const OutputAllocation& allocation_info, const xla::Shape& shape,
      const Args& args, const InputOutputAliasingMap& output_info);

  // Functions which check whether any resource variables need copying to/from
  // device
  StatusOr<bool> CheckMoveDeviceToHostRequired(const bool engine_changed);
  StatusOr<bool> CheckMoveHostToDeviceRequired(const bool engine_changed);

  // Create a new trace event object
  tensorflow::IpuTraceEvent NewTraceEvent();

  // Functions which move the resource variables to/from the device
  Status MoveDeviceToHost();
  Status MoveHostToDevice();

  // Functions which connect the streams to/from device
  void ConnectStreamedVariablesHostToDevice();
  void ConnectStreamedVariablesDeviceToHost();

  // Sometimes post process streamed data into the right host format
  void PostProcessStreamedVariablesDeviceToHost();

  // Takes a tensor and returns a pointer to a buffer with the data in the right
  // format
  static void* PreProcessBuffer(InputDef& id);
  // Convers the data into the right host format
  static void PostProcessBuffer(TensorControl* tc);

  // Connect buffers provided by infeed transfer manager to Poplar
  // HostToDevice FIFO
  void ConnectInfeedsToStreamCallback(const InfeedInfos& infeed_infos);

  // Connect buffers provided by transfer manager to Poplar
  // deviceToHostFIFO()
  void ConnectOutfeedToStreamCallback(se::StreamExecutor* executor,
                                      const OutfeedInfos& outfeed_infos,
                                      const uint32 replication_factor);

  // Creates and launches the thread which will fetch inputs from
  // the InfeedDatasetIterator and enqueue them in the TransferManager.
  // The thread is joined when the pointer is deleted.
  void LaunchInfeedThread(se::StreamExecutor* executor,
                          const InfeedInfos& infeed_infos);

  std::function<void()> CreateInfeedIOThreadFunction(
      se::StreamExecutor* executor, const InfeedInfos& infeed_infos);

  // Sets cancellation flags and notifies the threads running in thread_pool_
  void StopThreadPool();

  void DeferredDeallocation();

  int ordinal_;

  std::recursive_mutex mutex_;

  poplar::Engine* current_engine_;

  bool device_open_;

  poplar::Device poplar_device_;

  int64 poplar_device_hash_;

  poplar::OptionFlags option_flags_;

  poplar::OptionFlags conv_options_;

  poplar::OptionFlags pooling_options_;

  poplar::OptionFlags report_options_;

  std::list<TensorControl*> allocations_;

  ArgsHandleMap args_map_;
  OutputsHandleMap outputs_map_;

  bool hardware_configured_;

  IpuOptions current_config_;

  std::list<tensorflow::IpuTraceEvent> reports_;

  std::atomic<bool> infeed_thread_cancelled_;

  static const int NUM_THREADS = 1;
  tensorflow::thread::ThreadPool thread_pool_;

  struct InfeedDatasetIterator {
    std::unique_ptr<tensorflow::data::IteratorBase> iterator;
    std::unique_ptr<tensorflow::data::IteratorContext> iterator_ctx;
    const std::vector<xla::Shape> shapes;
    std::vector<bool> used;
    std::condition_variable tensors_dequeued_cv;

    std::vector<tensorflow::Tensor> tensors;

    // Need to acquire mutex before calling this function
    bool all_tensors_used() {
      return absl::c_all_of(used, [](bool v) { return v; });
    }

    // Mutex used to make sure only one callback is accessing the dataset
    // iterator.
    std::mutex mutex;

    InfeedDatasetIterator(
        std::unique_ptr<tensorflow::data::IteratorBase> iterator,
        std::unique_ptr<tensorflow::data::IteratorContext> iterator_ctx,
        const std::vector<xla::Shape>& shapes)
        : iterator(std::move(iterator)),
          iterator_ctx(std::move(iterator_ctx)),
          shapes(std::move(shapes)),
          used(shapes.size(), true) {}
  };

  absl::flat_hash_map<std::string, std::unique_ptr<InfeedDatasetIterator>>
      infeed_dataset_iterators_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
