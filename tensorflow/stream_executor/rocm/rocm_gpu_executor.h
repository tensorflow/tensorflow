/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// The ROCM implementation of the StreamExecutorInterface functionality.
// ROCM inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the ROCM streams
// programming model provided by the librocm.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_GPU_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_GPU_EXECUTOR_H_

#include <map>
#include <set>

#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/rocm/rocm_kernel.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace rocm {

// ROCM-platform implementation of the platform-agnostic
// StreamExecutorInferface.
class ROCMExecutor : public internal::StreamExecutorInterface {
 public:
  // sub_platform indicates the subplatform used in this executor; it must
  // be a ROCM type.
  explicit ROCMExecutor(const PluginConfig &plugin_config)
      : device_(0),
        device_ordinal_(0),
        version_(0),
        plugin_config_(plugin_config) {}

  // See the corresponding StreamExecutor methods for method comments on the
  // following overrides.

  ~ROCMExecutor() override;

  port::Status Init(int device_ordinal, DeviceOptions device_options) override;

  bool GetKernel(const MultiKernelLoaderSpec &spec,
                 KernelBase *kernel) override;

  bool Launch(Stream *stream, const ThreadDim &thread_dims,
              const BlockDim &block_dims, const KernelBase &k,
              const KernelArgsArrayBase &args) override;

  void *Allocate(uint64 size) override;

  void *AllocateSubBuffer(DeviceMemoryBase *mem, uint64 offset_bytes,
                          uint64 size_bytes) override;

  void Deallocate(DeviceMemoryBase *mem) override;

  // ROCM allocation/registration functions are necessary because the driver
  // internally sets up buffers for DMA operations (and page locks them).
  // There's no external interface for us to otherwise control these DMA
  // settings.
  void *HostMemoryAllocate(uint64 size) override {
    return ROCMDriver::HostAllocate(device_ordinal_, size);
  }

  void HostMemoryDeallocate(void *location) override {
    return ROCMDriver::HostDeallocate(device_ordinal_, location);
  }

  bool HostMemoryRegister(void *location, uint64 size) override;

  bool HostMemoryUnregister(void *location) override;

  bool SynchronizeAllActivity() override;

  bool SynchronousMemZero(DeviceMemoryBase *location, uint64 size) override;

  bool SynchronousMemSet(DeviceMemoryBase *location, int value,
                         uint64 size) override;

  port::Status SynchronousMemcpy(DeviceMemoryBase *gpu_dst,
                                 const void *host_src, uint64 size) override;

  port::Status SynchronousMemcpy(void *host_dst,
                                 const DeviceMemoryBase &gpu_src,
                                 uint64 size) override;

  port::Status SynchronousMemcpyDeviceToDevice(DeviceMemoryBase *gpu_dst,
                                               const DeviceMemoryBase &gpu_src,
                                               uint64 size) override;

  bool MemZero(Stream *stream, DeviceMemoryBase *location,
               uint64 size) override;
  bool Memset(Stream *stream, DeviceMemoryBase *location, uint8 pattern,
              uint64 size) override;
  bool Memset32(Stream *stream, DeviceMemoryBase *location, uint32 pattern,
                uint64 size) override;

  bool Memcpy(Stream *stream, void *host_dst, const DeviceMemoryBase &gpu_src,
              uint64 size) override;

  bool Memcpy(Stream *stream, DeviceMemoryBase *gpu_dst, const void *host_src,
              uint64 size) override;

  bool MemcpyDeviceToDevice(Stream *stream, DeviceMemoryBase *gpu_dst,
                            const DeviceMemoryBase &gpu_src,
                            uint64 size) override;

  bool HostCallback(Stream *stream, std::function<void()> callback) override;

  bool AllocateStream(Stream *stream) override;

  void DeallocateStream(Stream *stream) override;

  bool CreateStreamDependency(Stream *dependent, Stream *other) override;

  bool AllocateTimer(Timer *timer) override;

  void DeallocateTimer(Timer *timer) override;

  bool StartTimer(Stream *stream, Timer *timer) override;

  bool StopTimer(Stream *stream, Timer *timer) override;

  port::Status AllocateEvent(Event *event) override;

  port::Status DeallocateEvent(Event *event) override;

  port::Status RecordEvent(Stream *stream, Event *event) override;

  port::Status WaitForEvent(Stream *stream, Event *event) override;

  Event::Status PollForEventStatus(Event *event) override;

  port::Status BlockHostUntilDone(Stream *stream) override;

  int PlatformDeviceCount() override { return ROCMDriver::GetDeviceCount(); }

  port::Status EnablePeerAccessTo(StreamExecutorInterface *other) override;

  bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override;

  SharedMemoryConfig GetDeviceSharedMemoryConfig() override;

  port::Status SetDeviceSharedMemoryConfig(SharedMemoryConfig config) override;

  bool DeviceMemoryUsage(int64 *free, int64 *total) const override;

  // Search for the symbol and returns a device pointer and size.
  // Returns false if symbol does not exist.
  bool GetSymbol(const string &symbol_name, void **mem, size_t *bytes) override;

  DeviceDescription *PopulateDeviceDescription() const override;

  // Populates the block_dim_limit by querying the device driver API. If an
  // error occurs at any point while asking the driver for block dim limits, it
  // will be only partially populated as a result, and an error will be logged.
  bool FillBlockDimLimit(BlockDim *block_dim_limit) const;

  bool SupportsBlas() const override;

  blas::BlasSupport *CreateBlas() override;

  bool SupportsFft() const override;

  fft::FftSupport *CreateFft() override;

  bool SupportsRng() const override;

  rng::RngSupport *CreateRng() override;

  bool SupportsDnn() const override;

  dnn::DnnSupport *CreateDnn() override;

  std::unique_ptr<internal::EventInterface> CreateEventImplementation()
      override;

  std::unique_ptr<internal::KernelInterface> CreateKernelImplementation()
      override;

  std::unique_ptr<internal::StreamInterface> GetStreamImplementation() override;

  std::unique_ptr<internal::TimerInterface> GetTimerImplementation() override;

  void *GpuContextHack() override;

  int device_ordinal() const { return device_ordinal_; }

 private:
  // Attempts to find a more specific version of the file indicated by
  // filename by looking for AMDGPU ISA-specific suffixed versions.
  bool FindOnDiskForISAVersion(port::StringPiece filename,
                               port::StringPiece canonical_suffix,
                               string *found_filename) const;

  // Host callback landing routine invoked by ROCM.
  // data: User-provided callback provided to HostCallback() above, captured
  //       as a std::function<void()>. Allocated/initialized inside
  //       HostCallback() and owned and deleted by this call.
  static void InternalHostCallback(hipStream_t stream, hipError_t status,
                                   void *data);

  // Collects metadata for the specified kernel.
  bool GetKernelMetadata(ROCMKernel *rocm_kernel,
                         KernelMetadata *kernel_metadata);

  // Prints to VLOG(2) information about the kernel's occupancy and how it might
  // be improved.
  void VlogOccupancyInfo(const KernelBase &kernel, const ThreadDim &thread_dims,
                         const BlockDim &block_dims);

  // Guards the on-disk-module mapping.
  mutex disk_modules_mu_;

  // Mapping from filename to hipModule_t, if it was already retrieved.
  // Multiple hipFunction_t are usually obtained from a single hipModule_t so we
  // attempt to hit in this mapping first, before retrieving it.
  std::map<string, hipModule_t> disk_modules_ GUARDED_BY(disk_modules_mu_);

  // Guards the in-memory-module mapping.
  mutex in_memory_modules_mu_;

  std::map<const char *, hipModule_t> in_memory_modules_
      GUARDED_BY(in_memory_modules_mu_);

  // Guards the launched kernel set.
  mutex launched_kernels_mu_;

  // Keeps track of the set of launched kernels. Currently used to suppress the
  // occupancy check on subsequent launches.
  std::set<hipFunction_t> launched_kernels_ GUARDED_BY(launched_kernels_mu_);

  // Handle for the ROCM device being operated on. Immutable
  // post-initialization.
  hipDevice_t device_;

  // The device ordinal value that this executor was initialized with; recorded
  // for use in getting device metadata. Immutable post-initialization.
  int device_ordinal_;

  // AMDGPU ISA version for device_.
  int version_;

  // The plugin configuration associated with this instance.
  PluginConfig plugin_config_;

  SE_DISALLOW_COPY_AND_ASSIGN(ROCMExecutor);
};
}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_GPU_EXECUTOR_H_
