/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// The CUDA implementation of the StreamExecutorInterface functionality.
// CUDA inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the CUDA streams
// programming model provided by the libcuda.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_GPU_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_GPU_EXECUTOR_H_

#include <set>
#include <unordered_map>

#include "absl/strings/string_view.h"
#include "tensorflow/stream_executor/cuda/cuda_kernel.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace cuda {

// CUDA-platform implementation of the platform-agnostic
// StreamExecutorInferface.
class CUDAExecutor : public internal::StreamExecutorInterface {
 public:
  // sub_platform indicates the subplatform used in this executor; it must
  // be a CUDA type.
  explicit CUDAExecutor(const PluginConfig &plugin_config)
      : device_(0),
        context_(nullptr),
        device_ordinal_(0),
        cc_major_(0),
        cc_minor_(0),
        plugin_config_(plugin_config) {}

  // See the corresponding StreamExecutor methods for method comments on the
  // following overrides.

  ~CUDAExecutor() override;

  port::Status Init(int device_ordinal, DeviceOptions device_options) override;

  bool GetKernel(const MultiKernelLoaderSpec &spec,
                 KernelBase *kernel) override;
  void UnloadKernel(const KernelBase *kernel) override;
  bool LoadModule(const MultiModuleLoaderSpec &spec,
                  ModuleHandle *module_handle) override;
  bool UnloadModule(ModuleHandle module_handle) override;

  bool Launch(Stream *stream, const ThreadDim &thread_dims,
              const BlockDim &block_dims, const KernelBase &k,
              const KernelArgsArrayBase &args) override;

  int CalculateOccupancy(const DeviceDescription &device_description,
                         uint64 registers_per_thread,
                         uint64 shared_memory_per_block,
                         const ThreadDim &thread_dims, CUfunction func);

  int CompareOccupancy(int *initial_blocks,
                       const DeviceDescription &device_description,
                       uint64 registers_per_thread,
                       uint64 shared_memory_per_block,
                       const ThreadDim &thread_dims, CUfunction func);

  void *Allocate(uint64 size) override;

  void *AllocateSubBuffer(DeviceMemoryBase *mem, uint64 offset_bytes,
                          uint64 size_bytes) override;

  void Deallocate(DeviceMemoryBase *mem) override;

  void *UnifiedMemoryAllocate(uint64 size) override {
    return CUDADriver::UnifiedMemoryAllocate(context_, size);
  }

  void UnifiedMemoryDeallocate(void *location) override {
    return CUDADriver::UnifiedMemoryDeallocate(context_, location);
  }

  // CUDA allocation/registration functions are necessary because the driver
  // internally sets up buffers for DMA operations (and page locks them).
  // There's no external interface for us to otherwise control these DMA
  // settings.
  void *HostMemoryAllocate(uint64 size) override {
    return CUDADriver::HostAllocate(context_, size);
  }

  void HostMemoryDeallocate(void *location) override {
    return CUDADriver::HostDeallocate(context_, location);
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

  int PlatformDeviceCount() override { return CUDADriver::GetDeviceCount(); }

  port::Status EnablePeerAccessTo(StreamExecutorInterface *other) override;

  bool CanEnablePeerAccessTo(StreamExecutorInterface *other) override;

  SharedMemoryConfig GetDeviceSharedMemoryConfig() override;

  port::Status SetDeviceSharedMemoryConfig(SharedMemoryConfig config) override;

  bool DeviceMemoryUsage(int64 *free, int64 *total) const override;

  // Search for the symbol and returns a device pointer and size.
  // Returns false if symbol does not exist.
  bool GetSymbol(const string &symbol_name, ModuleHandle module_handle,
                 void **mem, size_t *bytes) override;

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

  CudaContext* cuda_context();

 private:
  // Attempts to find a more specific version of the file indicated by
  // filename by looking for compute-capability-specific suffixed versions; i.e.
  // looking for "foo.ptx" will check to see if "foo.ptx.cc30.ptx" is present if
  // we're on a compute capability 3.0 machine.
  bool FindOnDiskForComputeCapability(absl::string_view filename,
                                      absl::string_view canonical_suffix,
                                      string *found_filename) const;

  // Host callback landing routine invoked by CUDA.
  // data: User-provided callback provided to HostCallback() above, captured
  //       as a std::function<void()>. Allocated/initialized inside
  //       HostCallback() and owned and deleted by this call.
  static void InternalHostCallback(CUstream stream, CUresult status,
                                   void *data);

  // Collects metadata for the specified kernel.
  bool GetKernelMetadata(CUDAKernel *cuda_kernel,
                         KernelMetadata *kernel_metadata);

  // Prints to VLOG(2) information about the kernel's occupancy and how it might
  // be improved.
  void VlogOccupancyInfo(const KernelBase &kernel, const ThreadDim &thread_dims,
                         const BlockDim &block_dims);

  bool LoadModuleFromCuBin(const char *cubin, CUmodule *module)
      EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Loads the PTX text `ptx` as a CUDA module.  `ptx` must be null terminated.
  bool LoadModuleFromPtx(const char *ptx, CUmodule *module)
      EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  bool UnloadGpuBinary(const void *gpu_binary)
      EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);

  // Guards the in-memory-module mapping.
  mutex in_memory_modules_mu_;

  // Kernel -> loaded GPU binary. Many kernels may load the same binary.
  std::unordered_map<const KernelBase *, const void *> kernel_to_gpu_binary_
      GUARDED_BY(in_memory_modules_mu_);
  // GPU binary (PTX or CUBIN) -> {CUDA module, reference count}.
  std::unordered_map<const void *, std::pair<CUmodule, uint64>>
      gpu_binary_to_module_ GUARDED_BY(in_memory_modules_mu_);

  // Guards the launched kernel set.
  mutex launched_kernels_mu_;

  // Keeps track of the set of launched kernels. Currently used to suppress the
  // occupancy check on subsequent launches.
  std::set<CUfunction> launched_kernels_ GUARDED_BY(launched_kernels_mu_);

  // Handle for the CUDA device being operated on. Immutable
  // post-initialization.
  CUdevice device_;

  // Handle for session with the library/driver. Immutable post-initialization.
  CudaContext* context_;

  // The device ordinal value that this executor was initialized with; recorded
  // for use in getting device metadata. Immutable post-initialization.
  int device_ordinal_;

  // The major verion of the compute capability for device_.
  int cc_major_;

  // The minor verion of the compute capability for device_.
  int cc_minor_;

  // The plugin configuration associated with this instance.
  PluginConfig plugin_config_;

  SE_DISALLOW_COPY_AND_ASSIGN(CUDAExecutor);
};

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_GPU_EXECUTOR_H_
