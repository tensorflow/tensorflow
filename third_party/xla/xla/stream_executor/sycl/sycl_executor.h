/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_EXECUTOR_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/sycl/sycl_kernel.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace stream_executor::sycl {

// This class implements GpuExecutor for Intel GPUs that use SYCL libraries.
class SyclExecutor : public gpu::GpuExecutor {
 public:
  SyclExecutor(Platform* platform, int device_ordinal)
      : GpuExecutor(platform, device_ordinal) {}

  ~SyclExecutor() override;

  // Initializes the SYCL executor for the given device ordinal.
  // Returns OK on success, error status on failure.
  absl::Status Init() override;

  // Returns the DNN support implementation for SYCL.
  dnn::DnnSupport* AsDnn() override;

  // Loads a kernel from the given KernelLoaderSpec, which contains the SPIR-V
  // binary and kernel metadata.
  // Returns a unique_ptr to the loaded kernel or error status.
  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;

  // Unloads the given module, releasing associated resources.
  // Returns true if the module was unloaded, false otherwise.
  bool UnloadModule(ModuleHandle module_handle) override;

  // Unloads the given kernel and decrements the reference count of its module.
  // Unloads the module if no other kernels are using it.
  // If the kernel was never loaded, this is a no-op.
  void UnloadKernel(const Kernel* kernel) override;

  // Loads a module from the given MultiModuleLoaderSpec.
  // Returns a handle to the loaded module or error status.
  absl::StatusOr<ModuleHandle> LoadModule(
      const MultiModuleLoaderSpec& spec) override;

  // Returns a shared constant for the given content, creating it on the device
  // if it does not already exist.
  absl::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, absl::Span<const uint8_t> content) override;

  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space) override;

  void Deallocate(DeviceMemoryBase* mem) override;

  // Synchronizes all device activity.
  bool SynchronizeAllActivity() override;

  // Sets the specified device memory to zero synchronously.
  absl::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) override;

  // Copies memory from host to device synchronously.
  absl::Status SynchronousMemcpy(DeviceMemoryBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;

  // Copies memory from device to host synchronously.
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceMemoryBase& gpu_src,
                                 uint64_t size) override;

  // Returns the Stream for the given raw GPU stream pointer, or nullptr if
  // not found.
  Stream* FindAllocatedStream(void* gpu_stream) override {
    absl::MutexLock lock(&alive_gpu_streams_mu_);
    auto it = alive_gpu_streams_.find(gpu_stream);
    if (it == alive_gpu_streams_.end()) {
      return nullptr;
    }
    return it->second;
  }

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override {
    return SyclExecutor::CreateDeviceDescription(device_ordinal());
  }

  static absl::StatusOr<std::unique_ptr<DeviceDescription>>
  CreateDeviceDescription(int device_ordinal);

  // Creates a new SYCL stream with the given options.
  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      bool enable_multiple_streams,
      std::optional<std::variant<StreamPriority, int>> priority);

  // Creates a new SYCL stream with the given priority.
  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override {
    return CreateStream(/*enable_multiple_streams=*/true, priority);
  }

  // Creates a new SYCL event.
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override;

  // Deallocates the given stream.
  void DeallocateStream(Stream* stream) override;

  // Enables peer access to another StreamExecutor's device.
  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;

  // Returns true if peer access can be enabled to another StreamExecutor.
  bool CanEnablePeerAccessTo(StreamExecutor* other) override;

  // Returns the total and free device memory in bytes for the executor's
  // SYCL device.
  // Note: SYCL does not provide a standard way to query free device memory,
  // so a placeholder value of -1 is returned for free_bytes.
  // Returns true on success, false on failure.
  bool DeviceMemoryUsage(int64_t* free_bytes,
                         int64_t* total_bytes) const override;

  // Returns the SyclKernel for the given Kernel, or NotFound error.
  absl::StatusOr<const SyclKernel*> GetSyclKernel(const Kernel* kernel);

  // Return by value since sycl::device is a lightweight object.
  ::sycl::device GetDevice() const { return device_; }

  // Return by value since sycl::context is a lightweight object.
  absl::StatusOr<::sycl::context> GetContext() const {
    if (sycl_context_ == nullptr) {
      return absl::InternalError("GetContext: sycl_context_ is nullptr");
    }
    return sycl_context_->context();
  }

 private:
  // Handle for the SYCL device being operated on. Immutable
  // post-initialization.
  ::sycl::device device_;

  // SyclContext for this device.
  std::unique_ptr<SyclContext> sycl_context_;

  // Guards the in-memory-module mapping.
  absl::Mutex in_memory_modules_mu_;

  // Guards the shared constants map.
  absl::Mutex shared_constants_mu_;

  // On-device constants that can be shared between multiple executables. A
  // pointer for a given constant will expire when no executables require use
  // of that constant anymore.
  std::map<const absl::uint128, std::weak_ptr<DeviceMemoryBase>>
      shared_constants_ ABSL_GUARDED_BY(shared_constants_mu_);

  // Guards the alive streams map.
  absl::Mutex alive_gpu_streams_mu_;

  // Lookup map for alive streams, from raw stream pointers.
  absl::flat_hash_map<void*, Stream*> alive_gpu_streams_
      ABSL_GUARDED_BY(alive_gpu_streams_mu_);

  // Kernel -> loaded GPU binary. Many kernels may load the same binary.
  absl::flat_hash_map<const Kernel*, ModuleHandle> kernel_to_gpu_binary_
      ABSL_GUARDED_BY(in_memory_modules_mu_);

  // Loaded GPU binary handle -> {module, reference count}.
  absl::flat_hash_map<ModuleHandle, std::pair<ze_module_handle_t, uint64_t>>
      gpu_binary_to_module_ ABSL_GUARDED_BY(in_memory_modules_mu_);

  // Module handle -> loaded module.
  absl::flat_hash_map<ModuleHandle, ze_module_handle_t> in_memory_modules_
      ABSL_GUARDED_BY(in_memory_modules_mu_);

  // Set of loaded kernels. This contains all kernels loaded by this executor,
  // including in-process kernels.
  absl::flat_hash_set<const Kernel*> loaded_kernels_
      ABSL_GUARDED_BY(in_memory_modules_mu_);

  // Loads a SPIR-V binary into a Level Zero module for the current SYCL
  // context. If the module is already loaded, increments its reference count
  // and returns the existing handle. Otherwise, loads the module, caches it,
  // and sets its reference count to 1.
  // Internally acquires in_memory_modules_mu_; the caller should not hold it.
  // Returns a handle to the loaded module on success, or an error status.
  absl::StatusOr<ModuleHandle> LoadModuleFromSpirv(const char* spirv,
                                                   const size_t size);

  // Unloads the given SPIR-V module when its reference count reaches zero.
  // Removes the module from caches and destroys it.
  // REQUIRES: Caller must hold in_memory_modules_mu_.
  // Returns true if the module was unloaded, false otherwise.
  bool UnloadGpuBinary(ModuleHandle module_handle)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(in_memory_modules_mu_);
};

}  // namespace stream_executor::sycl

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_EXECUTOR_H_
