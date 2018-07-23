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

// ROCM userspace driver library wrapper functionality.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_

#include <stddef.h>
#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "rocm/include/hip/hip_runtime.h"

namespace stream_executor {
namespace rocm {

// Identifies the memory space where an allocation resides. See
// ROCMDriver::GetPointerMemorySpace().
enum class MemorySpace { kHost, kDevice };

// Returns a casual string, such as "host" for the provided memory space.
string MemorySpaceString(MemorySpace memory_space);

// ROCMDriver contains wrappers for calls to the userspace library driver. It's
// useful to isolate these calls and put basic wrappers around them to separate
// userspace library driver behaviors from the rest of the program.
//
// At the moment it's simply used as a namespace.
//
// The calls log any specific errors internally and return whether the operation
// was successful to the caller.
//
// Thread safety: these functions should not be used from signal handlers.
class ROCMDriver {
 public:
  // Wraps a call to hipInit with logging to help indicate what has gone wrong in
  // the case of failure. Safe to call multiple times; will be fast on all calls
  // after the first.
  static port::Status Init();

  // Creates a new ROCM stream associated with the given device via
  // hipStreamCreate.
  // stream is an outparam owned by the caller, must not be null.
  static bool CreateStream(int device_ordinal, hipStream_t *stream);

  // Destroys a ROCM stream associated with the given device.
  // stream is owned by the caller, must not be null, and *stream is set to null
  // if the stream is successfully destroyed.
  static void DestroyStream(int device_ordinal, hipStream_t *stream);

  // ROCM events can explicitly disable event TSC retrieval for some presumed
  // performance improvement if timing is unnecessary.
  enum class EventFlags { kDefault, kDisableTiming };

  // Creates a new event associated with the given device.
  // result is an outparam owned by the caller and must not be null.
  static port::Status CreateEvent(int device_ordinal, hipEvent_t *result,
                                  EventFlags flags);

  // Destroys *event and turns it into a nullptr. event may not be null, but
  // *event may be, via hipEventDestroy
  static port::Status DestroyEvent(int device_ordinal, hipEvent_t *event);

  // Allocates a GPU memory space of size bytes associated with the given
  // device via hipMemAlloc.
  static void *DeviceAllocate(int device_ordinal, uint64 bytes);

  // Deallocates a GPU memory space of size bytes associated with the given
  // device via hipMemFree.
  static void DeviceDeallocate(int device_ordinal, void *location);

  // Allocates page-locked and ROCM-registered memory on the host via
  // hipMemAllocHost.
  static void *HostAllocate(int device_ordinal, uint64 bytes);

  // Deallocates a location created by HostAllocate, via hipMemFreeHost.
  static void HostDeallocate(int device_ordinal, void *location);

  // Registers a memory region at location of size bytes via hipMemHostRegister.
  static bool HostRegister(int device_ordinal, void *location, uint64 bytes);

  // Unregisters a memory region that was previously registered at location via
  // hipMemHostUnregister.
  //
  //
  // TODO(leary) verify an error will be returned if the location wasn't
  // previously registered.
  static bool HostUnregister(int device_ordinal, void *location);

  // Given a device ordinal, returns a device handle into the device outparam,
  // which must not be null.
  //
  // N.B. these device handles do not have a corresponding destroy function in
  // the ROCM driver API.
  static port::Status GetDevice(int device_ordinal, hipDevice_t *device);

  // Given a device handle, returns the name reported by the driver for the
  // device.
  static bool GetDeviceName(hipDevice_t device, string *name_out);

  // Queries the runtime for the specified attribute of the specified function.
  static bool FuncGetAttribute(hipDeviceAttribute_t attribute,
                               hipFunction_t function, int *attribute_value);

  // Sets the preferred cache configuration for the specified function.
  static bool FuncSetCacheConfig(hipFunction_t function,
                                 hipFuncCache_t cache_config);

  // Gets the preferred shared memory bank configuration for the specified
  // DEVICE (not function!), either default or four- or eight-byte bank size.
  static port::StatusOr<hipSharedMemConfig> DeviceGetSharedMemConfig(
      int device_ordinal);

  // Sets the preferred shared memory bank configuration for the specified
  // DEVICE (not function!), either default or four- or eight-byte bank size.
  static port::Status DeviceSetSharedMemConfig(
      int device_ordinal, hipSharedMemConfig shared_mem_config);

  // Launches a HIP kernel via hipLaunchKernel.
  // TODO(leary) describe the structure of kernel_params and extra in a readable
  // way.
  static bool LaunchKernel(int device_ordinal, hipFunction_t function,
                           unsigned int grid_dim_x, unsigned int grid_dim_y,
                           unsigned int grid_dim_z, unsigned int block_dim_x,
                           unsigned int block_dim_y, unsigned int block_dim_z,
                           unsigned int shared_mem_bytes, hipStream_t stream,
                           void **kernel_params, void **extra);

  // Loads HSACO with the ROCM runtime and stores the resulting handle in
  // "module". Any error logs that are produced are logged internally.
  static bool LoadHsaco(int device_ordinal, const char *hsaco_contents,
                        hipModule_t *module);

  // Retrieves a named kernel from a loaded module, and places the resulting
  // handle into function (outparam) on success. Neither kernel_name nor
  // function may be null. No ownership is taken of kernel_name.
  static bool GetModuleFunction(int device_ordinal, hipModule_t module,
                                const char *kernel_name, hipFunction_t *function);

  // Retrieves a named global/constant symbol from a loaded module, and returns
  // a device pointer and size of the symbol on success. symbol_name may not be
  // null. At least one of dptr or bytes should not be null. No ownership is
  // taken of symbol_name.
  static bool GetModuleSymbol(int device_ordinal, hipModule_t module,
                              const char *symbol_name, hipDeviceptr_t *dptr,
                              size_t *bytes);

  // Unloads module from the current device via cuModuleUnload.
  // TODO(leary) the documentation doesn't say what kind of disasters happen
  // if you try to unload a module while its hipFunction_ts are in use.
  static void UnloadModule(int device_ordinal, hipModule_t module);

  // Performs a synchronous memset of the device memory segment via hipMemsetD8.
  static bool SynchronousMemsetUint8(int device_ordinal, hipDeviceptr_t location,
                                     uint8 value, size_t size);

  // Performs a synchronous memset of the device memory segment via hipMemsetD32.
  static bool SynchronousMemsetUint32(int device_ordinal,
                                      hipDeviceptr_t location, uint32 value,
                                      size_t uint32_count);

  // Performs an asynchronous memset of the device memory segment via
  // hipMemsetD8Async.
  static bool AsynchronousMemsetUint8(int device_ordinal, hipDeviceptr_t location,
                                      uint8 value, size_t uint32_count,
                                      hipStream_t stream);

  // Performs an asynchronous memset of the device memory segment via
  // hipMemsetD32Async.
  static bool AsynchronousMemsetUint32(int device_ordinal,
                                       hipDeviceptr_t location, uint32 value,
                                       size_t uint32_count, hipStream_t stream);

  // -- Synchronous memcopies.

  static port::Status SynchronousMemcpyD2H(int device_ordinal, void* host_dst,
                                           hipDeviceptr_t gpu_src, uint64 size);
  static port::Status SynchronousMemcpyH2D(int device_ordinal,
                                           hipDeviceptr_t gpu_dst,
                                           const void* host_src, uint64 size);
  static port::Status SynchronousMemcpyD2D(int device_ordinal,
                                           hipDeviceptr_t gpu_dst,
                                           hipDeviceptr_t gpu_src, uint64 size);

  // -- Asynchronous memcopies.

  static bool AsynchronousMemcpyD2H(int device_ordinal, void *host_dst,
                                    hipDeviceptr_t gpu_src, uint64 size,
                                    hipStream_t stream);
  static bool AsynchronousMemcpyH2D(int device_ordinal, hipDeviceptr_t gpu_dst,
                                    const void *host_src, uint64 size,
                                    hipStream_t stream);
  static bool AsynchronousMemcpyD2D(int device_ordinal, hipDeviceptr_t gpu_dst,
                                    hipDeviceptr_t gpu_src, uint64 size,
                                    hipStream_t stream);

  // The ROCM stream callback type signature.
  // The data passed to AddStreamCallback is subsequently passed to this
  // callback when it fires.
  //
  // Some notable things:
  // * Callbacks must not make any ROCM API calls.
  // * Callbacks from independent streams execute in an undefined order and may
  //   be serialized.
  typedef void (*StreamCallback)(hipStream_t stream, hipError_t status, void *data);

  // Enqueues a callback operation into stream.
  // See StreamCallback above ROCM documentation for additional
  // details.
  static bool AddStreamCallback(int device_ordinal, hipStream_t stream,
                                StreamCallback callback, void *data);

  // Causes stream to wait for event to trigger before proceeding via
  // hipStreamWaitEvent.
  static bool WaitStreamOnEvent(int device_ordinal, hipStream_t stream,
                                hipEvent_t event);

  // Blocks the calling thread until the operations enqueued onto stream have
  // been completed, via hipStreamSynchronize.
  //
  // TODO(leary) if a pathological thread enqueues operations onto the stream
  // while another thread blocks like this, can you wind up waiting an unbounded
  // amount of time?
  //
  static port::Status SynchronizeStream(int device_ordinal, hipStream_t stream);

  // Blocks the calling thread until the operations associated with the device
  // have been completed, via hipCtxSynchronize.
  //
  static bool SynchronizeDevice(int device_ordinal);

  // Returns true if all stream tasks have completed at time of the call. Note
  // the potential for races around this call (if another thread adds work to
  // the stream immediately after this returns).
  static bool IsStreamIdle(int device_ordinal, hipStream_t stream);

  // Returns whether code in the from device can access memory in the to
  // device via hipDeviceCanAccessPeer.
  static bool CanEnablePeerAccess(int from_device_ordinal, int to_device_ordinal);

  // Enables peer access per CanEnablePeerAccess, via hipDeviceEnablePeerAccess.
  static port::Status EnablePeerAccess(int from_device_ordinal, int to_device_ordinal);

  // Returns the elapsed milliseconds between start and stop via
  // hipEventElapsedTime.
  static bool GetEventElapsedTime(int device_ordinal,
                                  float *elapsed_milliseconds, hipEvent_t start,
                                  hipEvent_t stop);

  // Records that an event occurred when execution reaches the current point in
  // thestream via hipEventRecord.
  static port::Status RecordEvent(int device_ordinal, hipEvent_t event,
                                  hipStream_t stream);

  // Polls (without blocking) to determine the status of an event - pending or
  // complete (or an error status).
  static port::StatusOr<hipError_t> QueryEvent(int device_ordinal,
                                             hipEvent_t event);

  // -- Pointer-specific calls.

  // Returns the device associated with the context from GetPointerContext().
  static port::StatusOr<hipDevice_t> GetPointerDevice(hipDeviceptr_t pointer);

  // Returns the memory space addressed by pointer.
  static port::StatusOr<MemorySpace> GetPointerMemorySpace(hipDeviceptr_t pointer);

  // Returns the base address and size of the device pointer dptr.
  static port::Status GetPointerAddressRange(hipDeviceptr_t dptr,
                                             hipDeviceptr_t *base, size_t *size);

  // -- Device-specific calls.

  // Returns AMDGPU ISA version for the device; i.e 803, 900.
  static port::Status GetAMDGPUISAVersion(int *version,
                                          hipDevice_t device);

  // Returns the number of multiprocessors on the device (note that the device
  // may be multi-GPU-per-board).
  static port::StatusOr<int> GetMultiprocessorCount(hipDevice_t device);

  // Returns the limit on number of threads that can be resident in a single
  // multiprocessor.
  static port::StatusOr<int64> GetMaxThreadsPerMultiprocessor(hipDevice_t device);

  // Returns the limit on number of threads which may be resident for a single
  // block (cooperative thread array).
  static port::StatusOr<int64> GetMaxThreadsPerBlock(hipDevice_t device);

  // Returns the amount of shared memory available on a single GPU core (i.e.
  // CU on ROCM devices).
  static port::StatusOr<int64> GetMaxSharedMemoryPerCore(hipDevice_t device);

  // Returns the amount of shared memory available for a single block
  // (cooperative thread array).
  static port::StatusOr<int64> GetMaxSharedMemoryPerBlock(hipDevice_t device);

  // Returns the maximum supported number of registers per block.
  static port::StatusOr<int64> GetMaxRegistersPerBlock(hipDevice_t device);

  // Returns the number of threads per warp.
  static port::StatusOr<int64> GetThreadsPerWarp(hipDevice_t device);

  // Queries the grid limits for device with hipDeviceGetAttribute calls.
  static bool GetGridLimits(int *x, int *y, int *z, hipDevice_t device);

  // Returns a grab-bag of device properties in a caller-owned device_properties
  // structure for device_ordinal via hipDeviceGetProperties.
  static bool GetDeviceProperties(hipDeviceProp_t *device_properties,
                                  int device_ordinal);

  // Returns whether ECC is enabled for the given hipDevice_t via
  // hipDeviceGetattribute with CU_DEVICE_ATTRIBUTE_ECC_ENABLED.
  static bool IsEccEnabled(hipDevice_t device, bool *result);

  // Returns the total amount of memory available for allocation by the ROCM
  // device, in bytes, via hipDeviceTotalMem.
  static bool GetDeviceTotalMemory(hipDevice_t device, uint64 *result);

  // Returns the free amount of memory and total amount of memory, as reported
  // by hipMemGetInfo.
  static bool GetDeviceMemoryInfo(int device_ordinal, int64* free,
                                  int64* total);

  // Returns a PCI bus id string for the device.
  // [domain]:[bus]:[device].[function]
  static string GetPCIBusID(hipDevice_t device);

  // -- Context- and device-independent calls.

  // Returns the number of visible ROCM device via hipDeviceGetCount.
  // This should correspond to the set of device ordinals available.
  static int GetDeviceCount();

  // Returns the driver version number via cuDriverGetVersion.
  // This is, surprisingly, NOT the actual driver version (e.g. 331.79) but,
  // instead, the ROCM toolkit release number that this driver is compatible
  // with; e.g. 6000 (for a ROCM 6.0 compatible driver) or 6050 (for a ROCM 6.5
  // compatible driver).
  //
  static bool GetDriverVersion(int *driver_version);

  // -- Other calls

  // Returns the maximum number of blocks (per multiprocessor) occupied by the
  // specified kernel/hipFunction_t when launched with the specified parameters.
  static port::StatusOr<int> GetMaxOccupiedBlocksPerCore(
      int device_ordinal, hipFunction_t kernel, int threads_per_block,
      size_t dynamic_shared_memory_bytes);

  // Returns the current device ordinal set in HIP. This is done by calling the
  // HIP driver (e.g., this value is not our cached view of the current device).
  static int CurrentDeviceOrDie();

  // Seam for injecting an error at CUDA initialization time for testing
  // purposes.
  static bool driver_inject_init_error_;
};

// Ensures the given device is activated within a scope.
class ScopedActivateContext {
 public:
  // Activates the given device , if it is not the currently active device
  explicit ScopedActivateContext(int device_ordinal);

  // Checks that the device has remained activated for the duration of the
  // scope.
  ~ScopedActivateContext();

 private:
  int to_restore_ = -1;
};

}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
