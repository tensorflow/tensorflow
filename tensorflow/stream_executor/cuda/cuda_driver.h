/* Copyright 2015 Google Inc. All Rights Reserved.

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

// CUDA userspace driver library wrapper functionality.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_

#include <stddef.h>
#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "third_party/gpus/cuda/include/cuda.h"

namespace perftools {
namespace gputools {
namespace cuda {

// Identifies the memory space where an allocation resides. See
// CUDADriver::GetPointerMemorySpace().
enum class MemorySpace { kHost, kDevice };

// Returns a casual string, such as "host" for the provided memory space.
string MemorySpaceString(MemorySpace memory_space);

class CudaContext;

// CUDADriver contains wrappers for calls to the userspace library driver. It's
// useful to isolate these calls and put basic wrappers around them to separate
// userspace library driver behaviors from the rest of the program.
//
// At the moment it's simply used as a namespace.
//
// The calls log any specific errors internally and return whether the operation
// was successful to the caller.
//
// The order of parameters is generally kept symmetric with the underlying CUDA
// driver API.
//
// Links on functions are to specific documentation under
// http://docs.nvidia.com/cuda/cuda-driver-api/
//
// Thread safety: these functions should not be used from signal handlers.
class CUDADriver {
 public:
  // Wraps a call to cuInit with logging to help indicate what has gone wrong in
  // the case of failure. Safe to call multiple times; will be fast on all calls
  // after the first.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3
  static port::Status Init();

  // Returns the device associated with the given context.
  // device is an outparam owned by the caller, must not be null.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e
  static port::StatusOr<CUdevice> DeviceFromContext(CudaContext* context);

  // Creates a new CUDA stream associated with the given context via
  // cuStreamCreate.
  // stream is an outparam owned by the caller, must not be null.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4
  static bool CreateStream(CudaContext* context, CUstream *stream);

  // Destroys a CUDA stream associated with the given context.
  // stream is owned by the caller, must not be null, and *stream is set to null
  // if the stream is successfuly destroyed.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758
  static void DestroyStream(CudaContext* context, CUstream *stream);

  // CUDA events can explicitly disable event TSC retrieval for some presumed
  // performance improvement if timing is unnecessary.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db
  enum class EventFlags { kDefault, kDisableTiming };

  // Creates a new event associated with the given context.
  // result is an outparam owned by the caller and must not be null.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db
  static port::Status CreateEvent(CudaContext* context, CUevent *result,
                                  EventFlags flags);

  // Destroys *event and turns it into a nullptr. event may not be null, but
  // *event may be, via cuEventDestroy
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef
  static port::Status DestroyEvent(CudaContext* context, CUevent *event);

  // Allocates a GPU memory space of size bytes associated with the given
  // context via cuMemAlloc.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
  static void *DeviceAllocate(CudaContext* context, uint64 bytes);

  // Deallocates a GPU memory space of size bytes associated with the given
  // context via cuMemFree.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a
  static void DeviceDeallocate(CudaContext* context, void *location);

  // Allocates page-locked and CUDA-registered memory on the host via
  // cuMemAllocHost.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0
  static void *HostAllocate(CudaContext* context, uint64 bytes);

  // Deallocates a location created by HostAllocate, via cuMemFreeHost.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c
  static void HostDeallocate(CudaContext* context, void *location);

  // Registers a memory region at location of size bytes via cuMemHostRegister.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223
  static bool HostRegister(CudaContext* context, void *location, uint64 bytes);

  // Unregisters a memory region that was previously registered at location via
  // cuMemHostUnregister.
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g63f450c8125359be87b7623b1c0b2a14
  //
  // TODO(leary) verify an error will be returned if the location wasn't
  // previously registered.
  static bool HostUnregister(CudaContext* context, void *location);

  // Given a device ordinal, returns a device handle into the device outparam,
  // which must not be null.
  //
  // N.B. these device handles do not have a corresponding destroy function in
  // the CUDA driver API.
  static port::Status GetDevice(int device_ordinal, CUdevice *device);

  // Given a device handle, returns the name reported by the driver for the
  // device.
  static bool GetDeviceName(CUdevice device, string *name_out);

  // Given a device to create a context for, returns a context handle into the
  // context outparam, which must not be null.
  //
  // N.B. CUDA contexts are weird. They are implicitly associated with the
  // calling thread. Current documentation on contexts and their influence on
  // userspace processes is given here:
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
  static port::Status CreateContext(CUdevice device,
                                    DeviceOptions device_options,
                                    CudaContext** context);

  // Destroys the provided context via cuCtxDestroy.
  // Don't do this while clients could still be using the context, per the docs
  // bad things will happen.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
  static void DestroyContext(CudaContext* context);

  // Queries the runtime for the specified attribute of the specified function.
  // cuFuncGetAttribute (the underlying CUDA driver API routine) only operates
  // in terms of integer-sized values, so there's no potential for overrun (as
  // of CUDA 5.5).
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b
  static bool FuncGetAttribute(CUfunction_attribute attribute,
                               CUfunction function, int *attribute_value);

  // Sets the preferred cache configuration for the specified function.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681
  static bool FuncSetCacheConfig(CUfunction function,
                                 CUfunc_cache cache_config);

  // Gets the preferred shared memory bank configuration for the specified
  // CONTEXT (not function!), either default or four- or eight-byte bank size.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g17153a1b8b8c756f7ab8505686a4ad74
  static port::StatusOr<CUsharedconfig> ContextGetSharedMemConfig(
      CudaContext* context);

  // Sets the preferred shared memory bank configuration for the specified
  // CONTEXT (not function!), either default or four- or eight-byte bank size.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2574235fa643f8f251bf7bc28fac3692
  static port::Status ContextSetSharedMemConfig(
      CudaContext* context, CUsharedconfig shared_mem_config);

  // Launches a CUDA kernel via cuLaunchKernel.
  // TODO(leary) describe the structure of kernel_params and extra in a readable
  // way.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
  static bool LaunchKernel(CudaContext* context, CUfunction function,
                           unsigned int grid_dim_x, unsigned int grid_dim_y,
                           unsigned int grid_dim_z, unsigned int block_dim_x,
                           unsigned int block_dim_y, unsigned int block_dim_z,
                           unsigned int shared_mem_bytes, CUstream stream,
                           void **kernel_params, void **extra);

  // Loads ptx_contents with the CUDA driver's PTX JIT and stores the resulting
  // handle in "module". Any error logs that are produced are logged internally.
  static bool LoadPtx(CudaContext* context, const char *ptx_contents,
                      CUmodule *module);

  // Loads cubin_bytes with the CUDA driver's blob loading interface and stores
  // the resulting handle in "module".
  static port::Status LoadCubin(CudaContext* context, const char *cubin_bytes,
                                CUmodule *module);

  // Retrieves a named kernel from a loaded module, and places the resulting
  // handle into function (outparam) on success. Neither kernel_name nor
  // function may be null. No ownership is taken of kernel_name.
  static bool GetModuleFunction(CudaContext* context, CUmodule module,
                                const char *kernel_name, CUfunction *function);

  // Retrieves a named global/constant symbol from a loaded module, and returns
  // a device pointer and size of the symbol on success. symbol_name may not be
  // null. At least one of dptr or bytes should not be null. No ownership is
  // taken of symbol_name.
  static bool GetModuleSymbol(CudaContext* context, CUmodule module,
                              const char *symbol_name, CUdeviceptr *dptr,
                              size_t *bytes);

  // Unloads module from the current context via cuModuleUnload.
  // TODO(leary) the documentation doesn't say what kind of disasters happen
  // if you try to unload a module while its CUfunctions are in use.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b
  static void UnloadModule(CudaContext* context, CUmodule module);

  // Performs a synchronous memset of the device memory segment via cuMemsetD8.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b
  static bool SynchronousMemsetUint8(CudaContext* context, CUdeviceptr location,
                                     uint8 value, size_t size);

  // Performs a synchronous memset of the device memory segment via cuMemsetD32.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132
  static bool SynchronousMemsetUint32(CudaContext* context,
                                      CUdeviceptr location, uint32 value,
                                      size_t uint32_count);

  // Performs an asynchronous memset of the device memory segment via
  // cuMemsetD8Async.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627
  static bool AsynchronousMemsetUint8(CudaContext* context, CUdeviceptr location,
                                      uint8 value, size_t uint32_count,
                                      CUstream stream);

  // Performs an asynchronous memset of the device memory segment via
  // cuMemsetD32Async.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5
  static bool AsynchronousMemsetUint32(CudaContext* context,
                                       CUdeviceptr location, uint32 value,
                                       size_t uint32_count, CUstream stream);

  // -- Synchronous memcopies.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169

  static bool SynchronousMemcpyD2H(CudaContext* context, void *host_dst,
                                   CUdeviceptr gpu_src, uint64 size);
  static bool SynchronousMemcpyH2D(CudaContext* context, CUdeviceptr gpu_dst,
                                   const void *host_src, uint64 size);
  static bool SynchronousMemcpyD2D(CudaContext* context, CUdeviceptr gpu_dst,
                                   CUdeviceptr gpu_src, uint64 size);

  // -- Asynchronous memcopies.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362

  static bool AsynchronousMemcpyD2H(CudaContext* context, void *host_dst,
                                    CUdeviceptr gpu_src, uint64 size,
                                    CUstream stream);
  static bool AsynchronousMemcpyH2D(CudaContext* context, CUdeviceptr gpu_dst,
                                    const void *host_src, uint64 size,
                                    CUstream stream);
  static bool AsynchronousMemcpyD2D(CudaContext* context, CUdeviceptr gpu_dst,
                                    CUdeviceptr gpu_src, uint64 size,
                                    CUstream stream);

  // The CUDA stream callback type signature.
  // The data passed to AddStreamCallback is subsequently passed to this
  // callback when it fires.
  //
  // Some notable things:
  // * Callbacks must not make any CUDA API calls.
  // * Callbacks from independent streams execute in an undefined order and may
  //   be serialized.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483
  typedef void (*StreamCallback)(CUstream stream, CUresult status, void *data);

  // Enqueues a callback operation into stream.
  // See StreamCallback above and the NVIDIA documentation for additional
  // details.
  static bool AddStreamCallback(CudaContext* context, CUstream stream,
                                StreamCallback callback, void *data);

  // Causes stream to wait for event to trigger before proceeding via
  // cuStreamWaitEvent.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#axzz334nAXAhM
  static bool WaitStreamOnEvent(CudaContext* context, CUstream stream,
                                CUevent event);

  // Blocks the calling thread until the operations enqueued onto stream have
  // been completed, via cuStreamSynchronize.
  //
  // TODO(leary) if a pathological thread enqueues operations onto the stream
  // while another thread blocks like this, can you wind up waiting an unbounded
  // amount of time?
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad
  static bool SynchronizeStream(CudaContext* context, CUstream stream);

  // Blocks the calling thread until the operations associated with the context
  // have been completed, via cuCtxSynchronize.
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616
  static bool SynchronizeContext(CudaContext* context);

  // Returns true if all stream tasks have completed at time of the call. Note
  // the potential for races around this call (if another thread adds work to
  // the stream immediately after this returns).
  static bool IsStreamIdle(CudaContext* context, CUstream stream);

  // Returns whether code in the from context can access memory in the to
  // context via cuDeviceCanAccessPeer.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e
  static bool CanEnablePeerAccess(CudaContext* from, CudaContext* to);

  // Enables peer access per CanEnablePeerAccess, via cuCtxEnablePeerAccess.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a
  static port::Status EnablePeerAccess(CudaContext* from, CudaContext* to);

  // Returns the elapsed milliseconds between start and stop via
  // cuEventElapsedTime.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97
  static bool GetEventElapsedTime(CudaContext* context,
                                  float *elapsed_milliseconds, CUevent start,
                                  CUevent stop);

  // Records that an event occurred when execution reaches the current point in
  // thestream via cuEventRecord.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1
  static port::Status RecordEvent(CudaContext* context, CUevent event,
                                  CUstream stream);

  // Polls (without blocking) to determine the status of an event - pending or
  // complete (or an error status).
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef
  static port::StatusOr<CUresult> QueryEvent(CudaContext* context,
                                             CUevent event);

  // -- Pointer-specific calls.

  // Returns the context in which pointer was allocated or registered.
  static port::StatusOr<CudaContext*> GetPointerContext(CUdeviceptr pointer);

  // Returns the device associated with the context from GetPointerContext().
  static port::StatusOr<CUdevice> GetPointerDevice(CUdeviceptr pointer);

  // Returns the memory space addressed by pointer.
  static port::StatusOr<MemorySpace> GetPointerMemorySpace(CUdeviceptr pointer);

  // Returns the base address and size of the device pointer dptr.
  static port::Status GetPointerAddressRange(CUdeviceptr dptr,
                                             CUdeviceptr *base, size_t *size);

  // -- Device-specific calls.

  // Returns the compute capability for the device; i.e (3, 5).
  // This is currently done via the deprecated device API.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED_1ge2091bbac7e1fb18c2821612115607ea
  static port::Status GetComputeCapability(int *cc_major, int *cc_minor,
                                           CUdevice device);

  // Returns the number of multiprocessors on the device (note that the device
  // may be multi-GPU-per-board).
  static port::StatusOr<int> GetMultiprocessorCount(CUdevice device);

  // Returns the limit on number of threads that can be resident in a single
  // multiprocessor.
  static port::StatusOr<int64> GetMaxThreadsPerMultiprocessor(CUdevice device);

  // Returns the limit on number of threads which may be resident for a single
  // block (cooperative thread array).
  static port::StatusOr<int64> GetMaxThreadsPerBlock(CUdevice device);

  // Returns the amount of shared memory available on a single GPU core (i.e.
  // SM on NVIDIA devices).
  static port::StatusOr<int64> GetMaxSharedMemoryPerCore(CUdevice device);

  // Returns the amount of shared memory available for a single block
  // (cooperative thread array).
  static port::StatusOr<int64> GetMaxSharedMemoryPerBlock(CUdevice device);

  // Returns the maximum supported number of registers per block.
  static port::StatusOr<int64> GetMaxRegistersPerBlock(CUdevice device);

  // Returns the number of threads per warp.
  static port::StatusOr<int64> GetThreadsPerWarp(CUdevice device);

  // Queries the grid limits for device with cuDeviceGetAttribute calls.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
  static bool GetGridLimits(int *x, int *y, int *z, CUdevice device);

  // Returns a grab-bag of device properties in a caller-owned device_properties
  // structure for device_ordinal via cuDeviceGetProperties.
  // This call is deprecated in the NVIDIA driver API.
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED_1g65a5b4e25186bd257df80b98c98cffe6
  static bool GetDeviceProperties(CUdevprop *device_properties,
                                  int device_ordinal);

  // Returns whether ECC is enabled for the given CUdevice via
  // cuDeviceGetattribute with CU_DEVICE_ATTRIBUTE_ECC_ENABLED.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
  static bool IsEccEnabled(CUdevice device, bool *result);

  // Returns the total amount of memory available for allocation by the CUDA
  // context, in bytes, via cuDeviceTotalMem.
  static bool GetDeviceTotalMemory(CUdevice device, uint64 *result);

  // Returns the free amount of memory and total amount of memory, as reported
  // by cuMemGetInfo.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0
  static bool GetDeviceMemoryInfo(CudaContext* context, int64* free,
                                  int64* total);

  // Returns a PCI bus id string for the device.
  // [domain]:[bus]:[device].[function]
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g85295e7d9745ab8f0aa80dd1e172acfc
  static string GetPCIBusID(CUdevice device);

  // -- Context- and device-independent calls.

  // Returns the number of visible CUDA device via cuDeviceGetCount.
  // This should correspond to the set of device ordinals available.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74
  static int GetDeviceCount();

  // Returns the driver version number via cuDriverGetVersion.
  // This is, surprisingly, NOT the actual driver version (e.g. 331.79) but,
  // instead, the CUDA toolkit release number that this driver is compatible
  // with; e.g. 6000 (for a CUDA 6.0 compatible driver) or 6050 (for a CUDA 6.5
  // compatible driver).
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION_1g8b7a10395392e049006e61bcdc8ebe71
  static bool GetDriverVersion(int *driver_version);

  // -- Other calls

  // Returns the maximum number of blocks (per multiprocessor) occupied by the
  // specified kernel/CUfunction when launched with the specified parameters.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98
  static port::StatusOr<int> GetMaxOccupiedBlocksPerCore(
      CudaContext* context, CUfunction kernel, int threads_per_block,
      size_t dynamic_shared_memory_bytes);

  // Returns the current context set in CUDA. This is done by calling the cuda
  // driver (e.g., this value is not our cached view of the current context).
  static CUcontext CurrentContextOrDie();

  // Seam for injecting an error at CUDA initialization time for testing
  // purposes.
  static bool driver_inject_init_error_;
};

// Ensures a context is activated within a scope.
class ScopedActivateContext {
 public:
  // Activates the context via cuCtxSetCurrent, if it is not the currently
  // active context (a la cuCtxGetCurrent). Note the alternative push/pop
  // mechanism is said by NVIDIA to be relatively slow and deprecated.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7
  explicit ScopedActivateContext(CudaContext* context);

  // Checks that the context has remained activated for the duration of the
  // scope.
  ~ScopedActivateContext();

 private:
  CudaContext* to_restore_ = nullptr;
};

// CudaContext wraps a cuda CUcontext handle, and includes a unique id. The
// unique id is positive, and ids are not repeated within the process.
class CudaContext {
 public:
  CudaContext(CUcontext context, int64 id) : context_(context), id_(id) { }

  CUcontext context() const { return context_; }
  int64 id() const { return id_; }

  // Disallow copying and moving.
  CudaContext(CudaContext&&) = delete;
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(CudaContext&&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

 private:
  CUcontext const context_;
  const int64 id_;
};

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
