/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// CUDA/ROCm userspace driver library wrapper functionality.

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_DRIVER_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_DRIVER_H_

#include <stddef.h>

#include <cstdint>
#include <variant>

#include "absl/types/span.h"
#include "xla/stream_executor/device_options.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/platform.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

// Identifies the memory space where an allocation resides. See
// GpuDriver::GetPointerMemorySpace().
enum class MemorySpace { kHost, kDevice };

// Returns a casual string, such as "host" for the provided memory space.
std::string MemorySpaceString(MemorySpace memory_space);

class GpuContext;

// GpuDriver contains wrappers for calls to the userspace library driver. It's
// useful to isolate these calls and put basic wrappers around them to separate
// userspace library driver behaviors from the rest of the program.
//
// At the moment it's simply used as a namespace.
//
// The calls log any specific errors internally and return whether the operation
// was successful to the caller.
//
// The order of parameters is generally kept symmetric with the underlying
// CUDA/ROCm driver API.
//
// Links on functions are to specific documentation under
// http://docs.nvidia.com/cuda/cuda-driver-api/
// https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html
//
// Thread safety: these functions should not be used from signal handlers.
class GpuDriver {
 public:
  // Wraps a call to cuInit/hipInit with logging to help indicate what has gone
  // wrong in the case of failure. Safe to call multiple times; will be fast on
  // all calls after the first.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#initialization
  static tsl::Status Init();

  // Returns the device associated with the given context.
  // device is an outparam owned by the caller, must not be null.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e
  static tsl::StatusOr<GpuDeviceHandle> DeviceFromContext(GpuContext* context);

  // Creates a new CUDA/HIP stream associated with the given context via
  // cuStreamCreate/hipStreamCreateWithFlags.
  // stream is an outparam owned by the caller, must not be null.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#stream-management
  static bool CreateStream(GpuContext* context, GpuStreamHandle* stream,
                           int priority = 0);

  // Destroys a CUDA/HIP stream associated with the given context.
  // stream is owned by the caller, must not be null, and *stream is set to null
  // if the stream is successfully destroyed.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#stream-management
  static void DestroyStream(GpuContext* context, GpuStreamHandle* stream);

  // CUDA/HIP events can explicitly disable event TSC retrieval for some
  // presumed performance improvement if timing is unnecessary.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#cuda-driver-data-types
  enum class EventFlags { kDefault, kDisableTiming };

  // Creates a new event associated with the given context.
  // result is an outparam owned by the caller and must not be null.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#cuda-driver-data-types
  static tsl::Status InitEvent(GpuContext* context, GpuEventHandle* result,
                               EventFlags flags);

  // Destroys *event and turns it into a nullptr. event may not be null, but
  // *event may be, via cuEventDestroy/hipEventDestroy
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#event-management
  static tsl::Status DestroyEvent(GpuContext* context, GpuEventHandle* event);

  // Allocates a GPU memory space of size bytes associated with the given
  // context via cuMemAlloc/hipMalloc.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#memory-management
  static void* DeviceAllocate(GpuContext* context, uint64_t bytes);

  // Deallocates a GPU memory space of size bytes associated with the given
  // context via cuMemFree/hipFree.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#memory-management
  static void DeviceDeallocate(GpuContext* context, void* location);

  // Allocates a unified memory space of size bytes associated with the given
  // context via cuMemAllocManaged.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32
  // (supported on CUDA only)
  static void* UnifiedMemoryAllocate(GpuContext* context, uint64_t bytes);

  // Deallocates a unified memory space of size bytes associated with the given
  // context via cuMemFree.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a
  // (supported on CUDA only)
  static void UnifiedMemoryDeallocate(GpuContext* context, void* location);

  // Allocates page-locked and CUDA-registered memory on the host via
  // cuMemAllocHost/hipHostMalloc.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#memory-management
  static void* HostAllocate(GpuContext* context, uint64_t bytes);

  // Deallocates a location created by HostAllocate, via
  // cuMemFreeHost/hipHostFree.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#memory-management
  static void HostDeallocate(GpuContext* context, void* location);

  // Registers a memory region at location of size bytes via
  // cuMemHostRegister/hipHostRegister.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#memory-management
  static bool HostRegister(GpuContext* context, void* location, uint64_t bytes);

  // Unregisters a memory region that was previously registered at location via
  // cuMemHostUnregister/hipHostUnregister.
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g63f450c8125359be87b7623b1c0b2a14
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#memory-management
  //
  // TODO(leary) verify an error will be returned if the location wasn't
  // previously registered.
  static bool HostUnregister(GpuContext* context, void* location);

  // Queries the priority range and returns the corresponding integer value via
  // cuCtxGetStreamPriorityRange/hipDeviceGetStreamPriorityRange
  //
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#context-management
  static int GetGpuStreamPriority(
      GpuContext* context, stream_executor::StreamPriority stream_priority);

  // Virtual memory support was added to CUDA in 10.2
#if CUDA_VERSION >= 10020

  // Reserves a range of virtual device memory addresses via
  // cuMemAddressReserve. bytes must be a multiple of the host page size.
  // Returns nullptr base address in VmemSpan if the reservation fails.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b
  struct VmemSpan {
    GpuDevicePtr base;
    // Size in bytes.
    uint64_t size_bytes;
  };
  static tsl::StatusOr<VmemSpan> ReserveVirtualMemory(GpuContext* context,
                                                      uint64_t bytes);

  // Frees a range of virtual addresses that were previously reserved through
  // ReserveVirtualMemory via cuMemAddressFree.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1g6993ecea2ea03e1b802b8255edc2da5b
  static void FreeVirtualMemory(GpuContext* context, VmemSpan reservation);

  // Calculates the minimum alignment for memory allocations done through
  // cuMemCreate via cuMemGetAllocationGranularity.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1g30ee906c2cf66a0347b3dfec3d7eb31a
  static tsl::StatusOr<uint64_t> GetMinAllocationGranularity(
      GpuDeviceHandle device);

  // Allocates physical memory and returns a handle that can be mapped to
  // virtual addresses via cuMemCreate. bytes must be a multiple of the
  // granularity returned by GetMinAllocationGranularity.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c
  struct GenericMemoryHandle {
    uint64_t handle;
    uint64_t bytes;
  };
  static tsl::StatusOr<GenericMemoryHandle> CreateMemoryHandle(
      GpuContext* context, uint64_t bytes);

  // Frees memory represented by the provided MemoryHandle via cuMemRelease.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68
  static void ReleaseMemoryHandle(GpuContext* context,
                                  GenericMemoryHandle handle);

  // Maps a memory allocation handle to a reserved virtual address range via
  // cuMemMap and sets the appropriate access settings via cuMemSetAccess.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1
  static tsl::Status MapMemory(
      GpuContext* context, GpuDevicePtr va, const GenericMemoryHandle& handle,
      const std::vector<GpuDeviceHandle>& device_handles);

  // Unmaps the backing memory from the given virtual address range. This range
  // must fully unmap a memory handle that was mapped using MapMemory; partial
  // unmapping is not supported.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1gfb50aac00c848fd7087e858f59bf7e2a
  static void UnmapMemory(GpuContext* context, GpuDevicePtr va, uint64_t bytes);

#endif  // CUDA_VERSION >= 10200

  // Given a device ordinal, returns a device handle into the device outparam,
  // which must not be null.
  //
  // N.B. these device handles do not have a corresponding destroy function in
  // the CUDA/HIP driver API.
  static tsl::Status GetDevice(int device_ordinal, GpuDeviceHandle* device);

  // Given a device handle, returns the name reported by the driver for the
  // device.
  static tsl::Status GetDeviceName(GpuDeviceHandle device,
                                   std::string* device_name);

  // Given a device to create a context for, returns a context handle into the
  // context outparam, which must not be null.
  //
  // N.B. CUDA contexts are weird. They are implicitly associated with the
  // calling thread. Current documentation on contexts and their influence on
  // userspace processes is given here:
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
  static tsl::Status CreateContext(int device_ordinal, GpuDeviceHandle device,
                                   const DeviceOptions& device_options,
                                   GpuContext** context);

  // Destroys the provided context via cuCtxDestroy.
  // Don't do this while clients could still be using the context, per the docs
  // bad things will happen.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
  static void DestroyContext(GpuContext* context);

  // Returns the context handle (CUcontext for CUDA and hipCtx_t for ROCm) of a
  // GpuContext.
  static GpuContextHandle GetContextHandle(GpuContext* context);

  // Queries the runtime for the specified attribute of the specified function.
  // cuFuncGetAttribute (the underlying CUDA driver API routine) only operates
  // in terms of integer-sized values, so there's no potential for overrun (as
  // of CUDA 5.5).
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b
  static tsl::Status FuncGetAttribute(GpuFunctionAttribute attribute,
                                      GpuFunctionHandle function,
                                      int* attribute_value);

  // Sets the preferred cache configuration for the specified function.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681
  static tsl::Status FuncSetCacheConfig(GpuFunctionHandle function,
                                        GpuFuncCachePreference cache_config);

  // Gets the preferred shared memory bank configuration for the specified
  // CONTEXT (not function!), either default or four- or eight-byte bank size.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g17153a1b8b8c756f7ab8505686a4ad74
  // https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___execution.html
  static tsl::StatusOr<GpuSharedMemConfig> ContextGetSharedMemConfig(
      GpuContext* context);

  // Sets the preferred shared memory bank configuration for the specified
  // CONTEXT (not function!), either default or four- or eight-byte bank size.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2574235fa643f8f251bf7bc28fac3692
  // https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___execution.html
  static tsl::Status ContextSetSharedMemConfig(
      GpuContext* context, GpuSharedMemConfig shared_mem_config);

  // Launches a CUDA/ROCm kernel via cuLaunchKernel/hipModuleLaunchKernel.
  // TODO(leary) describe the structure of kernel_params and extra in a readable
  // way.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#execution-control
  static tsl::Status LaunchKernel(
      GpuContext* context, absl::string_view kernel_name,
      GpuFunctionHandle function, unsigned int grid_dim_x,
      unsigned int grid_dim_y, unsigned int grid_dim_z,
      unsigned int block_dim_x, unsigned int block_dim_y,
      unsigned int block_dim_z, unsigned int shared_mem_bytes,
      GpuStreamHandle stream, void** kernel_params, void** extra);

  // Creates a new GPU graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status CreateGraph(GpuGraphHandle* graph);

  // Destroys GPU graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status DestroyGraph(GpuGraphHandle graph);

  // Begins graph capture on a stream.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  enum class StreamCaptureMode { kGlobal, kThreadLocal, kRelaxed };
  static tsl::Status StreamBeginCapture(GpuStreamHandle stream,
                                        StreamCaptureMode mode);

  // Ends capture on a stream, returning the captured graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status StreamEndCapture(GpuStreamHandle stream,
                                      GpuGraphHandle* graph);

  // Graph instantiation flags.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g070bf5517d3a7915667c256eefce4956
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#cuda-driver-data-types
  struct GraphInstantiateFlags {
    // Automatically free memory allocated in a graph before relaunching.
    bool auto_free_on_launch = false;
    // Automatically upload the graph after instantiation.
    bool upload = false;
    // Instantiate the graph to be launchable from the device.
    bool device_launch = false;
    // Run the graph using the per-node priority attributes rather than the
    // priority of the stream it is launched into.
    bool use_node_prirotiy = false;
  };

  // Creates an executable graph from a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status GraphInstantiate(GpuGraphExecHandle* exec,
                                      GpuGraphHandle graph,
                                      const GraphInstantiateFlags& flags);

  // Launches an executable graph in a stream.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status GraphLaunch(GpuGraphExecHandle exec,
                                 GpuStreamHandle stream);

  // Graph update result.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g8edc8969ff6ae00b7cd5d7292f812c3c
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#cuda-driver-data-types
  enum class GraphExecUpdateResult {
    kSuccess,
    kError,
    kTopologyChanged,
    kNodeTypeChanged,
    kFunctionChanged,
    kParametersChanged,
    kNotSupported,
    kUnsupportedFunctionChange,
    kAttributesChanged
  };

  // Graph update result info.
  // https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphExecUpdateResultInfo__v1.html#structCUgraphExecUpdateResultInfo__v1
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  struct GraphExecUpdateResultInfo {
    GpuGraphNodeHandle error_from_node;
    GpuGraphNodeHandle error_node;
    GraphExecUpdateResult result;
  };

  // Check whether an executable graph can be updated with a graph and perform
  // the update if possible.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status GraphExecUpdate(GpuGraphExecHandle exec,
                                     GpuGraphHandle graph,
                                     GraphExecUpdateResultInfo* result);

  // Graph node type.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g0731a28f826922120d783d8444e154dc
  // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/group___graph.html#ga4727d20b89566832c74b762f987b9728
  enum class GraphNodeType {
    kKernel,
    kMemcpy,
    kMemset,
    kHost,
    kGraph,
    kEmpty,
    kWaitEvent,
    kEventRecord,
    kExtSemasSignal,
    kExtSemasWait,
    kMemAlloc,
    kMemFree,
    kBatchMemOp,
  };

  // Return the node type of the graph node.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdb1776d97aa1c9d5144774b29e4b8c3e
  // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/group___graph.html#ga87c68ae9408a6438d4a1101560ceea11
  static tsl::StatusOr<GraphNodeType> GraphNodeGetType(GpuGraphNodeHandle node);

  // Destroys an executable graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status DestroyGraphExec(GpuGraphExecHandle exec);

  // Write a DOT file describing graph structure.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status GraphDebugDotPrint(GpuGraphHandle graph, const char* path);

  // Returns a stream's capture status.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#stream-management
  static tsl::StatusOr<bool> StreamIsCapturing(GpuStreamHandle stream);

  // Free unused memory that was cached on the specified device for use with
  // graphs back to the OS.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g57c87f4ba6af41825627cdd4e5a8c52b
  static tsl::Status DeviceGraphMemTrim(GpuDeviceHandle device);

  // Creates a conditional handle.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gece6f3b9e85d0edb8484d625fe567376
  static tsl::Status GraphConditionalHandleCreate(
      GpuGraphConditionalHandle* handle, GpuGraphHandle graph,
      GpuContext* context, unsigned int default_launch_value,
      unsigned int flags);

  // Conditional node parameters.
  // https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__CONDITIONAL__NODE__PARAMS.html#structCUDA__CONDITIONAL__NODE__PARAMS
  struct GpuGraphConditionalNodeParams {
    // Conditional node type.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g04ade961d0263336423eb216fbe514da
    enum class Type { kIf };

    // A struct for returning output arguments back to the caller.
    struct Result {
      GpuGraphHandle graph;
    };

    Type type;
    GpuGraphConditionalHandle handle;
    GpuContext* context;
  };

  // Graph node parameters
  // https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphNodeParams.html#structCUgraphNodeParams
  using GpuGraphNodeParams = std::variant<GpuGraphConditionalNodeParams>;
  using GpuGraphNodeResult =
      std::variant<GpuGraphConditionalNodeParams::Result>;

  // Adds a node of arbitrary type to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4210c258cbba352040a26d1b4e658f9d
  static tsl::StatusOr<GpuGraphNodeResult> GraphAddNode(
      GpuGraphNodeHandle* node, GpuGraphHandle graph,
      absl::Span<GpuGraphNodeHandle> deps, const GpuGraphNodeParams& params);

  // Creates a kernel execution node and adds it to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static tsl::Status GraphAddKernelNode(
      GpuGraphNodeHandle* node, GpuGraphHandle graph,
      absl::Span<GpuGraphNodeHandle> deps, absl::string_view kernel_name,
      GpuFunctionHandle function, unsigned int grid_dim_x,
      unsigned int grid_dim_y, unsigned int grid_dim_z,
      unsigned int block_dim_x, unsigned int block_dim_y,
      unsigned int block_dim_z, unsigned int shared_mem_bytes,
      void** kernel_params, void** extra);

  // Sets the parameters for a kernel node in the given graph exec.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c
  // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/group___graph.html#ga5b1918dae65224863b7370e6d4ad3f2a
  static tsl::Status GraphExecKernelNodeSetParams(
      GpuGraphExecHandle exec, GpuGraphNodeHandle node,
      absl::string_view kernel_name, GpuFunctionHandle function,
      unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
      unsigned int block_dim_x, unsigned int block_dim_y,
      unsigned int block_dim_z, unsigned int shared_mem_bytes,
      void** kernel_params, void** extra);

  // Creates a memcpy node and adds it to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073
  static tsl::Status GraphAddMemcpyD2DNode(GpuContext* context,
                                           GpuGraphNodeHandle* node,
                                           GpuGraphHandle graph,
                                           absl::Span<GpuGraphNodeHandle> deps,
                                           GpuDevicePtr gpu_dst,
                                           GpuDevicePtr gpu_src, uint64_t size);

  // Creates a child graph node and adds it to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844
  static tsl::Status GraphAddChildNode(GpuGraphNodeHandle* node,
                                       GpuGraphHandle graph,
                                       absl::Span<GpuGraphNodeHandle> deps,
                                       GpuGraphHandle child);

  // Sets the parameters for a child graph node in the given graph exec.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff
  static tsl::Status GraphExecChildNodeSetParams(GpuGraphExecHandle exec,
                                                 GpuGraphNodeHandle node,
                                                 GpuGraphHandle child);

  // Loads ptx_contents with the CUDA driver's PTX JIT and stores the resulting
  // handle in "module". Any error logs that are produced are logged internally.
  // (supported on CUDA only)
  static tsl::Status LoadPtx(GpuContext* context, const char* ptx_contents,
                             GpuModuleHandle* module);

  // Loads cubin_bytes with the CUDA driver's blob loading interface and stores
  // the resulting handle in "module".
  // (supported on CUDA only)
  static tsl::Status LoadCubin(GpuContext* context, const char* cubin_bytes,
                               GpuModuleHandle* module);

  // Loads HSACO with the ROCM runtime and stores the resulting handle in
  // "module". Any error logs that are produced are logged internally.
  // (supported on ROCm only)
  static tsl::Status LoadHsaco(GpuContext* context, const char* hsaco_contents,
                               GpuModuleHandle* module);

  // Retrieves a named kernel from a loaded module, and places the resulting
  // handle into function (outparam) on success. Neither kernel_name nor
  // function may be null. No ownership is taken of kernel_name.
  static tsl::Status GetModuleFunction(GpuContext* context,
                                       GpuModuleHandle module,
                                       const char* kernel_name,
                                       GpuFunctionHandle* function);

  // Retrieves a named global/constant symbol from a loaded module, and returns
  // a device pointer and size of the symbol on success. symbol_name may not be
  // null. At least one of dptr or bytes should not be null. No ownership is
  // taken of symbol_name.
  static bool GetModuleSymbol(GpuContext* context, GpuModuleHandle module,
                              const char* symbol_name, GpuDevicePtr* dptr,
                              size_t* bytes);

  // Unloads module from the current context via cuModuleUnload.
  // TODO(leary) the documentation doesn't say what kind of disasters happen
  // if you try to unload a module while its GpuFunctionHandles are in use.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b
  static void UnloadModule(GpuContext* context, GpuModuleHandle module);

  // Performs a synchronous memset of the device memory segment via cuMemsetD8.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b
  static tsl::Status SynchronousMemsetUint8(GpuContext* context,
                                            GpuDevicePtr location,
                                            uint8_t value, size_t size);

  // Performs a synchronous memset of the device memory segment via cuMemsetD32.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132
  static tsl::Status SynchronousMemsetUint32(GpuContext* context,
                                             GpuDevicePtr location,
                                             uint32_t value,
                                             size_t uint32_count);

  // Performs an asynchronous memset of the device memory segment via
  // cuMemsetD8Async.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627
  static tsl::Status AsynchronousMemsetUint8(GpuContext* context,
                                             GpuDevicePtr location,
                                             uint8_t value, size_t uint32_count,
                                             GpuStreamHandle stream);

  // Performs an asynchronous memset of the device memory segment via
  // cuMemsetD32Async.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5
  static tsl::Status AsynchronousMemsetUint32(GpuContext* context,
                                              GpuDevicePtr location,
                                              uint32_t value,
                                              size_t uint32_count,
                                              GpuStreamHandle stream);

  // -- Synchronous memcopies.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169

  static tsl::Status SynchronousMemcpyD2H(GpuContext* context, void* host_dst,
                                          GpuDevicePtr gpu_src, uint64_t size);
  static tsl::Status SynchronousMemcpyH2D(GpuContext* context,
                                          GpuDevicePtr gpu_dst,
                                          const void* host_src, uint64_t size);
  static tsl::Status SynchronousMemcpyD2D(GpuContext* context,
                                          GpuDevicePtr gpu_dst,
                                          GpuDevicePtr gpu_src, uint64_t size);

  // -- Asynchronous memcopies.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362

  static bool AsynchronousMemcpyD2H(GpuContext* context, void* host_dst,
                                    GpuDevicePtr gpu_src, uint64_t size,
                                    GpuStreamHandle stream);
  static bool AsynchronousMemcpyH2D(GpuContext* context, GpuDevicePtr gpu_dst,
                                    const void* host_src, uint64_t size,
                                    GpuStreamHandle stream);
  static bool AsynchronousMemcpyD2D(GpuContext* context, GpuDevicePtr gpu_dst,
                                    GpuDevicePtr gpu_src, uint64_t size,
                                    GpuStreamHandle stream);

  // The CUDA stream callback type signature.
  // The data passed to AddStreamCallback is subsequently passed to this
  // callback when it fires.
  //
  // Some notable things:
  // * Callbacks must not make any CUDA API calls.
  // * Callbacks from independent streams execute in an undefined order and may
  //   be serialized.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f
  typedef void (*StreamCallback)(void* data);

  // Enqueues a callback operation into stream.
  // See StreamCallback above and the NVIDIA documentation for additional
  // details.
  static bool AddStreamCallback(GpuContext* context, GpuStreamHandle stream,
                                StreamCallback callback, void* data);

  // Causes stream to wait for event to trigger before proceeding via
  // cuStreamWaitEvent.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#axzz334nAXAhM
  static bool WaitStreamOnEvent(GpuContext* context, GpuStreamHandle stream,
                                GpuEventHandle event);

  // Blocks the calling thread until the operations enqueued onto stream have
  // been completed, via cuStreamSynchronize.
  //
  // TODO(leary) if a pathological thread enqueues operations onto the stream
  // while another thread blocks like this, can you wind up waiting an unbounded
  // amount of time?
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad
  static tsl::Status SynchronizeStream(GpuContext* context,
                                       GpuStreamHandle stream);

  // Blocks the calling thread until the operations associated with the context
  // have been completed, via cuCtxSynchronize.
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616
  static bool SynchronizeContext(GpuContext* context);

  // Returns true if all stream tasks have completed at time of the call. Note
  // the potential for races around this call (if another thread adds work to
  // the stream immediately after this returns).
  static bool IsStreamIdle(GpuContext* context, GpuStreamHandle stream);

  // Returns whether code in the from context can access memory in the to
  // context via cuDeviceCanAccessPeer.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e
  static bool CanEnablePeerAccess(GpuContext* from, GpuContext* to);

  // Returns whether the from device can access memory in the to
  // device via cuDeviceCanAccessPeer. Because of differences between ROCM and
  // CUDA, this API is not supported in ROCM builds and will result in a link
  // error if used.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e
  static bool CanEnablePeerAccess(GpuDeviceHandle from, GpuDeviceHandle to);

  // Enables peer access per CanEnablePeerAccess, via cuCtxEnablePeerAccess.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a
  static tsl::Status EnablePeerAccess(GpuContext* from, GpuContext* to);

  // Returns the elapsed milliseconds between start and stop via
  // cuEventElapsedTime.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97
  static bool GetEventElapsedTime(GpuContext* context,
                                  float* elapsed_milliseconds,
                                  GpuEventHandle start, GpuEventHandle stop);

  // Records that an event occurred when execution reaches the current point in
  // thestream via cuEventRecord.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1
  static tsl::Status RecordEvent(GpuContext* context, GpuEventHandle event,
                                 GpuStreamHandle stream);

  // Polls (without blocking) to determine the status of an event - pending or
  // complete (or an error status).
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef
  static tsl::StatusOr<GpuStatus> QueryEvent(GpuContext* context,
                                             GpuEventHandle event);

  // -- Pointer-specific calls.

  // Returns the context in which pointer was allocated or registered.
  static tsl::StatusOr<GpuContext*> GetPointerContext(GpuDevicePtr pointer);

  // Returns the device associated with the context from GetPointerContext().
  static tsl::StatusOr<GpuDeviceHandle> GetPointerDevice(GpuDevicePtr pointer);

  // Returns the memory space addressed by pointer.
  static tsl::StatusOr<MemorySpace> GetPointerMemorySpace(GpuDevicePtr pointer);

  // Returns the base address and size of the device pointer dptr.
  static tsl::Status GetPointerAddressRange(GpuDevicePtr dptr,
                                            GpuDevicePtr* base, size_t* size);

  // -- Device-specific calls.

  // Returns the compute capability for the device; i.e (3, 5).
  // This is currently done via the deprecated device API.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED_1ge2091bbac7e1fb18c2821612115607ea
  // (supported on CUDA only)
  static tsl::Status GetComputeCapability(int* cc_major, int* cc_minor,
                                          GpuDeviceHandle device);

  // Returns Gpu ISA version for the device; i.e 803, 900.
  // (supported on ROCm only)
  static tsl::Status GetGpuISAVersion(int* version, GpuDeviceHandle device);

  // Return the full GCN Architecture Name for the device
  // for eg: amdgcn-amd-amdhsa--gfx908:sramecc+:xnack-
  // (supported on ROCm only)
  static tsl::Status GetGpuGCNArchName(GpuDeviceHandle device,
                                       std::string* gcnArchName);

#if TENSORFLOW_USE_ROCM
  // tests the current device for MFMA insn support (ROCm only)
  static tsl::StatusOr<bool> GetMFMASupport();
#endif

  // Returns the number of multiprocessors on the device (note that the device
  // may be multi-GPU-per-board).
  static tsl::StatusOr<int> GetMultiprocessorCount(GpuDeviceHandle device);

  // Returns the limit on number of threads that can be resident in a single
  // multiprocessor.
  static tsl::StatusOr<int64_t> GetMaxThreadsPerMultiprocessor(
      GpuDeviceHandle device);

  // Returns the limit on number of threads which may be resident for a single
  // block (cooperative thread array).
  static tsl::StatusOr<int64_t> GetMaxThreadsPerBlock(GpuDeviceHandle device);

  // Returns the amount of shared memory available on a single GPU core (i.e.
  // SM on NVIDIA devices).
  static tsl::StatusOr<int64_t> GetMaxSharedMemoryPerCore(
      GpuDeviceHandle device);

  // Returns the amount of static shared memory available for a single block
  // (cooperative thread array).
  static tsl::StatusOr<int64_t> GetMaxSharedMemoryPerBlock(
      GpuDeviceHandle device);

  // Returns the total amount of shared memory available for a single block
  // (cooperative thread array).
  static tsl::StatusOr<int64_t> GetMaxSharedMemoryPerBlockOptin(
      GpuDeviceHandle device);

  // Returns the maximum supported number of registers per block.
  static tsl::StatusOr<int64_t> GetMaxRegistersPerBlock(GpuDeviceHandle device);

  // Returns the number of threads per warp.
  static tsl::StatusOr<int64_t> GetThreadsPerWarp(GpuDeviceHandle device);

  // Queries the grid limits for device with cuDeviceGetAttribute calls.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
  static tsl::Status GetGridLimits(int* x, int* y, int* z,
                                   GpuDeviceHandle device);

  // Returns a grab-bag of device properties in a caller-owned device_properties
  // structure for device_ordinal via cuDeviceGetProperties.
  //
  // This call is deprecated in the NVIDIA driver API; its replacement is
  // GetDeviceAttribute
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED_1g65a5b4e25186bd257df80b98c98cffe6
  static bool GetDeviceProperties(GpuDeviceProperty* device_properties,
                                  int device_ordinal);

  // Gets a specific integer-valued property about the given device.
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
  static tsl::StatusOr<int> GetDeviceAttribute(GpuDeviceAttribute attribute,
                                               GpuDeviceHandle device);

  // Returns whether ECC is enabled for the given GpuDeviceHandle via
  // cuDeviceGetattribute with CU_DEVICE_ATTRIBUTE_ECC_ENABLED.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
  static bool IsEccEnabled(GpuDeviceHandle device, bool* result);

  // Returns the total amount of memory available for allocation by the CUDA
  // context, in bytes, via cuDeviceTotalMem.
  static bool GetDeviceTotalMemory(GpuDeviceHandle device, uint64_t* result);

  // Returns the free amount of memory and total amount of memory, as reported
  // by cuMemGetInfo.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0
  static bool GetDeviceMemoryInfo(GpuContext* context, int64_t* free,
                                  int64_t* total);

  // Returns a PCI bus id string for the device.
  // [domain]:[bus]:[device].[function]
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g85295e7d9745ab8f0aa80dd1e172acfc
  static std::string GetPCIBusID(GpuDeviceHandle device);

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
  static bool GetDriverVersion(int* driver_version);

  // -- Other calls

  // Returns the maximum number of blocks (per multiprocessor) occupied by the
  // specified kernel/GpuFunctionHandle when launched with the specified
  // parameters.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98
  static tsl::StatusOr<int> GetMaxOccupiedBlocksPerCore(
      GpuContext* context, GpuFunctionHandle kernel, int threads_per_block,
      size_t dynamic_shared_memory_bytes);

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
  explicit ScopedActivateContext(GpuContext* context);

  // Checks that the context has remained activated for the duration of the
  // scope.
  ~ScopedActivateContext();

 private:
  GpuContext* to_restore_ = nullptr;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_DRIVER_H_
