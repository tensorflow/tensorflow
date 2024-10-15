/* Copyright 2019 The OpenXLA Authors.

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
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

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
  // Creates a new GPU graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::Status CreateGraph(GpuGraphHandle* graph);

  // Destroys GPU graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::Status DestroyGraph(GpuGraphHandle graph);

  // Begins graph capture on a stream.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  enum class StreamCaptureMode { kGlobal, kThreadLocal, kRelaxed };
  static absl::Status StreamBeginCapture(GpuStreamHandle stream,
                                         StreamCaptureMode mode);

  // Begins graph capture on a stream to an existing graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1gac495e0527d1dd6437f95ee482f61865
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::Status StreamBeginCaptureToGraph(GpuStreamHandle stream,
                                                GpuGraphHandle graph,
                                                StreamCaptureMode mode);

  // Ends capture on a stream, returning the captured graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::Status StreamEndCapture(GpuStreamHandle stream,
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
  static absl::Status GraphInstantiate(GpuGraphExecHandle* exec,
                                       GpuGraphHandle graph,
                                       const GraphInstantiateFlags& flags);

  // Launches an executable graph in a stream.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::Status GraphLaunch(GpuGraphExecHandle exec,
                                  GpuStreamHandle stream);

  // Enables or disables the specified node in the given exec.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g371b20eb0c0658731e38db7e68f12c78
  // https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___graph.html#ga8902200d9fed1df7644fc7a51c4d327b
  static absl::Status GraphNodeSetEnabled(GpuGraphExecHandle exec,
                                          GpuGraphNodeHandle node,
                                          bool enabled);

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
  static absl::Status GraphExecUpdate(GpuGraphExecHandle exec,
                                      GpuGraphHandle graph,
                                      GraphExecUpdateResultInfo* result);

  // Returns a node's dependencies.
  //
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g048f4c0babcbba64a933fc277cd45083
  static absl::StatusOr<std::vector<GpuGraphNodeHandle>>
  GraphNodeGetDependencies(GpuGraphNodeHandle node);

  // Destroys an executable graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::Status DestroyGraphExec(GpuGraphExecHandle exec);

  // Write a DOT file describing graph structure.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::StatusOr<std::string> GraphDebugDotPrint(
      GpuGraphHandle graph, const char* path,
      bool return_printed_graph = false);

  // Creates a conditional handle.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gece6f3b9e85d0edb8484d625fe567376
  static absl::Status GraphConditionalHandleCreate(
      GpuGraphConditionalHandle* handle, GpuGraphHandle graph, Context* context,
      unsigned int default_launch_value, unsigned int flags);

  // Conditional node parameters.
  // https://docs.nvidia.com/cuda/cuda-driver-api/structCUDA__CONDITIONAL__NODE__PARAMS.html#structCUDA__CONDITIONAL__NODE__PARAMS
  struct GpuGraphConditionalNodeParams {
    // Conditional node type.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g04ade961d0263336423eb216fbe514da
    enum class Type { kIf, kWhile };

    // A struct for returning output arguments back to the caller.
    struct Result {
      GpuGraphHandle graph;
    };

    Type type;
    GpuGraphConditionalHandle handle;
    Context* context;
  };

  // Graph node parameters
  // https://docs.nvidia.com/cuda/cuda-driver-api/structCUgraphNodeParams.html#structCUgraphNodeParams
  using GpuGraphNodeParams = std::variant<GpuGraphConditionalNodeParams>;
  using GpuGraphNodeResult =
      std::variant<GpuGraphConditionalNodeParams::Result>;

  // Adds a node of arbitrary type to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4210c258cbba352040a26d1b4e658f9d
  static absl::StatusOr<GpuGraphNodeResult> GraphAddNode(
      GpuGraphNodeHandle* node, GpuGraphHandle graph,
      absl::Span<const GpuGraphNodeHandle> deps,
      const GpuGraphNodeParams& params);

  // Creates an empty node and adds it to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223
  static absl::Status GraphAddEmptyNode(
      GpuGraphNodeHandle* node, GpuGraphHandle graph,
      absl::Span<const GpuGraphNodeHandle> deps);

  // Creates a kernel execution node and adds it to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::Status GraphAddKernelNode(
      GpuGraphNodeHandle* node, GpuGraphHandle graph,
      absl::Span<const GpuGraphNodeHandle> deps, absl::string_view kernel_name,
      GpuFunctionHandle function, unsigned int grid_dim_x,
      unsigned int grid_dim_y, unsigned int grid_dim_z,
      unsigned int block_dim_x, unsigned int block_dim_y,
      unsigned int block_dim_z, unsigned int shared_mem_bytes,
      void** kernel_params, void** extra);

  // Counts number of nodes in the graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5
  // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/group___graph.html#gaf006701d98164ed3492755bbb19bab83
  static absl::StatusOr<size_t> GraphGetNodeCount(GpuGraphHandle graph);

  // Sets the parameters for a kernel node in the given graph exec.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c
  // https://docs.amd.com/projects/HIP/en/docs-5.0.0/doxygen/html/group___graph.html#ga5b1918dae65224863b7370e6d4ad3f2a
  static absl::Status GraphExecKernelNodeSetParams(
      GpuGraphExecHandle exec, GpuGraphNodeHandle node,
      absl::string_view kernel_name, GpuFunctionHandle function,
      unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
      unsigned int block_dim_x, unsigned int block_dim_y,
      unsigned int block_dim_z, unsigned int shared_mem_bytes,
      void** kernel_params, void** extra);

  // Creates a memcpy node and adds it to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073
  static absl::Status GraphAddMemcpyD2DNode(
      Context* context, GpuGraphNodeHandle* node, GpuGraphHandle graph,
      absl::Span<const GpuGraphNodeHandle> deps, GpuDevicePtr gpu_dst,
      GpuDevicePtr gpu_src, uint64_t size);

  // Sets the parameters for a memcpy node in the given graphExec.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5
  static absl::Status GraphExecMemcpyD2DNodeSetParams(
      Context* context, GpuGraphExecHandle exec, GpuGraphNodeHandle node,
      GpuDevicePtr gpu_dst, GpuDevicePtr gpu_src, uint64_t size);

  // Creates a memset node and adds it to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3
  static absl::Status GraphAddMemsetNode(
      Context* context, GpuGraphNodeHandle* node, GpuGraphHandle graph,
      absl::Span<const GpuGraphNodeHandle> deps, GpuDevicePtr dst,
      std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
      uint64_t num_elements);

  // Sets the parameters for a memset node in the given graph exec.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739
  static absl::Status GraphExecMemsetNodeSetParams(
      Context* context, GpuGraphExecHandle exec, GpuGraphNodeHandle node,
      GpuDevicePtr dst, std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
      uint64_t num_elements);

  // Creates a child graph node and adds it to a graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844
  static absl::Status GraphAddChildNode(
      GpuGraphNodeHandle* node, GpuGraphHandle graph,
      absl::Span<const GpuGraphNodeHandle> deps, GpuGraphHandle child);

  // Sets the parameters for a child graph node in the given graph exec.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff
  static absl::Status GraphExecChildNodeSetParams(GpuGraphExecHandle exec,
                                                  GpuGraphNodeHandle node,
                                                  GpuGraphHandle child);

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

  // Blocks the calling thread until the operations enqueued onto stream have
  // been completed, via cuStreamSynchronize.
  //
  // TODO(leary) if a pathological thread enqueues operations onto the stream
  // while another thread blocks like this, can you wind up waiting an unbounded
  // amount of time?
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad
  static absl::Status SynchronizeStream(Context* context,
                                        GpuStreamHandle stream);

  // Blocks the calling thread until the operations associated with the context
  // have been completed, via cuCtxSynchronize.
  //
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616
  static absl::Status SynchronizeContext(Context* context);

  // Returns true if all stream tasks have completed at time of the call. Note
  // the potential for races around this call (if another thread adds work to
  // the stream immediately after this returns).
  static bool IsStreamIdle(Context* context, GpuStreamHandle stream);

  // Returns whether code in the from context can access memory in the to
  // context via cuDeviceCanAccessPeer.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e
  static bool CanEnablePeerAccess(Context* from, Context* to);

  // Returns whether the from device can access memory in the to
  // device via cuDeviceCanAccessPeer. Because of differences between ROCM and
  // CUDA, this API is not supported in ROCM builds and will result in a link
  // error if used.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e
  static bool CanEnablePeerAccess(GpuDeviceHandle from, GpuDeviceHandle to);

  // Enables peer access per CanEnablePeerAccess, via cuCtxEnablePeerAccess.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a
  static absl::Status EnablePeerAccess(Context* from, Context* to);

  // Returns the elapsed milliseconds between start and stop via
  // cuEventElapsedTime.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97
  static absl::StatusOr<float> GetEventElapsedTime(Context* context,
                                                   GpuEventHandle start,
                                                   GpuEventHandle stop);

  // Records that an event occurred when execution reaches the current point in
  // thestream via cuEventRecord.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1
  static absl::Status RecordEvent(Context* context, GpuEventHandle event,
                                  GpuStreamHandle stream);

  // -- Pointer-specific calls.

  // Returns the memory space addressed by pointer.
  static absl::StatusOr<MemoryType> GetPointerMemorySpace(GpuDevicePtr pointer);

  // Returns the base address and size of the device pointer dptr.
  static absl::Status GetPointerAddressRange(GpuDevicePtr dptr,
                                             GpuDevicePtr* base, size_t* size);

  // -- Device-specific calls.

  // Returns the compute capability for the device; i.e (3, 5).
  // This is currently done via the deprecated device API.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED_1ge2091bbac7e1fb18c2821612115607ea
  // (supported on CUDA only)
  static absl::Status GetComputeCapability(int* cc_major, int* cc_minor,
                                           GpuDeviceHandle device);

  // Returns Gpu ISA version for the device; i.e 803, 900.
  // (supported on ROCm only)
  static absl::Status GetGpuISAVersion(int* version, GpuDeviceHandle device);

  // Return the full GCN Architecture Name for the device
  // for eg: amdgcn-amd-amdhsa--gfx908:sramecc+:xnack-
  // (supported on ROCm only)
  static absl::Status GetGpuGCNArchName(GpuDeviceHandle device,
                                        std::string* gcnArchName);

  // Returns the number of multiprocessors on the device (note that the device
  // may be multi-GPU-per-board).
  static absl::StatusOr<int> GetMultiprocessorCount(GpuDeviceHandle device);

  // Returns the limit on number of threads that can be resident in a single
  // multiprocessor.
  static absl::StatusOr<int64_t> GetMaxThreadsPerMultiprocessor(
      GpuDeviceHandle device);

  // Returns the amount of shared memory available on a single GPU core (i.e.
  // SM on NVIDIA devices).
  static absl::StatusOr<int64_t> GetMaxSharedMemoryPerCore(
      GpuDeviceHandle device);

  // Returns the amount of static shared memory available for a single block
  // (cooperative thread array).
  static absl::StatusOr<int64_t> GetMaxSharedMemoryPerBlock(
      GpuDeviceHandle device);

  // Returns the total amount of shared memory available for a single block
  // (cooperative thread array).
  static absl::StatusOr<int64_t> GetMaxSharedMemoryPerBlockOptin(
      GpuDeviceHandle device);

  // Returns the maximum supported number of registers per block.
  static absl::StatusOr<int64_t> GetMaxRegistersPerBlock(
      GpuDeviceHandle device);

  // Returns the number of threads per warp.
  static absl::StatusOr<int64_t> GetThreadsPerWarp(GpuDeviceHandle device);

  // Queries the grid limits for device with cuDeviceGetAttribute calls.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
  static absl::Status GetGridLimits(int* x, int* y, int* z,
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
  static absl::StatusOr<int> GetDeviceAttribute(GpuDeviceAttribute attribute,
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
  static bool GetDeviceMemoryInfo(Context* context, int64_t* free,
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
  static absl::StatusOr<int32_t> GetDriverVersion();

  // -- Other calls

  // Returns the maximum number of blocks (per multiprocessor) occupied by the
  // specified kernel/GpuFunctionHandle when launched with the specified
  // parameters.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98
  static absl::StatusOr<int> GetMaxOccupiedBlocksPerCore(
      Context* context, GpuFunctionHandle kernel, int threads_per_block,
      size_t dynamic_shared_memory_bytes);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_DRIVER_H_
