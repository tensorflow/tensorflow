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

  // Returns a node's dependencies.
  //
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g048f4c0babcbba64a933fc277cd45083
  static absl::StatusOr<std::vector<GpuGraphNodeHandle>>
  GraphNodeGetDependencies(GpuGraphNodeHandle node);

  // Destroys an executable graph.
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7
  // https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Driver_API_functions_supported_by_HIP.html#graph-management
  static absl::Status DestroyGraphExec(GpuGraphExecHandle exec);

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
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_DRIVER_H_
