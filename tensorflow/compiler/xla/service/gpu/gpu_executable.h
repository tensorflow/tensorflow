/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// GPU-targeting implementation of the XLA Executable interface.
//
// Launches the given CUDA kernel via the StreamExecutor.
//
// This is an immutable data type after initialization, and thus thread safe.
class GpuExecutable : public Executable {
 public:
  // cubin (i.e. the compiled ptx) may be empty, in which case we leave
  // compilation up to the GPU driver.
  GpuExecutable(const string& ptx, const std::vector<uint8>& cubin,
                std::pair<int, int> compute_capability,
                std::unique_ptr<const ThunkSchedule> thunk_schedule,
                std::unique_ptr<HloModule> hlo_module,
                std::unique_ptr<const BufferAssignment> assignment,
                std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map);

  // This should be called after set_ir_module_string.
  const string& ir_module_string() const { return ir_module_string_; }

  // This should be called before ExecuteOnStream.
  void set_ir_module_string(const string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  // Returns the compiled PTX for the computation.
  const string& ptx() const { return ptx_; }

  // Returns the cubin (compiled PTX) stored in this GpuExecutable.  May be
  // empty, in which case compilation is left up to the GPU driver.
  const std::vector<uint8>& cubin() const { return cubin_; }

  // ExecuteOnStream will fail if the compute capability of the stream doesn't
  // match the compute capability passed to this object's constructor.
  StatusOr<ScopedShapedBuffer> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments) override;

 private:
  // If `block_host_until_done` is false, execution will not block the host
  // until the kernels have completed. This is used as an optimization for
  // clients, such as Tensorflow, that use a single stream of execution for
  // computations, and allow host-side deallocation from the allocator before
  // GPU execution completes.
  Status ExecuteThunks(const ServiceExecutableRunOptions* run_options,
                       const BufferAllocations& buffer_allocations,
                       bool block_host_until_done,
                       HloExecutionProfile* hlo_execution_profile);

  // Returns the points-to set of the root instruction of the entry
  // computation. Uses points-to analysis from buffer assignment.
  const PointsToSet& GetRootPointsToSet() const;

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceMemoryBase>;

  // Loads the PTX or CUBIN for this executable into `executor` and resolves the
  // globals corresponding to constant buffers.  Returns a map mapping buffer
  // allocation indices to GPU pointers.
  StatusOr<const BufferAllocToDeviceMemoryMap*> ResolveConstantGlobals(
      stream_executor::StreamExecutor* executor);

  // The LLVM IR, in string format, of the unoptimized module generated for this
  // GpuExecutable. We save a string instead of an llvm::Module* because leaving
  // llvm::Module* in a singleton can cause the heap checker to emit false
  // positives.
  //
  // This string should be modified only before ExecuteOnStream.
  string ir_module_string_;

  // The PTX for the computation.
  const string ptx_;

  // The GPU machine code for the computation, targeting GPUs at
  // compute_capability_.
  //
  // May be empty, in which case we leave compilation up to the GPU driver.
  const std::vector<uint8> cubin_;

  // The compute capability of the GPU we're targeting with this GpuExecutable.
  std::pair<int, int> compute_capability_;

  // The thunks to be invoked by this GpuExecutable. They are generated by the
  // IrEmitter.
  const std::unique_ptr<const ThunkSchedule> thunk_schedule_;

  // Owns the buffer data at runtime. It provides information to allocate
  // memory for every output/temp buffers.
  const std::unique_ptr<const BufferAssignment> assignment_;

  // Cache of module handles and constant buffer allocation maps used by
  // `ResolveConstantGlobals`.
  tensorflow::mutex module_handle_mutex_;
  std::map<stream_executor::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ GUARDED_BY(module_handle_mutex_);
  std::map<stream_executor::StreamExecutor*, BufferAllocToDeviceMemoryMap>
      module_globals_ GUARDED_BY(module_handle_mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(GpuExecutable);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
