/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_
#define XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

class ThunkExecutor;

// Owns executable-scoped buffer allocation state for one GpuExecutable.
//
// This base class implements the ALWAYS_UPDATE command buffer update mode,
// which needs no allocation-address policy beyond global constants. The
// SKIP_TEMP and SKIP_PROFILED update modes are implemented by
// GpuExecutableVaRemapAllocator (see gpu_executable_va_remap_allocator.h),
// which assigns stable addresses to selected command-buffer allocations via
// VMM VA remapping. The base-class behavior also serves as the runtime
// fallback for those modes when VA remapping is unavailable for an execution.
class GpuExecutableBufferAllocator {
 public:
  struct ParameterBuffer {
    se::DeviceAddressBase buffer;
    int64_t parameter_number = 0;
    bool allow_null_buffer = false;
  };

  // Resolves the device address backing an entry-computation-parameter
  // allocation. Returning `allow_null_buffer` is used for skipped tuple
  // index-table allocations.
  using ParameterBufferResolver =
      absl::FunctionRef<absl::StatusOr<ParameterBuffer>(
          const BufferAllocation& allocation)>;

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceAddressBase>;

  using AllocationIndexSet = absl::btree_set<BufferAllocation::Index>;

  // Per-run buffer allocation context created by `CreateExecutionScope`.
  // Callers first use it to build `BufferAllocations` from runtime parameters,
  // constants, temporary buffers, and output buffers, then use it to run the
  // executable with those allocations.
  //
  // This base class resolves allocation addresses without any VA remapping
  // and passes only global constants as persistent allocations. Subclasses
  // override the protected hooks to install a per-execution
  // allocation-address policy.
  class ExecutionScope {
   public:
    ExecutionScope(const ExecutionScope&) = delete;
    ExecutionScope& operator=(const ExecutionScope&) = delete;
    virtual ~ExecutionScope() = default;

    // True when command-buffer VA remapping is active for this execution.
    virtual bool va_remap_enabled() const { return false; }

    // Builds the BufferAllocations for an execution. Entry-computation
    // parameter buffers are obtained from `get_parameter_buffer`; all other
    // allocations are resolved internally, including alignment checking and
    // any subclass allocation-address policy.
    absl::StatusOr<BufferAllocations> GenerateBufferAllocations(
        const ServiceExecutableRunOptions* run_options,
        ParameterBufferResolver get_parameter_buffer,
        const BufferAllocToDeviceMemoryMap* globals,
        se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

    // Copy-protection for an aliased output that was not donated at runtime:
    // allocates a fresh result buffer for the output at `index`, copies the
    // contents of the aliased buffer (allocation `allocation`) into it, and
    // redirects the aliased entry in `buffer_allocations` to the fresh buffer.
    // Returns the newly allocated result buffer.
    absl::StatusOr<se::DeviceAddressBase> AllocateCopyProtectedOutputBuffer(
        const ServiceExecutableRunOptions* run_options,
        BufferAllocations& buffer_allocations, const ShapeIndex& index,
        const BufferAllocation& allocation, int device_ordinal,
        se::DeviceAddressAllocator* memory_allocator,
        absl::FunctionRef<absl::Status(absl::Status)> allocation_error);

    // Runs `execute` with the allocation-address policy for this execution.
    // The base implementation passes the command-buffer-referenced constant
    // allocations as the persistent allocation indices.
    virtual absl::Status ExecuteWithBufferAllocations(
        const BufferAllocations& owning_buffer_allocations, int device_ordinal,
        absl::FunctionRef<absl::Status(
            const BufferAllocations&,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices)>
            execute);

   protected:
    explicit ExecutionScope(const GpuExecutableBufferAllocator* owner)
        : owner_(owner) {}

    // Hook called once per GenerateBufferAllocations before any allocation is
    // resolved. The base implementation does nothing.
    virtual absl::Status Prepare(const ServiceExecutableRunOptions* run_options,
                                 int device_ordinal) {
      return absl::OkStatus();
    }

    // Hook that allocates a non-parameter, non-constant allocation of
    // `buffer_size` bytes (> 0). The base implementation allocates from
    // `memory_allocator`.
    virtual absl::StatusOr<se::DeviceAddressBase> AllocateTransientBuffer(
        int device_ordinal, const BufferAllocation& allocation,
        int64_t buffer_size, se::DeviceAddressAllocator* memory_allocator);

   private:
    friend class GpuExecutableBufferAllocator;

    absl::StatusOr<se::DeviceAddressBase> BufferForAllocation(
        ParameterBufferResolver get_parameter_buffer,
        const BufferAllocToDeviceMemoryMap* globals,
        const BufferAllocation& allocation,
        se::DeviceAddressAllocator* memory_allocator, int device_ordinal,
        int64_t arg_idx);

    const GpuExecutableBufferAllocator* owner_ = nullptr;
  };

  // Creates the buffer allocator implementing
  // `debug_options->xla_gpu_command_buffer_update_mode()`: this class for
  // ALWAYS_UPDATE, GpuExecutableVaRemapAllocator for SKIP_TEMP and
  // SKIP_PROFILED. Check-fails on any other mode.
  static std::unique_ptr<GpuExecutableBufferAllocator> Create(
      absl::string_view module_name,
      absl::Span<const BufferAllocation* const> allocations,
      const Shape& result_shape, const DebugOptions* debug_options,
      ThunkExecutor* thunk_executor);

  GpuExecutableBufferAllocator(
      absl::string_view module_name,
      absl::Span<const BufferAllocation* const> allocations,
      const Shape& result_shape, const DebugOptions* debug_options,
      ThunkExecutor* thunk_executor);
  virtual ~GpuExecutableBufferAllocator() = default;

  size_t command_buffer_allocation_count() const {
    return persistent_alloc_indices_.size();
  }

  virtual absl::StatusOr<std::unique_ptr<ExecutionScope>> CreateExecutionScope(
      const ServiceExecutableRunOptions* run_options,
      se::DeviceAddressAllocator* memory_allocator, int device_ordinal);

 protected:
  // Invokes `callback` for every valid, non-empty allocation referenced by a
  // command buffer thunk of `thunk_executor` (which may be null).
  static void ForEachCommandBufferAllocation(
      absl::Span<const BufferAllocation* const> allocations,
      const ThunkExecutor* thunk_executor,
      absl::FunctionRef<void(BufferAllocation::Index, const BufferAllocation&)>
          callback);

  const std::string& module_name() const { return module_name_; }
  absl::Span<const BufferAllocation* const> allocations() const {
    return allocations_;
  }
  const DebugOptions* debug_options() const { return debug_options_; }
  absl::Span<const BufferAllocation::Index> constant_alloc_indices() const {
    return constant_alloc_indices_;
  }
  absl::Span<const BufferAllocation::Index> persistent_alloc_indices() const {
    return persistent_alloc_indices_;
  }
  void set_persistent_alloc_indices(
      std::vector<BufferAllocation::Index> indices) {
    persistent_alloc_indices_ = std::move(indices);
  }

 private:
  std::string module_name_;
  std::vector<const BufferAllocation*> allocations_;
  Shape result_shape_;
  const DebugOptions* debug_options_ = nullptr;

  // Sorted indices of command-buffer-referenced constant allocations. Their
  // global addresses are stable without VMM remapping.
  std::vector<BufferAllocation::Index> constant_alloc_indices_;

  // Sorted indices of command-buffer-referenced allocations with stable
  // addresses across executions. Equals `constant_alloc_indices_` here;
  // subclasses extend it with VA-remapped allocations.
  std::vector<BufferAllocation::Index> persistent_alloc_indices_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_EXECUTABLE_BUFFER_ALLOCATOR_H_
