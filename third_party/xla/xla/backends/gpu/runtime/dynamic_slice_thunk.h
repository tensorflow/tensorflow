/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_DYNAMIC_SLICE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_DYNAMIC_SLICE_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// DynamicSliceThunk wraps the logic to compute dynamic offsets/sizes from
// dynamic-slice or DUS around some original thunks (e.g. custom call or NCCL
// thunks)
//
// DynamicSliceThunk assumes that the slices are contiguous.
class DynamicSliceThunk : public Thunk {
 public:
  // Dynamic slice offset can be either: (1) a statically known constant value
  // or (2) a truly dynamic offset that is computed on device and have to be
  // transferred to host or (3) a temporary HloModule that computes the offset
  // with a single induction variable as the input.
  using Offset = std::variant<int64_t, BufferAllocation::Slice, HloModule*>;

  struct OffsetAsFunctionOfIndvarModulesMetadata {
    // These two modules help keep track of the induction variable. The module
    // `indvar_init_` is a module with prototype `() -> integer[]`. It takes
    // no input, and returns the initial value of the induction variable. The
    // module `indvar_update_` is a module with prototype `(integer[]) ->
    // integer[]`. It takes the current value of the induction variable, and
    // returns the next value of the induction variable.
    std::unique_ptr<HloModule> indvar_init, indvar_update;

    // Extracted HloModules for computing dynamic offsets. The modules are
    // not used here, this is solely for keeping the modules alive and maintain
    // their ownership with the thunk while their raw pointers would be used
    // during execution from the `offsets_` vector. These modules are with
    // signature `(integer[]) -> integer[]`, where the input is the current
    // value of the loop induction variable, and the output is the offset value
    // for that iteration.
    std::vector<std::unique_ptr<HloModule>> extracted_offset_modules;

    OffsetAsFunctionOfIndvarModulesMetadata(
        std::unique_ptr<HloModule> indvar_init,
        std::unique_ptr<HloModule> indvar_update,
        std::vector<std::unique_ptr<HloModule>> extracted_offset_modules)
        : indvar_init(std::move(indvar_init)),
          indvar_update(std::move(indvar_update)),
          extracted_offset_modules(std::move(extracted_offset_modules)) {
      CHECK(this->indvar_init != nullptr && this->indvar_update != nullptr);
      Shape init_output_shape =
          this->indvar_init->entry_computation()->root_instruction()->shape();
      CHECK(this->indvar_init->entry_computation()->num_parameters() == 0 &&
            init_output_shape.IsInteger() &&
            ShapeUtil::IsScalar(init_output_shape))
          << "Induction variable init module expected with signature `() -> "
             "integer[]`.";
      Shape update_output_shape =
          this->indvar_update->entry_computation()->root_instruction()->shape();
      CHECK(this->indvar_update->entry_computation()->num_parameters() == 1 &&
            update_output_shape.IsInteger() &&
            ShapeUtil::IsScalar(update_output_shape))
          << "Induction variable update module expected with signature "
             "`(integer[]) -> integer[]`.";
      Shape update_input_shape = this->indvar_update->entry_computation()
                                     ->parameter_instruction(0)
                                     ->shape();
      CHECK(ShapeUtil::IsScalar(update_input_shape) &&
            update_input_shape.IsInteger())
          << "Induction variable update module expected with signature "
             "`(integer[]) -> integer[]`.";
    }
  };

  DynamicSliceThunk(
      ThunkInfo thunk_info, std::unique_ptr<ThunkSequence> embedded_thunk,
      std::vector<std::optional<BufferAllocation::Slice>> arguments,
      std::vector<std::unique_ptr<BufferAllocation>> fake_allocations,
      std::vector<std::optional<std::vector<Offset>>> offsets,
      std::vector<std::optional<Shape>> orig_shapes,
      std::vector<std::optional<Shape>> sliced_shapes,
      std::vector<std::optional<uint64_t>> offset_byte_sizes,
      std::optional<OffsetAsFunctionOfIndvarModulesMetadata>
          offset_as_function_of_indvar_metadata = std::nullopt);
  DynamicSliceThunk(const DynamicSliceThunk&) = delete;
  DynamicSliceThunk& operator=(const DynamicSliceThunk&) = delete;

  const Thunk* embedded_thunk() const { return embedded_thunk_.get(); }

  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequestsInterface& resource_requests) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  // Definition of a dynamic slice that extract a slice from the original buffer
  // defined by `embedded_thunk_argument` at given `offsets`.
  struct SliceDef {
    std::optional<BufferAllocation::Slice> embedded_thunk_argument;
    std::optional<std::vector<Offset>> offsets;
    std::optional<Shape> orig_shape;
    std::optional<Shape> sliced_shape;
    std::optional<uint64_t> offset_byte_size;
  };

  const SequentialThunk* get_embedded_thunk() const {
    return embedded_thunk_.get();
  }

  std::vector<std::optional<BufferAllocation::Slice>> get_arguments() const {
    return arguments_;
  }

  const std::vector<std::unique_ptr<BufferAllocation>>& get_fake_allocations()
      const {
    return fake_allocations_;
  }

  std::vector<std::optional<std::vector<Offset>>> get_offsets() const {
    return offsets_;
  }

  std::vector<std::optional<Shape>> get_orig_shapes() const {
    return orig_shapes_;
  }

  std::vector<std::optional<Shape>> get_sliced_shapes() const {
    return sliced_shapes_;
  }

  std::vector<std::optional<uint64_t>> get_offset_byte_sizes() const {
    return offset_byte_sizes_;
  }

  void ForAllThunks(absl::FunctionRef<void(const Thunk*)> fn) const override;

 private:
  std::unique_ptr<SequentialThunk> embedded_thunk_;
  std::vector<std::optional<BufferAllocation::Slice>> arguments_;
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_;
  std::vector<std::optional<std::vector<Offset>>> offsets_;
  std::vector<std::optional<Shape>> orig_shapes_;
  std::vector<std::optional<Shape>> sliced_shapes_;
  std::vector<std::optional<uint64_t>> offset_byte_sizes_;

  std::vector<SliceDef> slices_;

  // Pinned host memory for transferring offset values from device to host.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      offsets_allocs_ ABSL_GUARDED_BY(mutex_);

  // Pre-computed size requirement for `offsets_allocs_`.
  int64_t offsets_allocs_size_ = 0;

  // A mapping from argument index to the base offset in the `offsets_allocs_`.
  std::vector<int64_t> offsets_allocs_base_;

  // This structure holds the metadata for offset computations on host. It
  // stores a single induction variable initialization module, its update module
  // and the offsets that are a function of the induction variable.
  std::optional<OffsetAsFunctionOfIndvarModulesMetadata>
      offset_as_function_of_indvar_metadata_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_DYNAMIC_SLICE_THUNK_H_
