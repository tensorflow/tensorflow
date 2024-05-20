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

#include "xla/service/gpu/runtime/address_computation_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

AddressComputationThunk::AddressComputationThunk(
    ThunkInfo thunk_info, std::unique_ptr<ThunkSequence> embedded_thunk,
    std::vector<std::optional<const BufferAllocation::Slice>> arguments,
    std::vector<std::unique_ptr<BufferAllocation>> fake_allocations,
    std::vector<std::optional<std::vector<BufferAllocation::Slice>>>
        offset_buffer_indices,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<uint64_t>> offset_byte_sizes)
    : Thunk(Kind::kAddressComputation, thunk_info),
      embedded_thunk_(std::make_unique<SequentialThunk>(
          ThunkInfo(thunk_info.op), std::move(*embedded_thunk))),
      embedded_thunk_arguments_(std::move(arguments)),
      fake_allocations_(std::move(fake_allocations)),
      offset_buffer_indices_(std::move(offset_buffer_indices)),
      orig_shapes_(std::move(orig_shapes)),
      sliced_shapes_(std::move(sliced_shapes)),
      offset_byte_sizes_(std::move(offset_byte_sizes)) {}

absl::Status AddressComputationThunk::Prepare(
    const PrepareParams& params, ResourceRequests& resource_requests) {
  auto num_arguments = embedded_thunk_arguments_.size();
  TF_RET_CHECK(num_arguments == offset_buffer_indices_.size());
  TF_RET_CHECK(num_arguments == orig_shapes_.size());
  TF_RET_CHECK(num_arguments == sliced_shapes_.size());
  TF_RET_CHECK(num_arguments == offset_byte_sizes_.size());
  for (auto [argument, offset_slice, orig_shape, sliced_shape,
             offset_byte_size] :
       llvm::zip(embedded_thunk_arguments_, offset_buffer_indices_,
                 orig_shapes_, sliced_shapes_, offset_byte_sizes_)) {
    if (offset_slice.has_value()) {
      TF_RET_CHECK(argument.has_value());
      TF_RET_CHECK(orig_shape.has_value());
      TF_RET_CHECK(sliced_shape.has_value());
      TF_RET_CHECK(offset_byte_size.has_value());

      TF_RET_CHECK(orig_shape->IsArray());
      TF_RET_CHECK(sliced_shape->IsArray());

      TF_RET_CHECK(offset_slice->size() == orig_shape->rank());
      TF_RET_CHECK(sliced_shape->rank() == orig_shape->rank());
    }
  }

  TF_RETURN_IF_ERROR(embedded_thunk_->Prepare(params, resource_requests));
  return absl::OkStatus();
}

absl::Status AddressComputationThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(embedded_thunk_->Initialize(params));

  unsigned offset_count = 0;
  for (auto maybe_shape : sliced_shapes_) {
    offset_count += (maybe_shape == std::nullopt) ? 1 : maybe_shape->rank();
  }

  absl::MutexLock lock(&mutex_);
  if (auto it = offsets_.find(params.executor); it == offsets_.end()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::MemoryAllocation> allocation,
        params.executor->HostMemoryAllocate(offset_count * sizeof(int64_t)));
    offsets_.emplace(params.executor, std::move(allocation));
  }

  return absl::OkStatus();
}

absl::Status AddressComputationThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& stream = *params.stream;
  const BufferAllocations& orig_allocations = *params.buffer_allocations;
  std::vector<se::DeviceMemoryBase> new_buffers(
      embedded_thunk_arguments_.size(), se::DeviceMemoryBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_base = [&] {
    absl::MutexLock lock(&mutex_);
    return reinterpret_cast<int64_t*>(offsets_.at(stream.parent())->opaque());
  }();

  for (auto [argument_idx, values] : llvm::enumerate(
           llvm::zip(embedded_thunk_arguments_, offset_buffer_indices_,
                     orig_shapes_, sliced_shapes_, offset_byte_sizes_))) {
    auto [argument_slice, offset_slice, orig_shape, sliced_shape,
          offset_byte_size] = values;

    if (argument_slice == std::nullopt) {
      continue;
    }

    // `orig_argument` will contain the original offset for slice
    // `argument_slice` within `orig_allocations`
    se::DeviceMemoryBase orig_argument =
        orig_allocations.GetDeviceAddress(*argument_slice);

    if (offset_slice == std::nullopt) {
      new_buffers[argument_idx] = orig_argument;
      continue;
    }

    const Shape& src_shape = *orig_shape;
    const Shape& dst_shape = *sliced_shape;
    TF_RET_CHECK(IsContiguousSlice(src_shape, dst_shape));

    std::vector<int64_t> slice_starts;
    slice_starts.reserve(dst_shape.rank());

    // Get offset for `argument_idx`-th argument, which has `dst_shape.rank()`
    // components.
    for (auto [offset_idx, values] : llvm::enumerate(llvm::zip(
             *offset_slice, src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [slice, src_dim, dst_dim] = values;
      se::DeviceMemoryBase offset_src =
          orig_allocations.GetDeviceAddress(slice);
      int64_t* offset_dst = &offsets_base[argument_idx + offset_idx];
      // Copy the `offset_idx`-th component of the offset for the
      // `argument_idx`-th argument from device to host.
      TF_RETURN_IF_ERROR(
          stream.Memcpy(offset_dst, offset_src, offset_byte_size.value()));

      if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
        return absl::InternalError(absl::StrFormat(
            "Failed to retrieve all slice offset values on stream %p: %s",
            &stream, blocked.message()));
      }
      // Clamp start indices:
      // start_indices[i] = min(max(start_indices[i], 0),
      //                        operand.dimension_size[i] - size_indices[i])
      auto start_index = std::min(std::max(*offset_dst, 0L), src_dim - dst_dim);
      slice_starts.push_back(start_index);
    }

    // Compute new slice. No need to copy the content to new buffers as we can
    // reuse the original buffers since slices are contiguous.
    int64_t new_size = ShapeUtil::ByteSizeOf(dst_shape);

    int64_t new_offset = 0;
    for (auto [start, stride] :
         llvm::zip(slice_starts, *ShapeUtil::ByteStrides(src_shape))) {
      new_offset += start * stride;
    }

    new_buffers[argument_idx] =
        orig_argument.GetByteSlice(new_offset, new_size);
  }

  // Safe to create a local BufferAllocations here since buffers are only slices
  // of bigger ones allocated elsewhere.
  BufferAllocations new_allocations(new_buffers,
                                    orig_allocations.device_ordinal(),
                                    orig_allocations.memory_allocator());

  Thunk::ExecuteParams new_params =
      Thunk::ExecuteParams::CloneWithNewAllocations(params, new_allocations);

  // Execute the underlying custom call thunk with the new buffers.
  TF_RETURN_IF_ERROR(embedded_thunk_->ExecuteOnStream(new_params));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
