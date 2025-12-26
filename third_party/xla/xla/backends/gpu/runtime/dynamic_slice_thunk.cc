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

#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.pb.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// Indvar is a thread-local map that stores the induction variable for each
// dynamic slice thunk. The same thunk object in the memory is shared by
// multiple replicas of the same computation. So, each replica should have its
// own tracking of the induction variable (threadlocal). With threadlocal, we
// cannot embed this inside the dynamic slice thunk object, and so we have a
// static map. There could be multiple dynamic slice thunks in the same module,
// and so we need a map to store the induction variable for each thunk. The
// usage of threadlocal in this context is similar to `LoopCounters` in
// while_thunk.cc (b/343294327).
Literal& Indvar(DynamicSliceThunk* thunk) {
  static thread_local absl::flat_hash_map<DynamicSliceThunk*, Literal>
      indvar_map;
  return indvar_map[thunk];
}

using DynamicSliceOffsetProto =
    OptionalDynamicSliceOffsetsProto::DynamicSliceOffsetProto;

}  // namespace

absl::StatusOr<OffsetAsFunctionOfIndvarModulesMetadataProto>
DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata::ToProto() const {
  OffsetAsFunctionOfIndvarModulesMetadataProto proto;
  *proto.mutable_indvar_init() = indvar_init->ToProtoWithConfig();
  *proto.mutable_indvar_update() = indvar_update->ToProtoWithConfig();
  for (const auto& module : extracted_offset_modules) {
    *proto.add_extracted_offset_modules() = module->ToProtoWithConfig();
  }
  return proto;
}

absl::StatusOr<DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata>
DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata::FromProto(
    const OffsetAsFunctionOfIndvarModulesMetadataProto& proto) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> indvar_init,
      HloModule::CreateFromProtoWithConfig(proto.indvar_init()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> indvar_update,
      HloModule::CreateFromProtoWithConfig(proto.indvar_update()));
  std::vector<std::unique_ptr<HloModule>> extracted_offset_modules;
  for (const auto& module_proto : proto.extracted_offset_modules()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        HloModule::CreateFromProtoWithConfig(module_proto));
    extracted_offset_modules.push_back(std::move(module));
  }
  return OffsetAsFunctionOfIndvarModulesMetadata(
      std::move(indvar_init), std::move(indvar_update),
      std::move(extracted_offset_modules));
}

std::string DynamicSliceThunk::SliceDef::ToString() const {
  std::string result = "SliceDef{";

  // embedded_thunk_argument
  if (embedded_thunk_argument.has_value()) {
    absl::StrAppend(&result, "embedded_thunk_argument:",
                    embedded_thunk_argument->ToString());
  } else {
    absl::StrAppend(&result, "embedded_thunk_argument:null");
  }

  // offsets
  if (offsets.has_value()) {
    absl::StrAppend(&result, ", offsets:[");
    absl::StrAppend(
        &result,
        absl::StrJoin(*offsets, ", ", [](std::string* out, const auto& offset) {
          std::visit(
              [out](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, int64_t>) {
                  absl::StrAppend(out, value);
                } else if constexpr (std::is_same_v<T,
                                                    BufferAllocation::Slice>) {
                  absl::StrAppend(out, value.ToString());
                } else if constexpr (std::is_same_v<T, HloModule*>) {
                  absl::StrAppend(out, "HloModule*:", value->ToString());
                }
              },
              offset);
        }));
    absl::StrAppend(&result, "]");
  } else {
    absl::StrAppend(&result, ", offsets:null");
  }

  if (orig_shape.has_value()) {
    absl::StrAppend(&result, ", orig_shape:", orig_shape->ToString());
  } else {
    absl::StrAppend(&result, ", orig_shape:null");
  }

  if (sliced_shape.has_value()) {
    absl::StrAppend(&result, ", sliced_shape:", sliced_shape->ToString());
  } else {
    absl::StrAppend(&result, ", sliced_shape:null");
  }

  if (offset_primitive_type.has_value()) {
    absl::StrAppend(
        &result, ", offset_primitive_type:",
        primitive_util::LowercasePrimitiveTypeName(*offset_primitive_type));
  } else {
    absl::StrAppend(&result, ", offset_primitive_type:null");
  }

  absl::StrAppend(&result, "}");
  return result;
}

DynamicSliceThunk::DynamicSliceThunk(
    ThunkInfo thunk_info, std::unique_ptr<ThunkSequence> embedded_thunk,
    std::vector<std::optional<BufferAllocation::Slice>> arguments,
    std::vector<BufferAllocation> fake_allocations,
    std::vector<std::optional<std::vector<Offset>>> offsets,
    std::vector<std::optional<Shape>> orig_shapes,
    std::vector<std::optional<Shape>> sliced_shapes,
    std::vector<std::optional<PrimitiveType>> offset_primitive_types,
    std::optional<OffsetAsFunctionOfIndvarModulesMetadata>
        offset_as_function_of_indvar_metadata)
    : Thunk(Kind::kDynamicSlice, thunk_info),
      embedded_thunk_(std::make_unique<SequentialThunk>(
          ThunkInfo(), std::move(*embedded_thunk))),
      arguments_(arguments),
      fake_allocations_(std::move(fake_allocations)),
      offsets_(offsets),
      orig_shapes_(orig_shapes),
      sliced_shapes_(sliced_shapes),
      offset_primitive_types_(offset_primitive_types),
      offset_as_function_of_indvar_metadata_(
          std::move(offset_as_function_of_indvar_metadata)) {
  // Zip all arguments together to create a list of SliceDef.
  for (auto [arg, offsets, orig_shape, sliced_shape, offset_primitive_type] :
       llvm::zip_equal(arguments, offsets, orig_shapes, sliced_shapes,
                       offset_primitive_types)) {
    slices_.push_back(SliceDef{
        std::move(arg),
        std::move(offsets),
        std::move(orig_shape),
        std::move(sliced_shape),
        std::move(offset_primitive_type),
    });
  }

  // Find how many offsets we might have to transfer from device to host and
  // pre-compute host allocation requirements.
  for (SliceDef& slice : slices_) {
    offsets_allocs_base_.push_back(offsets_allocs_size_);
    if (slice.sliced_shape.has_value()) {
      offsets_allocs_size_ +=
          slice.sliced_shape->dimensions().size() * sizeof(int64_t);
    }
  }
}

absl::Status DynamicSliceThunk::Prepare(const PrepareParams& params) {
  for (SliceDef& slice : slices_) {
    VLOG(2) << "DynamicSliceThunk: slice: " << slice.ToString();
    if (slice.offsets.has_value()) {
      TF_RET_CHECK(slice.embedded_thunk_argument.has_value());
      TF_RET_CHECK(slice.orig_shape.has_value());
      TF_RET_CHECK(slice.sliced_shape.has_value());
      TF_RET_CHECK(slice.offset_primitive_type.has_value());

      TF_RET_CHECK(slice.orig_shape->IsArray());
      TF_RET_CHECK(slice.sliced_shape->IsArray());

      TF_RET_CHECK(slice.offsets->size() ==
                   slice.orig_shape->dimensions().size());
      TF_RET_CHECK(slice.sliced_shape->dimensions().size() ==
                   slice.orig_shape->dimensions().size());
    }
  }

  TF_RETURN_IF_ERROR(embedded_thunk_->Prepare(params));

  if (offset_as_function_of_indvar_metadata_ != std::nullopt) {
    Indvar(this) =
        HloEvaluator()
            .Evaluate(
                /*module=*/*offset_as_function_of_indvar_metadata_->indvar_init,
                /*args=*/{})
            .value();
    VLOG(2) << "Indvar init module: "
            << offset_as_function_of_indvar_metadata_->indvar_init->ToString();
    VLOG(2)
        << "Indvar update module: "
        << offset_as_function_of_indvar_metadata_->indvar_update->ToString();
    VLOG(2) << "Indvar initialized to " << Indvar(this).ToString();
  }
  return absl::OkStatus();
}

absl::Status DynamicSliceThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(embedded_thunk_->Initialize(params));

  absl::MutexLock lock(mutex_);
  if (offsets_allocs_.contains(params.executor)) {
    return absl::OkStatus();
  }

  VLOG(2) << "Allocate " << offsets_allocs_size_
          << " bytes for transferring offsets on executor: " << params.executor;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<se::MemoryAllocation> allocation,
      params.executor->HostMemoryAllocate(offsets_allocs_size_));
  offsets_allocs_.emplace(params.executor, std::move(allocation));

  return absl::OkStatus();
}

absl::Status DynamicSliceThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream& stream = *params.stream;
  const BufferAllocations& orig_allocations = *params.buffer_allocations;

  absl::InlinedVector<se::DeviceAddressBase, 8> slice_buffers(
      slices_.size(), se::DeviceAddressBase());

  // Get memory allocation for copying offsets from device.
  int64_t* offsets_alloc = [&] {
    absl::MutexLock lock(mutex_);
    return reinterpret_cast<int64_t*>(
        offsets_allocs_.at(stream.parent())->address().opaque());
  }();

  auto offset_value = [&](int64_t arg_idx, int64_t offset_idx) -> int64_t& {
    return offsets_alloc[offsets_allocs_base_.at(arg_idx) + offset_idx];
  };

  VLOG(2) << "Execute dynamic slice thunk: slices=" << slices_.size();
  for (auto [argument_idx, slice] : llvm::enumerate(slices_)) {
    // Skip arguments that do not have buffer slices (tokens).
    if (!slice.embedded_thunk_argument.has_value()) {
      continue;
    }

    // `argument_buffer` will contain the original offset for slice
    // `argument_slice` within `orig_allocations`
    se::DeviceAddressBase argument_buffer =
        orig_allocations.GetDeviceAddress(*slice.embedded_thunk_argument);

    // If argument is not sliced, just use the original buffer.
    if (!slice.offsets.has_value()) {
      slice_buffers[argument_idx] = argument_buffer;
      continue;
    }

    const Shape& src_shape = *slice.orig_shape;
    const Shape& dst_shape = *slice.sliced_shape;

    absl::InlinedVector<int64_t, 4> slice_starts;
    slice_starts.reserve(dst_shape.dimensions().size());

    // Number of issues d2h transfers to copy offset values from device to
    // host.
    int64_t num_transfers = 0;

    // Get offset for `argument_idx`-th argument, which has
    // `dst_shape.dimensions_size()` components.
    for (auto [offset_idx, values] : llvm::enumerate(llvm::zip(
             *slice.offsets, src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [offset, src_dim, dst_dim] = values;

      if (int64_t* const_offset = std::get_if<int64_t>(&offset)) {
        // Forward slice offsets that are known constant values
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: constant offset = " << *const_offset;
        offset_value(argument_idx, offset_idx) = *const_offset;

      } else if (HloModule** offset_module = std::get_if<HloModule*>(&offset)) {
        TF_ASSIGN_OR_RETURN(
            Literal offset,
            HloEvaluator().Evaluate(**offset_module, {&Indvar(this)}));
        auto offset_int = LiteralUtil::LiteralAsScalarInt64(offset);
        if (offset_int.has_value()) {
          offset_value(argument_idx, offset_idx) = *offset_int;
        } else {
          return absl::InternalError(
              absl::StrFormat("Unhandled type returned from offset module: %s",
                              offset.shape().ToString()));
        }
        VLOG(2) << "Offset value = " << offset_value(argument_idx, offset_idx);
      } else {
        // Transfer slice offset value from device to host.
        auto alloc_slice = std::get<BufferAllocation::Slice>(offset);
        VLOG(2) << "  - arg " << argument_idx << "[" << offset_idx
                << "]: transfer offset from device " << alloc_slice.ToString();

        se::DeviceAddressBase offset_src =
            orig_allocations.GetDeviceAddress(alloc_slice);
        int64_t* offset_dst = &offset_value(argument_idx, offset_idx);

        // Copy the `offset_idx`-th component of the offset for the
        // `argument_idx`-th argument from device to host.
        TF_RETURN_IF_ERROR(stream.Memcpy(
            offset_dst, offset_src,
            ShapeUtil::ByteSizeOfPrimitiveType(*slice.offset_primitive_type)));
        ++num_transfers;
      }
    }

    // Wait for the completion of all transfers.
    if (num_transfers > 0) {
      VLOG(2) << "Wait for completion of " << num_transfers << " transfer";
      TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
    }

    // Clamp start indices:
    // start_indices[i] = min(max(start_indices[i], 0),
    //                        operand.dimension_size[i] - size_indices[i])
    for (auto [offset_idx, values] : llvm::enumerate(
             llvm::zip(src_shape.dimensions(), dst_shape.dimensions()))) {
      auto [src_dim, dst_dim] = values;
      int64_t start_index =
          std::min(std::max(offset_value(argument_idx, offset_idx), int64_t{0}),
                   src_dim - dst_dim);
      VLOG(2) << "arg idx: " << argument_idx << " offset_idx " << offset_idx
              << " with offset_value " << offset_value(argument_idx, offset_idx)
              << " start_idx: " << start_index << " src_dim: " << src_dim
              << " dst_dim:" << dst_dim;
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

    VLOG(2) << "Create sliced argument " << argument_idx << " of shape "
            << slice.sliced_shape->ToString()
            << " by slicing argument of shape " << slice.orig_shape->ToString()
            << " at offset " << new_offset << " with " << new_size;
    slice_buffers[argument_idx] =
        argument_buffer.GetByteSlice(new_offset, new_size);
  }

  // Safe to create a local BufferAllocations here since buffers are only slices
  // of bigger ones allocated elsewhere.
  BufferAllocations slice_allocations(slice_buffers,
                                      orig_allocations.device_ordinal(),
                                      orig_allocations.memory_allocator());

  VLOG(2) << "DynamicSliceThunk: slice_allocations: "
          << slice_allocations.ToString();

  Thunk::ExecuteParams new_params =
      Thunk::ExecuteParams::CloneWithNewAllocations(params, slice_allocations);

  // Execute the underlying custom call thunk with the new buffers.
  TF_RETURN_IF_ERROR(embedded_thunk_->ExecuteOnStream(new_params));

  if (offset_as_function_of_indvar_metadata_ != std::nullopt) {
    Indvar(this) =
        HloEvaluator()
            .Evaluate(*offset_as_function_of_indvar_metadata_->indvar_update,
                      {&Indvar(this)})
            .value();
    VLOG(2) << "Update Indvar = " << Indvar(this).ToString();
  }

  return absl::OkStatus();
}

void DynamicSliceThunk::ForAllThunks(
    absl::FunctionRef<void(const Thunk*)> fn) const {
  fn(this);
  embedded_thunk_->ForAllThunks(fn);
}

void DynamicSliceThunk::ForAllThunksMutable(
    absl::FunctionRef<void(Thunk*)> fn) {
  fn(this);
  embedded_thunk_->ForAllThunksMutable(fn);
}

absl::Status DynamicSliceThunk::TransformAllNestedThunks(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<Thunk>>(std::unique_ptr<Thunk>)>
        fn) {
  TF_RETURN_IF_ERROR(embedded_thunk_->TransformAllNestedThunks(fn));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk,
                      fn(std::move(embedded_thunk_)));
  embedded_thunk_ = SequentialThunk::FromThunk(std::move(thunk));
  return absl::OkStatus();
}

Thunk::BufferUses DynamicSliceThunk::buffer_uses() const {
  Thunk::BufferUses res;
  res.reserve(slices_.size());
  for (const SliceDef& slice : slices_) {
    if (!slice.embedded_thunk_argument.has_value()) {
      continue;
    }
    res.push_back(
        BufferUse::Read(*slice.embedded_thunk_argument, *slice.orig_shape));

    if (!slice.offsets.has_value()) {
      continue;
    }
    for (const Offset& offset : *slice.offsets) {
      auto* alloc_slice = std::get_if<BufferAllocation::Slice>(&offset);
      if (!alloc_slice) {
        continue;
      }
      res.push_back(BufferUse::Read(
          *alloc_slice,
          ShapeUtil::MakeShape(*slice.offset_primitive_type, {})));
    }
  }
  return res;
}

absl::StatusOr<OptionalDynamicSliceOffsetsProto>
SerializeOptionalDynamicSliceOffsetsToProto(
    const std::optional<std::vector<DynamicSliceThunk::Offset>>& offsets_item,
    const std::optional<
        DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata>&
        offset_as_function_of_indvar_metadata) {
  OptionalDynamicSliceOffsetsProto offsets_proto;
  if (offsets_item.has_value()) {
    auto& offsets_inner = *offsets_proto.mutable_offsets();
    for (const auto& offset : *offsets_item) {
      auto& offset_proto = *offsets_inner.add_offsets();
      if (const int64_t* const_offset = std::get_if<int64_t>(&offset)) {
        offset_proto.set_const_offset(*const_offset);
      } else if (const BufferAllocation::Slice* slice_offset =
                     std::get_if<BufferAllocation::Slice>(&offset)) {
        TF_ASSIGN_OR_RETURN(*offset_proto.mutable_slice_offset(),
                            slice_offset->ToProto());
      } else if (const HloModule* const* module_offset =
                     std::get_if<HloModule*>(&offset)) {
        TF_RET_CHECK(offset_as_function_of_indvar_metadata.has_value());
        const std::vector<std::unique_ptr<HloModule>>& modules =
            offset_as_function_of_indvar_metadata->extracted_offset_modules;
        auto it = absl::c_find_if(modules, [&](const auto& module) {
          return module.get() == *module_offset;
        });
        TF_RET_CHECK(it != modules.end());
        offset_proto.set_hlo_module_offset_idx(it - modules.begin());
      } else {
        return absl::InternalError(
            "Unhandled offset type in DynamicSliceThunk::ToProto");
      }
    }
  }
  return offsets_proto;
}

absl::Status SerializeOffsetsToProto(
    const std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>&
        offsets,
    const std::optional<
        DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata>&
        offset_as_function_of_indvar_metadata,
    DynamicSliceThunkProto* proto) {
  for (const auto& offsets_item : offsets) {
    TF_ASSIGN_OR_RETURN(
        *proto->add_offsets(),
        SerializeOptionalDynamicSliceOffsetsToProto(
            offsets_item, offset_as_function_of_indvar_metadata));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<std::vector<DynamicSliceThunk::Offset>>>
DeserializeOptionalDynamicSliceOffsetsFromProto(
    const OptionalDynamicSliceOffsetsProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const std::optional<
        DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata>&
        offset_as_function_of_indvar_metadata) {
  if (!proto.has_offsets()) {
    return std::nullopt;
  }
  std::vector<DynamicSliceThunk::Offset> offsets;

  for (const auto& offset_proto : proto.offsets().offsets()) {
    switch (offset_proto.offset_case()) {
      case DynamicSliceOffsetProto::kConstOffset:
        offsets.push_back(offset_proto.const_offset());
        break;
      case DynamicSliceOffsetProto::kSliceOffset: {
        TF_ASSIGN_OR_RETURN(
            auto slice, BufferAllocation::Slice::FromProto(
                            offset_proto.slice_offset(), buffer_allocations));
        offsets.push_back(slice);
        break;
      }
      case DynamicSliceOffsetProto::kHloModuleOffsetIdx: {
        TF_RET_CHECK(offset_as_function_of_indvar_metadata.has_value());
        const std::vector<std::unique_ptr<HloModule>>& modules =
            offset_as_function_of_indvar_metadata->extracted_offset_modules;
        TF_RET_CHECK(modules.size() > offset_proto.hlo_module_offset_idx());
        offsets.push_back(modules[offset_proto.hlo_module_offset_idx()].get());
        break;
      }
      default:
        return absl::InternalError(
            "Offset not set in OptionalDynamicSliceOffsetsProto");
    }
  }
  return offsets;
}

absl::StatusOr<
    std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>>
DeserializeOffsetsFromProto(
    const DynamicSliceThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const std::optional<
        DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata>&
        offset_as_function_of_indvar_metadata) {
  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets;
  for (const auto& offsets_proto : proto.offsets()) {
    TF_ASSIGN_OR_RETURN(offsets.emplace_back(),
                        DeserializeOptionalDynamicSliceOffsetsFromProto(
                            offsets_proto, buffer_allocations,
                            offset_as_function_of_indvar_metadata));
  }
  return offsets;
}

absl::StatusOr<ThunkProto> DynamicSliceThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  DynamicSliceThunkProto* dynamic_slice_proto =
      proto.mutable_dynamic_slice_thunk();

  TF_ASSIGN_OR_RETURN(ThunkProto embedded_thunk_proto,
                      embedded_thunk_->ToProto());
  TF_RET_CHECK(embedded_thunk_proto.has_sequential_thunk());
  *dynamic_slice_proto->mutable_embedded_thunk() =
      std::move(embedded_thunk_proto.sequential_thunk());

  // arguments
  for (const auto& arg : arguments_) {
    auto& proto_arg = *dynamic_slice_proto->add_arguments();
    if (arg.has_value()) {
      TF_ASSIGN_OR_RETURN(*proto_arg.mutable_slice(), arg->ToProto());
    }
  }

  TF_RETURN_IF_ERROR(SerializeOffsetsToProto(
      offsets_, offset_as_function_of_indvar_metadata_, dynamic_slice_proto));

  // orig_shapes
  for (const auto& shape : orig_shapes_) {
    auto& proto_shape = *dynamic_slice_proto->add_orig_shapes();
    if (shape.has_value()) {
      *proto_shape.mutable_shape() = shape->ToProto();
    }
  }

  // sliced_shapes
  for (const auto& shape : sliced_shapes_) {
    auto& proto_shape = *dynamic_slice_proto->add_sliced_shapes();
    if (shape.has_value()) {
      *proto_shape.mutable_shape() = shape->ToProto();
    }
  }

  // offset_byte_sizes
  for (const std::optional<PrimitiveType>& primtive_type :
       offset_primitive_types_) {
    auto& proto_size = *dynamic_slice_proto->add_offset_primitive_types();
    if (primtive_type.has_value()) {
      proto_size.set_value(primtive_type.value());
    }
  }

  // offset_as_function_of_indvar_metadata
  if (offset_as_function_of_indvar_metadata_.has_value()) {
    TF_ASSIGN_OR_RETURN(
        *dynamic_slice_proto
             ->mutable_offset_as_function_of_indvar_modules_metadata(),
        offset_as_function_of_indvar_metadata_->ToProto());
  }

  // fake_allocations
  for (const auto& fake_allocation : fake_allocations_) {
    *dynamic_slice_proto->add_fake_allocations() = fake_allocation.ToProto();
  }

  return proto;
}

absl::StatusOr<std::unique_ptr<DynamicSliceThunk>> DynamicSliceThunk::FromProto(
    ThunkInfo thunk_info, const DynamicSliceThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const DeserializerWithCustomAllocations& deserializer) {
  // offset_as_function_of_indvar_metadata
  std::optional<OffsetAsFunctionOfIndvarModulesMetadata>
      offset_as_function_of_indvar_metadata;
  if (proto.has_offset_as_function_of_indvar_modules_metadata()) {
    TF_ASSIGN_OR_RETURN(
        offset_as_function_of_indvar_metadata,
        OffsetAsFunctionOfIndvarModulesMetadata::FromProto(
            proto.offset_as_function_of_indvar_modules_metadata()));
  }

  std::vector<std::optional<BufferAllocation::Slice>> arguments;
  for (auto& arg_proto : proto.arguments()) {
    arguments.push_back(std::nullopt);
    if (arg_proto.has_slice()) {
      TF_ASSIGN_OR_RETURN(arguments.back(),
                          BufferAllocation::Slice::FromProto(
                              arg_proto.slice(), buffer_allocations));
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::optional<std::vector<Offset>>> offsets,
      DeserializeOffsetsFromProto(proto, buffer_allocations,
                                  offset_as_function_of_indvar_metadata));

  std::vector<std::optional<Shape>> orig_shapes;
  for (auto& shape_proto : proto.orig_shapes()) {
    orig_shapes.push_back(std::nullopt);
    if (shape_proto.has_shape()) {
      TF_ASSIGN_OR_RETURN(orig_shapes.back(),
                          Shape::FromProto(shape_proto.shape()));
    }
  }

  std::vector<std::optional<Shape>> sliced_shapes;
  for (auto& shape_proto : proto.sliced_shapes()) {
    sliced_shapes.push_back(std::nullopt);
    if (shape_proto.has_shape()) {
      TF_ASSIGN_OR_RETURN(sliced_shapes.back(),
                          Shape::FromProto(shape_proto.shape()));
    }
  }

  std::vector<std::optional<PrimitiveType>> offset_primtitive_types;
  offset_primtitive_types.reserve(proto.offset_primitive_types_size());
  for (const OptionalPrimitiveType& type_proto :
       proto.offset_primitive_types()) {
    offset_primtitive_types.push_back(std::nullopt);
    if (type_proto.has_value()) {
      offset_primtitive_types.back() = type_proto.value();
    }
  }

  std::vector<BufferAllocation> fake_allocations;
  for (const auto& fake_allocation_proto : proto.fake_allocations()) {
    fake_allocations.push_back(
        BufferAllocation::FromProto(fake_allocation_proto));
  }

  std::vector<std::unique_ptr<Thunk>> embedded_thunks;
  for (const auto& thunk_proto : proto.embedded_thunk().thunks()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> embedded_thunk,
                        deserializer(thunk_proto, fake_allocations));
    embedded_thunks.push_back(std::move(embedded_thunk));
  }

  return std::make_unique<DynamicSliceThunk>(
      thunk_info, std::make_unique<ThunkSequence>(std::move(embedded_thunks)),
      std::move(arguments), std::move(fake_allocations), std::move(offsets),
      std::move(orig_shapes), std::move(sliced_shapes),
      std::move(offset_primtitive_types),
      std::move(offset_as_function_of_indvar_metadata));
}

}  // namespace gpu
}  // namespace xla
