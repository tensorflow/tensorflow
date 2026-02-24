/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/cub_sort_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
using buffer_assignment::BufferAllocationSliceProto;

namespace {

// N pairs of [start_offset, end_offset) require (N+1) storage.
// The size of each offset is selected to be 32-bits (int type).
uint64_t GetOffsetsSize(int64_t batch_size) {
  return (batch_size + 1) * sizeof(int);
}

// Copies segment offsets to the device memory.
absl::Status CopyOffsets(se::Stream* stream, se::DeviceAddressBase scratch,
                         int64_t batch_size, int64_t segment_size) {
  uint64_t offsets_size = GetOffsetsSize(batch_size);
  char* offsets_buffer =
      static_cast<char*>(scratch.opaque()) + scratch.size() - offsets_size;
  se::DeviceAddressBase d_offsets(offsets_buffer, offsets_size);
  std::vector<int> h_offsets(batch_size + 1);
  for (int i = 0; i <= batch_size; ++i) {
    h_offsets[i] = i * segment_size;
  }
  return stream->Memcpy(&d_offsets, h_offsets.data(), offsets_size);
}

// Template class for sorting a single tensor.
class CubSortKeysImpl : public CubSortRunnerInterface {
 public:
  explicit CubSortKeysImpl(ffi::HandlerRegistration sort_keys_fn,
                           PrimitiveType type)
      : sort_keys_fn_(sort_keys_fn), type_(type) {}

  absl::Status Run(se::DeviceAddressBase input_keys,
                   se::DeviceAddressBase input_values,
                   se::DeviceAddressBase output_keys,
                   se::DeviceAddressBase output_values,
                   se::DeviceAddressBase scratch, bool descending,
                   int64_t batch_size, se::Stream* stream) override;
  absl::Status Run(const Thunk::ExecuteParams& params,
                   const CubSortThunk* thunk) override;
  absl::StatusOr<int64_t> GetScratchSize(int64_t num_items,
                                         int64_t batch_size) override;

 private:
  ffi::HandlerRegistration sort_keys_fn_;
  PrimitiveType type_;
};

absl::Status CubSortKeysImpl::Run(se::DeviceAddressBase input_keys,
                                  se::DeviceAddressBase input_values,
                                  se::DeviceAddressBase output_keys,
                                  se::DeviceAddressBase output_values,
                                  se::DeviceAddressBase scratch,
                                  bool descending, int64_t batch_size,
                                  se::Stream* stream) {
  size_t temp_bytes = scratch.size();
  size_t num_items = input_keys.size() * 8 / primitive_util::BitWidth(type_);
  CHECK(input_values.is_null());
  CHECK(output_values.is_null());
  if (batch_size > 1) {
    TF_RETURN_IF_ERROR(
        CopyOffsets(stream, scratch, batch_size, num_items / batch_size));
    temp_bytes -= GetOffsetsSize(batch_size);
  }

  ffi::CallFrameBuilder builder(2, 1);
  builder.AddBufferArg(scratch, PrimitiveType::U8,
                       {static_cast<int64_t>(temp_bytes)});
  builder.AddBufferArg(input_keys, PrimitiveType::U8,
                       {static_cast<int64_t>(input_keys.size())});
  builder.AddBufferRet(output_keys, PrimitiveType::U8,
                       {static_cast<int64_t>(output_keys.size())});

  ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("num_items", static_cast<size_t>(num_items));
  attrs.Insert("descending", descending);
  attrs.Insert("batch_size", static_cast<size_t>(batch_size));
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  ffi::InvokeContext context{};
  context.backend_context = ffi::InvokeContext::GpuContext{stream, nullptr};
  return ffi::Invoke(ffi::GetXlaFfiApi(), sort_keys_fn_.bundle.execute,
                     call_frame, context, XLA_FFI_ExecutionStage_EXECUTE);
}

absl::Status CubSortKeysImpl::Run(const Thunk::ExecuteParams& params,
                                  const CubSortThunk* thunk) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  return Run(allocs.GetDeviceAddress(thunk->operand(0)),
             se::DeviceAddressBase(), allocs.GetDeviceAddress(thunk->result(0)),
             se::DeviceAddressBase(), allocs.GetDeviceAddress(thunk->scratch()),
             thunk->descending(), thunk->batch_size(), params.stream);
}

absl::StatusOr<int64_t> CubSortKeysImpl::GetScratchSize(int64_t num_items,
                                                        int64_t batch_size) {
  ffi::CallFrameBuilder builder(0, 0);

  ffi::CallFrameBuilder::AttributesBuilder attrs;
  size_t temp_bytes = 0;
  attrs.Insert("temp_bytes", absl::bit_cast<int64_t>(&temp_bytes));
  attrs.Insert("num_items", static_cast<size_t>(num_items));
  attrs.Insert("batch_size", static_cast<size_t>(batch_size));
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  TF_RETURN_IF_ERROR(ffi::Invoke(
      ffi::GetXlaFfiApi(), sort_keys_fn_.bundle.initialize, call_frame,
      ffi::InvokeContext{}, XLA_FFI_ExecutionStage_INITIALIZE));
  return temp_bytes;
}

// Template class for sorting a pair of tensors.
class CubSortPairsImpl : public CubSortRunnerInterface {
 public:
  explicit CubSortPairsImpl(ffi::HandlerRegistration sort_pairs_fn,
                            PrimitiveType type)
      : sort_pairs_fn_(sort_pairs_fn), type_(type) {}

  absl::Status Run(se::DeviceAddressBase input_keys,
                   se::DeviceAddressBase input_values,
                   se::DeviceAddressBase output_keys,
                   se::DeviceAddressBase output_values,
                   se::DeviceAddressBase scratch, bool descending,
                   int64_t batch_size, se::Stream* stream) override;
  absl::Status Run(const Thunk::ExecuteParams& params,
                   const CubSortThunk* thunk) override;
  absl::StatusOr<int64_t> GetScratchSize(int64_t num_items,
                                         int64_t batch_size) override;

 private:
  ffi::HandlerRegistration sort_pairs_fn_;
  PrimitiveType type_;
};

absl::Status CubSortPairsImpl::Run(se::DeviceAddressBase input_keys,
                                   se::DeviceAddressBase input_values,
                                   se::DeviceAddressBase output_keys,
                                   se::DeviceAddressBase output_values,
                                   se::DeviceAddressBase scratch,
                                   bool descending, int64_t batch_size,
                                   se::Stream* stream) {
  size_t temp_bytes = scratch.size();
  size_t num_items = input_keys.size() * 8 / primitive_util::BitWidth(type_);
  if (batch_size > 1) {
    TF_RETURN_IF_ERROR(
        CopyOffsets(stream, scratch, batch_size, num_items / batch_size));
    temp_bytes -= GetOffsetsSize(batch_size);
  }

  ffi::CallFrameBuilder builder(3, 2);
  builder.AddBufferArg(scratch, PrimitiveType::U8,
                       {static_cast<int64_t>(temp_bytes)});
  builder.AddBufferArg(input_keys, PrimitiveType::U8,
                       {static_cast<int64_t>(input_keys.size())});
  builder.AddBufferRet(output_keys, PrimitiveType::U8,
                       {static_cast<int64_t>(output_keys.size())});
  builder.AddBufferArg(input_values, PrimitiveType::U8,
                       {static_cast<int64_t>(input_values.size())});
  builder.AddBufferRet(output_values, PrimitiveType::U8,
                       {static_cast<int64_t>(output_values.size())});

  ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("num_items", static_cast<size_t>(num_items));
  attrs.Insert("descending", descending);
  attrs.Insert("batch_size", static_cast<size_t>(batch_size));
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  ffi::InvokeContext context{};
  context.backend_context = ffi::InvokeContext::GpuContext{stream, nullptr};
  return ffi::Invoke(ffi::GetXlaFfiApi(), sort_pairs_fn_.bundle.execute,
                     call_frame, context, XLA_FFI_ExecutionStage_EXECUTE);
}

absl::Status CubSortPairsImpl::Run(const Thunk::ExecuteParams& params,
                                   const CubSortThunk* thunk) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  return Run(allocs.GetDeviceAddress(thunk->operand(0)),
             allocs.GetDeviceAddress(thunk->operand(1)),
             allocs.GetDeviceAddress(thunk->result(0)),
             allocs.GetDeviceAddress(thunk->result(1)),
             allocs.GetDeviceAddress(thunk->scratch()), thunk->descending(),
             thunk->batch_size(), params.stream);
}

absl::StatusOr<int64_t> CubSortPairsImpl::GetScratchSize(int64_t num_items,
                                                         int64_t batch_size) {
  ffi::CallFrameBuilder::AttributesBuilder attrs;
  size_t temp_bytes = 0;
  // FFI expects a pointer to be passed as an int64_t.
  attrs.Insert("temp_bytes", absl::bit_cast<int64_t>(&temp_bytes));
  attrs.Insert("num_items", static_cast<size_t>(num_items));
  attrs.Insert("batch_size", static_cast<size_t>(batch_size));

  ffi::CallFrameBuilder builder(0, 0);
  builder.AddAttributes(attrs.Build());
  ffi::CallFrame call_frame = builder.Build();

  TF_RETURN_IF_ERROR(ffi::Invoke(
      ffi::GetXlaFfiApi(), sort_pairs_fn_.bundle.initialize, call_frame,
      ffi::InvokeContext{}, XLA_FFI_ExecutionStage_INITIALIZE));
  return temp_bytes;
}

absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> CreateCubSortRunner(
    PrimitiveType type, absl::string_view platform_name) {
  TF_ASSIGN_OR_RETURN(
      ffi::HandlerRegistration handler,
      ffi::FindHandler("xla.gpu.ext.cub_sort_keys_" +
                           primitive_util::LowercasePrimitiveTypeName(type),
                       platform_name));
  return std::make_unique<CubSortKeysImpl>(handler, type);
}

// Returns an interface for calling CubSortPairs on the given key and value
// types. key_type can be any unsigned integer types or F32. value_type can be
// any type of 16/32/64 bit width.
absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> CreateCubSortRunner(
    PrimitiveType key_type, PrimitiveType value_type,
    absl::string_view platform_name) {
  TF_ASSIGN_OR_RETURN(
      ffi::HandlerRegistration handler,
      ffi::FindHandler(
          absl::StrFormat("xla.gpu.ext.cub_sort_pairs_%s_b%d",
                          primitive_util::LowercasePrimitiveTypeName(key_type),
                          primitive_util::BitWidth(value_type)),
          platform_name));
  return std::make_unique<CubSortPairsImpl>(handler, key_type);
}

}  // namespace

absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>>
CubSortRunnerInterface::Create(PrimitiveType type,
                               std::optional<PrimitiveType> value_type,
                               absl::string_view platform_name) {
  return value_type.has_value()
             ? CreateCubSortRunner(type, *value_type, platform_name)
             : CreateCubSortRunner(type, platform_name);
}

absl::StatusOr<std::unique_ptr<CubSortThunk>> CubSortThunk::Create(
    ThunkInfo thunk_info, absl::InlinedVector<ShapedSlice, 2> operands,
    absl::InlinedVector<ShapedSlice, 2> results,
    BufferAllocation::Slice scratch, bool descending, int64_t batch_size,
    absl::string_view platform_name) {
  PrimitiveType type = operands[0].shape.element_type();
  std::optional<PrimitiveType> value_type =
      operands.size() == 2 ? std::optional(operands[1].shape.element_type())
                           : std::nullopt;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<CubSortRunnerInterface> runner,
      CubSortRunnerInterface::Create(type, value_type, platform_name));

  return absl::WrapUnique<CubSortThunk>(new CubSortThunk(
      thunk_info, std::move(runner), type, value_type, std::move(operands),
      std::move(results), scratch, descending, batch_size));
}

CubSortThunk::CubSortThunk(ThunkInfo thunk_info,
                           std::unique_ptr<CubSortRunnerInterface> runner,
                           PrimitiveType type,
                           std::optional<PrimitiveType> value_type,
                           absl::InlinedVector<ShapedSlice, 2> operands,
                           absl::InlinedVector<ShapedSlice, 2> results,
                           BufferAllocation::Slice scratch, bool descending,
                           int64_t batch_size)
    : Thunk(Thunk::kCubSort, thunk_info),
      runner_(std::move(runner)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      scratch_(scratch),
      type_(type),
      value_type_(value_type),
      descending_(descending),
      batch_size_(batch_size) {}

Thunk::BufferUses CubSortThunk::buffer_uses() const {
  Thunk::BufferUses res;
  res.reserve(operands_.size() + results_.size() + 1);
  for (const ShapedSlice& slice : operands_) {
    res.push_back(BufferUse::Read(slice.slice, slice.shape));
  }
  for (const ShapedSlice& slice : results_) {
    res.push_back(BufferUse::Write(slice.slice, slice.shape));
  }
  res.push_back(BufferUse::Scratch(
      scratch_, ShapeUtil::MakeShape(U8, {scratch_.size()})));
  return res;
}

absl::StatusOr<std::unique_ptr<CubSortThunk>> CubSortThunk::FromProto(
    ThunkInfo thunk_info, const CubSortThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    absl::string_view platform_name) {
  absl::InlinedVector<ShapedSlice, 2> operands;
  for (const ShapedSliceProto& slice_proto : proto.operands()) {
    TF_ASSIGN_OR_RETURN(
        operands.emplace_back(),
        ShapedSlice::FromProto(slice_proto, buffer_allocations));
  }

  absl::InlinedVector<ShapedSlice, 2> results;
  for (const ShapedSliceProto& slice_proto : proto.results()) {
    TF_ASSIGN_OR_RETURN(
        results.emplace_back(),
        ShapedSlice::FromProto(slice_proto, buffer_allocations));
  }

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice scratch,
      BufferAllocation::Slice::FromProto(proto.scratch(), buffer_allocations));

  return Create(thunk_info, operands, results, scratch, proto.descending(),
                proto.batch_size(), platform_name);
}

absl::StatusOr<ThunkProto> CubSortThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  CubSortThunkProto* cub_sort_proto = proto.mutable_cub_sort_thunk();

  for (const ShapedSlice& slice : operands_) {
    TF_ASSIGN_OR_RETURN(*cub_sort_proto->add_operands(), slice.ToProto());
  }
  for (const ShapedSlice& slice : results_) {
    TF_ASSIGN_OR_RETURN(*cub_sort_proto->add_results(), slice.ToProto());
  }
  TF_ASSIGN_OR_RETURN(*cub_sort_proto->mutable_scratch(), scratch_.ToProto());
  cub_sort_proto->set_descending(descending_);
  cub_sort_proto->set_batch_size(batch_size_);

  return proto;
}

}  // namespace gpu
}  // namespace xla
