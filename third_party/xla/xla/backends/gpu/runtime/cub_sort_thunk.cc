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
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// N pairs of [start_offset, end_offset) require (N+1) storage.
// The size of each offset is selected to be 32-bits (int type).
uint64_t GetOffsetsSize(int64_t batch_size) {
  return (batch_size + 1) * sizeof(int);
}

// Copies segment offsets to the device memory.
absl::Status CopyOffsets(se::Stream* stream, se::DeviceMemoryBase scratch,
                         int64_t batch_size, int64_t segment_size) {
  uint64_t offsets_size = GetOffsetsSize(batch_size);
  char* offsets_buffer =
      static_cast<char*>(scratch.opaque()) + scratch.size() - offsets_size;
  se::DeviceMemoryBase d_offsets(offsets_buffer, offsets_size);
  std::vector<int> h_offsets(batch_size + 1);
  for (int i = 0; i <= batch_size; ++i) h_offsets[i] = i * segment_size;
  return stream->Memcpy(&d_offsets, h_offsets.data(), offsets_size);
}

// Template class for sorting a single tensor.
class CubSortKeysImpl : public CubSortRunnerInterface {
 public:
  explicit CubSortKeysImpl(ffi::HandlerRegistration sort_keys_fn,
                           PrimitiveType type)
      : sort_keys_fn_(sort_keys_fn), type_(type) {}

  absl::Status Run(se::DeviceMemoryBase input_keys,
                   se::DeviceMemoryBase input_values,
                   se::DeviceMemoryBase output_keys,
                   se::DeviceMemoryBase output_values,
                   se::DeviceMemoryBase scratch, bool descending,
                   int64_t batch_size, se::Stream* stream) override;
  absl::Status Run(const Thunk::ExecuteParams& params,
                   const CubSortThunk* thunk) override;
  absl::StatusOr<int64_t> GetScratchSize(int64_t num_items,
                                         int64_t batch_size) override;

 private:
  ffi::HandlerRegistration sort_keys_fn_;
  PrimitiveType type_;
};

absl::Status CubSortKeysImpl::Run(se::DeviceMemoryBase input_keys,
                                  se::DeviceMemoryBase input_values,
                                  se::DeviceMemoryBase output_keys,
                                  se::DeviceMemoryBase output_values,
                                  se::DeviceMemoryBase scratch, bool descending,
                                  int64_t batch_size, se::Stream* stream) {
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

  ffi::CallOptions options{};
  options.backend_options = ffi::CallOptions::GpuOptions{stream, nullptr};
  return ffi::Call(sort_keys_fn_.bundle.execute, call_frame, options,
                   XLA_FFI_ExecutionStage_EXECUTE);
}

absl::Status CubSortKeysImpl::Run(const Thunk::ExecuteParams& params,
                                  const CubSortThunk* thunk) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  return Run(allocs.GetDeviceAddress(thunk->operand(0)), se::DeviceMemoryBase(),
             allocs.GetDeviceAddress(thunk->result(0)), se::DeviceMemoryBase(),
             allocs.GetDeviceAddress(thunk->scratch()), thunk->descending(),
             thunk->batch_size(), params.stream);
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

  TF_RETURN_IF_ERROR(ffi::Call(sort_keys_fn_.bundle.initialize, call_frame,
                               ffi::CallOptions{},
                               XLA_FFI_ExecutionStage_INITIALIZE));
  return temp_bytes;
}

// Template class for sorting a pair of tensors.
class CubSortPairsImpl : public CubSortRunnerInterface {
 public:
  explicit CubSortPairsImpl(ffi::HandlerRegistration sort_pairs_fn,
                            PrimitiveType type)
      : sort_pairs_fn_(sort_pairs_fn), type_(type) {}

  absl::Status Run(se::DeviceMemoryBase input_keys,
                   se::DeviceMemoryBase input_values,
                   se::DeviceMemoryBase output_keys,
                   se::DeviceMemoryBase output_values,
                   se::DeviceMemoryBase scratch, bool descending,
                   int64_t batch_size, se::Stream* stream) override;
  absl::Status Run(const Thunk::ExecuteParams& params,
                   const CubSortThunk* thunk) override;
  absl::StatusOr<int64_t> GetScratchSize(int64_t num_items,
                                         int64_t batch_size) override;

 private:
  ffi::HandlerRegistration sort_pairs_fn_;
  PrimitiveType type_;
};

absl::Status CubSortPairsImpl::Run(se::DeviceMemoryBase input_keys,
                                   se::DeviceMemoryBase input_values,
                                   se::DeviceMemoryBase output_keys,
                                   se::DeviceMemoryBase output_values,
                                   se::DeviceMemoryBase scratch,
                                   bool descending, int64_t batch_size,
                                   se::Stream* stream) {
  size_t temp_bytes = scratch.size();
  size_t num_items = input_keys.size() * 8 / primitive_util::BitWidth(type_);
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

  ffi::CallOptions options{};
  options.backend_options = ffi::CallOptions::GpuOptions{stream, nullptr};
  return ffi::Call(sort_pairs_fn_.bundle.execute, call_frame, options,
                   XLA_FFI_ExecutionStage_EXECUTE);
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

  TF_RETURN_IF_ERROR(ffi::Call(sort_pairs_fn_.bundle.initialize, call_frame,
                               ffi::CallOptions{},
                               XLA_FFI_ExecutionStage_INITIALIZE));
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

CubSortThunk::CubSortThunk(
    ThunkInfo thunk_info, PrimitiveType type,
    std::optional<PrimitiveType> value_type,
    absl::InlinedVector<BufferAllocation::Slice, 2> operands,
    absl::InlinedVector<BufferAllocation::Slice, 2> results,
    BufferAllocation::Slice scratch, bool descending, int64_t batch_size,
    absl::string_view platform_name)
    : Thunk(Thunk::kCubSort, thunk_info),
      runner_(CubSortRunnerInterface::Create(type, value_type, platform_name)
                  .value()),
      operands_(std::move(operands)),
      results_(std::move(results)),
      scratch_(scratch),
      descending_(descending),
      batch_size_(batch_size) {}

}  // namespace gpu
}  // namespace xla
