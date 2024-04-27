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

#include "xla/service/gpu/runtime/cub_sort_thunk.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/cub_sort_kernel.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// Template class for sorting a single tensor.
class CubSortKeysImpl : public CubSortRunnerInterface {
 public:
  using SortKeysFn = std::function<const char*(void*, size_t&, const void*,
                                               void*, size_t, bool)>;

  explicit CubSortKeysImpl(SortKeysFn sort_keys_fn, PrimitiveType type)
      : sort_keys_fn_(sort_keys_fn), type_(type) {}

  absl::Status Run(se::DeviceMemoryBase input_keys,
                   se::DeviceMemoryBase input_values,
                   se::DeviceMemoryBase output_keys,
                   se::DeviceMemoryBase output_values,
                   se::DeviceMemoryBase scratch, bool descending) override;
  absl::Status Run(const Thunk::ExecuteParams& params,
                   const CubSortThunk* thunk) override;
  absl::StatusOr<int64_t> GetScratchSize(int64_t num_items) override;

 private:
  SortKeysFn sort_keys_fn_;
  PrimitiveType type_;
};

absl::Status CubSortKeysImpl::Run(se::DeviceMemoryBase input_keys,
                                  se::DeviceMemoryBase input_values,
                                  se::DeviceMemoryBase output_keys,
                                  se::DeviceMemoryBase output_values,
                                  se::DeviceMemoryBase scratch,
                                  bool descending) {
  size_t temp_bytes = scratch.size();
  size_t num_items = input_keys.size() * 8 / primitive_util::BitWidth(type_);
  CHECK(input_values.is_null());
  CHECK(output_values.is_null());
  const char* error =
      sort_keys_fn_(scratch.opaque(), temp_bytes, input_keys.opaque(),
                    output_keys.opaque(), num_items, descending);
  if (error != nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("CubSortKeys error: ", error));
  }
  return absl::OkStatus();
}

absl::Status CubSortKeysImpl::Run(const Thunk::ExecuteParams& params,
                                  const CubSortThunk* thunk) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  return Run(allocs.GetDeviceAddress(thunk->operand(0)), se::DeviceMemoryBase(),
             allocs.GetDeviceAddress(thunk->result(0)), se::DeviceMemoryBase(),
             allocs.GetDeviceAddress(thunk->scratch()), thunk->descending());
}

absl::StatusOr<int64_t> CubSortKeysImpl::GetScratchSize(int64_t num_items) {
  size_t temp_bytes = 0;
  const char* error =
      sort_keys_fn_(nullptr, temp_bytes, nullptr, nullptr, num_items, false);
  if (error != nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("CubSortKeys error: ", error));
  }
  return temp_bytes;
}

// Template class for sorting a pair of tensors.
class CubSortPairsImpl : public CubSortRunnerInterface {
 public:
  using SortPairsFn = std::function<const char*(
      void*, size_t&, const void*, void*, const void*, void*, size_t, bool)>;

  explicit CubSortPairsImpl(SortPairsFn sort_pairs_fn, PrimitiveType type)
      : sort_pairs_fn_(sort_pairs_fn), type_(type) {}

  absl::Status Run(se::DeviceMemoryBase input_keys,
                   se::DeviceMemoryBase input_values,
                   se::DeviceMemoryBase output_keys,
                   se::DeviceMemoryBase output_values,
                   se::DeviceMemoryBase scratch, bool descending) override;
  absl::Status Run(const Thunk::ExecuteParams& params,
                   const CubSortThunk* thunk) override;
  absl::StatusOr<int64_t> GetScratchSize(int64_t num_items) override;

 private:
  SortPairsFn sort_pairs_fn_;
  PrimitiveType type_;
};

absl::Status CubSortPairsImpl::Run(se::DeviceMemoryBase input_keys,
                                   se::DeviceMemoryBase input_values,
                                   se::DeviceMemoryBase output_keys,
                                   se::DeviceMemoryBase output_values,
                                   se::DeviceMemoryBase scratch,
                                   bool descending) {
  size_t temp_bytes = scratch.size();
  size_t num_items = input_keys.size() * 8 / primitive_util::BitWidth(type_);
  const char* error = sort_pairs_fn_(
      scratch.opaque(), temp_bytes, input_keys.opaque(), output_keys.opaque(),
      input_values.opaque(), output_values.opaque(), num_items, descending);
  if (error != nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("CubSortPairs error: ", error));
  }
  return absl::OkStatus();
}

absl::Status CubSortPairsImpl::Run(const Thunk::ExecuteParams& params,
                                   const CubSortThunk* thunk) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  return Run(allocs.GetDeviceAddress(thunk->operand(0)),
             allocs.GetDeviceAddress(thunk->operand(1)),
             allocs.GetDeviceAddress(thunk->result(0)),
             allocs.GetDeviceAddress(thunk->result(1)),
             allocs.GetDeviceAddress(thunk->scratch()), thunk->descending());
}

absl::StatusOr<int64_t> CubSortPairsImpl::GetScratchSize(int64_t num_items) {
  size_t temp_bytes = 0;
  const char* error = sort_pairs_fn_(nullptr, temp_bytes, nullptr, nullptr,
                                     nullptr, nullptr, num_items, false);
  if (error != nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("CubSortPairs error: ", error));
  }
  return temp_bytes;
}

absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> CreateCubSortRunner(
    PrimitiveType type) {
  switch (type) {
    case F16:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_f16, F16);
    case F32:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_f32, F32);
    case F64:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_f64, F64);
    case S8:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_s8, S8);
    case S16:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_s16, S16);
    case S32:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_s32, S32);
    case S64:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_s64, S64);
    case U8:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_u8, U8);
    case U16:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_u16, U16);
    case U32:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_u32, U32);
    case U64:
      return std::make_unique<CubSortKeysImpl>(CubSortKeys_u64, U64);
    default:
      return InvalidArgument("Unsupported type of the sort kernel: %s",
                             primitive_util::LowercasePrimitiveTypeName(type));
  }
}

absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> CreateCubSortRunner(
    PrimitiveType key_type, PrimitiveType value_type) {
  // Values can be of any type of 16/32/64 bit width.
  int valueWidth = primitive_util::BitWidth(value_type);
  if (valueWidth != 16 && valueWidth != 32 && valueWidth != 64) {
    return InvalidArgument(
        "Unsupported value type of the sort kernel: %s",
        primitive_util::LowercasePrimitiveTypeName(value_type));
  }

  // Only unsigned integer types could be used for keys.
  switch (key_type) {
    case U16:
      if (valueWidth == 16) {
        return std::make_unique<CubSortPairsImpl>(CubSortPairs_u16_b16, U16);
      }
      if (valueWidth == 32) {
        return std::make_unique<CubSortPairsImpl>(CubSortPairs_u16_b32, U16);
      }
      return std::make_unique<CubSortPairsImpl>(CubSortPairs_u16_b64, U16);
    case U32:
      if (valueWidth == 16) {
        return std::make_unique<CubSortPairsImpl>(CubSortPairs_u32_b16, U32);
      }
      if (valueWidth == 32) {
        return std::make_unique<CubSortPairsImpl>(CubSortPairs_u32_b32, U32);
      }
      return std::make_unique<CubSortPairsImpl>(CubSortPairs_u32_b64, U32);
    case U64:
      if (valueWidth == 16) {
        return std::make_unique<CubSortPairsImpl>(CubSortPairs_u64_b16, U64);
      }
      if (valueWidth == 32) {
        return std::make_unique<CubSortPairsImpl>(CubSortPairs_u64_b32, U64);
      }
      return std::make_unique<CubSortPairsImpl>(CubSortPairs_u64_b64, U64);
    default:
      return InvalidArgument(
          "Unsupported key type of the sort kernel: %s",
          primitive_util::LowercasePrimitiveTypeName(key_type));
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>>
CubSortRunnerInterface::Create(PrimitiveType type,
                               std::optional<PrimitiveType> value_type) {
  return value_type.has_value() ? CreateCubSortRunner(type, *value_type)
                                : CreateCubSortRunner(type);
}

CubSortThunk::CubSortThunk(
    ThunkInfo thunk_info, PrimitiveType type,
    std::optional<PrimitiveType> value_type,
    absl::InlinedVector<BufferAllocation::Slice, 2> operands,
    absl::InlinedVector<BufferAllocation::Slice, 2> results,
    BufferAllocation::Slice scratch, bool descending)
    : Thunk(Thunk::kCubSort, thunk_info),
      runner_(CubSortRunnerInterface::Create(type, value_type).value()),
      operands_(std::move(operands)),
      results_(std::move(results)),
      scratch_(scratch),
      descending_(descending) {}

absl::Status RunCubSort(PrimitiveType type,
                        std::optional<PrimitiveType> value_type,
                        se::DeviceMemoryBase input_keys,
                        se::DeviceMemoryBase input_values,
                        se::DeviceMemoryBase output_keys,
                        se::DeviceMemoryBase output_values,
                        se::DeviceMemoryBase scratch, bool descending) {
  auto runner = CubSortRunnerInterface::Create(type, value_type).value();
  return runner->Run(input_keys, input_values, output_keys, output_values,
                     scratch, descending);
}

}  // namespace gpu
}  // namespace xla
