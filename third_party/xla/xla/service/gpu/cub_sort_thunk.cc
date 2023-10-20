/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/cub_sort_thunk.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/cub_sort_kernel.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

// Template class for sorting a single tensor.
class CubSortKeysImpl : public CubSortRunnerInterface {
 public:
  using SortKeysFn =
      std::function<Status(void*, size_t&, const void*, void*, size_t, bool)>;

  explicit CubSortKeysImpl(SortKeysFn sort_keys_fn, PrimitiveType type)
      : sort_keys_fn_(sort_keys_fn), type_(type) {}

  Status Run(se::DeviceMemoryBase input_keys, se::DeviceMemoryBase input_values,
             se::DeviceMemoryBase output_keys,
             se::DeviceMemoryBase output_values, se::DeviceMemoryBase scratch,
             bool descending) override;
  Status Run(const Thunk::ExecuteParams& params,
             const CubSortThunk* thunk) override;

 private:
  SortKeysFn sort_keys_fn_;
  PrimitiveType type_;
};

Status CubSortKeysImpl::Run(se::DeviceMemoryBase input_keys,
                            se::DeviceMemoryBase input_values,
                            se::DeviceMemoryBase output_keys,
                            se::DeviceMemoryBase output_values,
                            se::DeviceMemoryBase scratch, bool descending) {
  size_t temp_bytes = scratch.size();
  size_t num_items = input_keys.size() * 8 / primitive_util::BitWidth(type_);
  CHECK(input_values.is_null());
  CHECK(output_values.is_null());
  return sort_keys_fn_(scratch.opaque(), temp_bytes, input_keys.opaque(),
                       output_keys.opaque(), num_items, descending);
}

Status CubSortKeysImpl::Run(const Thunk::ExecuteParams& params,
                            const CubSortThunk* thunk) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  return Run(allocs.GetDeviceAddress(thunk->operand(0)), se::DeviceMemoryBase(),
             allocs.GetDeviceAddress(thunk->result(0)), se::DeviceMemoryBase(),
             allocs.GetDeviceAddress(thunk->scratch()), thunk->descending());
}

// Template class for sorting a pair of tensors.
class CubSortPairsImpl : public CubSortRunnerInterface {
 public:
  using SortPairsFn = std::function<Status(void*, size_t&, const void*, void*,
                                           const void*, void*, size_t, bool)>;

  explicit CubSortPairsImpl(SortPairsFn sort_pairs_fn, PrimitiveType type)
      : sort_pairs_fn_(sort_pairs_fn), type_(type) {}

  Status Run(se::DeviceMemoryBase input_keys, se::DeviceMemoryBase input_values,
             se::DeviceMemoryBase output_keys,
             se::DeviceMemoryBase output_values, se::DeviceMemoryBase scratch,
             bool descending) override;
  Status Run(const Thunk::ExecuteParams& params,
             const CubSortThunk* thunk) override;

 private:
  SortPairsFn sort_pairs_fn_;
  PrimitiveType type_;
};

Status CubSortPairsImpl::Run(se::DeviceMemoryBase input_keys,
                             se::DeviceMemoryBase input_values,
                             se::DeviceMemoryBase output_keys,
                             se::DeviceMemoryBase output_values,
                             se::DeviceMemoryBase scratch, bool descending) {
  size_t temp_bytes = scratch.size();
  size_t num_items = input_keys.size() * 8 / primitive_util::BitWidth(type_);
  return sort_pairs_fn_(scratch.opaque(), temp_bytes, input_keys.opaque(),
                        output_keys.opaque(), input_values.opaque(),
                        output_values.opaque(), num_items, descending);
}

Status CubSortPairsImpl::Run(const Thunk::ExecuteParams& params,
                             const CubSortThunk* thunk) {
  const BufferAllocations& allocs = *params.buffer_allocations;
  return Run(allocs.GetDeviceAddress(thunk->operand(0)),
             allocs.GetDeviceAddress(thunk->operand(1)),
             allocs.GetDeviceAddress(thunk->result(0)),
             allocs.GetDeviceAddress(thunk->result(1)),
             allocs.GetDeviceAddress(thunk->scratch()), thunk->descending());
}

std::unique_ptr<CubSortRunnerInterface> CreateCubSortRunner(
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
      CHECK(false) << "Unsupported type of the sort kernel: "
                   << primitive_util::LowercasePrimitiveTypeName(type);
  }
}

std::unique_ptr<CubSortRunnerInterface> CreateCubSortRunner(
    PrimitiveType key_type, PrimitiveType value_type) {
  // Values can be of any type of 16/32/64 bit width.
  int valueWidth = primitive_util::BitWidth(value_type);
  CHECK(valueWidth == 16 || valueWidth == 32 || valueWidth == 64)
      << "Unsupported value type of the sort kernel: "
      << primitive_util::LowercasePrimitiveTypeName(value_type);

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
      CHECK(false) << "Unsupported key type of the sort kernel: "
                   << primitive_util::LowercasePrimitiveTypeName(key_type);
  }
}

std::unique_ptr<CubSortRunnerInterface> CreateCubSortRunner(
    PrimitiveType type, std::optional<PrimitiveType> value_type) {
  return value_type.has_value() ? CreateCubSortRunner(type, *value_type)
                                : CreateCubSortRunner(type);
}

}  // namespace

CubSortThunk::CubSortThunk(ThunkInfo thunk_info, PrimitiveType type,
                           std::optional<PrimitiveType> value_type,
                           std::vector<BufferAllocation::Slice> operands,
                           std::vector<BufferAllocation::Slice> results,
                           BufferAllocation::Slice scratch, bool descending)
    : Thunk(Thunk::kCubSort, thunk_info),
      runner_(CreateCubSortRunner(type, value_type)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      scratch_(scratch),
      descending_(descending) {}

Status RunCubSort(PrimitiveType type, std::optional<PrimitiveType> value_type,
                  se::DeviceMemoryBase input_keys,
                  se::DeviceMemoryBase input_values,
                  se::DeviceMemoryBase output_keys,
                  se::DeviceMemoryBase output_values,
                  se::DeviceMemoryBase scratch, bool descending) {
  auto runner = CreateCubSortRunner(type, value_type);
  return runner->Run(input_keys, input_values, output_keys, output_values,
                     scratch, descending);
}

}  // namespace gpu
}  // namespace xla
