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

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cub/device/device_radix_sort.cuh>  // NOLINT(build/include_order)
#include <cub/device/device_segmented_radix_sort.cuh>  // NOLINT(build/include_order)
#include <cub/version.cuh>  // NOLINT(build/include_order)
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/libraries/cub/scratch_space_lookup_table.pb.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

template <typename KeyT, typename ValueT>
absl::StatusOr<int64_t> GetScratchSizeKeyValueSort(int64_t num_items) {
  size_t temp_storage_bytes = 0;
  cudaError_t status = cub::DeviceRadixSort::SortPairs<KeyT, ValueT>(
      nullptr, temp_storage_bytes, static_cast<KeyT*>(nullptr),
      static_cast<KeyT*>(nullptr), static_cast<ValueT*>(nullptr),
      static_cast<ValueT*>(nullptr), num_items);
  if (status == cudaSuccess) {
    return temp_storage_bytes;
  }
  return absl::InternalError(
      absl::StrCat("CUB SortPairs failed: ", cudaGetErrorString(status)));
}

template <typename KeyT, typename ValueT>
absl::StatusOr<int64_t> GetScratchSizeKeyValueSegmentedSort(
    int64_t num_items, int64_t num_segments) {
  size_t temp_storage_bytes = 0;
  cudaError_t status = cub::DeviceSegmentedRadixSort::SortPairs<KeyT, ValueT>(
      nullptr, temp_storage_bytes, static_cast<KeyT*>(nullptr),
      static_cast<KeyT*>(nullptr), static_cast<ValueT*>(nullptr),
      static_cast<ValueT*>(nullptr), num_items, num_segments,
      static_cast<int*>(nullptr), static_cast<int*>(nullptr));
  if (status == cudaSuccess) {
    return temp_storage_bytes;
  }
  return absl::InternalError(
      absl::StrCat("CUB SortPairs failed: ", cudaGetErrorString(status)));
}

template <typename KeyT>
absl::StatusOr<int64_t> GetScratchSizeKeySort(int64_t num_items) {
  size_t temp_storage_bytes = 0;
  cudaError_t status = cub::DeviceRadixSort::SortKeys<KeyT>(
      nullptr, temp_storage_bytes, static_cast<KeyT*>(nullptr),
      static_cast<KeyT*>(nullptr), num_items);
  if (status == cudaSuccess) {
    return temp_storage_bytes;
  }
  return absl::InternalError(
      absl::StrCat("CUB SortKeys failed: ", cudaGetErrorString(status)));
}

template <typename KeyT>
absl::StatusOr<int64_t> GetScratchSizeKeySegmentedSort(int64_t num_items,
                                                       int64_t num_segments) {
  size_t temp_storage_bytes = 0;
  cudaError_t status = cub::DeviceSegmentedRadixSort::SortKeys<KeyT>(
      nullptr, temp_storage_bytes, static_cast<KeyT*>(nullptr),
      static_cast<KeyT*>(nullptr), num_items, num_segments,
      static_cast<int*>(nullptr), static_cast<int*>(nullptr));
  if (status == cudaSuccess) {
    return temp_storage_bytes;
  }
  return absl::InternalError(
      absl::StrCat("CUB SortKeys failed: ", cudaGetErrorString(status)));
}

CubScratchSizeEntry CreateEntry(absl::string_view device_name, int32_t key_size,
                                std::optional<int32_t> value_size,
                                bool is_segmented,
                                const std::vector<int64_t>& num_items_list,
                                const std::vector<int64_t>& scratch_sizes) {
  CubScratchSizeEntry entry;
  entry.set_device_name(device_name);
  entry.set_key_type_size(key_size);
  if (value_size.has_value()) {
    entry.set_value_type_size(*value_size);
  }
  entry.set_is_segmented(is_segmented);

  entry.add_cub_version(absl::StrCat(CUB_MAJOR_VERSION, ".", CUB_MINOR_VERSION,
                                     ".", CUB_SUBMINOR_VERSION));

  for (size_t i = 0; i < num_items_list.size(); ++i) {
    const bool is_last_item = (i == num_items_list.size() - 1);
    const bool scratch_size_changes =
        !is_last_item && (scratch_sizes[i] != scratch_sizes[i + 1]);

    // Only record the last num_items for a given scratch size to compress the
    // data.
    if (is_last_item || scratch_size_changes) {
      CubScratchSizeEntry::ScratchSizeRecord* record =
          entry.add_scratch_size_recordings();
      record->set_num_items(num_items_list[i]);
      record->set_scratch_space_bytes(scratch_sizes[i]);
    }
  }
  return entry;
}

template <typename KeyT>
absl::StatusOr<CubScratchSizeEntry> GenerateDataForKeySegemented(
    absl::string_view device_name, const std::vector<int64_t>& num_items_list,
    bool is_segmented) {
  std::vector<int64_t> sizes;
  sizes.reserve(num_items_list.size());
  for (int64_t n : num_items_list) {
    int64_t size;
    if (is_segmented) {
      ASSIGN_OR_RETURN(
          size, (GetScratchSizeKeySegmentedSort<KeyT>(n, /*num_segments=*/2)));
    } else {
      ASSIGN_OR_RETURN(size, (GetScratchSizeKeySort<KeyT>(n)));
    }
    sizes.push_back(size);
  }

  return CreateEntry(device_name, sizeof(KeyT),
                     /*value_size=*/std::nullopt, is_segmented, num_items_list,
                     sizes);
}

// Returns the max number of items to be sorted that we are generating data for.
int64_t GetMaxNumOfItems(const cudaDeviceProp& prop, size_t element_size) {
  // Limit to 1 billion otherwise we might get integer overflows for small
  // Keys/Values types.
  return std::min<int64_t>(prop.totalGlobalMem / element_size, 1'000'000'000LL);
}

// Create a list of numbers that grows geometrically up to max_items.
std::vector<int64_t> CreateNumItemsList(int64_t max_items) {
  // Needed for the first few steps when n * kGrowthFactor rounds down to n.
  constexpr int64_t kMinStep = 10;
  constexpr double kGrowthFactor = 1.02;
  std::vector<int64_t> num_items_list;
  for (int64_t n = 1; n <= max_items;
       n = std::max(n + kMinStep, static_cast<int64_t>(n * kGrowthFactor))) {
    num_items_list.push_back(n);
  }
  return num_items_list;
}

// Generate the data for segmented and non-segmented sorts.
template <typename KeyT>
absl::Status GenerateDataForKeyOnlySort(CubScratchSizeLookupTable& lookup_table,
                                        absl::string_view device_name,
                                        const cudaDeviceProp& prop) {
  std::vector<int64_t> num_items_list =
      CreateNumItemsList(GetMaxNumOfItems(prop, sizeof(KeyT)));

  ASSIGN_OR_RETURN(
      *lookup_table.add_entries(),
      (GenerateDataForKeySegemented<KeyT>(device_name, num_items_list,
                                          /*is_segmented=*/false)));

  ASSIGN_OR_RETURN(
      *lookup_table.add_entries(),
      (GenerateDataForKeySegemented<KeyT>(device_name, num_items_list,
                                          /*is_segmented=*/true)));

  return absl::OkStatus();
}

template <typename KeyT, typename ValueT>
absl::StatusOr<CubScratchSizeEntry> GenerateDataForKeyValueSegemented(
    absl::string_view device_name, const std::vector<int64_t>& num_items_list,
    bool is_segmented) {
  std::vector<int64_t> sizes;
  sizes.reserve(num_items_list.size());
  for (int64_t n : num_items_list) {
    int64_t size;
    if (is_segmented) {
      ASSIGN_OR_RETURN(size, (GetScratchSizeKeyValueSegmentedSort<KeyT, ValueT>(
                                 n, /*num_segments=*/2)));
    } else {
      ASSIGN_OR_RETURN(size, (GetScratchSizeKeyValueSort<KeyT, ValueT>(n)));
    }
    sizes.push_back(size);
  }

  return CreateEntry(device_name, sizeof(KeyT), sizeof(ValueT), is_segmented,
                     num_items_list, sizes);
}

// Generate the data for segmented and non-segmented sorts.
template <typename KeyT, typename ValueT>
absl::Status GenerateDataForKeyValueSort(
    CubScratchSizeLookupTable& lookup_table, absl::string_view device_name,
    const cudaDeviceProp& prop) {
  std::vector<int64_t> num_items_list =
      CreateNumItemsList(GetMaxNumOfItems(prop, sizeof(KeyT) + sizeof(ValueT)));

  ASSIGN_OR_RETURN(*lookup_table.add_entries(),
                   (GenerateDataForKeyValueSegemented<KeyT, ValueT>(
                       device_name, num_items_list, /*is_segmented=*/false)));

  ASSIGN_OR_RETURN(*lookup_table.add_entries(),
                   (GenerateDataForKeyValueSegemented<KeyT, ValueT>(
                       device_name, num_items_list, /*is_segmented=*/true)));

  return absl::OkStatus();
}

absl::StatusOr<cudaDeviceProp> GetDeviceProperties() {
  int device = 0;
  cudaDeviceProp prop;
  cudaError_t status = cudaGetDeviceProperties(&prop, device);
  if (status == cudaSuccess) {
    return prop;
  }
  return absl::InternalError(
      absl::StrCat("Failed to get device properties for device ", device, ": ",
                   cudaGetErrorString(status)));
}

absl::Status WriteLookupTableToFile(
    const CubScratchSizeLookupTable& lookup_table) {
  std::string outputs_dir;
  if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
    return absl::NotFoundError("TEST_UNDECLARED_OUTPUTS_DIR not set");
  }

  std::string path =
      tsl::io::JoinPath(outputs_dir, "cub_scratch_size_lookup_table.textproto");
  tsl::Env* env = tsl::Env::Default();
  RETURN_IF_ERROR(tsl::WriteTextProto(env, path, lookup_table));

  std::cout << "Wrote lookup table to " << path << std::endl;
  return absl::OkStatus();
}

}  // namespace

absl::Status GenerateCubScratchSizeData() {
  ASSIGN_OR_RETURN(cudaDeviceProp prop, GetDeviceProperties());
  std::string device_name(prop.name, strnlen(prop.name, sizeof(prop.name)));

  CubScratchSizeLookupTable lookup_table;
  // Generate the data for 8, 16, 32, 64 bit, key only sorts
  RETURN_IF_ERROR(
      GenerateDataForKeyOnlySort<int8_t>(lookup_table, device_name, prop));
  RETURN_IF_ERROR(
      GenerateDataForKeyOnlySort<int16_t>(lookup_table, device_name, prop));
  RETURN_IF_ERROR(
      GenerateDataForKeyOnlySort<int32_t>(lookup_table, device_name, prop));
  RETURN_IF_ERROR(
      GenerateDataForKeyOnlySort<int64_t>(lookup_table, device_name, prop));

  // Generate the data for key value sorts for every key/value size combination.
  // I.e. keys of size 8, 16, 32, 64 bits, and values of size 8, 16, 32, 64

  // 8 bit keys
  // Skipping int8_t value sort since it isn't supported in XLA
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int8_t, int16_t>(
      lookup_table, device_name, prop)));
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int8_t, int32_t>(
      lookup_table, device_name, prop)));
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int8_t, int64_t>(
      lookup_table, device_name, prop)));

  // 16 bit keys
  // Skipping int8_t value sort since it isn't supported in XLA
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int16_t, int16_t>(
      lookup_table, device_name, prop)));
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int16_t, int32_t>(
      lookup_table, device_name, prop)));
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int16_t, int64_t>(
      lookup_table, device_name, prop)));

  // 32 bit keys
  // Skipping int8_t value sort since it isn't supported in XLA
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int32_t, int16_t>(
      lookup_table, device_name, prop)));
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int32_t, int32_t>(
      lookup_table, device_name, prop)));
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int32_t, int64_t>(
      lookup_table, device_name, prop)));

  // 64 bit keys
  // Skipping int8_t value sort since it isn't supported in XLA
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int64_t, int16_t>(
      lookup_table, device_name, prop)));
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int64_t, int32_t>(
      lookup_table, device_name, prop)));
  RETURN_IF_ERROR((GenerateDataForKeyValueSort<int64_t, int64_t>(
      lookup_table, device_name, prop)));

  return WriteLookupTableToFile(lookup_table);
}

}  // namespace xla::gpu
