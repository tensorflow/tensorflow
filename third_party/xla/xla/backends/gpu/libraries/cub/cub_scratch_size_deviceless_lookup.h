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

#ifndef XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SCRATCH_SIZE_DEVICELESS_LOOKUP_H_
#define XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SCRATCH_SIZE_DEVICELESS_LOOKUP_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/cub/scratch_space_lookup_table.pb.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla::gpu {

// A lookup table for that returns an estimate of the scratch space required by
// CUB sorts without requiring a GPU.
class CubScratchSizeDevicelessLookup {
 public:
  // Not copyable, only movable, since this is a singleton.
  CubScratchSizeDevicelessLookup(const CubScratchSizeDevicelessLookup&) =
      delete;
  CubScratchSizeDevicelessLookup& operator=(
      const CubScratchSizeDevicelessLookup&) = delete;
  CubScratchSizeDevicelessLookup(CubScratchSizeDevicelessLookup&&) = default;
  CubScratchSizeDevicelessLookup& operator=(CubScratchSizeDevicelessLookup&&) =
      default;

  static absl::StatusOr<CubScratchSizeDevicelessLookup> CreateFromProto(
      CubScratchSizeLookupTable proto);

  // Returns the singleton instance loaded from bundled data.
  static absl::StatusOr<const CubScratchSizeDevicelessLookup&> GetInstance();

  // Looks up the estimated scratch space CUB will need for the given
  // parameters. The estimated space will be >= to the actual space CUB will
  // need.
  //
  // Will return std::nullopt if we can't estimate it, i.e., if we have no
  // entries for the give parameters, or if the requested num_items is greater
  // than any recorded num_items for the given parameters.
  std::optional<int64_t> Lookup(stream_executor::SemanticVersion cub_version,
                                absl::string_view device_name,
                                int32_t key_type_size,
                                std::optional<int32_t> value_type_size,
                                int64_t num_items,
                                int64_t batch_size = 1) const;

  // Cheaper check to see if we have data for the given parameters.
  //
  // Returns true if a matching entry exists and the requested num_items does
  // not exceed the largest recorded num_items in that entry.
  bool CanLookup(stream_executor::SemanticVersion cub_version,
                 absl::string_view device_name, int32_t key_type_size,
                 std::optional<int32_t> value_type_size, int64_t num_items,
                 int64_t batch_size = 1) const;

 private:
  explicit CubScratchSizeDevicelessLookup(CubScratchSizeLookupTable proto);

  const CubScratchSizeEntry* FindEntry(
      stream_executor::SemanticVersion cub_version,
      absl::string_view device_name, int32_t key_type_size,
      std::optional<int32_t> value_type_size, bool is_segmented) const;

  CubScratchSizeLookupTable proto_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_LIBRARIES_CUB_CUB_SCRATCH_SIZE_DEVICELESS_LOOKUP_H_
