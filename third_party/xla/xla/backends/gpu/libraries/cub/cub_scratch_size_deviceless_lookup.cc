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

#include "xla/backends/gpu/libraries/cub/cub_scratch_size_deviceless_lookup.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/messages/parse_message.h"
#include "riegeli/zstd/zstd_reader.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_utils.h"
#include "xla/backends/gpu/libraries/cub/embed_cub_scratch_size_lookup_table.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/util/file_toc.h"

namespace xla::gpu {

namespace {
absl::StatusOr<CubScratchSizeDevicelessLookup> CreateFromBundledData() {
  const FileToc* file_toc = embed_cub_scratch_size_lookup_table_create();
  if (file_toc == nullptr || file_toc[0].name == nullptr) {
    return absl::InternalError("Failed to find embedded data");
  }
  absl::string_view compressed_data(file_toc[0].data, file_toc[0].size);
  riegeli::StringReader<> string_reader(compressed_data);
  riegeli::ZstdReader<> zstd_reader(&string_reader);

  CubScratchSizeLookupTable proto;
  absl::Status status = riegeli::ParseMessage(zstd_reader, proto);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to read bundled data: ", status.message()));
  }
  return CubScratchSizeDevicelessLookup::CreateFromProto(std::move(proto));
}
}  // namespace

absl::StatusOr<CubScratchSizeDevicelessLookup>
CubScratchSizeDevicelessLookup::CreateFromProto(
    CubScratchSizeLookupTable proto) {
  if (proto.entries_size() == 0) {
    return absl::InvalidArgumentError("No entries found in proto");
  }

  for (const auto& entry : proto.entries()) {
    for (int i = 1; i < entry.scratch_size_recordings_size(); ++i) {
      if (entry.scratch_size_recordings(i).num_items() <=
          entry.scratch_size_recordings(i - 1).num_items()) {
        return absl::InvalidArgumentError(
            "scratch_size_recordings must be sorted by num_items");
      }
    }
  }
  return CubScratchSizeDevicelessLookup(std::move(proto));
}

absl::StatusOr<const CubScratchSizeDevicelessLookup&>
CubScratchSizeDevicelessLookup::GetInstance() {
  static absl::NoDestructor<absl::StatusOr<CubScratchSizeDevicelessLookup>>
      instance(CreateFromBundledData());
  if (!instance->ok()) {
    return instance->status();
  }
  return **instance;
}

CubScratchSizeDevicelessLookup::CubScratchSizeDevicelessLookup(
    CubScratchSizeLookupTable proto)
    : proto_(std::move(proto)) {}

namespace {
absl::string_view StripMigSuffix(absl::string_view device_name) {
  size_t mig_pos = device_name.find(" MIG");
  if (mig_pos != absl::string_view::npos) {
    device_name = device_name.substr(0, mig_pos);
  }
  return device_name;
}
}  // namespace

const CubScratchSizeEntry* CubScratchSizeDevicelessLookup::FindEntry(
    stream_executor::SemanticVersion cub_version, absl::string_view device_name,
    int32_t key_type_size, std::optional<int32_t> value_type_size,
    bool is_segmented) const {
  // The scratch size doesn't actually differ between MIG and non-MIG devices,
  // so just lookup the non-MIG device name.
  device_name = StripMigSuffix(device_name);

  for (const CubScratchSizeEntry& entry : proto_.entries()) {
    bool version_matched =
        std::find(entry.cub_version().begin(), entry.cub_version().end(),
                  cub_version.ToString()) != entry.cub_version().end();

    if (version_matched && entry.device_name() == device_name &&
        entry.key_type_size() == key_type_size &&
        entry.value_type_size() == value_type_size.value_or(0) &&
        entry.is_segmented() == is_segmented) {
      return &entry;
    }
  }
  return nullptr;
}

std::optional<int64_t> CubScratchSizeDevicelessLookup::Lookup(
    stream_executor::SemanticVersion cub_version, absl::string_view device_name,
    int32_t key_type_size, std::optional<int32_t> value_type_size,
    int64_t num_items, int64_t batch_size) const {
  const CubScratchSizeEntry* entry =
      FindEntry(cub_version, device_name, key_type_size, value_type_size,
                /*is_segmented=*/batch_size > 1);
  if (entry == nullptr) {
    return std::nullopt;
  }

  auto it = std::lower_bound(
      entry->scratch_size_recordings().begin(),
      entry->scratch_size_recordings().end(), num_items,
      [](const CubScratchSizeEntry::ScratchSizeRecord& record,
         int64_t num_items) { return record.num_items() < num_items; });

  if (it == entry->scratch_size_recordings().end()) {
    return std::nullopt;
  }

  return AddSegmentedSortOffsetsToScratchSize(it->scratch_space_bytes(),
                                              batch_size);
}

bool CubScratchSizeDevicelessLookup::CanLookup(
    stream_executor::SemanticVersion cub_version, absl::string_view device_name,
    int32_t key_type_size, std::optional<int32_t> value_type_size,
    int64_t num_items, int64_t batch_size) const {
  const CubScratchSizeEntry* entry = FindEntry(
      cub_version, device_name, key_type_size, value_type_size, batch_size > 1);
  if (entry == nullptr || entry->scratch_size_recordings().empty()) {
    return false;
  }

  return entry->scratch_size_recordings()
             .Get(entry->scratch_size_recordings().size() - 1)
             .num_items() >= num_items;
}

}  // namespace xla::gpu
