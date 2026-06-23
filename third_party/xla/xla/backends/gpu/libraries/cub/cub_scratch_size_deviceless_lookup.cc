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
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/messages/parse_message.h"
#include "riegeli/zstd/zstd_reader.h"
#include "xla/backends/gpu/libraries/cub/cub_sort_utils.h"
#include "xla/backends/gpu/libraries/cub/embed_cub_scratch_size_lookup_table.h"
#include "xla/backends/gpu/libraries/cub/scratch_space_lookup_table.pb.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/util/file_toc.h"

namespace xla::gpu {
using internal::LookupKey;
using internal::ScratchSizeRecord;

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

  absl::flat_hash_map<LookupKey, std::vector<ScratchSizeRecord>> lookup_table;

  for (const auto& entry : proto.entries()) {
    for (int i = 1; i < entry.scratch_size_recordings_size(); ++i) {
      if (entry.scratch_size_recordings(i).num_items() <=
          entry.scratch_size_recordings(i - 1).num_items()) {
        return absl::InvalidArgumentError(
            "scratch_size_recordings must be sorted by num_items");
      }
    }

    std::vector<ScratchSizeRecord> recordings;
    recordings.reserve(entry.scratch_size_recordings_size());
    for (const auto& record : entry.scratch_size_recordings()) {
      recordings.push_back({record.num_items(), record.scratch_space_bytes()});
    }

    for (const std::string& version_str : entry.cub_version()) {
      auto cub_version_or =
          stream_executor::SemanticVersion::ParseFromString(version_str);
      if (!cub_version_or.ok()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to parse CUB version: ", version_str));
      }

      LookupKey key{*cub_version_or, entry.device_name(), entry.key_type_size(),
                    entry.value_type_size(), entry.is_segmented()};
      lookup_table[key] = recordings;
    }
  }

  return CubScratchSizeDevicelessLookup(std::move(lookup_table));
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
    absl::flat_hash_map<LookupKey, std::vector<ScratchSizeRecord>> lookup_table)
    : lookup_table_(std::move(lookup_table)) {}

namespace {
absl::string_view StripMigSuffix(absl::string_view device_name) {
  size_t mig_pos = device_name.find(" MIG");
  if (mig_pos != absl::string_view::npos) {
    device_name = device_name.substr(0, mig_pos);
  }
  return device_name;
}
}  // namespace

std::optional<int64_t> CubScratchSizeDevicelessLookup::Lookup(
    stream_executor::SemanticVersion cub_version, absl::string_view device_name,
    int32_t key_type_size, std::optional<int32_t> value_type_size,
    int64_t num_items, int64_t batch_size) const {
  // The scratch size doesn't actually differ between MIG and non-MIG devices,
  // so just lookup the non-MIG device name.
  device_name = StripMigSuffix(device_name);

  LookupKey key{cub_version, std::string(device_name), key_type_size,
                value_type_size.value_or(0),
                /*is_segmented=*/batch_size > 1};

  auto it = lookup_table_.find(key);
  if (it == lookup_table_.end()) {
    return std::nullopt;
  }

  const std::vector<ScratchSizeRecord>& recordings = it->second;

  auto rec_it =
      std::lower_bound(recordings.begin(), recordings.end(), num_items,
                       [](const ScratchSizeRecord& record, int64_t num_items) {
                         return record.num_items < num_items;
                       });

  if (rec_it == recordings.end()) {
    return std::nullopt;
  }

  return AddSegmentedSortOffsetsToScratchSize(rec_it->scratch_space_bytes,
                                              batch_size);
}

bool CubScratchSizeDevicelessLookup::CanLookup(
    stream_executor::SemanticVersion cub_version, absl::string_view device_name,
    int32_t key_type_size, std::optional<int32_t> value_type_size,
    int64_t num_items, int64_t batch_size) const {
  device_name = StripMigSuffix(device_name);

  LookupKey key{cub_version, std::string(device_name), key_type_size,
                value_type_size.value_or(0), batch_size > 1};

  auto it = lookup_table_.find(key);
  if (it == lookup_table_.end() || it->second.empty()) {
    return false;
  }

  return it->second.back().num_items >= num_items;
}

}  // namespace xla::gpu
