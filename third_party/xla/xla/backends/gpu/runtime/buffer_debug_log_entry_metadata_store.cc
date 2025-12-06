/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"

namespace xla::gpu {

BufferDebugLogEntryId BufferDebugLogEntryMetadataStore::AssignId(
    const BufferDebugLogEntryMetadataStore::Metadata& metadata) {
  absl::MutexLock lock{mutex_};
  size_t id = log_entry_metadata_.size();
  CHECK_LT(id, std::numeric_limits<uint32_t>::max())
      << "BufferDebugLogEntryId overflowed";

  log_entry_metadata_.push_back(std::move(metadata));
  return BufferDebugLogEntryId{static_cast<uint32_t>(id)};
}

std::optional<BufferDebugLogEntryMetadataStore::Metadata>
BufferDebugLogEntryMetadataStore::GetEntryMetadata(
    BufferDebugLogEntryId entry_id) {
  absl::MutexLock lock{mutex_};
  return GetEntryMetadataLocked(entry_id);
}

std::vector<std::optional<BufferDebugLogEntryMetadataStore::Metadata>>
BufferDebugLogEntryMetadataStore::GetEntryMetadataBatch(
    absl::Span<const BufferDebugLogEntryId> entry_ids) {
  absl::MutexLock lock{mutex_};
  std::vector<std::optional<Metadata>> result;
  result.reserve(entry_ids.size());
  for (BufferDebugLogEntryId entry_id : entry_ids) {
    result.push_back(GetEntryMetadataLocked(entry_id));
  }
  return result;
}

std::optional<BufferDebugLogEntryMetadataStore::Metadata>
BufferDebugLogEntryMetadataStore::GetEntryMetadataLocked(
    BufferDebugLogEntryId entry_id) {
  if (entry_id >= log_entry_metadata_.size()) {
    return std::nullopt;
  }
  return log_entry_metadata_[entry_id.value()];
}

BufferDebugLogProto BufferDebugLogEntryMetadataStore::EntriesToProto(
    absl::Span<const BufferDebugLogEntry> entries) {
  absl::MutexLock lock{mutex_};

  BufferDebugLogProto proto;
  for (const BufferDebugLogEntry& entry : entries) {
    std::optional<Metadata> metadata = GetEntryMetadataLocked(entry.entry_id);
    if (!metadata.has_value()) {
      continue;
    }

    BufferDebugLogEntryProto* entry_proto = proto.add_entries();
    entry_proto->set_thunk_id(metadata->thunk_id.value());
    entry_proto->set_buffer_idx(metadata->buffer_idx);
    entry_proto->set_execution_id(metadata->execution_id);
    entry_proto->set_is_input_buffer(metadata->is_input);
    entry_proto->set_checksum(entry.value);
    entry_proto->set_check_type(metadata->check_type);
  }
  return proto;
}

}  // namespace xla::gpu
