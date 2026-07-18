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

#ifndef XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_ENTRY_METADATA_STORE_H_
#define XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_ENTRY_METADATA_STORE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk_id.h"

namespace xla::gpu {

// Provides unique mapping between `BufferDebugLogEntry::entry_id` and
// additional information about the entry.
//
// For checksumming, the entry_id is transferred between host and device to
// identify the context of checksummed buffer. This class provides a way to
// store additional data about the entry without passing excessive information
// back and forth.
class BufferDebugLogEntryMetadataStore {
 public:
  // Metadata stored for each entry.
  struct Metadata {
    // ID of the thunk the entry relates to.
    ThunkId thunk_id;
    // Index of the thunk's buffer within the array returned by
    // `Thunk::buffer_uses()`.
    size_t buffer_idx;
    // ID of the execution of the thunk, to distinguish between different
    // executions of the same thunk, e.g. when it's used in a loop.
    size_t execution_id;
    // True if the entry represents a check made before the thunk executes.
    bool is_input;

    // The type of check that produced this entry.
    BufferDebugLogEntryProto::CheckType check_type;

    // Profile annotation of the HLO instruction that produced this entry.
    // This is used to identify the HLO instruction in HloModule that was under
    // the check. We need that to be able to log the HLO instruction when
    // a non-zero number of infs or nans were found.
    std::string profile_annotation;

    std::string ToString() const {
      return absl::StrCat(
          "thunk_id: ", thunk_id.value(), ", buffer_idx: ", buffer_idx,
          ", execution_id: ", execution_id,
          ", is_input: ", is_input ? "true" : "false", ", check_type: ",
          BufferDebugLogEntryProto::CheckType_Name(check_type));
    }
  };

  // Inserts `metadata` into the store and returns an ID that can be used to
  // retrieve it with `GetEntryMetadata`.
  //
  // The returned ID is guaranteed to be unique within the lifetime of this
  // store, and stays valid until the store gets destroyed.
  BufferDebugLogEntryId AssignId(const Metadata& metadata)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the metadata for the entry with `entry_id` previously returned by
  // `AssignId`, or `std::nullopt` if the ID is invalid.
  std::optional<Metadata> GetEntryMetadata(BufferDebugLogEntryId entry_id)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the metadata for the entries with `entry_ids` previously
  // returned by `AssignId`, or `std::nullopt` if the ID is invalid.
  std::vector<std::optional<Metadata>> GetEntryMetadataBatch(
      absl::Span<const BufferDebugLogEntryId> entry_ids)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Converts a list of `entries` with IDs assigned by this store to a
  // `BufferDebugLogProto` with additional metadata.
  BufferDebugLogProto EntriesToProto(
      absl::Span<const BufferDebugLogEntry> entries)
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  std::optional<Metadata> GetEntryMetadataLocked(BufferDebugLogEntryId entry_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  absl::Mutex mutex_;
  std::vector<Metadata> log_entry_metadata_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_ENTRY_METADATA_STORE_H_
