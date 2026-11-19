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

#ifndef XLA_HLO_IR_HLO_PAYLOAD_DEDUPLICATOR_H_
#define XLA_HLO_IR_HLO_PAYLOAD_DEDUPLICATOR_H_

#include <cstdint>
#include <deque>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/backend_config.h"

namespace xla {

// Helper for deduplicating backend_config payloads into a growing list
// during HLO module serialization.
class HloPayloadDeduplicator {
 public:
  // Initializes the deduplicator.
  // `base_offset` shifts the generated payload IDs on-the-fly. This is useful
  // when serializing into a proto that already has pre-existing payloads
  // (e.g. when appending to a pre-filled proto).
  explicit HloPayloadDeduplicator(int64_t base_offset = 0);

  // Fast path: deduplicates the backend config using the in-memory pointer
  // address of the BackendConfigWrapper. Returns the unique index (ID) of
  // the payload and stores it if not already stored.
  int64_t Deduplicate(const BackendConfigWrapper* wrapper);

  // Fallback path: deduplicates the backend config using raw string comparison.
  // Stores the given value if not already stored, and returns a unique index
  // (ID) referencing it.
  int64_t Deduplicate(absl::string_view value);

  // Returns the collected string payloads and transfers ownership of them
  // to the caller (moves the internal payloads list), avoiding copies.
  std::deque<std::string> TakePayloads();

  // Returns the payload at the given ID.
  const std::string& GetPayload(int64_t id) const;

  // Returns the number of payloads.
  int64_t size() const;

 private:
  // Shifts the generated payload IDs. This is useful when appending to a
  // pre-existing proto that already has some payloads.
  // Example: If offset_ is 5, the first deduplicated payload gets ID 5, the
  // second ID 6, etc.
  int64_t offset_;
  std::deque<std::string> payloads_;
  absl::flat_hash_map<absl::string_view, int64_t> string_map_;
  absl::flat_hash_map<const BackendConfigWrapper*, int64_t> pointer_map_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_PAYLOAD_DEDUPLICATOR_H_
