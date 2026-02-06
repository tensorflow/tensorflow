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

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(BufferDebugLogEntryMetadataStoreTest, RoundTrip) {
  BufferDebugLogEntryMetadataStore store;
  BufferDebugLogEntryMetadataStore::Metadata metadata = {
      /*thunk_id=*/ThunkId(123),
      /*buffer_idx=*/4,
      /*execution_id=*/5,
      /*is_input=*/true,
  };

  BufferDebugLogEntryId entry_id = store.AssignId(metadata);
  // Add a second entry to ensure mutating the store doesn't invalidate
  // previously assigned IDs.
  [[maybe_unused]] BufferDebugLogEntryId unused_id =
      store.AssignId(BufferDebugLogEntryMetadataStore::Metadata{});
  std::optional<BufferDebugLogEntryMetadataStore::Metadata> retrieved_metadata =
      store.GetEntryMetadata(entry_id);

  ASSERT_TRUE(retrieved_metadata.has_value());
  EXPECT_EQ(retrieved_metadata->thunk_id, metadata.thunk_id);
  EXPECT_EQ(retrieved_metadata->buffer_idx, metadata.buffer_idx);
  EXPECT_EQ(retrieved_metadata->execution_id, metadata.execution_id);
  EXPECT_EQ(retrieved_metadata->is_input, metadata.is_input);
}

TEST(BufferDebugLogEntryMetadataStoreTest, InvalidId) {
  BufferDebugLogEntryMetadataStore store;

  EXPECT_EQ(store.GetEntryMetadata(BufferDebugLogEntryId{123}), std::nullopt);
}

TEST(BufferDebugLogEntryMetadataStoreTest, EntriesToProto) {
  BufferDebugLogEntryMetadataStore store;
  const BufferDebugLogEntryId entry_id1 = store.AssignId({
      /*thunk_id=*/ThunkId(123),
      /*buffer_idx=*/4,
      /*execution_id=*/5,
      /*is_input=*/true,
      BufferDebugLogEntryProto::CHECK_TYPE_CHECKSUM,
  });
  const BufferDebugLogEntryId entry_id2 = store.AssignId({
      /*thunk_id=*/ThunkId(567),
      /*buffer_idx=*/8,
      /*execution_id=*/9,
      /*is_input=*/false,
      BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
  });
  std::vector<BufferDebugLogEntry> entries = {
      {
          /*entry_id=*/entry_id1,
          /*checksum=*/12341234,
      },
      {
          /*entry_id=*/entry_id2,
          /*checksum=*/56785678,
      },
  };

  BufferDebugLogProto log_proto = store.EntriesToProto(entries);

  EXPECT_THAT(log_proto, EqualsProto(R"pb(
                entries {
                  thunk_id: 123
                  buffer_idx: 4
                  execution_id: 5
                  is_input_buffer: true
                  checksum: 12341234
                  check_type: CHECK_TYPE_CHECKSUM
                }
                entries {
                  thunk_id: 567
                  buffer_idx: 8
                  execution_id: 9
                  is_input_buffer: false
                  checksum: 56785678,
                  check_type: CHECK_TYPE_FLOAT_CHECKS
                }
              )pb"));
}

}  // namespace
}  // namespace xla::gpu
