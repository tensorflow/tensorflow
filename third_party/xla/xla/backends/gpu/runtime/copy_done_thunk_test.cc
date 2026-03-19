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

#include "xla/backends/gpu/runtime/copy_done_thunk.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(CopyDoneThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  auto async_events = std::make_shared<CopyThunk::AsyncEvents>();
  CopyDoneThunk thunk(thunk_info, async_events,
                      /*copy_start_instr_id=*/456);

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());

  // We can't easily predict async_events_unique_id because it's the pointer
  // address. But we can check that it's set.
  EXPECT_TRUE(proto.copy_done_thunk().has_async_events_unique_id());
  EXPECT_EQ(proto.copy_done_thunk().copy_start_instr_id(), 456);
}

TEST(CopyDoneThunkTest, FromProto) {
  auto existing_events = std::make_shared<CopyThunk::AsyncEvents>();
  uint64_t unique_id = absl::bit_cast<uint64_t>(existing_events.get());

  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        copy_done_thunk { copy_start_instr_id: 456 }
      )pb");
  proto.mutable_copy_done_thunk()->set_async_events_unique_id(unique_id);

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  CopyThunk::AsyncEventsMap async_events_map;
  // Pre-populate map to check linkage
  async_events_map[AsyncEventsUniqueId{unique_id}] = existing_events;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CopyDoneThunk> thunk,
      CopyDoneThunk::FromProto(thunk_info, proto.copy_done_thunk(),
                               async_events_map));

  EXPECT_EQ(thunk->kind(), Thunk::kCopyDone);
  EXPECT_EQ(thunk->GetAsyncEventsUniqueId()->value(), unique_id);
}

}  // namespace
}  // namespace xla::gpu
