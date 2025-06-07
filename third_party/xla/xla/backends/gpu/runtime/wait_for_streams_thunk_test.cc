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

#include "xla/backends/gpu/runtime/wait_for_streams_thunk.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

TEST(WaitForStreamsThunkTest, RoundTrip) {
  Thunk::ThunkInfo thunk_info;
  ExecutionStreamId stream_id(1);
  ExecutionStreamId wait_for_stream_id(2);

  WaitForStreamsThunk original_thunk(thunk_info, stream_id, wait_for_stream_id);

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto thunk_proto, original_thunk.ToProto());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WaitForStreamsThunk> deserialized_thunk,
      WaitForStreamsThunk::FromProto(thunk_info,
                                     thunk_proto.wait_for_streams_thunk()));

  EXPECT_EQ(deserialized_thunk->stream_id(), stream_id);
  EXPECT_EQ(deserialized_thunk->wait_for_stream_id(), wait_for_stream_id);
}

}  // namespace
}  // namespace xla::gpu
