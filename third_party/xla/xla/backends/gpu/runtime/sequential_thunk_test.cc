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

#include "xla/backends/gpu/runtime/sequential_thunk.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla::gpu {
namespace {

constexpr ExecutionStreamId kExecutionStreamId{123};
constexpr absl::string_view kProfileAnnotation = "profile_annotation";

Thunk::ThunkInfo GetExampleThunkInfo() {
  Thunk::ThunkInfo thunk_info{};
  thunk_info.execution_stream_id = kExecutionStreamId;
  thunk_info.profile_annotation = kProfileAnnotation;
  return thunk_info;
}

TEST(SequentialThunkTest, EmptySequentialThunkToProto) {
  SequentialThunk thunk{GetExampleThunkInfo(), {}};
  ThunkProto proto;

  TF_ASSERT_OK(thunk.ToProto(&proto));
  ASSERT_TRUE(proto.has_sequential_thunk());
  EXPECT_EQ(proto.sequential_thunk().thunks_size(), 0);

  ASSERT_TRUE(proto.has_thunk_info());
  EXPECT_EQ(proto.thunk_info().execution_stream_id(), kExecutionStreamId);
  EXPECT_EQ(proto.thunk_info().profile_annotation(), kProfileAnnotation);
}

}  // namespace
}  // namespace xla::gpu
