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

#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pointer;
using ::testing::Property;
using ::testing::WhenDynamicCastTo;

TEST(ThunkProtoDeserializationTest, SequentialThunkChain) {
  constexpr ExecutionStreamId kExecutionStreamId{123};
  constexpr absl::string_view kProfileAnnotation = "profile_annotation";

  Thunk::ThunkInfo thunk_info{};
  thunk_info.execution_stream_id = kExecutionStreamId;
  thunk_info.profile_annotation = kProfileAnnotation;

  // This constructs the following thunk tree:
  // `SequentialThunk{SequentialThunk{}}`
  std::unique_ptr<Thunk> inner_thunk =
      std::make_unique<SequentialThunk>(thunk_info, ThunkSequence{});
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(inner_thunk));
  SequentialThunk outer_thunk(thunk_info, std::move(thunk_sequence));

  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, outer_thunk.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> new_thunk,
                          DeserializeThunkProto(proto, {}));

  EXPECT_THAT(new_thunk.get(),
              WhenDynamicCastTo<const SequentialThunk*>(Property(
                  &SequentialThunk::thunks,
                  ElementsAre(Pointer(WhenDynamicCastTo<const SequentialThunk*>(
                      Property(&SequentialThunk::thunks, IsEmpty())))))));
}

}  // namespace
}  // namespace xla::gpu
