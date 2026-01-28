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

#include "xla/backends/gpu/runtime/cudnn_thunk.h"

#include <array>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using tsl::proto_testing::EqualsProto;

TEST(CuDnnThunkTest, TestSerializationDeserialization) {
  CudnnThunkProto cudnn_thunk_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        fingerprint: "fingerprint"
        args {
          slice { offset: 123 size: 456 }
          shape { element_type: U8 }
        }
        args {
          slice { offset: 789 size: 1011 }
          shape { element_type: U8 }
        }
        output_args: false
        output_args: true
        sdpa_dropout_seed: 123456789
      )pb",
      &cudnn_thunk_proto));

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  ThunkProto thunk_proto;
  *thunk_proto.mutable_thunk_info() = thunk_info.ToProto();
  *thunk_proto.mutable_cudnn_thunk() = cudnn_thunk_proto;

  BufferAllocation alloc0(/*index=*/0, /*size=*/2048, /*color=*/0);
  std::array buffer_allocations = {alloc0};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CuDnnThunk> thunk,
      CuDnnThunk::FromProto(thunk_info, cudnn_thunk_proto, buffer_allocations));

  EXPECT_THAT(thunk->ToProto(),
              absl_testing::IsOkAndHolds(EqualsProto(thunk_proto)));
}

}  // namespace
}  // namespace xla::gpu
