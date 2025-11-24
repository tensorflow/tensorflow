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

#include "xla/backends/gpu/runtime/thunk.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {
using ::tsl::proto_testing::EqualsProto;

class TestThunk : public Thunk {
 public:
  explicit TestThunk(ThunkInfo thunk_info) : Thunk(kKernel, thunk_info) {}
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
};

TEST(ThunkTest, GetMetadataProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.thunk_id = 123;
  thunk_info.profile_annotation = "test_kernel";
  auto thunk = std::make_unique<TestThunk>(thunk_info);
  EXPECT_THAT(thunk->ToMetadataProto(), EqualsProto(R"pb(
                thunk_info { thunk_id: 123 profile_annotation: "test_kernel" }
                thunk_kind: "kKernel"
              )pb"));
}

TEST(ThunkTest, GetMetadataListProtoFromThunkGraph) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.thunk_id = 123;
  thunk_info.profile_annotation = "test_kernel";
  auto test_thunk = std::make_unique<TestThunk>(thunk_info);

  thunk_info.thunk_id = 456;
  thunk_info.profile_annotation = "";
  ThunkSequence thunks;
  thunks.push_back(std::move(test_thunk));

  SequentialThunk sequential_thunk(thunk_info, std::move(thunks));
  EXPECT_THAT(GetMetadataListProtoFromThunkGraph(sequential_thunk),
              EqualsProto(R"pb(
                thunk_metadata {
                  thunk_info { thunk_id: 456 }
                  thunk_kind: "kSequential"
                }
                thunk_metadata {
                  thunk_info { thunk_id: 123 profile_annotation: "test_kernel" }
                  thunk_kind: "kKernel"
                }
              )pb"));
}

}  // namespace
}  // namespace xla::gpu
