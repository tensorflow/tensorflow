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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::tsl::proto_testing::EqualsProto;

class TestThunk : public Thunk {
 public:
  explicit TestThunk(ThunkInfo thunk_info) : Thunk(kKernel, thunk_info) {}
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }
  BufferUses buffer_uses() const override { return {}; }
  absl::StatusOr<ThunkProto> ToProto() const override {
    return absl::UnimplementedError("TestThunk::ToProto is not implemented");
  }
};

class TestNestedThunk : public TestThunk {
 public:
  TestNestedThunk(ThunkInfo thunk_info, ThunkSequence nested)
      : TestThunk(thunk_info), nested_(std::move(nested)) {}

 protected:
  absl::Status WalkNested(Walker pre_order, Walker post_order) override {
    return nested_.WalkNested(pre_order, post_order);
  }

 private:
  ThunkSequence nested_;
};

TEST(ThunkTest, WalksInPreAndPostOrder) {
  auto make_info = [](std::string annotation) {
    Thunk::ThunkInfo info;
    info.profile_annotation = std::move(annotation);
    return info;
  };

  ThunkSequence nested;
  nested.push_back(std::make_unique<TestThunk>(make_info("grandchild")));

  ThunkSequence children;
  children.push_back(std::make_unique<TestThunk>(make_info("child")));
  children.push_back(std::make_unique<TestNestedThunk>(make_info("nested"),
                                                       std::move(nested)));

  TestNestedThunk root(make_info("root"), std::move(children));
  std::vector<std::string> visited;
  root.Walk(
      [&](const Thunk* thunk) {
        visited.push_back("pre " + std::string(thunk->profile_annotation()));
      },
      [&](const Thunk* thunk) {
        visited.push_back("post " + std::string(thunk->profile_annotation()));
      });

  EXPECT_THAT(visited,
              ElementsAre("pre root", "pre child", "post child", "pre nested",
                          "pre grandchild", "post grandchild", "post nested",
                          "post root"));
}

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

  EXPECT_THAT(GetMetadataListProtoFromThunkGraph(thunks), EqualsProto(R"pb(
                thunk_metadata {
                  thunk_info { thunk_id: 123 profile_annotation: "test_kernel" }
                  thunk_kind: "kKernel"
                }
              )pb"));
}

}  // namespace
}  // namespace xla::gpu
