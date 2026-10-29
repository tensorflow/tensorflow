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

#include "xla/hlo/pass/hlo_pass_options.h"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Optional;

static_assert(!std::is_copy_constructible_v<HloPassOptions>);
static_assert(std::is_move_constructible_v<HloPassOptions>);

HloPassOptions::ProtoMap MakeProtoMap() {
  HloPassOptions::ProtoMap proto;
  proto["flag"].set_bool_val(true);
  proto["count"].set_int_val(10);
  return proto;
}

TEST(HloPassOptionsTest, DefaultConstructedIsEmpty) {
  HloPassOptions options;

  EXPECT_EQ(options.Get<bool>("flag"), std::nullopt);
  EXPECT_EQ(options.Get<int64_t>("limit"), std::nullopt);
  EXPECT_THAT(options.CheckAllConsumed(), IsOk());
}

TEST(HloPassOptionsTest, CopiesProtoMap) {
  HloPassOptions options(MakeProtoMap());

  EXPECT_THAT(options.Get<bool>("flag"), Optional(true));
  EXPECT_THAT(options.Get<int64_t>("count"), Optional(10));
  EXPECT_EQ(options.Get<int64_t>("missing"), std::nullopt);
}

TEST(HloPassOptionsTest, InitializerListMapsIntegralTypes) {
  HloPassOptions options = {{"flag", true}, {"negative", -3}, {"unsigned", 7u}};

  EXPECT_THAT(options.Get<bool>("flag"), Optional(true));
  EXPECT_THAT(options.Get<int64_t>("negative"), Optional(-3));
  EXPECT_THAT(options.Get<int64_t>("unsigned"), Optional(7));
  EXPECT_THAT(options.CheckAllConsumed(), IsOk());
}

TEST(HloPassOptionsDeathTest, InitializerListRejectsDuplicateName) {
  EXPECT_DEATH(HloPassOptions({{"flag", true}, {"flag", false}}),
               "Duplicate pass option 'flag'");
}

TEST(HloPassOptionsTest, TypeMismatchIsReportedByCheckAllConsumed) {
  HloPassOptions options = {{"flag", true}};

  EXPECT_EQ(options.Get<int64_t>("flag"), std::nullopt);
  EXPECT_THAT(
      options.CheckAllConsumed(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Pass option 'flag' (bool) was not consumed")));
}

TEST(HloPassOptionsTest, UnsetValueIsReportedByCheckAllConsumed) {
  HloPassOptions::ProtoMap proto;
  proto["unset"];
  HloPassOptions options(proto);

  EXPECT_EQ(options.Get<bool>("unset"), std::nullopt);
  EXPECT_THAT(options.CheckAllConsumed(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("'unset' (unset)")));
}

TEST(HloPassOptionsTest, CheckAllConsumedDetectsUnconsumedKey) {
  HloPassOptions options = {{"used", true}, {"typo", 1}};

  EXPECT_THAT(options.Get<bool>("used"), Optional(true));
  EXPECT_THAT(
      options.CheckAllConsumed(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Pass option 'typo' (int64) was not consumed")));
}

TEST(HloPassOptionsTest, MoveKeepsConsumptionTracking) {
  HloPassOptions source = {{"used", true}, {"unused", 1}};
  EXPECT_THAT(source.Get<bool>("used"), Optional(true));

  HloPassOptions moved = std::move(source);

  EXPECT_THAT(
      moved.CheckAllConsumed(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("'unused'")));
  EXPECT_THAT(moved.Get<int64_t>("unused"), Optional(1));
  EXPECT_THAT(moved.CheckAllConsumed(), IsOk());
}

TEST(HloPassOptionsTest, ToProtoRoundTripsWithoutConsuming) {
  HloPassOptions options = {{"flag", true}, {"count", 10}};

  HloPassOptions round_tripped(options.ToProto());

  EXPECT_THAT(round_tripped.Get<bool>("flag"), Optional(true));
  EXPECT_THAT(round_tripped.Get<int64_t>("count"), Optional(10));
  EXPECT_THAT(round_tripped.CheckAllConsumed(), IsOk());
  EXPECT_THAT(options.CheckAllConsumed(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
