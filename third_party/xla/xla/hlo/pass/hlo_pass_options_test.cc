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
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::SizeIs;
using ::tsl::proto_testing::EqualsProto;

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

// HloPassOptionValue (bool and int64 fields with presence) and
// HloPipelineNodeProto (unrepresentable int32 and map fields) serve as option
// messages.

TEST(HloPassOptionsTest, FromMessageMapsSetFields) {
  HloPassOptionValue message;
  // Explicitly set to the default value; still mapped.
  message.set_bool_val(false);

  HloPassOptions options(message);

  EXPECT_THAT(options.ToProto(), SizeIs(1));
  EXPECT_THAT(options.Get<bool>("bool_val"), Optional(false));
}

TEST(HloPassOptionsTest, FromMessageSkipsUnsetFields) {
  EXPECT_THAT(HloPassOptions(HloPassOptionValue()).ToProto(), IsEmpty());
}

TEST(HloPassOptionsDeathTest, FromMessageRejectsUnsupportedType) {
  HloPipelineNodeProto message;
  message.set_max_fixed_point_iterations(2);

  EXPECT_DEATH(HloPassOptions{message},
               "max_fixed_point_iterations' of type int32");
}

TEST(HloPassOptionsDeathTest, FromMessageRejectsRepeatedField) {
  HloPipelineNodeProto message;
  (*message.mutable_options())["flag"].set_bool_val(true);

  EXPECT_DEATH(HloPassOptions{message}, "Repeated field '.*options'");
}

TEST(HloPassOptionsTest, MergeIntoSetsNamedField) {
  HloPassOptions options = {{"int_val", 17}};
  HloPassOptionValue message;

  options.MergeInto(message);

  EXPECT_THAT(message, EqualsProto(R"pb(int_val: 17)pb"));
  EXPECT_THAT(options.CheckAllConsumed(), IsOk());
}

TEST(HloPassOptionsTest, MergeIntoLeavesMistypedOptionUnconsumed) {
  HloPassOptions options = {{"bool_val", 1}};
  HloPassOptionValue message;

  options.MergeInto(message);

  EXPECT_FALSE(message.has_bool_val());
  EXPECT_THAT(
      options.CheckAllConsumed(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("'bool_val'")));
}

TEST(HloPassOptionsTest, MergeIntoLeavesUnrepresentableFieldOptionUnconsumed) {
  HloPassOptions options = {{"max_fixed_point_iterations", 1}, {"options", 1}};
  HloPipelineNodeProto message;

  options.MergeInto(message);

  EXPECT_THAT(message, EqualsProto(""));
  EXPECT_THAT(options.CheckAllConsumed(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HloPassOptionsTest, MergeIntoLeavesUnknownOptionUnconsumed) {
  HloPassOptions options = {{"no_such_field", true}};
  HloPassOptionValue message;

  options.MergeInto(message);

  EXPECT_THAT(options.CheckAllConsumed(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("'no_such_field'")));
}

TEST(HloPassOptionsTest, MessageRoundTrip) {
  HloPassOptionValue message;
  message.set_int_val(4);
  HloPassOptions options(message);

  HloPassOptionValue round_tripped;
  options.MergeInto(round_tripped);

  EXPECT_THAT(round_tripped, EqualsProto(message));
  EXPECT_THAT(options.CheckAllConsumed(), IsOk());
}

}  // namespace
}  // namespace xla
