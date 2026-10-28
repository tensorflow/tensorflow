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
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

static_assert(!std::is_copy_constructible_v<HloPassOptions>);
static_assert(std::is_move_constructible_v<HloPassOptions>);

HloPassOptions::ProtoMap MakeProtoMap() {
  HloPassOptions::ProtoMap proto;
  proto["flag"].set_bool_val(true);
  proto["count"].set_int_val(10);
  proto["unset"];
  return proto;
}

TEST(HloPassOptionsTest, DefaultConstructedIsEmpty) {
  HloPassOptions options;

  EXPECT_THAT(options.Get<bool>("flag", true), IsOkAndHolds(true));
  EXPECT_THAT(options.Get<int64_t>("limit", 42), IsOkAndHolds(42));
  EXPECT_THAT(options.CheckAllConsumed(), IsOk());
}

TEST(HloPassOptionsTest, CopiesProtoMap) {
  HloPassOptions options(MakeProtoMap());

  EXPECT_THAT(options.Get<bool>("flag", false), IsOkAndHolds(true));
  EXPECT_THAT(options.Get<int64_t>("count", 5), IsOkAndHolds(10));
  EXPECT_THAT(options.Get<int64_t>("missing", 5), IsOkAndHolds(5));
}

TEST(HloPassOptionsTest, InitializerListMapsIntegralTypes) {
  HloPassOptions options = {{"flag", true}, {"negative", -3}, {"unsigned", 7u}};

  EXPECT_THAT(options.Get<bool>("flag", false), IsOkAndHolds(true));
  EXPECT_THAT(options.Get<int64_t>("negative", 0), IsOkAndHolds(-3));
  EXPECT_THAT(options.Get<int64_t>("unsigned", 0), IsOkAndHolds(7));
  EXPECT_THAT(options.CheckAllConsumed(), IsOk());
}

TEST(HloPassOptionsDeathTest, InitializerListRejectsDuplicateName) {
  EXPECT_DEATH(HloPassOptions({{"flag", true}, {"flag", false}}),
               "Duplicate pass option 'flag'");
}

TEST(HloPassOptionsTest, TypeMismatchReturnsInvalidArgument) {
  HloPassOptions options(MakeProtoMap());

  EXPECT_THAT(options.Get<int64_t>("flag", 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("'flag' is not of type int64")));
  EXPECT_THAT(options.Get<bool>("count", false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("'count' is not of type bool")));
  EXPECT_THAT(options.Get<bool>("unset", false),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // Mistyped options are reported by Get(), not again as unconsumed.
  EXPECT_THAT(options.CheckAllConsumed(), IsOk());
}

TEST(HloPassOptionsTest, CheckAllConsumedDetectsUnconsumedKey) {
  HloPassOptions options = {{"used", true}, {"typo", 1}};

  EXPECT_THAT(options.Get<bool>("used", false), IsOkAndHolds(true));
  EXPECT_THAT(options.CheckAllConsumed(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown pass option 'typo'")));
}

TEST(HloPassOptionsTest, MoveKeepsConsumptionTracking) {
  HloPassOptions source = {{"used", true}, {"unused", 1}};
  EXPECT_THAT(source.Get<bool>("used", false), IsOkAndHolds(true));

  HloPassOptions moved = std::move(source);

  EXPECT_THAT(moved.CheckAllConsumed(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unknown pass option 'unused'")));
  EXPECT_THAT(moved.Get<int64_t>("unused", 0), IsOkAndHolds(1));
  EXPECT_THAT(moved.CheckAllConsumed(), IsOk());
}

}  // namespace
}  // namespace xla
