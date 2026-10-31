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

#include "xla/pjrt/pjrt_compatibility_versions.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/pjrt_build_stamp_writer.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::tsl::proto_testing::EqualsProto;

TEST(PjRtCompatibilityVersionsTest, SetAndGetMosaicVersion) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK(SetMosaicVersion(proto, 12, 3));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  PjRtCompatibilityVersionProto expected;
  expected.set_name("mosaic");
  expected.set_major_version(12);
  expected.set_minor_version(3);
  EXPECT_THAT(GetMosaicVersion(stamp), IsOkAndHolds(EqualsProto(expected)));
}

TEST(PjRtCompatibilityVersionsTest,
     SetMosaicVersionDefaultsMinorVersionToZero) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK(SetMosaicVersion(proto, 12));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  PjRtCompatibilityVersionProto expected;
  expected.set_name("mosaic");
  expected.set_major_version(12);
  expected.set_minor_version(0);
  EXPECT_THAT(GetMosaicVersion(stamp), IsOkAndHolds(EqualsProto(expected)));
}

TEST(PjRtCompatibilityVersionsTest,
     SetMosaicVersionTwiceWithSameVersionReturnsAlreadyExists) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK(SetMosaicVersion(proto, 12, 3));

  EXPECT_THAT(SetMosaicVersion(proto, 12, 3),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("'mosaic' is already set to 12.3")));
  EXPECT_EQ(proto.compatibility_versions_size(), 1);
}

TEST(PjRtCompatibilityVersionsTest,
     SetMosaicVersionWithConflictingVersionReturnsAlreadyExists) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK(SetMosaicVersion(proto, 12, 0));

  EXPECT_THAT(SetMosaicVersion(proto, 13, 1),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("'mosaic' is already set to 12.0")));
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));
  PjRtCompatibilityVersionProto expected;
  expected.set_name("mosaic");
  expected.set_major_version(12);
  expected.set_minor_version(0);
  EXPECT_THAT(GetMosaicVersion(stamp), IsOkAndHolds(EqualsProto(expected)));
}

TEST(PjRtCompatibilityVersionsTest,
     SetMosaicVersionWithNegativeVersionReturnsInvalidArgument) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");

  EXPECT_THAT(SetMosaicVersion(proto, -1, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version numbers must be non-negative")));
  EXPECT_THAT(SetMosaicVersion(proto, 1, -2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Version numbers must be non-negative")));
  EXPECT_EQ(proto.compatibility_versions_size(), 0);
}

TEST(PjRtCompatibilityVersionsTest, GetMosaicVersionWhenUnsetReturnsNotFound) {
  PjRtBuildStampProto proto;
  proto.set_source_hash("hash1");
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_THAT(GetMosaicVersion(stamp),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Version 'mosaic' not found.")));
}

}  // namespace
}  // namespace xla
