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

#include "xla/pjrt/pjrt_build_stamp_writer.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"

namespace xla {
namespace {

// Tests for `SetMosaicVersion` is in `pjrt_compatibility_versions_test.cc`.

using ::testing::IsEmpty;

TEST(PjRtBuildStampWriterTest, ProtoFromBuildDataPopulatesTimestamp) {
  PjRtBuildStampProto proto = PjRtBuildStampProtoFromBuildData();
  EXPECT_NE(proto.vcs_revision(), "<unknown>");

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::FromProto(std::move(proto)));

  EXPECT_NE(stamp.build_timestamp(), absl::InfinitePast());
  EXPECT_NE(stamp.vcs_revision(), "<unknown>");
  EXPECT_THAT(stamp.source_hash(), IsEmpty());
  EXPECT_THAT(stamp.ToProto().source_hash(), IsEmpty());
}

}  // namespace
}  // namespace xla
