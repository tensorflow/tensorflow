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

#include "xla/backends/gpu/libraries/cub/cub_scratch_size_deviceless_lookup.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/libraries/cub/scratch_space_lookup_table.pb.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/util/proto/parse_text_proto.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Optional;
using ::tsl::proto_testing::ParseTextProtoOrDie;

constexpr stream_executor::SemanticVersion kCubVersion1_15_0{1, 15, 0};
constexpr stream_executor::SemanticVersion kCubVersion1_16_0{1, 16, 0};

TEST(CubScratchSizeDevicelessLookupTest, LookupExactMatch) {
  ASSERT_OK_AND_ASSIGN(auto lookup,
                       CubScratchSizeDevicelessLookup::CreateFromProto(
                           ParseTextProtoOrDie<CubScratchSizeLookupTable>(R"pb(
                             entries {
                               cub_version: "1.15.0"
                               device_name: "sm_80"
                               key_type_size: 4
                               value_type_size: 4
                               scratch_size_recordings {
                                 num_items: 100
                                 scratch_space_bytes: 1024
                               }
                               scratch_size_recordings {
                                 num_items: 200
                                 scratch_space_bytes: 2048
                               }
                             }
                           )pb")));

  EXPECT_THAT(lookup.Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                            /*value_type_size=*/4, /*num_items=*/100),
              Optional(1024));
  EXPECT_THAT(lookup.Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                            /*value_type_size=*/4, /*num_items=*/200),
              Optional(2048));
}

TEST(CubScratchSizeDevicelessLookupTest, LookupClosestHigherMatch) {
  ASSERT_OK_AND_ASSIGN(auto lookup,
                       CubScratchSizeDevicelessLookup::CreateFromProto(
                           ParseTextProtoOrDie<CubScratchSizeLookupTable>(R"pb(
                             entries {
                               cub_version: "1.15.0"
                               device_name: "sm_80"
                               key_type_size: 4
                               value_type_size: 4
                               scratch_size_recordings {
                                 num_items: 100
                                 scratch_space_bytes: 1024
                               }
                               scratch_size_recordings {
                                 num_items: 200
                                 scratch_space_bytes: 2048
                               }
                             }
                           )pb")));

  // Request 50, should give 100's scratch (1024)
  EXPECT_THAT(lookup.Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                            /*value_type_size=*/4, /*num_items=*/50),
              Optional(1024));

  // Request 150, should give 200's scratch (2048)
  EXPECT_THAT(lookup.Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                            /*value_type_size=*/4, /*num_items=*/150),
              Optional(2048));
}

TEST(CubScratchSizeDevicelessLookupTest, LookupNoEntryForParams) {
  ASSERT_OK_AND_ASSIGN(auto lookup,
                       CubScratchSizeDevicelessLookup::CreateFromProto(
                           ParseTextProtoOrDie<CubScratchSizeLookupTable>(R"pb(
                             entries {
                               cub_version: "1.15.0"
                               device_name: "sm_80"
                               key_type_size: 4
                               value_type_size: 4
                               scratch_size_recordings {
                                 num_items: 100
                                 scratch_space_bytes: 1024
                               }
                             }
                           )pb")));

  // Mismatch version
  EXPECT_FALSE(lookup
                   .Lookup(kCubVersion1_16_0, "sm_80", /*key_type_size=*/4,
                           /*value_type_size=*/4, /*num_items=*/100)
                   .has_value());
  // Mismatch device
  EXPECT_FALSE(lookup
                   .Lookup(kCubVersion1_15_0, "sm_90", /*key_type_size=*/4,
                           /*value_type_size=*/4, /*num_items=*/100)
                   .has_value());
  // Mismatch key size
  EXPECT_FALSE(
      lookup.Lookup(kCubVersion1_15_0, "sm_80", 8, 4, 100).has_value());
}

TEST(CubScratchSizeDevicelessLookupTest, LookupItemsExceedRecordings) {
  ASSERT_OK_AND_ASSIGN(auto lookup,
                       CubScratchSizeDevicelessLookup::CreateFromProto(
                           ParseTextProtoOrDie<CubScratchSizeLookupTable>(R"pb(
                             entries {
                               cub_version: "1.15.0"
                               device_name: "sm_80"
                               key_type_size: 4
                               value_type_size: 4
                               scratch_size_recordings {
                                 num_items: 100
                                 scratch_space_bytes: 1024
                               }
                               scratch_size_recordings {
                                 num_items: 200
                                 scratch_space_bytes: 2048
                               }
                             }
                           )pb")));

  // Request 300 (greater than all recorded sizes), should return nullopt
  EXPECT_FALSE(lookup
                   .Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                           /*value_type_size=*/4, /*num_items=*/300)
                   .has_value());
}

TEST(CubScratchSizeDevicelessLookupTest, CanLookupTest) {
  ASSERT_OK_AND_ASSIGN(auto lookup,
                       CubScratchSizeDevicelessLookup::CreateFromProto(
                           ParseTextProtoOrDie<CubScratchSizeLookupTable>(R"pb(
                             entries {
                               cub_version: "1.15.0"
                               device_name: "sm_80"
                               key_type_size: 4
                               value_type_size: 4
                               scratch_size_recordings {
                                 num_items: 100
                                 scratch_space_bytes: 1024
                               }
                               scratch_size_recordings {
                                 num_items: 200
                                 scratch_space_bytes: 2048
                               }
                             }
                           )pb")));

  // Exact match for num_items
  EXPECT_TRUE(lookup.CanLookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                               /*value_type_size=*/4, /*num_items=*/100));
  EXPECT_TRUE(lookup.CanLookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                               /*value_type_size=*/4, /*num_items=*/200));

  // Less than max recorded
  EXPECT_TRUE(lookup.CanLookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                               /*value_type_size=*/4, /*num_items=*/50));
  EXPECT_TRUE(lookup.CanLookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                               /*value_type_size=*/4, /*num_items=*/150));

  // Greater than max recorded
  EXPECT_FALSE(lookup.CanLookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                                /*value_type_size=*/4, /*num_items=*/300));

  // Mismatched params
  EXPECT_FALSE(lookup.CanLookup(kCubVersion1_16_0, "sm_80", /*key_type_size=*/4,
                                /*value_type_size=*/4, /*num_items=*/100));
  EXPECT_FALSE(lookup.CanLookup(kCubVersion1_15_0, "sm_90", /*key_type_size=*/4,
                                /*value_type_size=*/4, /*num_items=*/100));
  EXPECT_FALSE(lookup.CanLookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/8,
                                /*value_type_size=*/4, /*num_items=*/100));
}

TEST(CubScratchSizeDevicelessLookupTest, LookupWithBatchSize) {
  ASSERT_OK_AND_ASSIGN(auto lookup,
                       CubScratchSizeDevicelessLookup::CreateFromProto(
                           ParseTextProtoOrDie<CubScratchSizeLookupTable>(R"pb(
                             entries {
                               cub_version: "1.15.0"
                               device_name: "sm_80"
                               key_type_size: 4
                               value_type_size: 4
                               is_segmented: true
                               scratch_size_recordings {
                                 num_items: 100
                                 scratch_space_bytes: 1024
                               }
                               scratch_size_recordings {
                                 num_items: 200
                                 scratch_space_bytes: 2048
                               }
                             }
                             entries {
                               cub_version: "1.15.0"
                               device_name: "sm_80"
                               key_type_size: 4
                               value_type_size: 4
                               is_segmented: false
                               scratch_size_recordings {
                                 num_items: 100
                                 scratch_space_bytes: 2048
                               }
                               scratch_size_recordings {
                                 num_items: 200
                                 scratch_space_bytes: 4096
                               }
                             }
                           )pb")));

  // batch_size = 1 (default), should return exact recorded size
  EXPECT_THAT(lookup.Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                            /*value_type_size=*/4, /*num_items=*/100),
              Optional(2048));
  EXPECT_THAT(lookup.Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                            /*value_type_size=*/4, /*num_items=*/100,
                            /*batch_size=*/1),
              Optional(2048));

  // batch_size = 2, base = 1024
  // padded to 1028
  // offsets = (2 + 1) * 4 = 12
  // total = 1040
  EXPECT_THAT(lookup.Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                            /*value_type_size=*/4, /*num_items=*/100,
                            /*batch_size=*/2),
              Optional(1040));

  // batch_size = 2, base = 2048 (for num_items = 200)
  // padded to 2052
  // offsets = (2 + 1) * 4 = 12
  // total = 2064
  EXPECT_THAT(lookup.Lookup(kCubVersion1_15_0, "sm_80", /*key_type_size=*/4,
                            /*value_type_size=*/4, /*num_items=*/200,
                            /*batch_size=*/2),
              Optional(2064));
}

TEST(CubScratchSizeDevicelessLookupTest, CreateFailsIfRecordingsNotSorted) {
  EXPECT_THAT(CubScratchSizeDevicelessLookup::CreateFromProto(
                  ParseTextProtoOrDie<CubScratchSizeLookupTable>(R"pb(
                    entries {
                      cub_version: "1.15.0"
                      device_name: "sm_80"
                      key_type_size: 4
                      value_type_size: 4
                      scratch_size_recordings {
                        num_items: 200
                        scratch_space_bytes: 2048
                      }
                      scratch_size_recordings {
                        num_items: 100
                        scratch_space_bytes: 1024
                      }
                    }
                  )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CubScratchSizeDevicelessLookupTest, CanBeLoadedFromBundledData) {
  absl::StatusOr<const CubScratchSizeDevicelessLookup&> lookup =
      CubScratchSizeDevicelessLookup::GetInstance();
  ASSERT_OK(lookup) << lookup.status().message();
}

}  // namespace
}  // namespace xla::gpu
