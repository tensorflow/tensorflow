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

#include "xla/pjrt/pjrt_device_dimensions.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "xla/pjrt/proto/pjrt_device_dimensions.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

TEST(PjRtDeviceDimensionsTest, Equality) {
  EXPECT_EQ((PjRtDeviceDimensions{1, 2, 3}), (PjRtDeviceDimensions{1, 2, 3}));
  EXPECT_NE((PjRtDeviceDimensions{1, 2, 3}), (PjRtDeviceDimensions{1, 2, 4}));
}

TEST(PjRtDeviceDimensionsTest, LessThan) {
  // Same size comparisons
  EXPECT_TRUE((PjRtDeviceDimensions{1, 2, 3}) <
              (PjRtDeviceDimensions{1, 2, 4}));
  EXPECT_TRUE((PjRtDeviceDimensions{1, 2, 3}) <
              (PjRtDeviceDimensions{1, 3, 0}));
  EXPECT_TRUE((PjRtDeviceDimensions{1, 2, 3}) <
              (PjRtDeviceDimensions{2, 0, 0}));
  EXPECT_FALSE((PjRtDeviceDimensions{1, 2, 3}) <
               (PjRtDeviceDimensions{1, 2, 3}));
  EXPECT_FALSE((PjRtDeviceDimensions{1, 2, 4}) <
               (PjRtDeviceDimensions{1, 2, 3}));

  // Different size comparisons (shorter is less than longer if prefixes match)
  EXPECT_TRUE((PjRtDeviceDimensions{1, 2}) < (PjRtDeviceDimensions{1, 2, 3}));
  EXPECT_FALSE((PjRtDeviceDimensions{1, 2, 3}) < (PjRtDeviceDimensions{1, 2}));
  EXPECT_TRUE((PjRtDeviceDimensions{}) < (PjRtDeviceDimensions{1}));
  EXPECT_FALSE((PjRtDeviceDimensions{1}) < (PjRtDeviceDimensions{}));

  // Different size comparisons with different prefixes
  EXPECT_TRUE((PjRtDeviceDimensions{1, 1}) < (PjRtDeviceDimensions{1, 2, 3}));
  EXPECT_TRUE((PjRtDeviceDimensions{0, 2}) < (PjRtDeviceDimensions{1, 2, 3}));
}

TEST(PjRtDeviceDimensionsTest, Ostream) {
  std::stringstream ss;
  ss << PjRtDeviceDimensions{1, 2, 3};
  EXPECT_EQ(ss.str(), "1,2,3");
}

TEST(PjRtDeviceDimensionsTest, AbslHashValue) {
  absl::flat_hash_set<PjRtDeviceDimensions> hash_set;
  hash_set.insert({1, 2, 3});
  hash_set.insert({0, 0, 0});
  hash_set.insert({1, 2, 3});  // Inserting again should not change size

  EXPECT_EQ(hash_set.size(), 2);
  EXPECT_TRUE(hash_set.contains({1, 2, 3}));
  EXPECT_TRUE(hash_set.contains({0, 0, 0}));
  EXPECT_FALSE(hash_set.contains({1, 2, 4}));
}

TEST(PjRtDeviceDimensionsTest, FromProto) {
  PjRtDeviceDimensionsProto proto;
  proto.add_dimensions(1);
  proto.add_dimensions(2);
  proto.add_dimensions(3);
  TF_ASSERT_OK_AND_ASSIGN(PjRtDeviceDimensions dims,
                          PjRtDeviceDimensions::FromProto(proto));
  EXPECT_EQ(dims, PjRtDeviceDimensions({1, 2, 3}));
}

TEST(PjRtDeviceDimensionsTest, ToProto) {
  PjRtDeviceDimensions bounds = {1, 2, 3};
  PjRtDeviceDimensionsProto proto = bounds.ToProto();
  EXPECT_THAT(proto.dimensions(), testing::ElementsAre(1, 2, 3));
}

TEST(AbslParseFlagTest, ValidInputs) {
  PjRtDeviceDimensions bounds;
  std::string err;

  EXPECT_TRUE(AbslParseFlag("1,2,3", &bounds, &err));
  EXPECT_EQ(bounds, (PjRtDeviceDimensions{1, 2, 3}));
  EXPECT_EQ(err, "");

  EXPECT_TRUE(AbslParseFlag("1,2", &bounds, &err));
  EXPECT_EQ(bounds, (PjRtDeviceDimensions{1, 2}));
  EXPECT_EQ(err, "");

  EXPECT_TRUE(AbslParseFlag("1,2,3,4", &bounds, &err));
  EXPECT_EQ(bounds, (PjRtDeviceDimensions{1, 2, 3, 4}));
  EXPECT_EQ(err, "");

  EXPECT_TRUE(AbslParseFlag("", &bounds, &err));
  EXPECT_EQ(bounds, (PjRtDeviceDimensions{}));
  EXPECT_EQ(err, "");
}

TEST(AbslParseFlagTest, InvalidInputs) {
  PjRtDeviceDimensions bounds;
  std::string err;

  EXPECT_FALSE(AbslParseFlag("1,a,3", &bounds, &err));
  EXPECT_THAT(err, HasSubstr("Number parsing error"));

  EXPECT_FALSE(AbslParseFlag("1,2.5,3", &bounds, &err));
  EXPECT_THAT(err, HasSubstr("Number parsing error"));
}

TEST(AbslUnparseFlagTest, ConvertsCorrectly) {
  EXPECT_EQ(AbslUnparseFlag(PjRtDeviceDimensions{1, 2, 3}), "1,2,3");
  EXPECT_EQ(AbslUnparseFlag(PjRtDeviceDimensions{0, 0, 0}), "0,0,0");
}

TEST(PjRtDeviceDimensionsTest, Iterator) {
  const PjRtDeviceDimensions const_dims = {4, 5, 6};
  int i = 4;
  for (int d : const_dims) {
    EXPECT_EQ(d, i);
    i++;
  }

  PjRtDeviceDimensions mutable_dims = {7, 8, 9};
  for (int& d : mutable_dims) {
    d *= 2;
  }
  EXPECT_EQ(mutable_dims, (PjRtDeviceDimensions{14, 16, 18}));
}

TEST(PjRtDeviceDimensionsTest, SubscriptAccess) {
  PjRtDeviceDimensions dims = {10, 20, 30};
  EXPECT_EQ(dims[0], 10);
  EXPECT_EQ(dims[1], 20);
  EXPECT_EQ(dims[2], 30);

  dims[1] = 25;
  EXPECT_EQ(dims[1], 25);
  EXPECT_EQ(dims, (PjRtDeviceDimensions{10, 25, 30}));

  const PjRtDeviceDimensions const_dims = {1, 2, 3};
  EXPECT_EQ(const_dims[0], 1);
}

TEST(PjRtDeviceDimensionsTest, Size) {
  EXPECT_EQ((PjRtDeviceDimensions{1, 2, 3}).size(), 3);
  EXPECT_EQ((PjRtDeviceDimensions{1, 2}).size(), 2);
  EXPECT_EQ((PjRtDeviceDimensions{}).size(), 0);
}

}  // namespace
}  // namespace xla
