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

#include "xla/python/ifrt/abstract_array_spec.h"

#include <memory>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/abstract_array_spec.pb.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_spec.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;

using AbstractArraySpecTestParam =
    std::tuple<SerDesVersion, test_util::DeviceTestParam>;

class AbstractArraySpecTest
    : public testing::TestWithParam<AbstractArraySpecTestParam> {
 public:
  AbstractArraySpecTest()
      : version_(std::get<0>(GetParam())), fixture_(std::get<1>(GetParam())) {}

  SerDesVersion version() const { return version_; }

  Client* client() { return fixture_.client(); }
  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

 private:
  SerDesVersion version_;
  test_util::DeviceTestFixture fixture_;
};

TEST_P(AbstractArraySpecTest, CreateAndAccessors) {
  DType dtype(DType::kS32);
  Shape shape({4, 2});
  ShardingSpecRef sharding_spec = SingleDeviceShardingSpec::Create();
  MemoryKind memory_kind("device");
  auto layout = std::make_shared<xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(2));

  ASSERT_OK_AND_ASSIGN(AbstractArraySpec abstract_array_spec,
                       AbstractArraySpec::Create(dtype, shape, sharding_spec,
                                                 memory_kind, layout));

  EXPECT_EQ(abstract_array_spec.dtype(), dtype);
  EXPECT_EQ(abstract_array_spec.shape(), shape);
  EXPECT_EQ(abstract_array_spec.sharding_spec(), sharding_spec);
  EXPECT_EQ(abstract_array_spec.memory_kind(), memory_kind);
  EXPECT_EQ(abstract_array_spec.layout(), layout);
}

TEST_P(AbstractArraySpecTest, CreateNullShardingSpecFails) {
  DType dtype(DType::kS32);
  Shape shape({4, 2});
  MemoryKind memory_kind("device");
  ShardingSpecRef null_sharding_spec;

  EXPECT_THAT(AbstractArraySpec::Create(dtype, shape, null_sharding_spec,
                                        memory_kind, /*layout=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(AbstractArraySpecTest, EqualityAndHash) {
  DType dtype(DType::kS32);
  Shape shape({4, 2});
  ShardingSpecRef sharding_spec1 = SingleDeviceShardingSpec::Create();
  ShardingSpecRef sharding_spec2 = ConcreteEvenShardingSpec::Create(
      /*num_shards=*/2, /*shape=*/shape, /*shard_shape=*/Shape({2, 2}));
  MemoryKind memory_kind("device");
  auto layout = std::make_shared<xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(2));

  ASSERT_OK_AND_ASSIGN(AbstractArraySpec abstract_array_spec1,
                       AbstractArraySpec::Create(dtype, shape, sharding_spec1,
                                                 memory_kind, layout));

  ASSERT_OK_AND_ASSIGN(AbstractArraySpec abstract_array_spec1_duplicate,
                       AbstractArraySpec::Create(dtype, shape, sharding_spec1,
                                                 memory_kind, layout));

  ASSERT_OK_AND_ASSIGN(AbstractArraySpec abstract_array_spec2,
                       AbstractArraySpec::Create(dtype, shape, sharding_spec2,
                                                 memory_kind, layout));

  EXPECT_EQ(abstract_array_spec1, abstract_array_spec1_duplicate);
  EXPECT_NE(abstract_array_spec1, abstract_array_spec2);

  absl::flat_hash_set<AbstractArraySpec> set;
  set.insert(abstract_array_spec1);
  EXPECT_TRUE(set.contains(abstract_array_spec1_duplicate));
  EXPECT_FALSE(set.contains(abstract_array_spec2));
}

TEST_P(AbstractArraySpecTest, SupportsAbslHash) {
  Shape shape({4, 2});
  ASSERT_OK_AND_ASSIGN(
      AbstractArraySpec abstract_array_spec1,
      AbstractArraySpec::Create(DType(DType::kS32), shape,
                                SingleDeviceShardingSpec::Create(),
                                MemoryKind(), /*layout=*/nullptr));
  ASSERT_OK_AND_ASSIGN(
      AbstractArraySpec abstract_array_spec2,
      AbstractArraySpec::Create(
          DType(DType::kS32), shape,
          ConcreteEvenShardingSpec::Create(/*num_shards=*/2, /*shape=*/shape,
                                           /*shard_shape=*/Shape({2, 2})),
          MemoryKind("device"),
          std::make_shared<xla::PjRtLayout>(
              xla::LayoutUtil::MakeDescendingLayout(2))));

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {abstract_array_spec1, abstract_array_spec2}));
}

TEST_P(AbstractArraySpecTest, ToArraySpecAndRoundTrip) {
  DeviceListRef device_list = GetDevices({0, 1});
  DType dtype(DType::kS32);
  Shape shape({4, 2});
  Shape shard_shape({2, 2});
  ShardingSpecRef sharding_spec = ConcreteEvenShardingSpec::Create(
      /*num_shards=*/2, /*shape=*/shape, /*shard_shape=*/shard_shape);
  MemoryKind memory_kind("device");

  ASSERT_OK_AND_ASSIGN(
      AbstractArraySpec abstract_array_spec,
      AbstractArraySpec::Create(dtype, shape, sharding_spec, memory_kind,
                                /*layout=*/nullptr));

  ASSERT_OK_AND_ASSIGN(ArraySpec array_spec,
                       abstract_array_spec.ToArraySpec(device_list));

  EXPECT_EQ(array_spec.dtype, dtype);
  EXPECT_EQ(array_spec.shape, shape);
  EXPECT_EQ(*array_spec.sharding->devices(), *device_list);
  EXPECT_EQ(array_spec.sharding->memory_kind(), memory_kind);
  EXPECT_EQ(array_spec.layout, nullptr);

  ASSERT_OK_AND_ASSIGN(AbstractArraySpec roundtrip_abstract_array_spec,
                       array_spec.ToAbstractArraySpec());

  EXPECT_EQ(abstract_array_spec, roundtrip_abstract_array_spec);
}

TEST_P(AbstractArraySpecTest, ToCanonicalizedAbstractArraySpec) {
  DeviceListRef device_list = GetDevices({0, 1});
  DType dtype(DType::kS32);
  Shape shape({4, 2});
  Shape shard_shape({2, 2});
  MemoryKind memory_kind;
  ShardingRef sharding = ConcreteEvenSharding::Create(
      device_list, memory_kind, /*shape=*/shape, /*shard_shape=*/shard_shape);

  ArraySpec array_spec = {dtype, shape, sharding};
  ASSERT_OK_AND_ASSIGN(AbstractArraySpec abstract_array_spec,
                       array_spec.ToAbstractArraySpec());

  EXPECT_EQ(abstract_array_spec.dtype(), dtype);
  EXPECT_EQ(abstract_array_spec.shape(), shape);
  EXPECT_TRUE(abstract_array_spec.sharding_spec()->HasSamePartitioning(
      *sharding->sharding_spec()));
  EXPECT_EQ(abstract_array_spec.memory_kind(), MemoryKind("host"));
}

TEST_P(AbstractArraySpecTest, ToFromProto) {
  DType dtype(DType::kS32);
  Shape shape({4, 2});
  Shape shard_shape({2, 2});
  ShardingSpecRef sharding_spec = ConcreteEvenShardingSpec::Create(
      /*num_shards=*/2, /*shape=*/shape, /*shard_shape=*/shard_shape);
  MemoryKind memory_kind("device");
  auto layout = std::make_shared<xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(2));

  ASSERT_OK_AND_ASSIGN(AbstractArraySpec abstract_array_spec,
                       AbstractArraySpec::Create(dtype, shape, sharding_spec,
                                                 memory_kind, layout));

  if (version().version_number() >= SerDesVersionNumber(5)) {
    ASSERT_OK_AND_ASSIGN(AbstractArraySpecProto proto,
                         abstract_array_spec.ToProto(version()));
    ASSERT_OK_AND_ASSIGN(AbstractArraySpec abstract_array_spec_copy,
                         AbstractArraySpec::FromProto(proto));
    EXPECT_EQ(abstract_array_spec, abstract_array_spec_copy);
  } else {
    EXPECT_THAT(abstract_array_spec.ToProto(version()),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }
}

INSTANTIATE_TEST_SUITE_P(
    AbstractArraySpecTests, AbstractArraySpecTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(test_util::DeviceTestParam{
                         /*num_devices=*/2, /*num_addressable_devices=*/2})),
    [](const testing::TestParamInfo<AbstractArraySpecTestParam>& info) {
      return absl::StrCat(std::get<0>(info.param).version_number().value());
    });

}  // namespace
}  // namespace ifrt
}  // namespace xla
