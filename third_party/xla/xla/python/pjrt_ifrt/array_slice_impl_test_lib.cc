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

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/pjrt_ifrt/array_test_util.h"
#include "xla/python/pjrt_ifrt/xla_sharding_spec.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;

class ArraySliceTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(client_, test_util::GetClient());
  }

  absl::StatusOr<xla::ifrt::ShardingRef> MakeSharding(
      xla::ifrt::ShardingSpecRef sharding_spec,
      xla::ifrt::MemoryKind memory_kind) {
    std::vector<xla::ifrt::Device*> devices;
    devices.reserve(sharding_spec->num_shards());
    for (int i = 0; i < sharding_spec->num_shards(); ++i) {
      devices.push_back(client_->addressable_devices()[i]);
    }
    ABSL_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                     client_->MakeDeviceList(devices));
    return sharding_spec->ToSharding(std::move(device_list), memory_kind);
  }

  std::shared_ptr<Client> client_;
};

struct ArraySliceTestParam {
  absl::string_view name;
  Shape shape;
  xla::ifrt::ShardingSpecRef sharding;
  IndexDomain index_domain;
  xla::ifrt::ShardingSpecRef slice_sharding;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ArraySliceTestParam& param) {
    sink.Append(param.name);
  }
};

class ArraySliceParameterizedTest
    : public ArraySliceTest,
      public testing::WithParamInterface<std::tuple<
          ArraySliceTestParam, xla::PrimitiveType, xla::ifrt::MemoryKind>> {};

TEST_P(ArraySliceParameterizedTest, CopySlice) {
  const auto& [param, primitive_type, memory_kind] = GetParam();
  if (client_->addressable_devices().size() < param.sharding->num_shards()) {
    GTEST_SKIP() << "Insufficient devices for the test case";
  }

  ASSERT_OK_AND_ASSIGN(
      const xla::Literal literal,
      test_util::CreateIotaLiteral(primitive_type, param.shape.dims()));

  ASSERT_OK_AND_ASSIGN(const xla::ifrt::ShardingRef sharding,
                       MakeSharding(param.sharding, memory_kind));
  ASSERT_OK_AND_ASSIGN(const xla::ifrt::ShardingRef expected_slice_sharding,
                       MakeSharding(param.slice_sharding, memory_kind));

  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef array,
      test_util::MakeArrayFromLiteral(client_.get(), literal, sharding));

  ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> sliced,
      client_->CopyArraySlices(absl::MakeSpan(&array, 1),
                               absl::MakeSpan(&param.index_domain, 1)));
  ASSERT_EQ(sliced.size(), 1);
  EXPECT_EQ(sliced[0]->shape(), param.index_domain.shape());

  ASSERT_OK_AND_ASSIGN(const std::vector<IndexDomain> actual_index_domains,
                       sliced[0]->sharding().IndexDomains(
                           sliced[0]->shape(),
                           xla::ifrt::SingleDeviceShardSemantics::kAllShards));
  ASSERT_OK_AND_ASSIGN(const std::vector<IndexDomain> expected_index_domains,
                       expected_slice_sharding->IndexDomains(
                           param.index_domain.shape(),
                           xla::ifrt::SingleDeviceShardSemantics::kAllShards));
  EXPECT_EQ(actual_index_domains, expected_index_domains);

  const Index limit =
      param.index_domain.origin() + Index(param.index_domain.shape().dims());
  const xla::Literal expected =
      literal.Slice(param.index_domain.origin().elements(), limit.elements());
  EXPECT_THAT(test_util::CopyArrayToLiteral(sliced[0]),
              IsOkAndHolds(Eq(std::cref(expected))));
}

TEST_P(ArraySliceParameterizedTest, UpdateSlice) {
  const auto& [param, primitive_type, memory_kind] = GetParam();
  if (client_->addressable_devices().size() < param.sharding->num_shards()) {
    GTEST_SKIP() << "Insufficient devices for the test case";
  }

  ASSERT_OK_AND_ASSIGN(
      const xla::Literal target_literal,
      test_util::CreateIotaLiteral(primitive_type, param.shape.dims()));
  ASSERT_OK_AND_ASSIGN(const xla::Literal slice_literal,
                       test_util::CreateIotaLiteral(
                           primitive_type, param.index_domain.shape().dims(),
                           /*start=*/3));

  ASSERT_OK_AND_ASSIGN(const xla::ifrt::ShardingRef sharding,
                       MakeSharding(param.sharding, memory_kind));
  ASSERT_OK_AND_ASSIGN(const xla::ifrt::ShardingRef slice_sharding,
                       MakeSharding(param.slice_sharding, memory_kind));

  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef target,
      test_util::MakeArrayFromLiteral(client_.get(), target_literal, sharding));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef slice,
                       test_util::MakeArrayFromLiteral(
                           client_.get(), slice_literal, slice_sharding));

  const Index origin = param.index_domain.origin();
  ASSERT_OK_AND_ASSIGN(std::vector<xla::ifrt::ArrayRef> updated,
                       client_->UpdateArraySlices(absl::MakeSpan(&target, 1),
                                                  absl::MakeSpan(&slice, 1),
                                                  absl::MakeSpan(&origin, 1)));
  ASSERT_EQ(updated.size(), 1);
  EXPECT_TRUE(target->IsDeleted());
  EXPECT_FALSE(slice->IsDeleted());

  ASSERT_OK_AND_ASSIGN(const xla::Literal actual_literal,
                       test_util::CopyArrayToLiteral(updated[0]));

  const Index limit = origin + Index(param.index_domain.shape().dims());
  EXPECT_EQ(actual_literal.Slice(origin.elements(), limit.elements()),
            slice_literal);

  xla::Literal expected = target_literal.Clone();
  ASSERT_OK(expected.CopySliceFrom(
      slice_literal,
      /*src_base=*/
      Index::Zeros(slice_literal.shape().dimensions().size()).elements(),
      /*dest_base=*/origin.elements(),
      /*copy_size=*/param.index_domain.shape().dims()));
  EXPECT_EQ(actual_literal, expected);
}

INSTANTIATE_TEST_SUITE_P(
    ArraySliceParameterizedTest, ArraySliceParameterizedTest,
    testing::Combine(     //
        testing::Values(  //
            ArraySliceTestParam{
                /*name=*/"SingleDevice",
                /*shape=*/Shape({32, 256}),
                /*sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    1, xla::HloSharding::Replicate()),
                /*index_domain=*/IndexDomain(Index({8, 128}), Shape({16, 128})),
                /*slice_sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    1, xla::HloSharding::Replicate()),
            },
            ArraySliceTestParam{
                /*name=*/"MultiDeviceAllShards",
                /*shape=*/Shape({64, 256}),
                /*sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    2, xla::HloSharding::IotaTile({2, 1})),
                /*index_domain=*/IndexDomain(Index({16, 0}), Shape({32, 128})),
                /*slice_sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    2, xla::HloSharding::IotaTile({2, 1})),
            },
            ArraySliceTestParam{
                /*name=*/"MultiDeviceSingleShardSubset",
                /*shape=*/Shape({64, 256}),
                /*sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    2, xla::HloSharding::IotaTile({2, 1})),
                /*index_domain=*/IndexDomain(Index({8, 0}), Shape({16, 128})),
                /*slice_sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    1, xla::HloSharding::Replicate()),
            },
            ArraySliceTestParam{
                /*name=*/"MultiDeviceMultiShardSubset",
                /*shape=*/Shape({128, 256}),
                /*sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    4, xla::HloSharding::IotaTile({4, 1})),
                /*index_domain=*/IndexDomain(Index({16, 0}), Shape({32, 128})),
                /*slice_sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    2, xla::HloSharding::IotaTile({2, 1})),
            },
            ArraySliceTestParam{
                /*name=*/"FullyReplicated",
                /*shape=*/Shape({32, 256}),
                /*sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    4, xla::HloSharding::Replicate()),
                /*index_domain=*/IndexDomain(Index({8, 128}), Shape({16, 128})),
                /*slice_sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    4, xla::HloSharding::Replicate()),
            },
            ArraySliceTestParam{
                /*name=*/"PartiallyReplicatedAllShards",
                /*shape=*/Shape({64, 256}),
                /*sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    4, xla::HloSharding::PartialTile(  //
                           xla::TileAssignment({2, 1, 2}))),
                /*index_domain=*/IndexDomain(Index({16, 0}), Shape({32, 128})),
                /*slice_sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    4, xla::HloSharding::PartialTile(  //
                           xla::TileAssignment({2, 1, 2}))),
            },
            ArraySliceTestParam{
                /*name=*/"PartiallyReplicatedSubsetOfShards",
                /*shape=*/Shape({64, 256}),
                /*sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    4, xla::HloSharding::PartialTile(  //
                           xla::TileAssignment({2, 1, 2}))),
                /*index_domain=*/IndexDomain(Index({8, 0}), Shape({16, 128})),
                /*slice_sharding=*/
                xla::ifrt::HloShardingSpec::Create(
                    2, xla::HloSharding::Replicate()),
            }),
        testing::Values(xla::S4, xla::S32),
        testing::Values(xla::ifrt::MemoryKind("pinned_host"),
                        xla::ifrt::MemoryKind("unpinned_host"))),
    ([](const testing::TestParamInfo<ArraySliceParameterizedTest::ParamType>&
            info) {
      const auto& [param, primitive_type, memory_kind] = info.param;
      return absl::StrCat(param.name, "_",
                          xla::PrimitiveType_Name(primitive_type), "_",
                          memory_kind.value());
    }));

TEST_F(ArraySliceTest, CopySliceBatchMultipleArrays) {
  ASSERT_OK_AND_ASSIGN(
      const xla::Literal literal0,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {32, 128}));
  ASSERT_OK_AND_ASSIGN(
      const xla::Literal literal1,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {32, 256}));
  xla::ifrt::Device* const device = client_->addressable_devices()[0];
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device, xla::ifrt::MemoryKind("pinned_host"));

  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef array0,
      test_util::MakeArrayFromLiteral(client_.get(), literal0, sharding));
  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef array1,
      test_util::MakeArrayFromLiteral(client_.get(), literal1, sharding));

  std::vector<xla::ifrt::ArrayRef> arrays = {array0, array1};
  std::vector<IndexDomain> index_domains = {
      IndexDomain(Index({8, 0}), Shape({8, 128})),
      IndexDomain(Index({8, 128}), Shape({16, 128})),
  };

  ASSERT_OK_AND_ASSIGN(std::vector<xla::ifrt::ArrayRef> sliced,
                       client_->CopyArraySlices(absl::MakeSpan(arrays),
                                                absl::MakeSpan(index_domains)));
  ASSERT_EQ(sliced.size(), 2);

  const xla::Literal expected0 = literal0.Slice({8, 0}, {16, 128});
  EXPECT_THAT(test_util::CopyArrayToLiteral(sliced[0]),
              IsOkAndHolds(Eq(std::cref(expected0))));

  const xla::Literal expected1 = literal1.Slice({8, 128}, {24, 256});
  EXPECT_THAT(test_util::CopyArrayToLiteral(sliced[1]),
              IsOkAndHolds(Eq(std::cref(expected1))));
}

TEST_F(ArraySliceTest, CopySliceInvalidArguments) {
  ASSERT_OK_AND_ASSIGN(
      const xla::Literal literal,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {32, 128}));
  xla::ifrt::Device* const device = client_->addressable_devices()[0];
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device, xla::ifrt::MemoryKind("pinned_host"));

  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef array,
      test_util::MakeArrayFromLiteral(client_.get(), literal, sharding));

  // Mismatched counts
  {
    std::vector<IndexDomain> domains = {
        IndexDomain(Shape({16, 128})),
        IndexDomain(Shape({16, 128})),
    };
    EXPECT_THAT(client_->CopyArraySlices(absl::MakeSpan(&array, 1),
                                         absl::MakeSpan(domains)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Rank mismatch
  {
    IndexDomain domain(Shape({16}));
    EXPECT_THAT(client_->CopyArraySlices(absl::MakeSpan(&array, 1),
                                         absl::MakeSpan(&domain, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Out of bounds
  {
    IndexDomain domain(Index({24, 0}), Shape({16, 128}));
    EXPECT_THAT(client_->CopyArraySlices(absl::MakeSpan(&array, 1),
                                         absl::MakeSpan(&domain, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Negative origin
  {
    IndexDomain domain(Index({-8, 0}), Shape({16, 128}));
    EXPECT_THAT(client_->CopyArraySlices(absl::MakeSpan(&array, 1),
                                         absl::MakeSpan(&domain, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Negative extent
  {
    IndexDomain domain(Index({0, 0}), Shape({-8, 128}));
    EXPECT_THAT(client_->CopyArraySlices(absl::MakeSpan(&array, 1),
                                         absl::MakeSpan(&domain, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  if (client_->addressable_devices().size() >= 2) {
    std::vector<xla::ifrt::Device*> devices = {
        client_->addressable_devices()[0],
        client_->addressable_devices()[1],
    };
    ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                         client_->MakeDeviceList(devices));

    // No intersecting shards: ConcreteSharding on [64, 256] with shard 0 at
    // [0..16, 0..256] and shard 1 at [48..64, 0..256], sliced at [16, 0] with
    // shape [16, 256].
    {
      Shape test_shape({64, 256});
      Shape s_shape({16, 256});
      std::vector<Shape> s_shapes = {s_shape, s_shape};
      std::vector<IndexDomain> s_domains = {
          IndexDomain(Index({0, 0}), s_shape),
          IndexDomain(Index({48, 0}), s_shape),
      };
      xla::ifrt::ShardingRef s_sharding = xla::ifrt::ConcreteSharding::Create(
          device_list, xla::ifrt::MemoryKind("pinned_host"), test_shape,
          std::move(s_shapes), std::move(s_domains));

      ASSERT_OK_AND_ASSIGN(const xla::Literal test_literal,
                           test_util::CreateIotaLiteral(xla::PrimitiveType::S32,
                                                        test_shape.dims()));
      ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef disjoint_array,
                           test_util::MakeArrayFromLiteral(
                               client_.get(), test_literal, s_sharding));

      IndexDomain disjoint_domain(Index({16, 0}), Shape({16, 256}));
      EXPECT_THAT(client_->CopyArraySlices(absl::MakeSpan(&disjoint_array, 1),
                                           absl::MakeSpan(&disjoint_domain, 1)),
                  StatusIs(absl::StatusCode::kInvalidArgument));
    }

    // Uneven shard shapes across shards: target of shape [64, 256] sharded
    // across 2 devices with IotaTile({2, 1}), sliced at [8, 0] with shape [32,
    // 128]. Shard 0 gets [24, 128] and shard 1 gets [8, 128].
    {
      Shape test_shape({64, 256});
      ASSERT_OK_AND_ASSIGN(
          xla::ifrt::ShardingRef sharding,
          MakeSharding(xla::ifrt::HloShardingSpec::Create(
                           2, xla::HloSharding::IotaTile({2, 1})),
                       xla::ifrt::MemoryKind("pinned_host")));
      ASSERT_OK_AND_ASSIGN(const xla::Literal test_literal,
                           test_util::CreateIotaLiteral(xla::PrimitiveType::S32,
                                                        test_shape.dims()));
      ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef array,
                           test_util::MakeArrayFromLiteral(
                               client_.get(), test_literal, sharding));

      IndexDomain domain(Index({8, 0}), Shape({32, 128}));
      EXPECT_THAT(client_->CopyArraySlices(absl::MakeSpan(&array, 1),
                                           absl::MakeSpan(&domain, 1)),
                  StatusIs(absl::StatusCode::kInvalidArgument));
    }
  }
}

TEST_F(ArraySliceTest, UpdateSliceBatchMultipleArrays) {
  ASSERT_OK_AND_ASSIGN(
      const xla::Literal target_literal0,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {32, 128}));
  ASSERT_OK_AND_ASSIGN(
      const xla::Literal target_literal1,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {32, 256}));
  ASSERT_OK_AND_ASSIGN(const xla::Literal slice_literal0,
                       test_util::CreateIotaLiteral(xla::PrimitiveType::S32,
                                                    {8, 128}, /*start=*/3));
  ASSERT_OK_AND_ASSIGN(const xla::Literal slice_literal1,
                       test_util::CreateIotaLiteral(xla::PrimitiveType::S32,
                                                    {16, 128}, /*start=*/5));

  xla::ifrt::Device* const device0 = client_->addressable_devices().front();
  xla::ifrt::ShardingRef sharding0 = xla::ifrt::SingleDeviceSharding::Create(
      device0, xla::ifrt::MemoryKind("pinned_host"));
  xla::ifrt::Device* const device1 = client_->addressable_devices().back();
  xla::ifrt::ShardingRef sharding1 = xla::ifrt::SingleDeviceSharding::Create(
      device1, xla::ifrt::MemoryKind("pinned_host"));

  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef target0,
                       test_util::MakeArrayFromLiteral(
                           client_.get(), target_literal0, sharding0));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef target1,
                       test_util::MakeArrayFromLiteral(
                           client_.get(), target_literal1, sharding1));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef slice0,
                       test_util::MakeArrayFromLiteral(
                           client_.get(), slice_literal0, sharding0));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef slice1,
                       test_util::MakeArrayFromLiteral(
                           client_.get(), slice_literal1, sharding1));

  std::vector<xla::ifrt::ArrayRef> targets = {target0, target1};
  std::vector<xla::ifrt::ArrayRef> slices = {slice0, slice1};
  std::vector<Index> origins = {Index({8, 0}), Index({8, 128})};

  ASSERT_OK_AND_ASSIGN(std::vector<xla::ifrt::ArrayRef> updated,
                       client_->UpdateArraySlices(absl::MakeSpan(targets),
                                                  absl::MakeSpan(slices),
                                                  absl::MakeSpan(origins)));
  ASSERT_EQ(updated.size(), 2);
  EXPECT_TRUE(target0->IsDeleted());
  EXPECT_TRUE(target1->IsDeleted());
  EXPECT_FALSE(slice0->IsDeleted());
  EXPECT_FALSE(slice1->IsDeleted());

  xla::Literal expected0 = target_literal0.Clone();
  ASSERT_OK(expected0.CopySliceFrom(
      slice_literal0,
      /*src_base=*/Index::Zeros(2).elements(),
      /*dest_base=*/origins[0].elements(),
      /*copy_size=*/slice_literal0.shape().dimensions()));
  EXPECT_THAT(test_util::CopyArrayToLiteral(updated[0]),
              IsOkAndHolds(Eq(std::cref(expected0))));

  xla::Literal expected1 = target_literal1.Clone();
  ASSERT_OK(expected1.CopySliceFrom(
      slice_literal1,
      /*src_base=*/Index::Zeros(2).elements(),
      /*dest_base=*/origins[1].elements(),
      /*copy_size=*/slice_literal1.shape().dimensions()));
  EXPECT_THAT(test_util::CopyArrayToLiteral(updated[1]),
              IsOkAndHolds(Eq(std::cref(expected1))));
}

TEST_F(ArraySliceTest, UpdateSliceInvalidArguments) {
  ASSERT_OK_AND_ASSIGN(
      const xla::Literal target_literal,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {32, 128}));
  ASSERT_OK_AND_ASSIGN(
      const xla::Literal slice_literal,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {16, 128}));
  xla::ifrt::Device* const device = client_->addressable_devices()[0];
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device, xla::ifrt::MemoryKind("pinned_host"));

  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef target,
      test_util::MakeArrayFromLiteral(client_.get(), target_literal, sharding));
  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef slice,
      test_util::MakeArrayFromLiteral(client_.get(), slice_literal, sharding));

  // Mismatched slice count
  {
    std::vector<xla::ifrt::ArrayRef> slices = {slice, slice};
    Index origin({0, 0});
    EXPECT_THAT(client_->UpdateArraySlices(absl::MakeSpan(&target, 1),
                                           absl::MakeSpan(slices),
                                           absl::MakeSpan(&origin, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Rank mismatch
  {
    ASSERT_OK_AND_ASSIGN(
        const xla::Literal rank1_literal,
        test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {16}));
    xla::ifrt::ShardingRef sds = xla::ifrt::SingleDeviceSharding::Create(
        device, xla::ifrt::MemoryKind("pinned_host"));
    ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ArrayRef rank1_slice,
        test_util::MakeArrayFromLiteral(client_.get(), rank1_literal, sds));
    Index origin({0});
    EXPECT_THAT(client_->UpdateArraySlices(absl::MakeSpan(&target, 1),
                                           absl::MakeSpan(&rank1_slice, 1),
                                           absl::MakeSpan(&origin, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Out of bounds
  {
    Index origin({24, 0});
    EXPECT_THAT(client_->UpdateArraySlices(absl::MakeSpan(&target, 1),
                                           absl::MakeSpan(&slice, 1),
                                           absl::MakeSpan(&origin, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Negative origin
  {
    Index origin({-8, 0});
    EXPECT_THAT(client_->UpdateArraySlices(absl::MakeSpan(&target, 1),
                                           absl::MakeSpan(&slice, 1),
                                           absl::MakeSpan(&origin, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Slice sharding with mismatched index domains
  if (client_->addressable_devices().size() >= 2) {
    Shape array_shape({4, 4});
    ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ShardingRef target_sharding,
        MakeSharding(xla::ifrt::HloShardingSpec::Create(
                         2, xla::HloSharding::IotaTile({2, 1})),
                     xla::ifrt::MemoryKind("pinned_host")));
    ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ShardingRef slice_sharding,
        MakeSharding(xla::ifrt::HloShardingSpec::Create(
                         2, xla::HloSharding::IotaTile({1, 2})),
                     xla::ifrt::MemoryKind("pinned_host")));

    ASSERT_OK_AND_ASSIGN(const xla::Literal sharded_target_literal,
                         test_util::CreateIotaLiteral(xla::PrimitiveType::S32,
                                                      array_shape.dims()));
    ASSERT_OK_AND_ASSIGN(const xla::Literal sharded_slice_literal,
                         test_util::CreateIotaLiteral(xla::PrimitiveType::S32,
                                                      array_shape.dims(),
                                                      /*start=*/3));

    ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ArrayRef sharded_target,
        test_util::MakeArrayFromLiteral(client_.get(), sharded_target_literal,
                                        target_sharding));
    ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ArrayRef sharded_slice,
        test_util::MakeArrayFromLiteral(client_.get(), sharded_slice_literal,
                                        slice_sharding));

    Index origin({0, 0});
    EXPECT_THAT(client_->UpdateArraySlices(absl::MakeSpan(&sharded_target, 1),
                                           absl::MakeSpan(&sharded_slice, 1),
                                           absl::MakeSpan(&origin, 1)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST_F(ArraySliceTest, UpdateSliceRejectsMismatchedLayout) {
  ASSERT_OK_AND_ASSIGN(
      const xla::Literal target_literal,
      test_util::CreateIotaLiteral(xla::PrimitiveType::S32, {32, 256}));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      client_->addressable_devices()[0], xla::ifrt::MemoryKind("pinned_host"));

  ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef target,
      test_util::MakeArrayFromLiteral(client_.get(), target_literal, sharding));

  xla::Layout layout = xla::LayoutUtil::MakeLayout({0, 1});
  if (absl::StatusOr<std::shared_ptr<Topology>> ifrt_topology =
          client_->GetTopologyForDevices(sharding->devices());
      ifrt_topology.ok()) {
    const xla::PjRtTopologyDescription* topology =
        (*ifrt_topology)->description().get();
    ASSERT_OK_AND_ASSIGN(
        xla::Shape shape,
        topology->MakeCanonicalShapeForMemorySpace(
            xla::PinnedHostMemorySpace::kKindId,
            xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32, {32, 256}),
            &layout));
    layout = shape.layout();
  }

  xla::ifrt::ArraySpec slice_spec = {
      /*dtype=*/xla::ifrt::DType(xla::ifrt::DType::kS32),
      /*shape=*/Shape({32, 256}),
      /*sharding=*/sharding,
      /*layout=*/std::make_shared<const xla::PjRtLayout>(std::move(layout)),
  };
  ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> slices,
      client_->MakeErrorArrays(absl::InternalError("unused"),
                               absl::MakeSpan(&slice_spec, 1)));

  Index origin({0, 0});
  EXPECT_THAT(client_->UpdateArraySlices(absl::MakeSpan(&target, 1),
                                         absl::MakeSpan(slices),
                                         absl::MakeSpan(&origin, 1)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
