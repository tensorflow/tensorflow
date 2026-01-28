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

#include "xla/python/pjrt_ifrt/xla_sharding_spec.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;

class HloShardingSpecTest : public testing::Test {};

TEST_F(HloShardingSpecTest, CreateWithNumShardsMismatch) {
  auto xla_hlo_sharding = xla::HloSharding::IotaTile({2, 3});
  EXPECT_DEATH(
      HloShardingSpec::Create(/*num_shards=*/5, xla_hlo_sharding),
      "`num_shards` and `xla_hlo_sharding`'s `num_devices` does not match");
}

TEST_F(HloShardingSpecTest, IsFullyReplicated) {
  int num_shards = 6;
  {
    // Fully replicated HloSharding is fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::Replicate();
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_TRUE(spec->IsFullyReplicated());
  }
  {
    // Single-tile HloSharding is fully replicated.
    int num_shards = 1;
    auto xla_hlo_sharding = xla::HloSharding::IotaTile({1, 1});
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_TRUE(spec->IsFullyReplicated());
  }
  {
    // Multi-tile HloSharding with last_dim_replicate where all replices are on
    // the last tile dimension is fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::PartialTile(
        xla::TileAssignment(xla::IotaTileAssignment::Create({1, 6})));
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_TRUE(spec->IsFullyReplicated());
  }
  {
    // Multi-tile HloSharding with last_dim_replicate where not all replices are
    // on the last tile dimension is not fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::PartialTile(
        xla::TileAssignment(xla::IotaTileAssignment::Create({2, 3})));
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
  {
    // Multi-tile HloSharding with no last_dim_replicate is not fully
    // replicated.
    auto xla_hlo_sharding = xla::HloSharding::IotaTile({1, 6});
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
  {
    // Maximal HloSharding with a single device is fully replicated.
    int num_shards = 1;
    auto xla_hlo_sharding = xla::HloSharding::AssignDevice(/*device_id=*/0);
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_TRUE(spec->IsFullyReplicated());
  }
  {
    // Maximal HloSharding with more than one device is not fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::AssignDevice(/*device_id=*/0);
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
  {
    // Manual HloSharding is not fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::Manual();
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
  {
    // Unknown HloSharding is not fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::Unknown();
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
}

TEST_F(HloShardingSpecTest, GetShardShape) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::IotaTile({2, 3});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);
  EXPECT_THAT(spec->GetShardShape(Shape({6, 6})),
              absl_testing::IsOkAndHolds(Shape({3, 2})));
  EXPECT_THAT(spec->GetShardShape(Shape({6, 6, 6})),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("Numbers of dimensions don't match. From "
                            "Shape 3 vs from HloSharding 2")));
}

TEST_F(HloShardingSpecTest, HasSamePartitioning) {
  int num_shards0 = 6;
  auto xla_hlo_sharding0 = xla::HloSharding::IotaTile({2, 3});
  std::shared_ptr<const HloShardingSpec> spec0 =
      HloShardingSpec::Create(num_shards0, xla_hlo_sharding0);

  EXPECT_TRUE(spec0->HasSamePartitioning(*spec0));
  {
    // Different number of shards.
    int num_shards1 = 3;
    auto xla_hlo_sharding1 = xla::HloSharding::IotaTile({3, 1});
    std::shared_ptr<const HloShardingSpec> spec1 =
        HloShardingSpec::Create(num_shards1, xla_hlo_sharding1);
    EXPECT_FALSE(spec0->HasSamePartitioning(*spec1));
  }
  // Different HloSharding.
  {
    auto xla_hlo_sharding1 = xla::HloSharding::IotaTile({3, 2});
    std::shared_ptr<const HloShardingSpec> spec1 =
        HloShardingSpec::Create(num_shards0, xla_hlo_sharding1);
    EXPECT_FALSE(spec0->HasSamePartitioning(*spec1));
  }

  // Replicated sharding with different numbers of devices.
  {
    int num_shards1 = 3;
    std::shared_ptr<const HloShardingSpec> spec0 =
        HloShardingSpec::Create(num_shards0, xla::HloSharding::Replicate());
    std::shared_ptr<const HloShardingSpec> spec1 =
        HloShardingSpec::Create(num_shards1, xla::HloSharding::Replicate());
    EXPECT_FALSE(spec0->HasSamePartitioning(*spec1));
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithReplication) {
  int num_shards = 6;
  // Fully replicated.
  auto xla_hlo_sharding = xla::HloSharding::Replicate();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(
      index_domains,
      ElementsAre(IndexDomain(shape), IndexDomain(shape), IndexDomain(shape),
                  IndexDomain(shape), IndexDomain(shape), IndexDomain(shape)));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, DisassembleWithReplication) {
  int num_shards = 6;
  // Fully replicated.
  auto xla_hlo_sharding = xla::HloSharding::Replicate();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({10, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({12, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({2, 20})),
                          IndexDomain(Index({2, 0}), Shape({2, 20})),
                          IndexDomain(Index({4, 0}), Shape({2, 20})),
                          IndexDomain(Index({6, 0}), Shape({2, 20})),
                          IndexDomain(Index({8, 0}), Shape({2, 20})),
                          IndexDomain(Index({10, 0}), Shape({2, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, DisassembleWithTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({12, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({2, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithUnevenTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({11, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({2, 20})),
                          IndexDomain(Index({2, 0}), Shape({2, 20})),
                          IndexDomain(Index({4, 0}), Shape({2, 20})),
                          IndexDomain(Index({6, 0}), Shape({2, 20})),
                          IndexDomain(Index({8, 0}), Shape({2, 20})),
                          IndexDomain(Index({10, 0}), Shape({1, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, DisassembleWithUnevenTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({11, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    if (i < 5) {
      EXPECT_EQ(shape, Shape({2, 20}));
    } else {
      EXPECT_EQ(shape, Shape({1, 20}));
    }
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithPartialTile) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding =
      xla::HloSharding::PartialTile(xla::TileAssignment({2, 1, 3}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, DisassembleWithPartialTile) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding =
      xla::HloSharding::PartialTile(xla::TileAssignment({2, 1, 3}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({5, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithSubgroupReplicated) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding = xla::HloSharding::Subgroup(
      xla::TileAssignment({2, 1, 3}), {xla::OpSharding::REPLICATED});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, DisassembleWithSubgroupReplicated) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding = xla::HloSharding::Subgroup(
      xla::TileAssignment({2, 1, 3}), {xla::OpSharding::REPLICATED});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({5, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithSubgroupMaximalSlowPath) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // maximal-replicated by 3 times, device#0 in each replication is maximal.
  auto xla_hlo_sharding = xla::HloSharding::Subgroup(
      xla::TileAssignment({2, 1, 3}), {xla::OpSharding::MAXIMAL});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, DisassembleWithSubgroupMaximalSlowPath) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // maximal-replicated by 3 times, device#0 in each replication is maximal.
  auto xla_hlo_sharding = xla::HloSharding::Subgroup(
      xla::TileAssignment({2, 1, 3}), {xla::OpSharding::MAXIMAL});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({5, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithTileTranspose) {
  int num_shards = 4;
  auto xla_hlo_sharding =
      xla::HloSharding::IotaTile(/*tile_assignment_dims=*/{2, 2},
                                 /*reshape_dims=*/{2, 2},
                                 /*transpose_perm=*/{1, 0});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);
  Shape shape({4, 4});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, IndexDomainsWithUnreduced) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::Unreduced();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  EXPECT_THAT(
      spec->IndexDomains(shape).status(),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("Unreduced sharding does not support IndexDomains")));
}

TEST_F(HloShardingSpecTest, IndexDomainsWithManual) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::Manual();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  EXPECT_THAT(spec->IndexDomains(shape).status(),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("Manual sharding does not support IndexDomains")));
}

TEST_F(HloShardingSpecTest, DisassembleWithManual) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::Manual();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({10, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, DisassembleFailsWithMismatchingShapeDimsSize) {
  int num_shards = 2;
  // 2-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({2, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10});
  EXPECT_THAT(
      spec->Disassemble(shape),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("shape must have 2 dimensions, but has 1 dimensions")));
}

TEST_F(HloShardingSpecTest, DisassembleFailsWithDynamicShape) {
  int num_shards = 2;
  auto xla_hlo_sharding =
      xla::HloSharding::Tile(xla::TileAssignment(absl::Span<const int64_t>{2}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({10}), BoundedDynamicShapeTag({true})));
  EXPECT_THAT(
      spec->Disassemble(dynamic_shape),
      absl_testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                             HasSubstr("can only disassemble static shape")));
}

TEST_F(HloShardingSpecTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      HloShardingSpec::Create(6, xla::HloSharding::Replicate()),
      HloShardingSpec::Create(1, xla::HloSharding::Replicate()),
      HloShardingSpec::Create(6,
                              xla::HloSharding::AssignDevice(/*device_id=*/0)),
      HloShardingSpec::Create(
          6, xla::HloSharding::PartialTile(
                 xla::TileAssignment(xla::IotaTileAssignment::Create({2, 3})))),
  }));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
