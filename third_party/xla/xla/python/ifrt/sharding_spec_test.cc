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

#include "xla/python/ifrt/sharding_spec.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/status/status_matchers.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;

struct ShardingSpecTestParam {
  int num_shards;
};

class ShardingSpecTest : public testing::TestWithParam<ShardingSpecTestParam> {
 public:
  int num_shards() const { return GetParam().num_shards; }
};

class SingleDeviceShardingSpecTest : public ShardingSpecTest {};
class OpaqueShardingSpecTest : public ShardingSpecTest {};
class ConcreteShardingSpecTest : public ShardingSpecTest {};
class ConcreteEvenShardingSpecTest : public ShardingSpecTest {};
class ShardingParamShardingSpecTest : public ShardingSpecTest {};

TEST_P(SingleDeviceShardingSpecTest, IsFullyReplicated) {
  ShardingSpecRef sharding = SingleDeviceShardingSpec::Create();
  EXPECT_TRUE(sharding->IsFullyReplicated());
}

TEST_P(SingleDeviceShardingSpecTest, GetShardShape) {
  ShardingSpecRef sharding = SingleDeviceShardingSpec::Create();
  EXPECT_THAT(sharding->GetShardShape(Shape({10, 20})),
              absl_testing::IsOkAndHolds(Shape({10, 20})));
}

TEST_P(SingleDeviceShardingSpecTest, HasSamePartitioning) {
  ShardingSpecRef sharding0 = SingleDeviceShardingSpec::Create();

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    ShardingSpecRef sharding1 = SingleDeviceShardingSpec::Create();
    EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(SingleDeviceShardingSpecTest, IndexDomains) {
  ShardingSpecRef sharding = SingleDeviceShardingSpec::Create();

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, sharding->IndexDomains(shape));
  EXPECT_THAT(index_domains, ElementsAre(IndexDomain(shape)));
}

TEST_P(SingleDeviceShardingSpecTest, Disassemble) {
  ShardingSpecRef sharding = SingleDeviceShardingSpec::Create();

  {  // Disassemble static shape.
    Shape shape({10, 20});
    TF_ASSERT_OK_AND_ASSIGN(auto disassembled, sharding->Disassemble(shape));
    ASSERT_THAT(disassembled, SizeIs(1));
    const auto& [result_shape, result_sharding] = disassembled[0];
    EXPECT_EQ(shape, result_shape);
    EXPECT_EQ(*result_sharding, *sharding);
  }
  {  // Disassemble dynamic shape.
    TF_ASSERT_OK_AND_ASSIGN(
        DynamicShape dynamic_shape,
        DynamicShape::Create(Shape({10, 20}),
                             BoundedDynamicShapeTag({true, true})));
    TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                            sharding->Disassemble(dynamic_shape));
    ASSERT_THAT(disassembled, SizeIs(1));
    const auto& [result_shape, result_sharding] = disassembled[0];
    EXPECT_EQ(dynamic_shape, result_shape);
    EXPECT_EQ(*result_sharding, *sharding);
  }
}

TEST_P(OpaqueShardingSpecTest, IsFullyReplicated) {
  ShardingSpecRef sharding = OpaqueShardingSpec::Create(2);
  EXPECT_FALSE(sharding->IsFullyReplicated());
}

TEST_P(OpaqueShardingSpecTest, GetShardShape) {
  ShardingSpecRef sharding = OpaqueShardingSpec::Create(num_shards());
  EXPECT_THAT(
      sharding->GetShardShape(Shape({10, 20})),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "OpaqueShardingSpec does not have shard shape information")));
}

TEST_P(OpaqueShardingSpecTest, HasSamePartitioning) {
  ShardingSpecRef sharding0 = OpaqueShardingSpec::Create(2);

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  // OpaqueShardingSpec::HasSamePartitioning currently only returns true for the
  // same object.
  ShardingSpecRef sharding1 = OpaqueShardingSpec::Create(2);
  EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
}

TEST_P(OpaqueShardingSpecTest, FailedToDisassemble) {
  ShardingSpecRef sharding = OpaqueShardingSpec::Create(num_shards());

  EXPECT_THAT(
      sharding->Disassemble(Shape({30})),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "OpaqueShardingSpec does not have shard shape information")));

  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({30}), BoundedDynamicShapeTag({true})));
  EXPECT_THAT(
      sharding->Disassemble(dynamic_shape),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "OpaqueShardingSpec does not have shard shape information")));
}

TEST_P(OpaqueShardingSpecTest, IndexDomainsFails) {
  ShardingSpecRef sharding = OpaqueShardingSpec::Create(num_shards());

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "OpaqueShardingSpec does not have index domain information")));
}

TEST_P(OpaqueShardingSpecTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *OpaqueShardingSpec::Create(2),
      *OpaqueShardingSpec::Create(3),
  }));
}

TEST_P(ConcreteShardingSpecTest, IsFullyReplicated) {
  std::vector<Shape> shard_shapes{Shape({10}), Shape({20})};
  ShardingSpecRef sharding =
      ConcreteShardingSpec::Create(Shape({30}), shard_shapes);
  EXPECT_FALSE(sharding->IsFullyReplicated());
}

TEST_P(ConcreteShardingSpecTest, GetShardShapeSuccess) {
  Shape shard_shape({30});
  std::vector<Shape> shard_shapes(2, shard_shape);
  ShardingSpecRef sharding =
      ConcreteShardingSpec::Create(Shape({30}), shard_shapes);
  EXPECT_THAT(sharding->GetShardShape(Shape({30})),
              absl_testing::IsOkAndHolds(shard_shape));
}

TEST_P(ConcreteShardingSpecTest, GetShardShapeFailure) {
  std::vector<Shape> shard_shapes{Shape({10}), Shape({20})};
  ShardingSpecRef sharding =
      ConcreteShardingSpec::Create(Shape({30}), shard_shapes);
  EXPECT_THAT(
      sharding->GetShardShape(Shape({30})),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("ConcreteShardingSpec does not have a fixed shard shape")));
}

TEST_P(ConcreteShardingSpecTest, HasSamePartitioning) {
  std::vector<Shape> shard_shapes0{Shape({10}), Shape({20})};
  ShardingSpecRef sharding0 =
      ConcreteShardingSpec::Create(Shape({30}), shard_shapes0);

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    std::vector<Shape> shard_shapes1{Shape({10}), Shape({20})};
    ShardingSpecRef sharding1 =
        ConcreteShardingSpec::Create(Shape({30}), shard_shapes1);
    EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different number of shards.
  {
    std::vector<Shape> shard_shapes1{Shape({10}), Shape({20}), Shape({30})};
    ShardingSpecRef sharding1 =
        ConcreteShardingSpec::Create(Shape({60}), shard_shapes1);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Difference shape.
  {
    std::vector<Shape> shard_shapes1{Shape({10}), Shape({20})};
    ShardingSpecRef sharding1 =
        ConcreteShardingSpec::Create(Shape({40}), shard_shapes1);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different shard shapes.
  {
    std::vector<Shape> shard_shapes1{Shape({10000}), Shape({20})};
    ShardingSpecRef sharding1 =
        ConcreteShardingSpec::Create(Shape({30}), shard_shapes1);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(ConcreteShardingSpecTest, Disassemble) {
  std::vector<Shape> shard_shapes{Shape({3}), Shape({7}), Shape({3}),
                                  Shape({7})};
  ShardingSpecRef sharding =
      ConcreteShardingSpec::Create(Shape({20}), shard_shapes);

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                          sharding->Disassemble(Shape({20})));
  ASSERT_THAT(disassembled, SizeIs(4));
  for (int i = 0; i < 4; ++i) {
    const auto& [shape, result_sharding] = disassembled[i];
    EXPECT_EQ(shape, shard_shapes[i]);
    EXPECT_EQ(*result_sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_P(ConcreteShardingSpecTest, DisassembleDynamicShape) {
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({20}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape0,
      DynamicShape::Create(Shape({3}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape1,
      DynamicShape::Create(Shape({7}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape2,
      DynamicShape::Create(Shape({3}), BoundedDynamicShapeTag({true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape3,
      DynamicShape::Create(Shape({7}), BoundedDynamicShapeTag({true})));
  std::vector<DynamicShape> shard_dynamic_shapes{
      std::move(shard_dynamic_shape0),
      std::move(shard_dynamic_shape1),
      std::move(shard_dynamic_shape2),
      std::move(shard_dynamic_shape3),
  };
  auto sharding =
      ConcreteShardingSpec::Create(dynamic_shape, shard_dynamic_shapes);
  EXPECT_THAT(sharding->Disassemble(Shape({20})),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("ConcreteShardingSpec holds dynamic shape")));
  {
    TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                            sharding->Disassemble(DynamicShape(dynamic_shape)));
    ASSERT_THAT(disassembled, SizeIs(4));
    for (int i = 0; i < 4; ++i) {
      const auto& [result_dynamic_shape, result_sharding] = disassembled[i];
      EXPECT_EQ(result_dynamic_shape, shard_dynamic_shapes[i]);
      EXPECT_EQ(*result_sharding, *SingleDeviceShardingSpec::Create());
    }
  }
}

TEST_P(ConcreteShardingSpecTest, DisassembleFailsForUnexpectedShape) {
  std::vector<Shape> shard_shapes{Shape({10}), Shape({20})};
  ShardingSpecRef sharding =
      ConcreteShardingSpec::Create(Shape({30}), shard_shapes);

  EXPECT_THAT(sharding->Disassemble(Shape({40})),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("ConcreteShardingSpec can only disassemble")));
}

TEST_P(ConcreteShardingSpecTest, IndexDomains) {
  std::vector<Shape> shard_shapes = {Shape({1}), Shape({2})};
  std::vector<IndexDomain> index_domains{
      IndexDomain(Index({0}), Shape({1})),
      IndexDomain(Index({1}), Shape({2})),
  };
  ShardingSpecRef sharding =
      ConcreteShardingSpec::Create(Shape({15}), shard_shapes, index_domains);

  EXPECT_THAT(sharding->IndexDomains(Shape({15})),
              absl_testing::IsOkAndHolds(ElementsAreArray(index_domains)));
}

TEST_P(ConcreteShardingSpecTest, IndexDomainsMissing) {
  std::vector<Shape> shard_shapes{Shape({10}), Shape({20})};
  ShardingSpecRef sharding =
      ConcreteShardingSpec::Create(Shape({30}), shard_shapes);

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "ConcreteShardingSpec does not have index domain information")));
}

TEST_P(ConcreteShardingSpecTest, Hash) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto dynamic_shape,
      DynamicShape::Create(Shape({30}), BoundedDynamicShapeTag({true})));
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *ConcreteShardingSpec::Create(Shape({30}), {Shape({10}), Shape({20})}),
      *ConcreteShardingSpec::Create(dynamic_shape,
                                    {dynamic_shape, dynamic_shape}),
  }));
}

TEST_P(ConcreteEvenShardingSpecTest, IsFullyReplicated) {
  {
    // Fully replicated.
    ShardingSpecRef sharding = ConcreteEvenShardingSpec::Create(
        /*num_shards=*/2, Shape({30}), Shape({15}),
        /*is_fully_replicated=*/true);
    EXPECT_TRUE(sharding->IsFullyReplicated());
  }
  {
    // Not fully replicated.
    ShardingSpecRef sharding = ConcreteEvenShardingSpec::Create(
        /*num_shards=*/2, Shape({30}), Shape({15}),
        /*is_fully_replicated=*/false);
    EXPECT_FALSE(sharding->IsFullyReplicated());
  }
}

TEST_P(ConcreteEvenShardingSpecTest, GetShardShape) {
  ShardingSpecRef sharding = ConcreteEvenShardingSpec::Create(
      /*num_shards=*/2, Shape({30}), Shape({15}),
      /*is_fully_replicated=*/true);
  EXPECT_THAT(sharding->GetShardShape(Shape({30})),
              absl_testing::IsOkAndHolds(Shape({15})));
  EXPECT_THAT(
      sharding->GetShardShape(Shape({45})),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "ConcreteEvenShardingSpec has a shard shape for shape [30], "
              "but was asked to get a shard shape for shape [45]")));
}

TEST_P(ConcreteEvenShardingSpecTest, HasSamePartitioning) {
  ShardingSpecRef sharding0 = ConcreteEvenShardingSpec::Create(
      /*num_shards=*/2, Shape({30}), Shape({15}),
      /*is_fully_replicated=*/true);

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    ShardingSpecRef sharding1 = ConcreteEvenShardingSpec::Create(
        /*num_shards=*/2, Shape({30}), Shape({15}),
        /*is_fully_replicated=*/true);
    EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different number of shards.
  {
    ShardingSpecRef sharding1 = ConcreteEvenShardingSpec::Create(
        /*num_shards=*/3, Shape({30}), Shape({15}),
        /*is_fully_replicated=*/true);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Difference shape.
  {
    ShardingSpecRef sharding1 = ConcreteEvenShardingSpec::Create(
        /*num_shards=*/2, Shape({45}), Shape({15}),
        /*is_fully_replicated=*/true);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different shard shape.
  {
    ShardingSpecRef sharding1 = ConcreteEvenShardingSpec::Create(
        /*num_shards=*/2, Shape({30}), Shape({10}),
        /*is_fully_replicated=*/true);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different is_fully_replicated.
  {
    ShardingSpecRef sharding1 = ConcreteEvenShardingSpec::Create(
        /*num_shards=*/2, Shape({30}), Shape({15}),
        /*is_fully_replicated=*/false);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(ConcreteEvenShardingSpecTest, Disassemble) {
  ShardingSpecRef sharding =
      ConcreteEvenShardingSpec::Create(num_shards(), Shape({30}), Shape({5}),
                                       /*is_fully_replicated=*/false);

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                          sharding->Disassemble(Shape({30})));
  ASSERT_THAT(disassembled, SizeIs(num_shards()));
  for (int i = 0; i < num_shards(); ++i) {
    const auto& [shape, result_sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({5}));
    EXPECT_EQ(*result_sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_P(ConcreteEvenShardingSpecTest, DisassembleFailsForUnexpectedShape) {
  ShardingSpecRef sharding = ConcreteEvenShardingSpec::Create(
      /*num_shards=*/2, Shape({30}), Shape({15}),
      /*is_fully_replicated=*/false);

  EXPECT_THAT(sharding->Disassemble(Shape({40})),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("ConcreteEvenShardingSpec can only disassemble")));
}

TEST_P(ConcreteEvenShardingSpecTest, IndexDomainsFails) {
  ShardingSpecRef sharding = ConcreteEvenShardingSpec::Create(
      /*num_shards=*/2, Shape({30}), Shape({5}),
      /*is_fully_replicated=*/false);

  EXPECT_THAT(
      sharding->IndexDomains(Shape({30})),
      absl_testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                             HasSubstr("ConcreteEvenShardingSpec does not have "
                                       "index domain information")));
}

TEST_P(ConcreteEvenShardingSpecTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *ConcreteEvenShardingSpec::Create(
          /*num_shards=*/2, Shape({30}), Shape({30}),
          /*is_fully_replicated=*/true),
      *ConcreteEvenShardingSpec::Create(/*num_shards=*/2, Shape({30}),
                                        Shape({15}),
                                        /*is_fully_replicated=*/false),
      *ConcreteEvenShardingSpec::Create(
          /*num_shards=*/3, Shape({30}), Shape({10}),
          /*is_fully_replicated=*/false),
  }));
}

TEST_P(ShardingParamShardingSpecTest, IsFullyReplicated) {
  {
    // Fully replicated.
    ShardingParam param{/*dim_shards=*/{1, 1},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);
    EXPECT_TRUE(param_sharding->IsFullyReplicated());
  }
  {
    // Not fully replicated.
    ShardingParam param{/*dim_shards=*/{1, 6},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);
    EXPECT_FALSE(param_sharding->IsFullyReplicated());
  }
  {
    // Not fully replicated.
    ShardingParam param{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);
    EXPECT_FALSE(param_sharding->IsFullyReplicated());
  }
}

TEST_P(ShardingParamShardingSpecTest, GetShardShape) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  ShardingSpecRef sharding = ShardingParamShardingSpec::Create(param);
  EXPECT_THAT(sharding->GetShardShape(Shape({6, 6})),
              absl_testing::IsOkAndHolds(Shape({3, 2})));
  EXPECT_THAT(sharding->GetShardShape(Shape({6, 6, 6})),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("Numbers of dimensions don't match. From "
                            "Shape 3 vs from ShardingParam 2")));
}

TEST_P(ShardingParamShardingSpecTest, HasSamePartitioning) {
  ShardingParam param0{/*dim_shards=*/{2, 3},
                       {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  ShardingSpecRef sharding0 = ShardingParamShardingSpec::Create(param0);

  EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding0));
  {
    ShardingParam param1{/*dim_shards=*/{2, 3},
                         {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
    ShardingSpecRef sharding1 = ShardingParamShardingSpec::Create(param1);
    EXPECT_TRUE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different number of shards.
  {
    ShardingParam param1{/*dim_shards=*/{3, 1},
                         {/*permutation=*/{1, 0}, /*axis_sizes=*/{1, 3}}};
    ShardingSpecRef sharding1 = ShardingParamShardingSpec::Create(param1);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
  // Different sharding param.
  {
    ShardingParam param1{/*dim_shards=*/{3, 2},
                         {/*permutation=*/{0, 1}, /*axis_sizes=*/{3, 2}}};
    ShardingSpecRef sharding1 = ShardingParamShardingSpec::Create(param1);
    EXPECT_FALSE(sharding0->HasSamePartitioning(*sharding1));
  }
}

TEST_P(ShardingParamShardingSpecTest, Disassemble) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);

  {
    TF_ASSERT_OK_AND_ASSIGN(auto disassembled,
                            param_sharding->Disassemble(Shape({6, 6})));
    ASSERT_THAT(disassembled, SizeIs(6));
    for (int i = 0; i < 6; ++i) {
      const auto& [shape, sharding] = disassembled[i];
      EXPECT_EQ(shape, Shape({3, 2}));
      EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
    }
  }
}

TEST_P(ShardingParamShardingSpecTest, DisassembleFailsWhenRankNotMatch) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);

  EXPECT_THAT(param_sharding->Disassemble(Shape({6, 6, 6})),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("Numbers of dimensions don't match. From "
                            "Shape 3 vs from ShardingParam 2")));
}

TEST_P(ShardingParamShardingSpecTest, DisassembleFailsForUnevenSharding) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);

  EXPECT_THAT(
      param_sharding->Disassemble(Shape({7, 6})),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("Uneven shard is not supported. dim: 7, dim_shards: 2")));
}

TEST_P(ShardingParamShardingSpecTest, IndexDomain) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);

  {
    TF_ASSERT_OK_AND_ASSIGN(auto index_domains,
                            param_sharding->IndexDomains(Shape({6, 6})));
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 2})),
                            IndexDomain(Index({0, 2}), Shape({3, 2})),
                            IndexDomain(Index({0, 4}), Shape({3, 2})),
                            IndexDomain(Index({3, 0}), Shape({3, 2})),
                            IndexDomain(Index({3, 2}), Shape({3, 2})),
                            IndexDomain(Index({3, 4}), Shape({3, 2}))));
  }
}

TEST_P(ShardingParamShardingSpecTest, IndexDomainWithPermutation) {
  ShardingParam param{/*dim_shards=*/{2, 3},
                      {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}};
  ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);

  {
    TF_ASSERT_OK_AND_ASSIGN(auto index_domains,
                            param_sharding->IndexDomains(Shape({6, 6})));
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 2})),
                            IndexDomain(Index({0, 4}), Shape({3, 2})),
                            IndexDomain(Index({3, 2}), Shape({3, 2})),
                            IndexDomain(Index({0, 2}), Shape({3, 2})),
                            IndexDomain(Index({3, 0}), Shape({3, 2})),
                            IndexDomain(Index({3, 4}), Shape({3, 2}))));
  }
}

TEST_P(ShardingParamShardingSpecTest, IndexDomainWithReplication) {
  ShardingParam param{/*dim_shards=*/{2, 1},
                      {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 3}}};
  ShardingSpecRef param_sharding = ShardingParamShardingSpec::Create(param);

  {
    TF_ASSERT_OK_AND_ASSIGN(auto index_domains,
                            param_sharding->IndexDomains(Shape({6, 6})));
    EXPECT_THAT(index_domains,
                ElementsAre(IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({0, 0}), Shape({3, 6})),
                            IndexDomain(Index({3, 0}), Shape({3, 6})),
                            IndexDomain(Index({3, 0}), Shape({3, 6})),
                            IndexDomain(Index({3, 0}), Shape({3, 6}))));
  }
}

TEST_P(ShardingParamShardingSpecTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      *ShardingParamShardingSpec::Create(
          ShardingParam{/*dim_shards=*/{2, 3},
                        {/*permutation=*/{1, 0}, /*axis_sizes=*/{3, 2}}}),
      *ShardingParamShardingSpec::Create(
          ShardingParam{/*dim_shards=*/{3, 2},
                        {/*permutation=*/{0, 1}, /*axis_sizes=*/{3, 2}}}),
  }));
}

INSTANTIATE_TEST_SUITE_P(NumShards, SingleDeviceShardingSpecTest,
                         testing::Values(ShardingSpecTestParam{
                             /*num_shards=*/6}));
INSTANTIATE_TEST_SUITE_P(NumShards, OpaqueShardingSpecTest,
                         testing::Values(ShardingSpecTestParam{
                             /*num_shards=*/6}));
INSTANTIATE_TEST_SUITE_P(NumShards, ConcreteShardingSpecTest,
                         testing::Values(ShardingSpecTestParam{
                             /*num_shards=*/6}));
INSTANTIATE_TEST_SUITE_P(NumShards, ConcreteEvenShardingSpecTest,
                         testing::Values(ShardingSpecTestParam{
                             /*num_shards=*/6}));
INSTANTIATE_TEST_SUITE_P(NumShards, ShardingParamShardingSpecTest,
                         testing::Values(ShardingSpecTestParam{
                             /*num_shards=*/6}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
