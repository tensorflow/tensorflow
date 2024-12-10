/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/remap_plan.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

class RemapPlanTest : public test_util::DeviceTest {
 public:
  ArraySpec GetDummySpec() {
    return ArraySpec{
        /*dtype=*/DType(DType::kS32),
        /*shape=*/Shape({4, 3}),
        /*sharding=*/
        ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                     /*shape=*/Shape({4, 3}),
                                     /*shard_shape=*/Shape({2, 3}))};
  }
};

TEST_P(RemapPlanTest, ToFromProto) {
  RemapPlan plan;

  Shape shape({20, 20});
  Shape shard_shape({5, 20});
  tsl::RCReference<DeviceList> devices = GetDevices({0, 1, 2, 3});
  std::shared_ptr<const Sharding> sharding =
      ConcreteEvenSharding::Create(devices, MemoryKind(), /*shape=*/shape,
                                   /*shard_shape=*/shard_shape);

  plan.input_specs.reserve(2);
  plan.input_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kF32),
                                       /*shape=*/shape, /*sharding=*/sharding});
  plan.input_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kF32),
                                       /*shape=*/shape, /*sharding=*/sharding});

  plan.output_specs.reserve(2);
  plan.output_specs.push_back(ArraySpec{
      /*dtype=*/DType(DType::kF32), /*shape=*/shape, /*sharding=*/sharding});
  plan.output_specs.push_back(ArraySpec{
      /*dtype=*/DType(DType::kF32), /*shape=*/shape, /*sharding=*/sharding});

  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->reserve(2);
  plan.mappings->push_back(RemapPlan::Mapping{
      /*in_array=*/0, /*out_array=*/1,
      /*from=*/{RemapPlan::Interval{0, 2, 1}, RemapPlan::Interval{2, 4, 1}},
      /*to=*/{RemapPlan::Interval{1, 4, 2}, RemapPlan::Interval{0, 4, 2}}});
  plan.mappings->push_back(RemapPlan::Mapping{
      /*in_array=*/1, /*out_array=*/0,
      /*from=*/{RemapPlan::Interval{0, 4, 2}, RemapPlan::Interval{1, 4, 2}},
      /*to=*/{RemapPlan::Interval{0, 2, 1}, RemapPlan::Interval{2, 4, 1}}});

  TF_ASSERT_OK_AND_ASSIGN(RemapPlanProto plan_proto, plan.ToProto());
  TF_ASSERT_OK_AND_ASSIGN(
      RemapPlan plan_copy,
      RemapPlan::FromProto(absl::bind_front(&Client::LookupDevice, client()),
                           plan_proto));

  EXPECT_THAT(*plan_copy.mappings, ElementsAreArray(*plan.mappings));

  EXPECT_THAT(plan_copy.output_specs, SizeIs(2));
  for (const auto& spec : plan_copy.input_specs) {
    EXPECT_EQ(spec.dtype, DType(DType::kF32));
    EXPECT_EQ(spec.shape, shape);
    const auto* sharding_copy =
        llvm::dyn_cast<ConcreteEvenSharding>(spec.sharding.get());
    ASSERT_NE(sharding_copy, nullptr);
    EXPECT_EQ(*sharding_copy->devices(), *devices);
    EXPECT_EQ(sharding_copy->shape(), shape);
    EXPECT_EQ(sharding_copy->shard_shape(), shard_shape);
  }
  for (const auto& spec : plan_copy.output_specs) {
    EXPECT_EQ(spec.dtype, DType(DType::kF32));
    EXPECT_EQ(spec.shape, shape);
    const auto* sharding_copy =
        llvm::dyn_cast<ConcreteEvenSharding>(spec.sharding.get());
    ASSERT_NE(sharding_copy, nullptr);
    EXPECT_EQ(*sharding_copy->devices(), *devices);
    EXPECT_EQ(sharding_copy->shape(), shape);
    EXPECT_EQ(sharding_copy->shard_shape(), shard_shape);
  }
}

TEST_P(RemapPlanTest, EmptyMappings) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kF32),  // dtype differs
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  EXPECT_THAT(plan.Validate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Must have at least one mapping")));
}

TEST_P(RemapPlanTest, MixedDtype) {
  RemapPlan plan;

  ArraySpec array_spec_s32{
      /*dtype=*/DType(DType::kS32),
      /*shape=*/Shape({2, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                   /*shape=*/Shape({2, 3}),
                                   /*shard_shape=*/Shape({2, 3}))};
  ArraySpec array_spec_f32{
      /*dtype=*/DType(DType::kF32),
      /*shape=*/Shape({2, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                   /*shape=*/Shape({2, 3}),
                                   /*shard_shape=*/Shape({2, 3}))};

  plan.input_specs.push_back(array_spec_s32);
  plan.input_specs.push_back(array_spec_f32);
  plan.output_specs.push_back(array_spec_f32);
  plan.output_specs.push_back(array_spec_s32);

  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/1,
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/1,
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});

  TF_EXPECT_OK(plan.Validate());
}

TEST_P(RemapPlanTest, InvalidOutputDtype) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kF32),  // dtype differs
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  EXPECT_THAT(plan.Validate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input and output must have the same dtype")));
}

TEST_P(RemapPlanTest, InvalidOutputDtypeFromMixedInputDtypes) {
  RemapPlan plan;
  ArraySpec array_spec_s32{
      /*dtype=*/DType(DType::kS32),
      /*shape=*/Shape({4, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                   /*shape=*/Shape({4, 3}),
                                   /*shard_shape=*/Shape({2, 3}))};
  ArraySpec array_spec_f32{
      /*dtype=*/DType(DType::kF32),
      /*shape=*/Shape({4, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                   /*shape=*/Shape({4, 3}),
                                   /*shard_shape=*/Shape({2, 3}))};
  plan.input_specs.push_back(array_spec_s32);
  plan.input_specs.push_back(array_spec_f32);
  plan.output_specs.push_back(array_spec_f32);

  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();

  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/1,
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{1, 2, 1}},
                         /*to=*/{RemapPlan::Interval{1, 2, 1}}});

  EXPECT_THAT(plan.Validate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input and output must have the same dtype")));
}

TEST_P(RemapPlanTest, InvalidInputArrayIndex) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/1,  // Invalid in_array
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  EXPECT_THAT(
      plan.Validate(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("mappings[0].in_array must be in [0, 0], but is 1")));
}

TEST_P(RemapPlanTest, InvalidOutputArrayIndex) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/1,  // Invalid out_array
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  EXPECT_THAT(
      plan.Validate(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("mappings[0].out_array must be in [0, 0], but is 1")));
}

TEST_P(RemapPlanTest, InvalidIntervalCount) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(RemapPlan::Mapping{
      /*in_array=*/0,
      /*out_array=*/0,
      /*from=*/{RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{0, 1, 1}},
      /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  EXPECT_THAT(
      plan.Validate(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("mappings[0].from and mappings[0].to must have the same "
                    "number of intervals, but has 2 and 1 intervals")));
}

TEST_P(RemapPlanTest, InvalidShardIndex) {
  auto run = [&](RemapPlan::Interval from, RemapPlan::Interval to) {
    RemapPlan plan;
    plan.input_specs.push_back(
        ArraySpec{/*dtype=*/DType(DType::kS32),
                  /*shape=*/Shape({2, 3}),
                  /*sharding=*/
                  ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                               /*shape=*/Shape({2, 3}),
                                               /*shard_shape=*/Shape({2, 3}))});
    plan.output_specs.push_back(
        ArraySpec{/*dtype=*/DType(DType::kS32),
                  /*shape=*/Shape({2, 3}),
                  /*sharding=*/
                  ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                               /*shape=*/Shape({2, 3}),
                                               /*shard_shape=*/Shape({2, 3}))});
    plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
    plan.mappings->push_back(RemapPlan::Mapping{/*in_array=*/0, /*out_array=*/0,
                                                /*from=*/{from},
                                                /*to=*/{to}});
    return plan.Validate();
  };

  EXPECT_THAT(run(RemapPlan::Interval{-1, 1, 1}, RemapPlan::Interval{0, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("start must be in [0, 0], but is -1")));
  EXPECT_THAT(run(RemapPlan::Interval{1, 1, 1}, RemapPlan::Interval{0, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("start must be in [0, 0], but is 1")));
  EXPECT_THAT(run(RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{-1, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("start must be in [0, 0], but is -1")));
  EXPECT_THAT(run(RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{1, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("start must be in [0, 0], but is 1")));

  EXPECT_THAT(run(RemapPlan::Interval{0, -1, 1}, RemapPlan::Interval{0, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("end must be in [0, 1], but is -1")));
  EXPECT_THAT(run(RemapPlan::Interval{0, 2, 1}, RemapPlan::Interval{0, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("end must be in [0, 1], but is 2")));
  EXPECT_THAT(run(RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{0, -1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("end must be in [0, 1], but is -1")));
  EXPECT_THAT(run(RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{0, 2, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("end must be in [0, 1], but is 2")));

  EXPECT_THAT(run(RemapPlan::Interval{0, 1, 0}, RemapPlan::Interval{0, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("step must be positive, but is 0")));
  EXPECT_THAT(run(RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{0, 1, -1}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("step must be positive, but is -1")));
}

TEST_P(RemapPlanTest, AlreadyUsedInputShard) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                             /*shape=*/Shape({4, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(RemapPlan::Mapping{
      /*in_array=*/0,
      /*out_array=*/0,
      /*from=*/{RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{0, 1, 1}},
      /*to=*/{RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{1, 2, 1}}});
  EXPECT_THAT(plan.Validate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input array 0 shard 0 is already used")));
}

TEST_P(RemapPlanTest, UnassignedOutputShard) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                             /*shape=*/Shape({4, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 1, 1}},
                         /*to=*/{RemapPlan::Interval{0, 1, 1}}});
  EXPECT_THAT(plan.Validate(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Output array 0 shard 1 is unassigned")));
}

TEST_P(RemapPlanTest, AlreadyAssignedOutputShard) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                             /*shape=*/Shape({4, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(RemapPlan::Mapping{
      /*in_array=*/0,
      /*out_array=*/0,
      /*from=*/{RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{1, 2, 1}},
      /*to=*/{RemapPlan::Interval{0, 1, 1}, RemapPlan::Interval{0, 1, 1}}});
  EXPECT_THAT(
      plan.Validate(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Output array 0 shard 0 is already assigned")));
}

TEST_P(RemapPlanTest, InvalidOutputDevices) {
  RemapPlan plan;
  plan.input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                             /*shape=*/Shape({4, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({4, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({1, 0}), MemoryKind(),
                                             /*shape=*/Shape({4, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  plan.mappings->push_back(
      RemapPlan::Mapping{/*in_array=*/0,
                         /*out_array=*/0,
                         /*from=*/{RemapPlan::Interval{0, 2, 1}},
                         /*to=*/{RemapPlan::Interval{0, 2, 1}}});
  EXPECT_THAT(
      plan.Validate(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Output array 0 devices and sharding devices do not match")));
}

TEST_P(RemapPlanTest, CheckOneInputToOneOutput) {
  ArraySpec dummy_spec = GetDummySpec();

  RemapPlan plan;
  plan.input_specs.push_back(dummy_spec);
  plan.output_specs.push_back(dummy_spec);
  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  *plan.mappings = {RemapPlan::Mapping{/*in_array=*/0,
                                       /*out_array=*/0,
                                       /*from=*/{RemapPlan::Interval{0, 2, 1}},
                                       /*to=*/{RemapPlan::Interval{0, 2, 1}}}};

  TF_EXPECT_OK(
      plan.CheckArrayCopySemantics(xla::ifrt::ArrayCopySemantics::kReuseInput));
  TF_EXPECT_OK(plan.CheckArrayCopySemantics(
      xla::ifrt::ArrayCopySemantics::kDonateInput));
}

TEST_P(RemapPlanTest, CheckOneInputToMultipleOutputs) {
  ArraySpec dummy_spec = GetDummySpec();

  RemapPlan plan;
  plan.input_specs = {dummy_spec};
  plan.output_specs = {dummy_spec, dummy_spec};

  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  *plan.mappings = {RemapPlan::Mapping{/*in_array=*/0,
                                       /*out_array=*/0,
                                       /*from=*/{RemapPlan::Interval{0, 2, 1}},
                                       /*to=*/{RemapPlan::Interval{0, 2, 1}}},
                    RemapPlan::Mapping{/*in_array=*/0,
                                       /*out_array=*/1,
                                       /*from=*/{RemapPlan::Interval{0, 2, 1}},
                                       /*to=*/{RemapPlan::Interval{0, 2, 1}}}};

  TF_EXPECT_OK(
      plan.CheckArrayCopySemantics(xla::ifrt::ArrayCopySemantics::kReuseInput));
  TF_EXPECT_OK(plan.CheckArrayCopySemantics(
      xla::ifrt::ArrayCopySemantics::kDonateInput));
}

TEST_P(RemapPlanTest, CheckMultipleInputsToOneOutput) {
  ArraySpec dummy_spec = GetDummySpec();

  RemapPlan plan;
  plan.input_specs = {dummy_spec, dummy_spec};
  plan.output_specs = {dummy_spec};

  plan.mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  *plan.mappings = {RemapPlan::Mapping{/*in_array=*/0,
                                       /*out_array=*/0,
                                       /*from=*/{RemapPlan::Interval{0, 2, 1}},
                                       /*to=*/{RemapPlan::Interval{0, 2, 1}}},
                    RemapPlan::Mapping{/*in_array=*/1,
                                       /*out_array=*/0,
                                       /*from=*/{RemapPlan::Interval{0, 2, 1}},
                                       /*to=*/{RemapPlan::Interval{0, 2, 1}}}};

  EXPECT_THAT(
      plan.CheckArrayCopySemantics(xla::ifrt::ArrayCopySemantics::kReuseInput),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("kDonateInput is required if multiple inputs are "
                         "mapped to one output")));
  TF_EXPECT_OK(plan.CheckArrayCopySemantics(
      xla::ifrt::ArrayCopySemantics::kDonateInput));
}

INSTANTIATE_TEST_SUITE_P(NumDevices, RemapPlanTest,
                         testing::Values(test_util::DeviceTestParam{
                             /*num_devices=*/4,
                             /*num_addressable_devices=*/4}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
