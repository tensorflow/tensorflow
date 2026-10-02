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
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.pb.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/serdes_any_version_accessor.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;

class RemapPlanTest
    : public testing::TestWithParam<test_util::DeviceTestParam> {
 public:
  RemapPlanTest() : fixture_(GetParam()) {}

  Client* client() { return fixture_.client(); }
  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

  ArraySpec GetDummySpec() {
    return ArraySpec{
        /*dtype=*/DType(DType::kS32),
        /*shape=*/Shape({4, 3}),
        /*sharding=*/
        ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                     /*shape=*/Shape({4, 3}),
                                     /*shard_shape=*/Shape({2, 3}))};
  }

 private:
  test_util::DeviceTestFixture fixture_;
};

TEST_P(RemapPlanTest, EmptyInputSpecs) {
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  RemapPlan plan(/*input_specs=*/{}, std::move(output_specs),
                 /*input_devices_for_output_map=*/{});
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Must have at least one input")));
  EXPECT_THAT(plan.Validate(), absl_testing::StatusIs(
                                   absl::StatusCode::kInvalidArgument,
                                   HasSubstr("Must have at least one input")));
}

TEST_P(RemapPlanTest, EmptyOutputSpecs) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  RemapPlan plan(std::move(input_specs), /*output_specs=*/{},
                 /*input_devices_for_output_map=*/{});
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Must have at least one output")));
  EXPECT_THAT(plan.Validate(), absl_testing::StatusIs(
                                   absl::StatusCode::kInvalidArgument,
                                   HasSubstr("Must have at least one output")));
}

TEST_P(RemapPlanTest, NullInputSharding) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  input_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kS32),
                                  /*shape=*/Shape({2, 3}),
                                  /*sharding=*/nullptr});
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map[0] = {{0, output_specs[0].sharding->devices()}};
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Input array 1 has null sharding")));
  EXPECT_THAT(
      plan.Validate(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Input array 1 has null sharding")));
}

TEST_P(RemapPlanTest, NullOutputSharding) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  output_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kS32),
                                   /*shape=*/Shape({2, 3}),
                                   /*sharding=*/nullptr});
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map[0] = {{0, input_specs[0].sharding->devices()}};
  input_devices_for_output_map[1] = {{0, input_specs[0].sharding->devices()}};
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Output array 1 has null sharding")));
  EXPECT_THAT(
      plan.Validate(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Output array 1 has null sharding")));
}

TEST_P(RemapPlanTest, EmptyInputDevicesForOutputMap) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kF32),  // dtype differs
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 /*input_devices_for_output_map=*/{});
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Must have at least one mapping in "
                                       "`input_devices_for_output_map`")));
  EXPECT_THAT(
      plan.Validate(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Must have at least one mapping in "
                                       "`input_devices_for_output_map`")));
}

TEST_P(RemapPlanTest, InvalidInputArrayIndex) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert({0, {{/*in_array=*/1, GetDevices({0})}}});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Input buffer index 1 in `input_devices_for_output_map` is "
                    "out of range [0, 0]")));
  EXPECT_THAT(
      plan.Validate(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Input buffer index 1 in `input_devices_for_output_map` is "
                    "out of range [0, 0]")));
}

TEST_P(RemapPlanTest, InvalidOutputArrayIndex) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert({1, {{/*in_array=*/0, GetDevices({0})}}});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Output buffer index 1 in `input_devices_for_output_map` "
                    "is out of range [0, 0]")));
  EXPECT_THAT(
      plan.Validate(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Output buffer index 1 in `input_devices_for_output_map` "
                    "is out of range [0, 0]")));
}

TEST_P(RemapPlanTest, InvalidOutputDtype) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kF32),  // dtype differs
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert({0, {{/*in_array=*/0, GetDevices({0})}}});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(plan.ValidateArraySpecs(),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("Input and output must have the same dtype")));
  EXPECT_THAT(plan.Validate(),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("Input and output must have the same dtype")));
}

TEST_P(RemapPlanTest, InvalidOutputDtypeFromMixedInputDtypes) {
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
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(array_spec_s32);
  input_specs.push_back(array_spec_f32);
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(array_spec_f32);

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert(
      {0,
       {{/*in_array=*/0, GetDevices({0})}, {/*in_array=*/1, GetDevices({1})}}});

  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));

  EXPECT_THAT(plan.ValidateArraySpecs(),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("Input and output must have the same dtype")));
  EXPECT_THAT(plan.Validate(),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("Input and output must have the same dtype")));
}

TEST_P(RemapPlanTest, InvalidShardShape) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({3, 2}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({3, 2}),
                                             /*shard_shape=*/Shape({3, 2}))});
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert({0, {{/*in_array=*/0, GetDevices({0})}}});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Input and output must have the same shard shape")));
  EXPECT_THAT(
      plan.Validate(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Input and output must have the same shard shape")));
}

TEST_P(RemapPlanTest, InvalidMemoryKind) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(ArraySpec{
      /*dtype=*/DType(DType::kS32),
      /*shape=*/Shape({2, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind("host"),
                                   /*shape=*/Shape({2, 3}),
                                   /*shard_shape=*/Shape({2, 3}))});
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(
      ArraySpec{/*dtype=*/DType(DType::kS32),
                /*shape=*/Shape({2, 3}),
                /*sharding=*/
                ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                             /*shape=*/Shape({2, 3}),
                                             /*shard_shape=*/Shape({2, 3}))});
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert({0, {{/*in_array=*/0, GetDevices({0})}}});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(
      plan.ValidateArraySpecs(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Input and output must have the same memory kind")));
  EXPECT_THAT(
      plan.Validate(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Input and output must have the same memory kind")));
}

TEST_P(RemapPlanTest, InvalidLayout) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(ArraySpec{
      /*dtype=*/DType(DType::kS32),
      /*shape=*/Shape({2, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                   /*shape=*/Shape({2, 3}),
                                   /*shard_shape=*/Shape({2, 3})),
      /*layout=*/
      std::make_shared<xla::PjRtLayout>(
          xla::LayoutUtil::MakeDescendingLayout(2)),
  });
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(ArraySpec{
      /*dtype=*/DType(DType::kS32),
      /*shape=*/Shape({2, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                   /*shape=*/Shape({2, 3}),
                                   /*shard_shape=*/Shape({2, 3})),
      /*layout=*/
      std::make_shared<xla::PjRtLayout>(
          xla::LayoutUtil::MakeAscendingLayout(2)),  // layout differs
  });
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert({0, {{/*in_array=*/0, GetDevices({0})}}});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(plan.ValidateArraySpecs(),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("Input and output must have the same layout")));
  EXPECT_THAT(plan.Validate(),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasSubstr("Input and output must have the same layout")));
}

TEST_P(RemapPlanTest, ValidLayoutFromDifferentLayoutObjects) {
  std::vector<ArraySpec> input_specs;
  input_specs.push_back(ArraySpec{
      /*dtype=*/DType(DType::kS32),
      /*shape=*/Shape({2, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                   /*shape=*/Shape({2, 3}),
                                   /*shard_shape=*/Shape({2, 3})),
      /*layout=*/
      std::make_shared<xla::PjRtLayout>(
          xla::LayoutUtil::MakeAscendingLayout(2)),
  });
  std::vector<ArraySpec> output_specs;
  output_specs.push_back(ArraySpec{
      /*dtype=*/DType(DType::kS32),
      /*shape=*/Shape({2, 3}),
      /*sharding=*/
      ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                   /*shape=*/Shape({2, 3}),
                                   /*shard_shape=*/Shape({2, 3})),
      /*layout=*/
      std::make_shared<xla::PjRtLayout>(
          xla::LayoutUtil::MakeAscendingLayout(2)),  // same layout
  });
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert({0, {{/*in_array=*/0, GetDevices({0})}}});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_OK(plan.ValidateArraySpecs());
  EXPECT_OK(plan.Validate());
}

TEST_P(RemapPlanTest, UnassignedOutput) {
  ArraySpec dummy_spec = GetDummySpec();
  std::vector<ArraySpec> input_specs = {dummy_spec};
  std::vector<ArraySpec> output_specs = {dummy_spec, dummy_spec};
  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert(
      {0, {{/*in_array=*/0, dummy_spec.sharding->devices()}}});
  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));
  EXPECT_THAT(
      plan.Validate(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "`input_devices_for_output_map` has 1 outputs, but expected "
              "2 outputs")));
}

TEST_P(RemapPlanTest, InputDevicesForOutputMap) {
  ArraySpec dummy_spec = GetDummySpec();

  std::vector<ArraySpec> input_specs = {dummy_spec, dummy_spec};
  std::vector<ArraySpec> output_specs = {dummy_spec};

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert(
      {0,
       {{/*in_array=*/0, GetDevices({0})}, {/*in_array=*/1, GetDevices({1})}}});

  RemapPlan plan(input_specs, output_specs, input_devices_for_output_map);
  EXPECT_EQ(plan.input_devices_for_output_map().size(), 1);
  EXPECT_OK(plan.Validate());
}

TEST_P(RemapPlanTest, InvalidInputDevicesForOutputMap) {
  ArraySpec dummy_spec = GetDummySpec();

  std::vector<ArraySpec> input_specs = {dummy_spec, dummy_spec};
  std::vector<ArraySpec> output_specs = {dummy_spec};

  {
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
    map.insert({1, {{0, GetDevices({0})}}});
    RemapPlan plan(input_specs, output_specs, std::move(map));
    EXPECT_THAT(plan.Validate(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Output buffer index 1")));
  }

  {
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
    map.insert({0, {{2, GetDevices({0})}}});
    RemapPlan plan(input_specs, output_specs, std::move(map));
    EXPECT_THAT(plan.Validate(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Input buffer index 2")));
  }

  {
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
    map.insert(
        {0,
         {RemapPlan::InputDeviceRange{/*in_array=*/0,
                                      /*input_devices=*/DeviceListRef()}}});
    RemapPlan plan(input_specs, output_specs, std::move(map));
    EXPECT_THAT(plan.Validate(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("null input_devices")));
  }

  {
    ArraySpec f32_spec{
        /*dtype=*/DType(DType::kF32),
        /*shape=*/Shape({4, 3}),
        /*sharding=*/
        ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                     /*shape=*/Shape({4, 3}),
                                     /*shard_shape=*/Shape({2, 3}))};
    std::vector<ArraySpec> in_specs = {f32_spec};
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
    map.insert({0, {{0, GetDevices({0})}}});
    RemapPlan plan(in_specs, output_specs, std::move(map));
    EXPECT_THAT(plan.Validate(),
                absl_testing::StatusIs(
                    absl::StatusCode::kInvalidArgument,
                    HasSubstr("Input and output must have the same dtype")));
  }

  {
    ArraySpec layout_spec{
        /*dtype=*/DType(DType::kS32),
        /*shape=*/Shape({4, 3}),
        /*sharding=*/
        ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind(),
                                     /*shape=*/Shape({4, 3}),
                                     /*shard_shape=*/Shape({2, 3})),
        /*layout=*/
        std::make_shared<xla::PjRtLayout>(
            xla::LayoutUtil::MakeAscendingLayout(2))};
    std::vector<ArraySpec> in_specs = {layout_spec};
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
    map.insert({0, {{0, GetDevices({0})}}});
    RemapPlan plan(in_specs, output_specs, std::move(map));
    EXPECT_THAT(plan.Validate(),
                absl_testing::StatusIs(
                    absl::StatusCode::kInvalidArgument,
                    HasSubstr("Input and output must have the same layout")));
  }

  {
    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
    map.insert({0, {{0, GetDevices({2})}}});
    RemapPlan plan(input_specs, output_specs, std::move(map));
    EXPECT_THAT(
        plan.Validate(),
        absl_testing::StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("not in the input array's addressable device list")));
  }
}

TEST_P(RemapPlanTest, CheckOneInputToOneOutput) {
  ArraySpec dummy_spec = GetDummySpec();

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
  map.insert({0, {{0, dummy_spec.sharding->devices()}}});
  RemapPlan plan({dummy_spec}, {dummy_spec}, std::move(map));

  EXPECT_OK(
      plan.CheckArrayCopySemantics(xla::ifrt::ArrayCopySemantics::kReuseInput));
  EXPECT_OK(plan.CheckArrayCopySemantics(
      xla::ifrt::ArrayCopySemantics::kDonateInput));
}

TEST_P(RemapPlanTest, CheckOneInputToMultipleOutputs) {
  ArraySpec dummy_spec = GetDummySpec();

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
  map.insert({0, {{0, dummy_spec.sharding->devices()}}});
  map.insert({1, {{0, dummy_spec.sharding->devices()}}});
  RemapPlan plan({dummy_spec}, {dummy_spec, dummy_spec}, std::move(map));

  EXPECT_OK(
      plan.CheckArrayCopySemantics(xla::ifrt::ArrayCopySemantics::kReuseInput));
  EXPECT_OK(plan.CheckArrayCopySemantics(
      xla::ifrt::ArrayCopySemantics::kDonateInput));
}

TEST_P(RemapPlanTest, CheckMultipleInputsToOneOutput) {
  ArraySpec dummy_spec = GetDummySpec();

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>> map;
  map.insert({0, {{0, GetDevices({0})}, {1, GetDevices({1})}}});
  RemapPlan plan({dummy_spec, dummy_spec}, {dummy_spec}, std::move(map));

  EXPECT_THAT(
      plan.CheckArrayCopySemantics(xla::ifrt::ArrayCopySemantics::kReuseInput),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("kDonateInput is required if multiple inputs are "
                    "mapped to one output")));
  EXPECT_OK(plan.CheckArrayCopySemantics(
      xla::ifrt::ArrayCopySemantics::kDonateInput));
}

TEST_P(RemapPlanTest, Hash) {
  std::vector<RemapPlan> plans;
  plans.push_back(RemapPlan());
  {
    std::vector<ArraySpec> input_specs;
    input_specs.push_back(
        ArraySpec{/*dtype=*/DType(DType::kS32),
                  /*shape=*/Shape({2, 3}),
                  /*sharding=*/
                  ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                               /*shape=*/Shape({2, 3}),
                                               /*shard_shape=*/Shape({2, 3}))});
    input_specs.push_back(
        ArraySpec{/*dtype=*/DType(DType::kF32),  // dtype differs
                  /*shape=*/Shape({2, 3}),
                  /*sharding=*/
                  ConcreteEvenSharding::Create(GetDevices({0}), MemoryKind(),
                                               /*shape=*/Shape({2, 3}),
                                               /*shard_shape=*/Shape({2, 3}))});

    plans.push_back(RemapPlan(input_specs, /*output_specs=*/{},
                              /*input_devices_for_output_map=*/{}));
  }
  {
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

    std::vector<ArraySpec> input_specs;
    input_specs.push_back(array_spec_s32);
    input_specs.push_back(array_spec_f32);
    std::vector<ArraySpec> output_specs;
    output_specs.push_back(array_spec_f32);
    output_specs.push_back(array_spec_s32);

    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
        input_devices_for_output_map;
    input_devices_for_output_map.insert({0, {{1, GetDevices({0})}}});
    input_devices_for_output_map.insert({1, {{0, GetDevices({0})}}});

    plans.push_back(
        RemapPlan(input_specs, output_specs, input_devices_for_output_map));

    absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
        input_devices_for_output_map2;
    input_devices_for_output_map2.insert({0, {{0, GetDevices({0})}}});
    input_devices_for_output_map2.insert({1, {{1, GetDevices({0})}}});

    plans.push_back(
        RemapPlan(input_specs, output_specs, input_devices_for_output_map2));
  }

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(plans));
}

INSTANTIATE_TEST_SUITE_P(NumDevices, RemapPlanTest,
                         testing::Values(test_util::DeviceTestParam{
                             /*num_devices=*/4,
                             /*num_addressable_devices=*/4}));

using RemapPlanSerDesTestParam =
    std::tuple<SerDesVersion, test_util::DeviceTestParam>;

class RemapPlanSerDesTest
    : public testing::TestWithParam<RemapPlanSerDesTestParam> {
 public:
  RemapPlanSerDesTest()
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

TEST_P(RemapPlanSerDesTest, ToFromProto) {
  Shape shape({20, 20});
  Shape shard_shape({5, 20});
  DeviceListRef devices = GetDevices({0, 1, 2, 3});
  DeviceListRef devices_2 = GetDevices({1, 2});
  ShardingRef sharding =
      ConcreteEvenSharding::Create(devices, MemoryKind(), /*shape=*/shape,
                                   /*shard_shape=*/shard_shape);

  std::vector<ArraySpec> input_specs;
  input_specs.reserve(2);
  input_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kF32),
                                  /*shape=*/shape, /*sharding=*/sharding});
  input_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kF32),
                                  /*shape=*/shape, /*sharding=*/sharding});

  std::vector<ArraySpec> output_specs;
  output_specs.reserve(2);
  output_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kF32),
                                   /*shape=*/shape, /*sharding=*/sharding});
  output_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kF32),
                                   /*shape=*/shape, /*sharding=*/sharding});

  absl::flat_hash_map<int, std::vector<RemapPlan::InputDeviceRange>>
      input_devices_for_output_map;
  input_devices_for_output_map.insert({0, {{1, devices}}});
  input_devices_for_output_map.insert({1, {{0, devices}, {1, devices_2}}});

  RemapPlan plan(std::move(input_specs), std::move(output_specs),
                 std::move(input_devices_for_output_map));

  ASSERT_OK_AND_ASSIGN(RemapPlanProto plan_proto, plan.ToProto(version()));
  if (version().version_number() >= SerDesVersionNumber(6)) {
    EXPECT_EQ(plan_proto.version_number(), SerDesVersionNumber(6).value());
    EXPECT_THAT(plan_proto.mappings(), IsEmpty());
  } else {
    EXPECT_EQ(plan_proto.version_number(), SerDesVersionNumber(0).value());
    EXPECT_THAT(plan_proto.mappings(), Not(IsEmpty()));

    RemapPlanProto mappings_only_proto = plan_proto;
    mappings_only_proto.clear_input_devices_for_output();
    ASSERT_OK_AND_ASSIGN(RemapPlan plan_from_mappings,
                         RemapPlan::FromProto(client(), mappings_only_proto));
    EXPECT_EQ(plan, plan_from_mappings);
  }
  ASSERT_OK_AND_ASSIGN(RemapPlan plan_copy,
                       RemapPlan::FromProto(client(), plan_proto));

  EXPECT_EQ(plan, plan_copy);
  ASSERT_EQ(plan.input_devices_for_output_map().size(),
            plan_copy.input_devices_for_output_map().size());
  for (const auto& [out_array, input_devices] :
       // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
       plan.input_devices_for_output_map()) {
    ASSERT_TRUE(plan_copy.input_devices_for_output_map().contains(out_array));
    const auto& copy_input_devices =
        plan_copy.input_devices_for_output_map().at(out_array);
    ASSERT_EQ(copy_input_devices.size(), input_devices.size());
    for (int i = 0; i < input_devices.size(); ++i) {
      EXPECT_EQ(copy_input_devices[i].in_array, input_devices[i].in_array);
      EXPECT_EQ(copy_input_devices[i].input_devices,
                input_devices[i].input_devices);
    }
  }

  EXPECT_THAT(plan_copy.output_specs(), SizeIs(2));
  for (const auto& spec : plan_copy.input_specs()) {
    EXPECT_EQ(spec.dtype, DType(DType::kF32));
    EXPECT_EQ(spec.shape, shape);
    const auto* sharding_copy =
        dyn_cast<ConcreteEvenSharding>(spec.sharding.get());
    ASSERT_NE(sharding_copy, nullptr);
    EXPECT_EQ(*sharding_copy->devices(), *devices);
    EXPECT_EQ(sharding_copy->shape(), shape);
    EXPECT_EQ(sharding_copy->shard_shape(), shard_shape);
  }
  for (const auto& spec : plan_copy.output_specs()) {
    EXPECT_EQ(spec.dtype, DType(DType::kF32));
    EXPECT_EQ(spec.shape, shape);
    const auto* sharding_copy =
        dyn_cast<ConcreteEvenSharding>(spec.sharding.get());
    ASSERT_NE(sharding_copy, nullptr);
    EXPECT_EQ(*sharding_copy->devices(), *devices);
    EXPECT_EQ(sharding_copy->shape(), shape);
    EXPECT_EQ(sharding_copy->shard_shape(), shard_shape);
  }
}

TEST_P(RemapPlanSerDesTest, FromProtoWithMappings) {
  if (version().version_number() != SerDesVersionNumber(0)) {
    GTEST_SKIP() << "FromProtoWithMappings only tests SerDes version 0.";
  }
  Shape shape({20, 20});
  Shape shard_shape({5, 20});
  DeviceListRef devices = GetDevices({0, 1, 2, 3});
  ShardingRef sharding =
      ConcreteEvenSharding::Create(devices, MemoryKind(), /*shape=*/shape,
                                   /*shard_shape=*/shard_shape);

  std::vector<ArraySpec> input_specs;
  input_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kF32),
                                  /*shape=*/shape, /*sharding=*/sharding});

  std::vector<ArraySpec> output_specs;
  output_specs.push_back(ArraySpec{/*dtype=*/DType(DType::kF32),
                                   /*shape=*/shape, /*sharding=*/sharding});

  RemapPlanProto plan_proto;
  plan_proto.set_version_number(SerDesVersionNumber(0).value());
  const SerDesVersion version_0 =
      SerDesAnyVersionAccessor::Get(SerDesVersionNumber(0));
  ASSERT_OK(input_specs[0].ToProto(*plan_proto.add_input_specs(), version_0));
  ASSERT_OK(output_specs[0].ToProto(*plan_proto.add_output_specs(), version_0));

  // Test contiguous interval mapping.
  {
    RemapPlanProto proto = plan_proto;
    auto* mapping_proto = proto.add_mappings();
    mapping_proto->set_in_array(0);
    mapping_proto->set_out_array(0);
    mapping_proto->add_from_start(0);
    mapping_proto->add_from_end(4);
    mapping_proto->add_from_step(1);
    mapping_proto->add_to_start(0);
    mapping_proto->add_to_end(4);
    mapping_proto->add_to_step(1);

    ASSERT_OK_AND_ASSIGN(RemapPlan plan, RemapPlan::FromProto(client(), proto));
    EXPECT_OK(plan.Validate());

    ASSERT_EQ(plan.input_devices_for_output_map().size(), 1);
    ASSERT_TRUE(plan.input_devices_for_output_map().contains(0));
    const auto& input_devices = plan.input_devices_for_output_map().at(0);
    ASSERT_EQ(input_devices.size(), 1);
    EXPECT_EQ(input_devices[0].in_array, 0);
    EXPECT_EQ(input_devices[0].input_devices, devices);
  }

  // Test strided interval mapping where end > num_shards (e.g. start=1, end=5,
  // step=2 on a 4-shard array).
  {
    RemapPlanProto proto = plan_proto;
    auto* mapping_proto = proto.add_mappings();
    mapping_proto->set_in_array(0);
    mapping_proto->set_out_array(0);
    // Maps shards {1, 3} of input to shards {0, 2} of output.
    mapping_proto->add_from_start(1);
    mapping_proto->add_from_end(5);
    mapping_proto->add_from_step(2);
    mapping_proto->add_to_start(0);
    mapping_proto->add_to_end(4);
    mapping_proto->add_to_step(2);
    // Maps shards {0, 2} of input to shards {1, 3} of output.
    mapping_proto->add_from_start(0);
    mapping_proto->add_from_end(4);
    mapping_proto->add_from_step(2);
    mapping_proto->add_to_start(1);
    mapping_proto->add_to_end(5);
    mapping_proto->add_to_step(2);

    ASSERT_OK_AND_ASSIGN(RemapPlan plan, RemapPlan::FromProto(client(), proto));
    EXPECT_OK(plan.Validate());

    ASSERT_EQ(plan.input_devices_for_output_map().size(), 1);
    ASSERT_TRUE(plan.input_devices_for_output_map().contains(0));
    const auto& input_devices = plan.input_devices_for_output_map().at(0);
    ASSERT_EQ(input_devices.size(), 1);
    EXPECT_EQ(input_devices[0].in_array, 0);
    DeviceListRef expected_devices = GetDevices({1, 3, 0, 2});
    EXPECT_EQ(input_devices[0].input_devices, expected_devices);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumDevices, RemapPlanSerDesTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(test_util::DeviceTestParam{
                         /*num_devices=*/4,
                         /*num_addressable_devices=*/4})),
    [](const testing::TestParamInfo<RemapPlanSerDesTestParam>& info) {
      return absl::StrCat("version_",
                          std::get<0>(info.param).version_number().value(),
                          "_num_devices_", std::get<1>(info.param).num_devices,
                          "_num_addressable_devices_",
                          std::get<1>(info.param).num_addressable_devices);
    });

}  // namespace
}  // namespace ifrt
}  // namespace xla
