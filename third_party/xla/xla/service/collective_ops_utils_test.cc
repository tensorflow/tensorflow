/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/collective_ops_utils.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(CollectiveOpsUtilsTest, GetParticipatingIDs_NoReplicaGroups) {
  std::vector<int> actual =
      GetParticipatingIDs(CollectiveOpGroupMode::kFlattenedID,
                          /*current_id=*/0, /*total_participant_count=*/3,
                          /*groups=*/{})
          .value();
  std::vector<int> expected = {0, 1, 2};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, GetParticipatingIDs_ReplicaGroups) {
  std::vector<ReplicaGroup> replica_groups(3);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(4);
  replica_groups[1].add_replica_ids(1);
  replica_groups[1].add_replica_ids(5);
  replica_groups[2].add_replica_ids(2);
  replica_groups[2].add_replica_ids(3);

  std::vector<int> actual =
      GetParticipatingIDs(CollectiveOpGroupMode::kFlattenedID,
                          /*current_id=*/1,
                          /*total_participant_count=*/std::nullopt,
                          replica_groups)
          .value();
  std::vector<int> expected = {1, 5};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, CollectiveWithChannelId) {
  absl::string_view hlo_string = R"(
  HloModule module, is_scheduled=true

  ENTRY %cluster {
    %param0 = f32[512]{0} parameter(0)
    %copy0 = f32[512]{0} copy(param0)
    %reshape0 = f32[1,1,512]{2,0,1} reshape(f32[512]{0} %copy0)
    %all-gather = f32[1,4,512]{2,0,1} all-gather(f32[1,1,512]{2,0,1} %reshape0), channel_id=3621, replica_groups={{0,1,2,3}}, dimensions={1}, use_global_device_ids=true
    %copy1 = f32[1,4,512]{2,0,1} copy(all-gather)
    ROOT root = f32[1,4,512]{2,1,0} copy(%copy1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  HloInstruction *all_gather =
      module->entry_computation()->GetInstructionWithName("all-gather");

  EXPECT_TRUE(IsCollectiveWithChannelId(all_gather));
}

TEST(CollectiveOpsUtilsTest, CollectiveWithChannelId2) {
  ReplicaGroup group;
  for (int64_t i = 0; i < 8; i++) {
    group.add_replica_ids(i);
  }

  auto builder = HloComputation::Builder("CollectiveWithChannelId2");
  TF_ASSERT_OK_AND_ASSIGN(
      HloInstruction * param_0,
      builder.AddParameter(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(BF16, {1, 512, 4096}), "p0")));
  HloInstruction *instr =
      builder.AddInstruction(HloInstruction::CreateAllGather(
          ShapeUtil::MakeShape(BF16, {1, 4096, 4096}), {param_0}, 1, {group},
          true, 231, true));
  auto computation = builder.Build(
      builder.AddInstruction(HloInstruction::CreateTuple({instr})));
  auto fusion =
      HloInstruction::CreateFusion(ShapeUtil::MakeShape(BF16, {1, 4096, 4096}),
                                   HloInstruction::FusionKind::kOutput,
                                   {param_0}, computation.get(), "fusion");

  EXPECT_TRUE(IsCollectiveWithChannelId(fusion.get()));
}

}  // namespace

// Tests for GetCollectOpGroupMode
namespace GetCollectiveOpGroupModeTest {
struct TestCase {
  bool has_channel_id;
  std::optional<bool> use_global_device_ids;
  std::optional<xla::CollectiveOpGroupMode> expected;

  std::string ToString() const {
    std::ostringstream s;
    s << (has_channel_id ? "chnl" : "nochnl");
    s << "_"
      << (use_global_device_ids
              ? (*use_global_device_ids ? "ugdi_true" : "ugdi_false")
              : "nougdi");
    return s.str();
  }
};

std::vector<TestCase> GetTestCases() {
  const std::vector<TestCase> test_cases = {
      // clang-format off
      // has_channel_id, use_global_device_ids, expected mode
      {false, std::nullopt, CollectiveOpGroupMode::kCrossReplica},
      {false, false,         CollectiveOpGroupMode::kCrossReplica},
      {false, true,          std::nullopt},
      {true,  std::nullopt, CollectiveOpGroupMode::kCrossPartition},
      {true,  false,         CollectiveOpGroupMode::kCrossReplicaAndPartition},
      {true,  true,          CollectiveOpGroupMode::kFlattenedID},
      // clang-format on
  };
  return test_cases;
}

class GetCollectOpGroupModeTest : public testing::TestWithParam<TestCase> {};

TEST_P(GetCollectOpGroupModeTest, Test) {
  const TestCase &tc = GetParam();
  absl::StatusOr<CollectiveOpGroupMode> actual =
      GetCollectiveOpGroupMode(tc.has_channel_id, tc.use_global_device_ids);
  if (tc.expected) {
    TF_ASSERT_OK(actual.status());
    EXPECT_EQ(*actual, *tc.expected);
  } else {
    EXPECT_FALSE(actual.ok());
  }
}

INSTANTIATE_TEST_SUITE_P(GetCollectOpGroupMode, GetCollectOpGroupModeTest,
                         testing::ValuesIn(GetTestCases()));
}  // namespace GetCollectiveOpGroupModeTest

// Tests for GetParticipatingDevices
namespace GetParticipatingDevicesTest {

// Test case for GetParticipatingDevices. Describes all the inputs to the
// function and for a given "setup", multiple "current_id" values and the
// expected output corresponding to those values.
struct TestCase {
  xla::Array2D<int> device_assignment;
  std::vector<std::vector<int>> replica_groups;
  bool has_channel_id;
  std::optional<bool> use_global_device_ids;

  // For a given test case, its useful to test multiple 'current_id' inputs.
  struct CurrentIdAndOutput {
    int current_id;
    std::vector<int> expected_output;
  };
  std::vector<CurrentIdAndOutput> subtests;

  std::vector<std::vector<int>> participating_device_groups;
  bool expected_failure;

  std::string ToString() const;
};

// Please see the comment for GetParticipatingDevices() for a description of
// modes and their behavior.
std::string TestCase::ToString() const {
  std::ostringstream s;
  absl::StatusOr<CollectiveOpGroupMode> group_mode =
      GetCollectiveOpGroupMode(has_channel_id, use_global_device_ids);
  if (group_mode.ok()) {
    s << CollectiveOpGroupModeToString(*group_mode);
  } else {
    s << "Invalid";
  }

  s << "_" << device_assignment.n1() << "x" << device_assignment.n2();
  s << "_" << (replica_groups.empty() ? "NoRG" : "RG");
  s << "_" << subtests.size() << "SubTests";
  return s.str();
}

std::ostream &operator<<(std::ostream &os, const TestCase &tc) {
  os << tc.ToString();
  return os;
}

std::vector<TestCase> GetTestCases() {
  std::vector<TestCase> test_cases;
  // clang-format off
  const std::vector<TestCase> cross_replica_test_cases = {
    // with empty replica groups, 1 partition.
    {
      {{33}, {44}, {55}},     // 3 replicas, 1 partition.
      {},                     // empty replica groups
      false,                  // has_channel_id
      false,                  // use_global_device_ids
      {                       // subtests
        // for empty replica group, any id should return all ids.
        {33, {33, 44, 55}},
        {44, {33, 44, 55}},
      },
      {{33, 44, 55}},          // participating device groups
      false                    // expected_failure
    },

    // empty replica groups, > 1 partition
    {
      {{33, 34}, {44, 45}, {55, 56}},  // 3r, 2p
      {},                              // empty replica groups
      false,                           // has_channel_id
      false,                           // use_global_device_ids
      // for empty replica group, any id should return all replicas within that
      // partition.
      {                                // subtests
        {33, {33, 44, 55}},
        {34, {34, 45, 56}},
        {45, {34, 45, 56}},
      },
      {{33, 44, 55}, {34, 45, 56}},    // participating device groups
      false                            // expected_failure
    },

    // non-empty replica groups, 1 partition.
    {
      {{33}, {44}, {55}},   // 3r, 1p.
      {{0}, {1, 2}},        // replica groups
      false,                // has_channel_id
      false,                // use_global_device_ids
      {                     // subtests
        // 33 is r0, so it's a singleton group.
        {33, {33}},
        // 44 is r1, so it should give {r1, r2}.
        {44, {44, 55}},
      },
      {{ 33 }, {44, 55}},    // participating device groups
      false                  // expected_failure
    },

    // non-empty, > 1 partition
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {{0}, {1, 2}},                    // replica groups
      false,                            // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        // 33 is r0p0, so should be singleton.
        {33, {33}},
        // 34 is r0p1, so should be singleton.
        {34, {34}},
        // 45 is r1p1, so should get r1p1 and r2p1.
        {45, {45, 56}},
      },
      {{33}, {34}, {44, 55}, {45, 56}},  // participating device groups
      false                              // expected_failure
    },
  };

  // replica groups contain partition ids.
  const std::vector<TestCase> cross_partition_test_cases = {
    {
      // 3x4 device assignment
      {
        {33, 34, 35, 36}, {44, 45, 46, 47}, {55, 56, 57, 58}
      },
      {{0, 1}, {2, 3}},          // replica groups
      true,                      // has_channel_id
      std::nullopt,             // use_global_device_ids
      {                          // subtests
        // 33 is r0p0, p0 group has p0, p1 so we get r0p0 and r0p1.
        {33, {33, 34}},
        // 35 is r0p2, so we get r0p2 and r0p3
        {35, {35, 36}},
        {45, {44, 45}},
        {47, {46, 47}},
        {58, {57, 58}},
      },
      {{33, 34}, {44, 45}, {55, 56},
       {35, 36}, {46, 47}, {57, 58}},  // participating device groups
      false                            // expected_failure
    }
  };


  const std::vector<TestCase> cross_replica_and_partition_test_cases = {
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {{0}, {1, 2}},                    // replica groups
      true,                             // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        // 33 is r0p0, so should get r0 from all partitions.
        {33, {33, 34}},
        // 34 is r0p1, so should get r0 from all partitions.
        {34, {33, 34}},
        // 45 is r1p1, so should get r1, r2 from all partitions.
        {45, {44, 45, 55, 56}},
      },
      {{33, 34}, {44, 45, 55, 56}},   // participating device groups
      false
    },

    // empty replica group = all replicas, so we should get all devices.
    {
      {{33, 34}, {44, 45}, {55, 56}},   // 3r, 2p
      {},                               // replica groups
      true,                             // has_channel_id
      false,                            // use_global_device_ids
      {                                 // subtests
        {33, {33, 34, 44, 45, 55, 56}},
        {34, {33, 34, 44, 45, 55, 56}},
        {56, {33, 34, 44, 45, 55, 56}},
      },
      {{33, 34, 44, 45, 55, 56}},        // participating device groups
      false                              // expected_failure
    },
  };

  // Replica groups are flattened ids. For a 3x2 device assignment
  // used in these tests, the flattened ID and deviceId correspondence is as
  // follows:
  //   r0p0 = f#0 = d#33
  //   r0p1 = f#1 = d#34
  //   r1p0 = f#2 = d#44
  //   r1p1 = f#3 = d#45
  //   r2p0 = f#4 = d#55
  //   r2p1 = f#5 = d#56
  const std::vector<TestCase> flattened_id_test_cases = {
    {
      {{33, 34}, {44, 45}, {55, 56}},  // 3r, 2p
      {{0}, {1, 2}, {3, 4, 5}},        // replica groups
      true,                            // has_channel_id
      true,                            // use_global_device_ids
      {                                // subtests
        {33, {33}},
        {34, {34, 44}},
        {44, {34, 44}},
        {45, {45, 55, 56}},
        {55, {45, 55, 56}},
        {56, {45, 55, 56}},
      },
      {{33}, {34, 44}, {45, 55, 56}},  // participating device groups
      false                            // expected_failure
    },
    {
      {{33}},
      {},         // empty replica groups not allowed.
      true,       // has_channel_id
      true,       // use_global_device_ids
      {           // subtests
        {33, {33}},
      },
      {{33}},      // participating device groups
      true         // expected_failure
    },
  };

  const std::vector<TestCase> failure_test_cases = {
    // No channel id, use_global_device_ids = true;
    {
      {{33}, {44}, {55}},   // 3r, 1p
      {},                   // replica groups
      false,                // has_channel_id
      true,                 // use_global_device_ids
      {                     // subtests
        {33, {}},
      },
      {{33, 44, 55}},       // participating device groups
      true                  // expected_failure
    },
  };
  // clang-format on

  test_cases.insert(test_cases.end(), cross_replica_test_cases.begin(),
                    cross_replica_test_cases.end());
  // When use_global_device_ids is not present and channel_id is not present,
  // that implies cross replica mode as well.
  for (TestCase tc : cross_replica_test_cases) {
    tc.use_global_device_ids = std::nullopt;
    test_cases.push_back(tc);
  }

  test_cases.insert(test_cases.end(), cross_partition_test_cases.begin(),
                    cross_partition_test_cases.end());
  test_cases.insert(test_cases.end(),
                    cross_replica_and_partition_test_cases.begin(),
                    cross_replica_and_partition_test_cases.end());
  test_cases.insert(test_cases.end(), flattened_id_test_cases.begin(),
                    flattened_id_test_cases.end());
  test_cases.insert(test_cases.end(), failure_test_cases.begin(),
                    failure_test_cases.end());

  return test_cases;
}

class GetParticipatingDevicesTest : public testing::TestWithParam<TestCase> {};

TEST_P(GetParticipatingDevicesTest, Test) {
  const TestCase &tc = GetParam();

  int64_t num_replicas = tc.device_assignment.n1();
  int64_t num_partitions = tc.device_assignment.n2();
  DeviceAssignment device_assignment(num_replicas, num_partitions);

  for (int64_t replica_id = 0; replica_id < num_replicas; ++replica_id) {
    for (int64_t partition_id = 0; partition_id < num_partitions;
         ++partition_id) {
      device_assignment(replica_id, partition_id) =
          tc.device_assignment(replica_id, partition_id);
    }
  }

  std::vector<ReplicaGroup> replica_groups;
  absl::c_transform(tc.replica_groups, std::back_inserter(replica_groups),
                    [](const std::vector<int> &ids) {
                      ReplicaGroup group;
                      for (int id : ids) {
                        group.add_replica_ids(id);
                      }
                      return group;
                    });

  absl::StatusOr<CollectiveOpGroupMode> group_mode =
      GetCollectiveOpGroupMode(tc.has_channel_id, tc.use_global_device_ids);

  if (!group_mode.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }

  // Execute each sub-test.
  for (const TestCase::CurrentIdAndOutput &subtest : tc.subtests) {
    absl::StatusOr<std::vector<GlobalDeviceId>> actual =
        GetParticipatingDevices(GlobalDeviceId(subtest.current_id),
                                device_assignment, replica_groups, *group_mode);
    if (!actual.ok()) {
      EXPECT_TRUE(tc.expected_failure);
      continue;
    }
    std::vector<GlobalDeviceId> expected;
    expected.reserve(subtest.expected_output.size());
    absl::c_transform(subtest.expected_output, std::back_inserter(expected),
                      [](int id) { return GlobalDeviceId(id); });
    EXPECT_EQ(*actual, expected);
  }

  absl::StatusOr<std::vector<std::vector<GlobalDeviceId>>>
      actual_device_groups = GetParticipatingDevicesGroups(
          device_assignment, replica_groups, *group_mode);

  if (!actual_device_groups.ok()) {
    EXPECT_TRUE(tc.expected_failure);
    return;
  }

  std::vector<std::vector<GlobalDeviceId>> expect_device_groups;
  expect_device_groups.reserve(tc.participating_device_groups.size());

  for (auto subgroup : tc.participating_device_groups) {
    std::vector<GlobalDeviceId> subgroup_device_ids;
    subgroup_device_ids.reserve(subgroup.size());
    absl::c_transform(subgroup, std::back_inserter(subgroup_device_ids),
                      [](int id) { return GlobalDeviceId(id); });

    expect_device_groups.push_back(subgroup_device_ids);
  }

  EXPECT_THAT(*actual_device_groups,
              testing::UnorderedElementsAreArray(expect_device_groups));
}

INSTANTIATE_TEST_SUITE_P(GetParticipatingDevices, GetParticipatingDevicesTest,
                         testing::ValuesIn(GetTestCases()));

}  // namespace GetParticipatingDevicesTest
}  // namespace xla
