/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/collective_ops_utils.h"

#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

TEST(CollectiveOpsUtilsTest, GetParticipatingReplicas_NoReplicaGroups) {
  std::vector<int> actual =
      GetParticipatingReplicas(
          /*replica_id=*/0, /*total_replica_count=*/3, /*replica_groups=*/{})
          .ConsumeValueOrDie();
  std::vector<int> expected = {0, 1, 2};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, GetParticipatingReplicas_ReplicaGroups) {
  std::vector<ReplicaGroup> replica_groups(3);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(4);
  replica_groups[1].add_replica_ids(1);
  replica_groups[1].add_replica_ids(5);
  replica_groups[2].add_replica_ids(2);
  replica_groups[2].add_replica_ids(3);

  std::vector<int> actual =
      GetParticipatingReplicas(
          /*replica_id=*/1, /*total_replica_count=*/6, replica_groups)
          .ConsumeValueOrDie();
  std::vector<int> expected = {1, 5};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, GetParticipatingDevices_NoReplicaGroups) {
  DeviceAssignment device_assignment(/*replica_count=*/3,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 42;
  device_assignment(1, 0) = 43;
  device_assignment(2, 0) = 44;

  std::vector<GlobalDeviceId> actual =
      GetParticipatingDevices(GlobalDeviceId(42), device_assignment,
                              /*total_replica_count=*/3, /*replica_groups=*/{})
          .ConsumeValueOrDie();
  std::vector<GlobalDeviceId> expected = {
      GlobalDeviceId(42), GlobalDeviceId(43), GlobalDeviceId(44)};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, GetParticipatingDevices_ReplicaGroups) {
  DeviceAssignment device_assignment(/*replica_count=*/4,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 42;
  device_assignment(1, 0) = 43;
  device_assignment(2, 0) = 44;
  device_assignment(3, 0) = 45;

  std::vector<ReplicaGroup> replica_groups(2);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(3);
  replica_groups[1].add_replica_ids(1);
  replica_groups[1].add_replica_ids(2);

  std::vector<GlobalDeviceId> actual =
      GetParticipatingDevices(GlobalDeviceId(42), device_assignment,
                              /*total_replica_count=*/4, replica_groups)
          .ConsumeValueOrDie();
  std::vector<GlobalDeviceId> expected = {GlobalDeviceId(42),
                                          GlobalDeviceId(45)};
  EXPECT_EQ(actual, expected);
}

TEST(CollectiveOpsUtilsTest, GetParticipatingDevices_MultipleComputations) {
  DeviceAssignment device_assignment(/*replica_count=*/2,
                                     /*computation_count=*/2);
  device_assignment(0, 0) = 42;
  device_assignment(1, 0) = 43;
  device_assignment(0, 1) = 44;
  device_assignment(1, 1) = 45;

  std::vector<GlobalDeviceId> actual =
      GetParticipatingDevices(GlobalDeviceId(44), device_assignment,
                              /*total_replica_count=*/2, /*replica_groups=*/{})
          .ConsumeValueOrDie();
  std::vector<GlobalDeviceId> expected = {GlobalDeviceId(44),
                                          GlobalDeviceId(45)};
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace xla
