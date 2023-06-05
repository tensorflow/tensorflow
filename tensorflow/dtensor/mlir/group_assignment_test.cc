/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/group_assignment.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"

namespace tensorflow {
namespace dtensor {
namespace {

mlir::DenseIntElementsAttr CreateGroupAssignmentAttr(
    mlir::MLIRContext& context,
    const std::vector<std::vector<int>>& replica_ids) {
  int num_groups = replica_ids.size();
  int group_size = replica_ids.front().size();
  llvm::SmallVector<int32, 4> flat_replica_ids;
  flat_replica_ids.reserve(num_groups * group_size);
  for (const std::vector<int>& group : replica_ids) {
    CHECK_EQ(group.size(), group_size);
    flat_replica_ids.insert(flat_replica_ids.end(), group.begin(), group.end());
  }
  auto shaped_type = mlir::RankedTensorType::get(
      {num_groups, group_size}, mlir::IntegerType::get(&context, 32));
  return mlir::DenseIntElementsAttr::get(shaped_type, flat_replica_ids);
}

GroupAssignment CreateGroupAssignment(
    mlir::MLIRContext& context,
    const std::vector<std::vector<int>>& replica_ids, int num_slices,
    int slice_size) {
  mlir::DenseIntElementsAttr group_assignment_attr =
      CreateGroupAssignmentAttr(context, replica_ids);
  StatusOr<GroupAssignment> group_assignment = GroupAssignment::FromMLIR(
      group_assignment_attr,
      GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
          num_slices, slice_size));
  CHECK(group_assignment.ok());
  return *group_assignment;
}

GroupAssignment CreateGroupAssignment(
    mlir::MLIRContext& context,
    const std::vector<std::vector<int>>& replica_ids,
    absl::flat_hash_map<GroupAssignment::ReplicaId, GroupAssignment::DeviceId>
        map) {
  mlir::DenseIntElementsAttr group_assignment_attr =
      CreateGroupAssignmentAttr(context, replica_ids);
  StatusOr<GroupAssignment> group_assignment = GroupAssignment::FromMLIR(
      group_assignment_attr,
      GroupAssignment::ReplicaToDeviceMap(std::move(map)));
  CHECK(group_assignment.ok());
  return *group_assignment;
}

TEST(DTensorGroupAssignmentTest, InputOutput) {
  mlir::MLIRContext context;

  mlir::DenseIntElementsAttr group_assignment_attr_in =
      CreateGroupAssignmentAttr(context,
                                /*replica_ids=*/{{0, 1, 2, 3, 4, 5, 6, 7}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto group_assignment,
      GroupAssignment::FromMLIR(
          group_assignment_attr_in,
          GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
              /*num_slices=*/1, /*slice_size=*/8)));
  EXPECT_EQ(group_assignment.replica_ids(),
            std::vector<std::vector<int>>({{0, 1, 2, 3, 4, 5, 6, 7}}));

  mlir::DenseIntElementsAttr group_assignment_attr_out =
      group_assignment.GlobalToMLIR(context);
  EXPECT_EQ(group_assignment_attr_out, group_assignment_attr_in);

  group_assignment_attr_out =
      group_assignment.SliceToMLIR(context, /*slice_id=*/0).value();
  EXPECT_EQ(group_assignment_attr_out, group_assignment_attr_in);
}

TEST(DTensorGroupAssignmentTest, BadInput) {
  mlir::MLIRContext context;

  mlir::DenseIntElementsAttr indivisible_donut_size =
      CreateGroupAssignmentAttr(context,
                                /*replica_ids=*/{{0, 1, 2, 3, 4, 5, 6, 7, 8}});
  EXPECT_FALSE(
      GroupAssignment::FromMLIR(
          indivisible_donut_size,
          GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
              /*num_slices=*/1, /*slice_size=*/8))
          .ok());

  mlir::DenseIntElementsAttr duplicate_replica_ids =
      CreateGroupAssignmentAttr(context,
                                /*replica_ids=*/{{0, 1, 2, 3, 4, 5, 6, 6}});
  EXPECT_FALSE(
      GroupAssignment::FromMLIR(
          duplicate_replica_ids,
          GroupAssignment::ReplicaToDeviceMap::DefaultReplicaToDeviceMap(
              /*num_slices=*/1, /*slice_size=*/8))
          .ok());
}

TEST(DTensorGroupAssignmentTest, Properties) {
  mlir::MLIRContext context;
  GroupAssignment group_assignment =
      CreateGroupAssignment(context,
                            /*replica_ids=*/{{0, 1, 2, 3}},
                            /*num_slices=*/1, /*slice_size=*/4);
  EXPECT_EQ(group_assignment.num_groups(), 1);
  EXPECT_EQ(group_assignment.group_size(), 4);
  EXPECT_EQ(group_assignment.num_replica_ids(), 4);
  EXPECT_EQ(group_assignment.replica_ids(),
            std::vector<std::vector<int>>({{0, 1, 2, 3}}));
}

TEST(DTensorGroupAssignmentTest, GlobalAllReduceSingleDonut) {
  mlir::MLIRContext context;
  GroupAssignment group_assignment =
      CreateGroupAssignment(context,
                            /*replica_ids=*/{{0, 1, 2, 3, 4, 5, 6, 7}},
                            /*num_slices=*/1, /*slice_size=*/8);
  EXPECT_TRUE(group_assignment.IsWithinSlices());
  EXPECT_EQ(group_assignment.replica_ids(),
            std::vector<std::vector<int>>({{0, 1, 2, 3, 4, 5, 6, 7}}));
  EXPECT_EQ(group_assignment.replica_ids(0),
            std::vector<std::vector<int>>({{0, 1, 2, 3, 4, 5, 6, 7}}));
}

TEST(DTensorGroupAssignmentTest, GlobalAllReduceTwoDonuts) {
  mlir::MLIRContext context;
  GroupAssignment group_assignment =
      CreateGroupAssignment(context,
                            /*replica_ids=*/{{1, 2, 0, 3}},
                            /*num_slices=*/2, /*slice_size=*/2);
  EXPECT_FALSE(group_assignment.IsWithinSlices());
  EXPECT_EQ(group_assignment.replica_ids(),
            std::vector<std::vector<int>>({{1, 2, 0, 3}}));
  EXPECT_EQ(group_assignment.replica_ids(0),
            std::vector<std::vector<int>>({{1, 0}}));
  EXPECT_EQ(group_assignment.replica_ids(1),
            std::vector<std::vector<int>>({{0, 1}}));
}

TEST(DTensorGroupAssignmentTest, SubgroupAllReduceFourDonuts) {
  mlir::MLIRContext context;
  std::vector<std::vector<int>> global(
      {{0, 4, 8, 12}, {1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}});
  GroupAssignment group_assignment =
      CreateGroupAssignment(context,
                            /*replica_ids=*/global,
                            /*map=*/
                            {
                                {0, {0, 0}},
                                {1, {0, 1}},
                                {2, {1, 0}},
                                {3, {1, 1}},
                                {4, {0, 2}},
                                {5, {0, 3}},
                                {6, {1, 2}},
                                {7, {1, 3}},
                                {8, {2, 0}},
                                {9, {2, 1}},
                                {10, {3, 0}},
                                {11, {3, 1}},
                                {12, {2, 2}},
                                {13, {2, 3}},
                                {14, {3, 2}},
                                {15, {3, 3}},
                            });
  EXPECT_FALSE(group_assignment.IsWithinSlices());
  EXPECT_EQ(group_assignment.replica_ids(), global);
  EXPECT_EQ(group_assignment.host_replica_ids(0),
            std::vector<std::vector<int>>({{0}, {1}}));
  EXPECT_EQ(group_assignment.replica_ids(0),
            std::vector<std::vector<int>>({{0, 2}, {1, 3}}));
  EXPECT_EQ(group_assignment.host_replica_ids(1),
            std::vector<std::vector<int>>({{2}, {3}}));
  EXPECT_EQ(group_assignment.replica_ids(1),
            std::vector<std::vector<int>>({{0, 2}, {1, 3}}));
  EXPECT_EQ(group_assignment.host_replica_ids(2),
            std::vector<std::vector<int>>({{8}, {9}}));
  EXPECT_EQ(group_assignment.replica_ids(2),
            std::vector<std::vector<int>>({{0, 2}, {1, 3}}));
  EXPECT_EQ(group_assignment.host_replica_ids(3),
            std::vector<std::vector<int>>({{10}, {11}}));
  EXPECT_EQ(group_assignment.replica_ids(3),
            std::vector<std::vector<int>>({{0, 2}, {1, 3}}));
}

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow
