/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/all_reduce_blueconnect.h"

#include <memory>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::tsl::testing::IsOkAndHolds;
namespace op = xla::testing::opcode_matchers;

using AllReduceBlueConnectTest = HloTestBase;

void SetModuleConfig(HloModule& module, size_t replica_count) {
  DeviceAssignment device_assignment(replica_count, /*computation_count=*/1);
  device_assignment.FillIota(0);
  module.config().set_replica_count(replica_count);
  module.config().set_static_device_assignment(device_assignment);
}

TEST_F(AllReduceBlueConnectTest, OneStage) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  ROOT crs = f32[4,4] all-reduce(p0), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));

  // clang-format off
  std::vector<std::vector<int64_t>> scatter_gather_groups = {
      {0, 1, 2, 3}, {4, 5, 6, 7}};
  std::vector<std::vector<int64_t>> new_all_reduce_groups = {
      {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  // clang-format on

  auto bitcast = AllOf(op::Shape("f32[16]"), op::Bitcast(op::Parameter(0)));
  auto reduce_scatter = AllOf(op::Shape("f32[4]"), op::ReduceScatter(bitcast),
                              op::ReplicaGroups(scatter_gather_groups));
  auto all_reduce = AllOf(op::Shape("f32[4]"), op::AllReduce(reduce_scatter),
                          op::ReplicaGroups(new_all_reduce_groups));
  auto all_gather = AllOf(op::Shape("f32[16]"), op::AllGather(all_reduce),
                          op::ReplicaGroups(scatter_gather_groups));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Shape("f32[4,4]"), op::Bitcast(all_gather)));
}

TEST_F(AllReduceBlueConnectTest, TwoStage) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  ROOT crs = f32[4,4] all-reduce(p0), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/16);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));

  std::vector<std::vector<int64_t>> outer_scatter_gather_groups = {
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  std::vector<std::vector<int64_t>> inner_scatter_gather_groups = {
      {0, 4}, {8, 12}, {1, 5}, {9, 13}, {2, 6}, {10, 14}, {3, 7}, {11, 15}};
  std::vector<std::vector<int64_t>> new_all_reduce_groups = {
      {0, 8}, {4, 12}, {1, 9}, {5, 13}, {2, 10}, {6, 14}, {3, 11}, {7, 15}};

  auto bitcast0 = AllOf(op::Shape("f32[16]"), op::Bitcast(op::Parameter(0)));
  auto reduce_scatter0 = AllOf(op::Shape("f32[4]"), op::ReduceScatter(bitcast0),
                               op::ReplicaGroups(outer_scatter_gather_groups));
  auto bitcast1 = AllOf(op::Shape("f32[4]"), op::Bitcast(reduce_scatter0));
  auto reduce_scatter1 = AllOf(op::Shape("f32[2]"), op::ReduceScatter(bitcast1),
                               op::ReplicaGroups(inner_scatter_gather_groups));
  auto all_reduce = AllOf(op::Shape("f32[2]"), op::AllReduce(reduce_scatter1),
                          op::ReplicaGroups(new_all_reduce_groups));
  auto all_gather0 = AllOf(op::Shape("f32[4]"), op::AllGather(all_reduce),
                           op::ReplicaGroups(inner_scatter_gather_groups));
  auto bitcast2 = AllOf(op::Shape("f32[4]"), op::Bitcast(all_gather0));
  auto all_gather1 = AllOf(op::Shape("f32[16]"), op::AllGather(bitcast2),
                           op::ReplicaGroups(outer_scatter_gather_groups));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Shape("f32[4,4]"), op::Bitcast(all_gather1)));
}

TEST_F(AllReduceBlueConnectTest, TwoOperands) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  p1 = f32[4,4,2] parameter(1)
  ROOT crs = (f32[4,4], f32[4,4,2]) all-reduce(p0, p1), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));

  // clang-format off
  std::vector<std::vector<int64_t>> scatter_gather_groups = {
      {0, 1, 2, 3}, {4, 5, 6, 7}};
  std::vector<std::vector<int64_t>> new_all_reduce_groups = {
      {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  // clang-format on

  auto bitcast0 = AllOf(op::Shape("f32[16]"), op::Bitcast(op::Parameter(0)));
  auto bitcast1 = AllOf(op::Shape("f32[32]"), op::Bitcast(op::Parameter(1)));
  auto reduce_scatter = AllOf(op::Shape("(f32[4], f32[8])"),
                              op::ReduceScatter(bitcast0, bitcast1),
                              op::ReplicaGroups(scatter_gather_groups));
  auto all_reduce = AllOf(op::Shape("(f32[4], f32[8])"),
                          op::AllReduce(op::GetTupleElement(reduce_scatter, 0),
                                        op::GetTupleElement(reduce_scatter, 1)),
                          op::ReplicaGroups(new_all_reduce_groups));
  auto all_gather = AllOf(op::Shape("(f32[16], f32[32])"),
                          op::AllGather(op::GetTupleElement(all_reduce, 0),
                                        op::GetTupleElement(all_reduce, 1)),
                          op::ReplicaGroups(scatter_gather_groups));
  auto bitcast2 = AllOf(op::Shape("f32[4,4]"),
                        op::Bitcast(op::GetTupleElement(all_gather, 0)));
  auto bitcast3 = AllOf(op::Shape("f32[4,4,2]"),
                        op::Bitcast(op::GetTupleElement(all_gather, 1)));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(bitcast2, bitcast3));
}

TEST_F(AllReduceBlueConnectTest, DifferentNumLocalDevicesWithinReplicaGroup) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  ROOT crs = f32[4,4] all-reduce(p0),
    replica_groups={{0,1,2,7},{3,4,5,6}}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(AllReduceBlueConnectTest, DifferentNumLocalDevicesAcrossReplicaGroups) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  ROOT crs = f32[4,4] all-reduce(p0),
    replica_groups={{0,1,4,5},{2,3,6,7},{8,9,10,11},{12,13,14,15}}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/16);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(AllReduceBlueConnectTest, OperandIndivisible) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  p1 = f32[9] parameter(1)
  ROOT crs = (f32[4,4], f32[9]) all-reduce(p0, p1), to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
