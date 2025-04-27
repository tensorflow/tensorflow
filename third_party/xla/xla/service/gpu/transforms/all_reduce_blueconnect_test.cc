/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/all_reduce_blueconnect.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/computation_placer.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using ::tsl::testing::IsOkAndHolds;
namespace m = ::xla::match;

using AllReduceBlueConnectTest = HloHardwareIndependentTestBase;

HloPredicate MatchChannelId(std::optional<int64_t> channel_id) {
  return [channel_id](const HloInstruction* instruction) {
    return instruction->channel_id() == channel_id;
  };
}

void SetModuleConfig(HloModuleConfig* module_config, size_t replica_count,
                     size_t partition_count = 1) {
  DeviceAssignment device_assignment(replica_count,
                                     /*computation_count=*/partition_count);
  device_assignment.FillIota(0);
  module_config->set_replica_count(replica_count);
  module_config->set_num_partitions(partition_count);
  module_config->set_static_device_assignment(device_assignment);
}

void SetModuleConfig(HloModule& module, size_t replica_count,
                     size_t partition_count = 1) {
  SetModuleConfig(&module.mutable_config(), replica_count, partition_count);
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

  auto bitcast = m::Bitcast(m::Parameter(0)).WithShape(F32, {16});
  auto reduce_scatter = m::ReduceScatter(bitcast)
                            .WithShape(F32, {4})
                            .WithReplicaGroups(scatter_gather_groups)
                            .WithPredicate(MatchChannelId(std::nullopt));
  auto all_reduce = m::AllReduce(reduce_scatter)
                        .WithShape(F32, {4})
                        .WithReplicaGroups(new_all_reduce_groups)
                        .WithPredicate(MatchChannelId(std::nullopt));
  auto all_gather = m::AllGather(all_reduce)
                        .WithShape(F32, {16})
                        .WithReplicaGroups(scatter_gather_groups)
                        .WithPredicate(MatchChannelId(std::nullopt));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Bitcast(all_gather).WithShape(F32, {4, 4})));
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

  auto bitcast0 = m::Bitcast(m::Parameter(0)).WithShape(F32, {16});
  auto reduce_scatter0 =
      m::ReduceScatter(bitcast0).WithShape(F32, {4}).WithReplicaGroups(
          outer_scatter_gather_groups);
  auto bitcast1 = m::Bitcast(reduce_scatter0).WithShape(F32, {4});
  auto reduce_scatter1 =
      m::ReduceScatter(bitcast1).WithShape(F32, {2}).WithReplicaGroups(
          inner_scatter_gather_groups);
  auto all_reduce = m::AllReduce(reduce_scatter1)
                        .WithShape(F32, {2})
                        .WithReplicaGroups(new_all_reduce_groups);
  auto all_gather0 = m::AllGather(all_reduce)
                         .WithShape(F32, {4})
                         .WithReplicaGroups(inner_scatter_gather_groups);
  auto bitcast2 = m::Bitcast(all_gather0).WithShape(F32, {4});
  auto all_gather1 =
      m::AllGather(bitcast2).WithShape(F32, {16}).WithReplicaGroups(
          outer_scatter_gather_groups);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Bitcast(all_gather1).WithShape(F32, {4, 4})));
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

  auto bitcast0 = m::Bitcast(m::Parameter(0)).WithShape(F32, {16});
  auto bitcast1 = m::Bitcast(m::Parameter(1)).WithShape(F32, {32});

  Shape expected0 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {4}), ShapeUtil::MakeShape(F32, {8})});
  Shape expected1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {16}), ShapeUtil::MakeShape(F32, {32})});
  auto reduce_scatter = m::ReduceScatter(bitcast0, bitcast1)
                            .WithShapeEqualTo(&expected0)
                            .WithReplicaGroups(scatter_gather_groups);
  auto all_reduce = m::AllReduce(m::GetTupleElement(reduce_scatter, 0),
                                 m::GetTupleElement(reduce_scatter, 1))
                        .WithShapeEqualTo(&expected0)
                        .WithReplicaGroups(new_all_reduce_groups);
  auto all_gather = m::AllGather(m::GetTupleElement(all_reduce, 0),
                                 m::GetTupleElement(all_reduce, 1))
                        .WithShapeEqualTo(&expected1)
                        .WithReplicaGroups(scatter_gather_groups);
  auto bitcast2 =
      m::Bitcast(m::GetTupleElement(all_gather, 0)).WithShape(F32, {4, 4});
  auto bitcast3 =
      m::Bitcast(m::GetTupleElement(all_gather, 1)).WithShape(F32, {4, 4, 2});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(bitcast2, bitcast3)));
}

TEST_F(AllReduceBlueConnectTest, MultiplePartitionsFilecheck) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[8,8] parameter(0)
  ROOT crs = f32[8,8] all-reduce(p0), channel_id=1,
    replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=add
})";
  HloModuleConfig module_config;
  SetModuleConfig(&module_config, /*replica_count=*/1, /*partition_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  // Note: When matching strings like "replica_groups={{0,1,2,3}}", FileCheck
  // interprets the string inside the double braces as regex. So to match such
  // strings, we use "replica_groups={{..0,1,2,3..}}", where the dots match the
  // opening and closing braces.
  RunAndFilecheckHloRewrite(hlo_string, std::move(pass), R"(
  CHECK:       %p0 = f32[8,8]{1,0} parameter(0)
  CHECK-NEXT:  [[bitcast:%[^ ]+]] = f32[64]{0} bitcast(%p0)
  CHECK-NEXT:  [[reduce_scatter:%[^ ]+]] = f32[16]{0} reduce-scatter([[bitcast]]), channel_id=2, replica_groups={{..0,1,2,3.,.4,5,6,7..}}, use_global_device_ids=true, dimensions={0}, to_apply=%add
  CHECK-NEXT:  [[all_reduce:%[^ ]+]] = f32[16]{0} all-reduce([[reduce_scatter]]), channel_id=1, replica_groups={{..0,4.,.1,5.,.2,6.,.3,7..}}, use_global_device_ids=true, to_apply=%add
  CHECK-NEXT:  [[all_gather:%[^ ]+]] = f32[64]{0} all-gather([[all_reduce]]), channel_id=3, replica_groups={{..0,1,2,3.,.4,5,6,7..}}, dimensions={0}, use_global_device_ids=true
  CHECK-NEXT:  ROOT [[output:%[^ ]+]] = f32[8,8]{1,0} bitcast([[all_gather]])
}
)",
                            /*after_pass_checks=*/nullptr, &module_config);
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

TEST_F(AllReduceBlueConnectTest, ControlDeps) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[4,4] parameter(0)
  p1 = f32[4,4] parameter(1)
  add = f32[4,4] add(p0, p1)
  crs = f32[4,4] all-reduce(p0), to_apply=add, control-predecessors={add}
  ROOT add1 = f32[4,4] add(crs, add), control-predecessors={crs}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  // Remember all-reduce's control succ and preds.
  const HloInstruction* ar =
      module->entry_computation()->root_instruction()->operand(0);
  auto expected_preds = ar->control_predecessors();
  auto expected_succs = ar->control_successors();

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(true));

  // clang-format off
  std::vector<std::vector<int64_t>> scatter_gather_groups = {
      {0, 1, 2, 3}, {4, 5, 6, 7}};
  std::vector<std::vector<int64_t>> new_all_reduce_groups = {
      {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  // clang-format on

  const HloInstruction *matched_rs, *matched_bitcast;
  auto bitcast = m::Bitcast(m::Parameter(0)).WithShape(F32, {16});
  auto reduce_scatter = m::ReduceScatter(&matched_rs, bitcast)
                            .WithShape(F32, {4})
                            .WithReplicaGroups(scatter_gather_groups);
  auto all_reduce = m::AllReduce(reduce_scatter)
                        .WithShape(F32, {4})
                        .WithReplicaGroups(new_all_reduce_groups);
  auto all_gather = m::AllGather(all_reduce)
                        .WithShape(F32, {16})
                        .WithReplicaGroups(scatter_gather_groups);
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Add()));

  EXPECT_THAT(
      root->operand(0),
      GmockMatch(
          m::Bitcast(&matched_bitcast, all_gather).WithShape(F32, {4, 4})));

  // Verify that control dependencies are transferred correctly.
  EXPECT_THAT(matched_rs, GmockMatch(m::Op().WithControlDeps(
                              absl::MakeSpan(expected_preds), {})));
  EXPECT_THAT(matched_bitcast, GmockMatch(m::Op().WithControlDeps(
                                   {}, absl::MakeSpan(expected_succs))));
}

TEST_F(AllReduceBlueConnectTest, ReduceScatterUnchanged) {
  // Tests that this pass does not affect reduce-scatter. In principle, the
  // BlueConnect algorithm could be applied to reduce-scatter, but for now it
  // doesn't.
  constexpr absl::string_view hlo_string = R"(
HloModule module

%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %comp {
  p0 = f32[8,4] parameter(0)
  ROOT crs = f32[1,4] reduce-scatter(p0), dimensions={0}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SetModuleConfig(*module, /*replica_count=*/8);

  AllReduceBlueConnect pass(/*num_devices_per_host=*/4);
  EXPECT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla
