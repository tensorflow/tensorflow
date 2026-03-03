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

#include "xla/service/replica_group_canonicalizer.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ReplicaGroupCanonicalizerTest : public HloPjRtTestBase {
 protected:
  std::unique_ptr<HloComputation> MakeAddComputation(const Shape& shape) {
    HloComputation::Builder builder("reduction");
    auto lhs = builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "lhs"));
    auto rhs = builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "rhs"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));
    return builder.Build();
  }
};

TEST_F(ReplicaGroupCanonicalizerTest, ConvertsV3ToV1CollectiveDeviceList) {
  Mesh mesh({2}, {"x"});

  std::vector<AxisRef> axes = {AxisRef(0)};
  MeshAxesReplicaGroupList mesh_axes_list(mesh, axes);

  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();

  HloComputation* reduction_comp = module->AddEmbeddedComputation(
      MakeAddComputation(ShapeUtil::MakeShape(F32, {8})));

  HloComputation::Builder builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {8}), "p0"));

  builder.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeShape(F32, {8}), {input}, reduction_comp, mesh_axes_list,
      /*constrain_layout=*/false, /*channel_id=*/1,
      /*use_global_device_ids=*/true));

  module->AddEntryComputation(builder.Build());

  ReplicaGroupCanonicalizer pass;
  auto result = pass.Run(module.get());
  ASSERT_OK(result);
  EXPECT_TRUE(*result);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAllReduce);
  auto* collective = Cast<HloCollectiveInstruction>(root);
  const auto& device_list = collective->device_list();
  // Expect fallback to ListOfLists (which is default for CollectiveDeviceList).
  EXPECT_EQ(device_list.version(), CollectiveDeviceListVersion::kListOfLists);
}

}  // namespace
}  // namespace xla
