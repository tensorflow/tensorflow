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

#include "xla/backends/gpu/transforms/collectives/collective_fusion.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/collectives/collective_kernel_strategy_annotator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

// Template for a module with an AllReduce.
constexpr absl::string_view kAllReduceHloTemplate = R"(
  HloModule all_reduce_test

  add {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    ROOT r = f32[] add(p0, p1)
  }

  ENTRY e {
    p0 = %1$s parameter(0)
    ROOT ar = %1$s all-reduce(p0),
        replica_groups={{0,1,2,3,4,5,6,7}},
        to_apply=add
  }
)";

absl::StatusOr<bool> AnnotateAndFuse(HloModule* module,
                                     const GpuTopology& gpu_topology) {
  CollectiveKernelStrategyAnnotator annotator(gpu_topology,
                                              /*is_multimem_enabled=*/false);
  RETURN_IF_ERROR(annotator.Run(module).status());
  CollectiveFusion fusion(gpu_topology);
  return fusion.Run(module);
}

class CollectiveFusionTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    stream_executor::GpuTargetConfigProto target_config_proto;
    target_config_proto.set_platform_name("CUDA");
    *target_config_proto.mutable_gpu_device_info() =
        TestGpuDeviceInfo::H100SXMDeviceInfo().ToProto();
    ASSERT_OK_AND_ASSIGN(gpu::GpuTargetConfig target_config,
                         gpu::GpuTargetConfig::FromProto(target_config_proto));
    gpu_topology_ = std::make_unique<GpuTopology>(
        "platform_version", /*num_partitions=*/1,
        /*num_hosts_per_partition=*/1,
        /*num_devices_per_host=*/16, target_config);
  }

  std::unique_ptr<GpuTopology> gpu_topology_;
};

TEST_F(CollectiveFusionTest, FusesSmallAllReduceOneShot) {
  constexpr int64_t kNumElements = 32768;  // 128 KB -> OneShot
  Shape shape = ShapeUtil::MakeShape(F32, {kNumElements});
  const std::string hlo =
      absl::StrFormat(kAllReduceHloTemplate, shape.ToString());
  SCOPED_TRACE(::testing::Message() << "hlo: " << hlo);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       AnnotateAndFuse(module.get(), *gpu_topology_));
  EXPECT_TRUE(changed);

  // Verify fusion instruction.
  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();  // Now the fusion itself!
  ASSERT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kCustom);

  // Verify backend config of the fusion.
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig cfg,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(cfg.fusion_backend_config().kind(), kTritonCollectiveFusionKind);
  EXPECT_TRUE(cfg.fusion_backend_config().has_block_level_fusion_config());

  // Verify inner instruction is the all-reduce.
  HloInstruction* inner_ar =
      Cast<HloFusionInstruction>(root)->fused_expression_root();
  EXPECT_EQ(inner_ar->opcode(), HloOpcode::kAllReduce);
  // Shape should remain 1D.
  EXPECT_EQ(inner_ar->shape().dimensions().size(), 1);
}

TEST_F(CollectiveFusionTest, FusesAndFlattensMediumAllReduceTwoShot) {
  // Use a 2D shape to test flattening: [2, 131072] = 262144 elements (1 MB ->
  // TwoShot)
  Shape shape = xla::ShapeUtil::MakeShape(F32, {2, 131072});
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, shape.ToString());
  SCOPED_TRACE(::testing::Message() << "hlo: " << hlo);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       AnnotateAndFuse(module.get(), *gpu_topology_));
  EXPECT_TRUE(changed);

  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kBitcast);  // Flattening bitcast.
  EXPECT_EQ(root->shape().dimensions().size(), 2);
  // The operand of the bitcast should be the fusion.
  HloInstruction* ar_fusion = root->mutable_operand(0);
  ASSERT_EQ(ar_fusion->opcode(), HloOpcode::kFusion);

  // Since it was 2D and TWO_SHOT, the fusion shape should be 1D [262144].
  EXPECT_EQ(ar_fusion->shape().dimensions().size(), 1);
  EXPECT_EQ(ar_fusion->shape().dimensions(0), 262144);

  // The fusion operand should also be a bitcast (flattening to 1D).
  HloInstruction* bitcast_to_1d = ar_fusion->mutable_operand(0);
  ASSERT_EQ(bitcast_to_1d->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(bitcast_to_1d->shape().dimensions().size(), 1);
  EXPECT_EQ(bitcast_to_1d->operand(0), entry->parameter_instruction(0));  // p0
}

TEST_F(CollectiveFusionTest, DoesNotFuseUnannotated) {
  constexpr int64_t kNumElements = 32768;
  Shape shape = ShapeUtil::MakeShape(F32, {kNumElements});
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, shape.ToString());
  SCOPED_TRACE(::testing::Message() << "hlo: " << hlo);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  // Only run fusion pass without annotating.
  CollectiveFusion fusion(*gpu_topology_);
  ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  EXPECT_FALSE(changed);

  // Verify no fusion was created.
  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAllReduce);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
}

TEST_F(CollectiveFusionTest, Idempotent) {
  constexpr int64_t kNumElements = 32768;
  Shape shape = ShapeUtil::MakeShape(F32, {kNumElements});
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, shape.ToString());
  SCOPED_TRACE(::testing::Message() << "hlo: " << hlo);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  // First run: should make changes.
  ASSERT_OK_AND_ASSIGN(bool changed1,
                       AnnotateAndFuse(module.get(), *gpu_topology_));
  EXPECT_TRUE(changed1);

  ASSERT_OK_AND_ASSIGN(bool changed2,
                       AnnotateAndFuse(module.get(), *gpu_topology_));
  EXPECT_FALSE(changed2);
}

}  // namespace
}  // namespace xla::gpu
