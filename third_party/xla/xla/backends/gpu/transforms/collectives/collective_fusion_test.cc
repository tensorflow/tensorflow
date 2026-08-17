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
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/backends/gpu/transforms/collectives/collective_kernel_strategy_annotator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
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

// Pipeline runs annotator and then the fusion pass.
class AnnotateAndFusePipeline : public HloModulePass {
 public:
  explicit AnnotateAndFusePipeline(const GpuTopology& gpu_topology)
      : gpu_topology_(gpu_topology) {}
  absl::string_view name() const override { return "annotate-and-fuse"; }
  using HloPassInterface::Run;

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    bool changed = false;
    CollectiveKernelStrategyAnnotator annotator(gpu_topology_,
                                                /*is_multimem_enabled=*/false);
    ABSL_ASSIGN_OR_RETURN(bool changed_annotator,
                     annotator.Run(module, execution_threads));
    changed |= changed_annotator;
    CollectiveFusion fusion(gpu_topology_);
    ABSL_ASSIGN_OR_RETURN(bool changed_fusion,
                     fusion.Run(module, execution_threads));
    changed |= changed_fusion;
    return changed;
  }

 private:
  const GpuTopology& gpu_topology_;
};

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
  static constexpr absl::string_view kExpected = R"(
    // CHECK: %[[FUSION_COMPUTATION:.*]] (param_0: f32[32768]) -> f32[32768] {
    // CHECK:   %[[P0:.*]] = f32[32768]{0} parameter(0)
    // CHECK:   ROOT {{.*}} = f32[32768]{0} all-reduce(%[[P0]]),
    // CHECK-SAME: to_apply=%add
    // CHECK: }
    //
    // CHECK: ENTRY %e (p0.1: f32[32768]) -> f32[32768] {
    // CHECK:   %[[P0:.*]] = f32[32768]{0} parameter(0)
    // CHECK:   ROOT {{.*}} = f32[32768]{0} fusion(%[[P0]]),
    // CHECK-SAME: kind=kCustom,
    // CHECK-SAME: calls=%[[FUSION_COMPUTATION]],
    // CHECK-SAME: backend_config={
    // CHECK-SAME: "kind":"__triton_collective"
    // CHECK-SAME: "block_level_fusion_config"
  )";
  HloModuleConfig config = GetModuleConfigForTest(/*replica_count=*/8);
  RunAndFilecheckHloRewrite(hlo, AnnotateAndFusePipeline(*gpu_topology_),
                            kExpected, /*after_pass_checks=*/nullptr, &config);
}

TEST_F(CollectiveFusionTest, FusesAndFlattensMediumAllReduceTwoShot) {
  // Use a 2D shape to test flattening: [2, 131072] = 262144 elements (1 MB ->
  // TwoShot)
  Shape shape = xla::ShapeUtil::MakeShape(F32, {2, 131072});
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, shape.ToString());
  SCOPED_TRACE(::testing::Message() << "hlo: " << hlo);
  static constexpr absl::string_view kExpected = R"(
    // CHECK: %[[FUSION_COMPUTATION:.*]] (param_0: f32[262144]) -> f32[262144] {
    // CHECK:   %[[P0:.*]] = f32[262144]{0} parameter(0)
    // CHECK:   ROOT {{.*}} = f32[262144]{0} all-reduce(%[[P0]]),
    // CHECK-SAME: to_apply=%add
    // CHECK: }
    //
    // CHECK: ENTRY %e (p0.1: f32[2,131072]) -> f32[2,131072] {
    // CHECK:   %[[P0:.*]] = f32[2,131072]{1,0} parameter(0)
    // CHECK:   %[[BITCAST_TO_1D:.*]] = f32[262144]{0} bitcast(%[[P0]])
    // CHECK:   %[[FUSION:.*]] = f32[262144]{0} fusion(%[[BITCAST_TO_1D]]),
    // CHECK-SAME: kind=kCustom,
    // CHECK-SAME: calls=%[[FUSION_COMPUTATION]],
    // CHECK-SAME: backend_config={
    // CHECK-SAME: "kind":"__triton_collective"
    // CHECK-SAME: "block_level_fusion_config"
    // CHECK:   ROOT {{.*}} = f32[2,131072]{1,0} bitcast(%[[FUSION]])
  )";
  HloModuleConfig config = GetModuleConfigForTest(/*replica_count=*/8);
  RunAndFilecheckHloRewrite(hlo, AnnotateAndFusePipeline(*gpu_topology_),
                            kExpected, /*after_pass_checks=*/nullptr, &config);
}

TEST_F(CollectiveFusionTest, DoesNotFuseUnannotated) {
  constexpr int64_t kNumElements = 32768;
  Shape shape = ShapeUtil::MakeShape(F32, {kNumElements});
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, shape.ToString());
  SCOPED_TRACE(::testing::Message() << "hlo: " << hlo);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));
  const std::string original_hlo = module->ToString();
  // Only run fusion pass without annotating.
  CollectiveFusion fusion(*gpu_topology_);
  ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->ToString(), original_hlo);
}

TEST_F(CollectiveFusionTest, Idempotent) {
  constexpr int64_t kNumElements = 32768;
  Shape shape = ShapeUtil::MakeShape(F32, {kNumElements});
  std::string hlo = absl::StrFormat(kAllReduceHloTemplate, shape.ToString());
  SCOPED_TRACE(::testing::Message() << "hlo: " << hlo);
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo, /*replica_count=*/8));

  // First run: should make changes.
  AnnotateAndFusePipeline pipeline(*gpu_topology_);
  ASSERT_OK_AND_ASSIGN(bool changed1, pipeline.Run(module.get()));
  EXPECT_TRUE(changed1);

  ASSERT_OK_AND_ASSIGN(bool changed2, pipeline.Run(module.get()));
  EXPECT_FALSE(changed2);
}

TEST_F(CollectiveFusionTest, NormalizeAsyncCandidate) {
  constexpr absl::string_view kAllReduceHloTemplate = R"(
    HloModule all_reduce_test

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT r = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = $0 parameter(0)
      %ar-start = $0 all-reduce-start(p0),
          replica_groups={{0,1,2,3,4,5,6,7}},
          to_apply=add
      ROOT %ar-done = $0 all-reduce-done(%ar-start)
    }
  )";

  constexpr int64_t kNumElements = 32768;
  Shape shape = ShapeUtil::MakeShape(F32, {kNumElements});
  std::string hlo = absl::Substitute(kAllReduceHloTemplate, shape.ToString());
  SCOPED_TRACE(::testing::Message() << "hlo: " << hlo);

  // Note that fusion-start/done are syntactic sugar for async-start/done.
  // In reality this is async-start calls = async_comp
  // where async_comp = fusion calls = fusion_computation.
  // and fusion_computation = all-reduce as shown before.
  static constexpr absl::string_view kExpected = R"(
    // CHECK: %[[FUSION_COMPUTATION:.*]] ({{.*}}: f32[32768]) -> f32[32768] {
    // CHECK:   %[[P0:.*]] = f32[32768]{0} parameter(0)
    // CHECK:   ROOT {{.*}} = f32[32768]{0} all-reduce(%[[P0]]),
    // CHECK-SAME: to_apply=%add
    // CHECK: }
    //
    // CHECK: ENTRY %e ({{.*}}: f32[32768]) -> f32[32768] {
    // CHECK:   %[[P0_ENTRY:.*]] = f32[32768]{0} parameter(0)
    // CHECK:   %[[ASYNC_START:.*]] = ((f32[32768]{0}), f32[32768]{0})
    // CHECK-SAME: fusion-start(%[[P0_ENTRY]]), kind=kCustom
    // CHECK-SAME: , calls=%[[FUSION_COMPUTATION]]
    // CHECK:   ROOT {{.*}} = f32[32768]{0} fusion-done(%[[ASYNC_START]])
    // CHECK: }
  )";

  HloModuleConfig config = GetModuleConfigForTest(/*replica_count=*/8);
  RunAndFilecheckHloRewrite(hlo, AnnotateAndFusePipeline(*gpu_topology_),
                            kExpected, /*after_pass_checks=*/nullptr, &config);
}

}  // namespace
}  // namespace xla::gpu
