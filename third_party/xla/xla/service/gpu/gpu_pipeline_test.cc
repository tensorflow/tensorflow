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

#include "xla/service/gpu/gpu_pipeline.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/core/host_offloading/hlo_host_device_type_call_wrapper.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/pass/hlo_pipeline_builder.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::TestParamInfo;
using ::testing::Values;
using ::testing::WithParamInterface;

void CollectPasses(const HloPipelineNodeSequenceProto& sequence,
                   std::vector<const HloPipelineNodeProto*>& passes) {
  for (const HloPipelineNodeProto& node : sequence.nodes()) {
    if (node.has_sub_nodes()) {
      CollectPasses(node.sub_nodes(), passes);
    } else {
      passes.push_back(&node);
    }
  }
}

struct PipelineConfigCase {
  std::string name;
  int64_t num_partitions;
  bool use_spmd_partitioning;
  bool use_shardy_partitioner;
  bool enable_enzyme_comms_opt;
};

using DefaultPipelineCreatabilityTest =
    ::testing::TestWithParam<PipelineConfigCase>;

// Checks that every pass of the default pipeline is registered and consumes all
// of its options across configuration branches.
TEST_P(DefaultPipelineCreatabilityTest, PassesAreCreatable) {
  const PipelineConfigCase& test_case = GetParam();
  HloModuleConfig config;
  config.set_num_partitions(test_case.num_partitions);
  config.set_replica_count(1);
  config.set_use_spmd_partitioning(test_case.use_spmd_partitioning);
  config.set_use_shardy_partitioner(test_case.use_shardy_partitioner);
  config.mutable_debug_options().set_xla_enable_enzyme_comms_opt(
      test_case.enable_enzyme_comms_opt);

  const se::DeviceDescription device_description =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  const GpuTopology gpu_topology("test", /*num_partitions=*/1,
                                 /*num_hosts_per_partition=*/1,
                                 /*num_devices_per_host=*/8);
  // Creatability doesn't depend on what the backend hooks do.
  const auto noop = [](HloInstruction*) { return absl::OkStatus(); };
  const HloHostDeviceTypeCallWrapper::Options host_device_type_options{
      /*set_backend_config_fn=*/noop,
      /*clear_backend_config_device_type=*/noop,
  };
  HloPassContext context;
  ASSERT_OK(context.Set(device_description));
  ASSERT_OK(context.Set(gpu_topology));
  ASSERT_OK(context.Set(host_device_type_options));

  ASSERT_OK_AND_ASSIGN(const HloPipelineProto proto,
                       GetDefaultCompilationPipeline(config, context));
  std::vector<const HloPipelineNodeProto*> passes;
  CollectPasses(proto.root(), passes);
  ASSERT_THAT(passes, Not(IsEmpty()));

  for (const HloPipelineNodeProto* pass : passes) {
    HloPassOptions options(pass->options());
    EXPECT_OK(HloPassRegistry::Global().Create(pass->name(), context, options))
        << pass->name();
  }
  EXPECT_OK(BuildHloPassPipeline(proto, "root", context));
}

INSTANTIATE_TEST_SUITE_P(
    AllConfigs, DefaultPipelineCreatabilityTest,
    Values(PipelineConfigCase{"SinglePartitionGspmd", 1, false, false, false},
           PipelineConfigCase{"SinglePartitionShardy", 1, false, true, false},
           PipelineConfigCase{"SpmdGspmd", 8, true, false, false},
           PipelineConfigCase{"SpmdGspmdEnzyme", 8, true, false, true},
           PipelineConfigCase{"SpmdShardy", 8, true, true, false}),
    [](const TestParamInfo<PipelineConfigCase>& info) {
      return info.param.name;
    });

class GpuSpmdPartitioningTest : public HloHardwareIndependentTestBase,
                                public WithParamInterface<bool> {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>> PartitionComputation(
      absl::string_view hlo_text, int64_t num_devices) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/1, /*num_partitions=*/num_devices);
    config.set_use_spmd_partitioning(true);
    config.set_use_shardy_partitioner(GetParam());
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text, config));
    if (GetParam()) {
      module->add_frontend_attribute(std::string(sdy::kImportMhloShardings),
                                     "t");
    }

    const se::DeviceDescription device_description =
        TestGpuDeviceInfo::RTXA6000DeviceInfo();
    const GpuTopology gpu_topology("test", /*num_partitions=*/1,
                                   /*num_hosts_per_partition=*/1,
                                   /*num_devices_per_host=*/8);
    HloPassContext context;
    ABSL_RETURN_IF_ERROR(context.Set(device_description));
    ABSL_RETURN_IF_ERROR(context.Set(gpu_topology));

    ABSL_ASSIGN_OR_RETURN(const HloPipelineProto full_proto,
                     GetDefaultCompilationPipeline(module->config(), context));
    const auto it = absl::c_find_if(full_proto.root().nodes(),
                                    [](const HloPipelineNodeProto& node) {
                                      return node.name() == "spmd-partitioner";
                                    });
    TF_RET_CHECK(it != full_proto.root().nodes().end() && it->has_sub_nodes());

    HloPipelineProto spmd_proto;
    *spmd_proto.mutable_root() = it->sub_nodes();
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<HloPassPipeline> spmd_pipeline,
        BuildHloPassPipeline(spmd_proto, "spmd-partitioner", context));
    ABSL_RETURN_IF_ERROR(spmd_pipeline->Run(module.get()).status());
    return module;
  }
};

TEST_P(GpuSpmdPartitioningTest, DotWithEntryComputationLayout) {
  constexpr absl::string_view kHloModule = R"hlo(
    HloModule module,
      entry_computation_layout={(f32[8,16]{0,1}, f32[16,24]{1,0})
                                ->f32[8,24]{1,0}}

    ENTRY main {
      %p0 = f32[8,16] parameter(0), sharding={devices=[1,8]<=[8]}
      %p1 = f32[16,24] parameter(1), sharding={devices=[8,1]<=[8]}
      ROOT %dot = f32[8,24] dot(%p0, %p1), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    })hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       PartitionComputation(kHloModule, /*num_devices=*/8));

  EXPECT_EQ(module->config().entry_computation_layout().parameter_shape(0),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 2}, {0, 1}));
  EXPECT_EQ(module->config().entry_computation_layout().parameter_shape(1),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 24}, {1, 0}));
  EXPECT_EQ(module->config().entry_computation_layout().result_shape(),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 24}, {1, 0}));
}

INSTANTIATE_TEST_SUITE_P(All, GpuSpmdPartitioningTest, Values(true, false),
                         [](const TestParamInfo<bool>& info) {
                           return info.param ? "Shardy" : "GSPMD";
                         });

TEST(GpuPipelineTest, MissingDeviceDescriptionInContextFails) {
  const GpuTopology gpu_topology("test", /*num_partitions=*/1,
                                 /*num_hosts_per_partition=*/1,
                                 /*num_devices_per_host=*/8);
  HloPassContext context;
  ASSERT_OK(context.Set(gpu_topology));
  EXPECT_THAT(
      GetDefaultCompilationPipeline(HloModuleConfig(), context),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("DeviceDescription")));
}

TEST(GpuPipelineTest, MissingGpuTopologyInContextFails) {
  const se::DeviceDescription device_description =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  HloPassContext context;
  ASSERT_OK(context.Set(device_description));
  EXPECT_THAT(GetDefaultCompilationPipeline(HloModuleConfig(), context),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("GpuTopology")));
}

using GpuPipelineAlgebraicSimplifierTest = HloHardwareIndependentTestBase;

TEST_F(GpuPipelineAlgebraicSimplifierTest,
       ConvolutionCanonicalizationRespectsFastMinMax) {
  constexpr absl::string_view kHloText = R"hlo(
    HloModule minmax

    ENTRY main {
      p = f32[] parameter(0)
      negative_infinity = f32[] constant(-inf)
      positive_infinity = f32[] constant(inf)
      maximum = f32[] maximum(p, negative_infinity)
      minimum = f32[] minimum(p, positive_infinity)
      ROOT result = (f32[], f32[]) tuple(maximum, minimum)
    })hlo";
  for (bool fast_min_max : {false, true}) {
    SCOPED_TRACE(fast_min_max);
    for (bool is_cuda_platform : {false, true}) {
      SCOPED_TRACE(is_cuda_platform);
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                           ParseAndReturnVerifiedModule(kHloText));
      DebugOptions debug_options = GetDebugOptionsForTest();
      debug_options.set_xla_gpu_enable_fast_min_max(fast_min_max);
      AlgebraicSimplifier simplifier(GetAlgebraicSimplifierOptions(
          AlgebraicSimplifierMode::kGpuConvoluationCanonicalization,
          debug_options, is_cuda_platform));
      EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(!fast_min_max));
      const HloInstruction* root =
          module->entry_computation()->root_instruction();
      if (fast_min_max) {
        EXPECT_THAT(root, GmockMatch(m::Tuple(
                              m::Maximum(m::Parameter(0), m::Constant()),
                              m::Minimum(m::Parameter(0), m::Constant()))));
      } else {
        EXPECT_THAT(root,
                    GmockMatch(m::Tuple(m::Parameter(0), m::Parameter(0))));
      }
    }
  }
}

}  // namespace
}  // namespace xla::gpu
