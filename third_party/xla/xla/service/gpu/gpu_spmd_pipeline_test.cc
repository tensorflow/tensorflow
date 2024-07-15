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

#include "xla/service/gpu/gpu_spmd_pipeline.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class GpuSpmdPartitioningTest : public HloTestBase,
                                public ::testing::WithParamInterface<bool> {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> PartitionComputation(
      const char* hlo_module, int64_t num_devices) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/1, /*num_partitions=*/num_devices);
    config.set_num_partitions(num_devices);
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_module, config));
    EXPECT_FALSE(config.debug_options().xla_use_shardonnay())
        << "Shardonnay not supported yet";

    HloPassPipeline spmd_pipeline("spmd-partitioner");
    se::CudaComputeCapability ampere(8, 0);
    AlgebraicSimplifierOptions alg_simplifier_options;
    // Ampere Core_count from tensorflow/compiler/xla/tools/hlo_opt/gpu_specs/.
    AddSPMDPasses(module.get(), alg_simplifier_options, ampere, spmd_pipeline,
                  std::nullopt);
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(module.get()).status());
    XLA_VLOG_LINES(10, module->ToString());
    return module;
  }

 protected:
  bool UseShardonnay() const { return GetParam(); }

  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_use_shardonnay(UseShardonnay());
    return debug_options;
  }
};

TEST_P(GpuSpmdPartitioningTest, DotWithEntryComputationLayout) {
  if (UseShardonnay()) {
    GTEST_SKIP() << "Shardonnay not supported yet";
  }

  const char* const kHloModule = R"(
  HloModule module,
   entry_computation_layout={(f32[8,16]{0,1}, f32[16,24]{1,0})
   ->f32[8,24]{1,0}}

  ENTRY main {
    %p0 = f32[8,16]  parameter(0), sharding={devices=[1,8]<=[8]}
    %p1 = f32[16,24] parameter(1), sharding={devices=[8,1]<=[8]}
    ROOT %dot = f32[8,24] dot(%p0, %p1), lhs_contracting_dims={1},
     rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          PartitionComputation(kHloModule, /*num_devices=*/8));

  EXPECT_EQ(module->config().entry_computation_layout().parameter_shape(0),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 2}, {0, 1}));
  EXPECT_EQ(module->config().entry_computation_layout().parameter_shape(1),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 24}, {1, 0}));
  EXPECT_EQ(module->config().entry_computation_layout().result_shape(),
            ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 24}, {1, 0}));
}

std::string TestParamToString(
    const ::testing::TestParamInfo<bool>& param_info) {
  return param_info.param ? "Shardonnay" : "GSPMD";
}

INSTANTIATE_TEST_SUITE_P(All, GpuSpmdPartitioningTest,
                         ::testing::Values(true, false), TestParamToString);

}  // namespace
}  // namespace gpu
}  // namespace xla
