/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features_fake.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class ParallelTaskAssignmentTest : public HloVerifiedTestBase {
 protected:
  const HloCostAnalysis::ShapeSizeFunction shape_size_func_ =
      cpu::CpuExecutable::ShapeSizeBytes;

  // Use any value larger than 2 since we only test whether a module is
  // parallelized or not
  const int max_parallelism_ = 10;

  cpu::TargetMachineFeaturesWithFakeAlignmentLogic target_machine_features_;

  ParallelTaskAssignmentTest()
      : HloVerifiedTestBase(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false),
        target_machine_features_([](int64 shape_size) {
          return cpu::TargetMachineFeatures::kEigenExpectedTensorAlignment;
        }) {}

  StatusOr<bool> RunParallelTaskAssigner(HloModule* module) {
    return cpu::ParallelTaskAssigner(max_parallelism_, shape_size_func_,
                                     &target_machine_features_)
        .Run(module);
  }
};

TEST_F(ParallelTaskAssignmentTest, DotOperationNotParallelized) {
  const string hlo_string = R"(
    HloModule TestTaskParallel_Dot
    ENTRY Dot {
      dot_lhs = f32[196614,2]{1,0} parameter(0)
      dot_rhs = f32[2,1]{1,0} parameter(1)
      ROOT dot = f32[196614,1]{1,0} dot(dot_lhs, dot_rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  ParseAndVerifyModule(hlo_string);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(&module()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest,
       FusedComputationWithDotOperationNotParallelized) {
  const string hlo_string = R"(
    HloModule TestTaskParallel_DotNestedInFusedComp
    fused_computation.0 {
      parameter.0 = f32[196614,2]{1,0} parameter(0)
      parameter.0.1 = f32[2,1]{1,0} parameter(1)
      parameter.0.2 = f32[196614,1]{1,0} parameter(2)
      dot.0 = f32[196614,1]{1,0} dot(parameter.0, parameter.0.1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT add.0 = f32[196614,1]{1,0} add(dot.0, parameter.0.2)

    }
    ENTRY DotNestedInFusedComp {
      parameter = f32[196614,2]{1,0} parameter(0)
      parameter.1 = f32[2,1]{1,0} parameter(1)
      parameter.2 = f32[196614,1]{1,0} parameter(2)
      ROOT fusion = f32[196614,1]{1,0} fusion(parameter, parameter.1,
        parameter.2), kind=kOutput, calls=fused_computation.0
    }
  )";

  ParseAndVerifyModule(hlo_string);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(&module()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, RngOperationNotParallelized) {
  const string hlo_string = R"(
    HloModule TestTaskParallel_rng
    ENTRY Rng {
      src0 = f32[] parameter(0)
      src1 = f32[] parameter(1)
      ROOT rng0 = f32[1234567,2]{1,0} rng(f32[] src0, f32[] src1),
      distribution=rng_uniform
    }
  )";

  ParseAndVerifyModule(hlo_string);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(&module()));
  EXPECT_FALSE(changed);
}

TEST_F(ParallelTaskAssignmentTest, InfeedOutfeedOperationNotParallelized) {
  const string hlo_string = R"(
    HloModule TestTaskParallel_infeed_outfeed
    ENTRY InfeedOutfeed {
      token = token[] after-all()
      infeed0 = (u32[12345678,2]{1,0}, token[]) infeed(token)
      infeed0.data = u32[12345678,2]{1,0} get-tuple-element((u32[12345678,2]{1,0}, token[]) infeed0), index=0
      ROOT outfeed0 = token[] outfeed(infeed0.data, token)
    }
  )";

  ParseAndVerifyModule(hlo_string);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunParallelTaskAssigner(&module()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
