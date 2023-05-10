/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding.h"

#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "ortools/linear_solver/linear_solver.h"

namespace op = xla::testing::opcode_matchers;

using MPConstraint = operations_research::MPConstraint;
using MPSolver = operations_research::MPSolver;
using MPSolverParameters = operations_research::MPSolverParameters;
using MPVariable = operations_research::MPVariable;

namespace xla {
namespace spmd {
namespace {

using DummyAutoShardingTest = HloTestBase;

TEST_F(DummyAutoShardingTest, ReplicatedShardingDummy) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[5,7,11,13]{3,2,1,0} parameter(0)
  %param1 = f32[5,7,11,13]{3,2,1,0} parameter(1)
  %add = f32[5,7,11,13]{3,2,1,0} add(%param0, %param1)
  ROOT %copy = f32[5,7,11,13]{3,2,1,0} copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, DummyAutoSharding().Run(module.get()));
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "param0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{replicated}"));
}

TEST(MIPSolverTest, TwoVariableToyExample) {
  // SAT or SCIP
  std::unique_ptr<MPSolver> solver(
      std::make_unique<MPSolver>("", MPSolver::GLPK_MIXED_INTEGER_PROGRAMMING));
  solver->MutableObjective()->SetMaximization();
  ASSERT_TRUE(solver);
  // Test with the following integer programming problem:
  //   max  x + 2y
  //   s.t. 6x + 2y <= 19
  //        0 <= x <= 3
  //        0 <= y <= 2
  MPVariable* x = solver->MakeIntVar(0.0, 3.0, "x");
  MPVariable* y = solver->MakeIntVar(0.0, 2.0, "y");
  MPConstraint* constraint =
      solver->MakeRowConstraint(-MPSolver::infinity(), 19.0);
  constraint->SetCoefficient(x, 6.0);
  constraint->SetCoefficient(y, 2.0);
  solver->MutableObjective()->SetCoefficient(x, 1.0);
  solver->MutableObjective()->SetCoefficient(y, 2.0);
  MPSolver::ResultStatus solve_status = solver->Solve();
  EXPECT_EQ(solve_status, MPSolver::OPTIMAL);
  EXPECT_DOUBLE_EQ(x->solution_value(), 2.0);
  EXPECT_DOUBLE_EQ(y->solution_value(), 2.0);
}

class AutoShardingTest : public HloTestBase {
 protected:
  const char* const dot_hlo_string_ = R"(
HloModule module
ENTRY matmul {
  parameter.1 = f32[32,64]{1,0} parameter(0)
  parameter.2 = f32[64,128]{1,0} parameter(1)
  ROOT root = f32[32,128]{1,0} dot(parameter.1, parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  const char* const add_hlo_string_ = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[16,32,64]{2,1,0} parameter(0)
  %param1 = f32[16,32,64]{2,1,0} parameter(1)
  ROOT root = f32[16,32,64]{2,1,0} add(%param0, %param1)
})";
  void RunMatMulAutoShardingWithOptions(
      AutoShardingOption option, size_t expected_num_tiles,
      size_t expected_sharded_dimensions = 1) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(dot_hlo_string_));
    RunAutoShardingWithOptions(module.get(), option, expected_num_tiles,
                               expected_sharded_dimensions);
  }

  void RunAddAutoShardingWithOptions(AutoShardingOption option,
                                     size_t expected_num_tiles,
                                     size_t expected_sharded_dimensions = 1) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(add_hlo_string_));
    RunAutoShardingWithOptions(module.get(), option, expected_num_tiles,
                               expected_sharded_dimensions);
  }

  void RunAutoShardingWithOptions(HloModule* module, AutoShardingOption option,
                                  size_t expected_num_tiles,
                                  size_t expected_sharded_dimensions = 1) {
    TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module));
    EXPECT_TRUE(changed);
    // To simplify the test, only checking the sharding of root.
    auto* root = FindInstruction(module, "root");
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->sharding().NumTiles(), expected_num_tiles);
    EXPECT_EQ(VectorGreaterThanOneElementCount(
                  root->sharding().tile_assignment().dimensions(),
                  root->sharding().ReplicateOnLastTileDim()),
              expected_sharded_dimensions);
  }

  void RunMatMulAutoShardingWithOptionsExpectFail(AutoShardingOption option) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(dot_hlo_string_));
    RunAutoShardingWithOptionsExpectFail(module.get(), option);
  }

  void RunAutoShardingWithOptionsExpectFail(HloModule* module,
                                            AutoShardingOption option) {
    EXPECT_FALSE(AutoSharding(option).Run(module).ok());
  }

  void RunMatMulAutoShardingWithOptionsNoDeviceIds(
      AutoShardingOption option, std::vector<int64_t> expected_tile,
      bool expeted_last_dim_replicate = false) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(dot_hlo_string_));
    RunAutoShardingWithOptionsNoDeviceIds(module.get(), option, expected_tile,
                                          expeted_last_dim_replicate);
  }

  void RunAutoShardingWithOptionsNoDeviceIds(HloModule* module,
                                             AutoShardingOption option,
                                             std::vector<int64_t> expected_tile,
                                             bool expeted_last_dim_replicate) {
    TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module));
    EXPECT_TRUE(changed);
    // To simplify the test, only checking the sharding of root.
    HloInstruction* root = FindInstruction(module, "root");
    ASSERT_NE(root, nullptr);
    EXPECT_EQ(root->sharding().ReplicateOnLastTileDim(),
              expeted_last_dim_replicate);
    EXPECT_THAT(root->sharding().tile_assignment().dimensions(),
                ::testing::ElementsAreArray(expected_tile));
  }
};

TEST_F(AutoShardingTest, DISABLED_ElementWiseOperator) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[128,128]{0,1} parameter(0)
  %param1 = f32[128,128]{0,1} parameter(1)
  %add = f32[128,128]{0,1} add(%param0, %param1)
  ROOT %copy = f32[128,128]{0,1} copy(%add)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* instruction = FindInstruction(module.get(), "param0");
  ASSERT_NE(instruction, nullptr);
  EXPECT_THAT(instruction, op::Sharding("{devices=[2,2]0,2,1,3}"));
}

TEST_F(AutoShardingTest, DotLHSTwoNonContractingDims) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0)
  %param1 = f32[64,32]{0,1} parameter(1)
  %dot = f32[4,256,32]{2,1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[64,32]{0,1} %param1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT %copy = f32[4,256,32]{2,1,0} copy(%dot)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(2) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param0 = FindInstruction(module.get(), "param0");
  auto* param1 = FindInstruction(module.get(), "param1");
  auto* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(param0, nullptr);
  ASSERT_NE(param1, nullptr);
  ASSERT_NE(dot, nullptr);
  EXPECT_THAT(
      std::make_tuple(param0, param1, dot),
      AnyOf(
          ::testing::FieldsAre(
              op::Sharding(
                  "{devices=[1,2,1,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,1,2,3}")),
          ::testing::FieldsAre(
              op::Sharding(
                  "{devices=[1,2,1,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,2,1,3}")),
          ::testing::FieldsAre(
              op::Sharding(
                  "{devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[2,1,2]0,1,2,3}")),
          ::testing::FieldsAre(
              op::Sharding(
                  "{devices=[2,1,1,2]0,2,1,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"),
              op::Sharding("{devices=[2,1,2]0,2,1,3}"))));
}

TEST_F(AutoShardingTest, DotRHSTwoNonContractingDims) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,32]{2,1,0} parameter(0)
  %param1 = f32[4,256,4,8]{1,3,2,0} parameter(1)
  %dot = f32[32,4,8]{2,1,0} dot(f32[4,256,32]{2,1,0} %param0, f32[4,256,4,8]{1,3,2,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
  ROOT %copy = f32[32,4,8]{2,1,0} copy(%dot)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(2) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param0 = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0, nullptr);
  EXPECT_THAT(
      param0,
      op::Sharding("{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* param1 = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1, nullptr);
  EXPECT_THAT(
      param1,
      op::Sharding("{devices=[1,1,2,1,2]0,2,1,3 last_tile_dim_replicate}"));
  auto* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);
  EXPECT_THAT(dot, op::Sharding("{devices=[2,2,1]0,1,2,3}"));
}

TEST_F(AutoShardingTest, DotTwoContractingDims) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0)
  %param1 = f32[4,256,32]{2,1,0} parameter(1)
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
  ROOT %copy = f32[64,32]{1,0} copy(%dot)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(2) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param0 = FindInstruction(module.get(), "param0");
  auto* param1 = FindInstruction(module.get(), "param1");
  auto* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(param0, nullptr);
  ASSERT_NE(param1, nullptr);
  ASSERT_NE(dot, nullptr);
  EXPECT_THAT(
      std::make_tuple(param0, param1, dot),
      AnyOf(::testing::FieldsAre(
                op::Sharding(
                    "{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"),
                op::Sharding(
                    "{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"),
                op::Sharding("{devices=[2,2]0,2,1,3}")),
            ::testing::FieldsAre(
                op::Sharding(
                    "{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"),
                op::Sharding(
                    "{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"),
                op::Sharding("{devices=[2,2]0,1,2,3}"))));
}

TEST_F(AutoShardingTest, TwoMatmul) {
  const char* const hlo_string = R"(
HloModule module
ENTRY twomatmul {
  parameter.1 = f32[64,64]{1,0} parameter(0)
  parameter.2 = f32[64,128]{1,0} parameter(1)
  dot.4 = f32[64,128]{1,0} dot(parameter.1, parameter.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  parameter.3 = f32[128,64]{1,0} parameter(2)
  ROOT dot.5 = f32[64,64]{1,0} dot(dot.4, parameter.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* param1 = FindInstruction(module.get(), "parameter.1");
  ASSERT_NE(param1, nullptr);
  EXPECT_THAT(param1,
              op::Sharding("{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}"));
  auto* param2 = FindInstruction(module.get(), "parameter.2");
  ASSERT_NE(param2, nullptr);
  EXPECT_THAT(param2,
              op::Sharding("{devices=[1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* param3 = FindInstruction(module.get(), "parameter.3");
  ASSERT_NE(param3, nullptr);
  EXPECT_THAT(param3,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* dot4 = FindInstruction(module.get(), "dot.4");
  ASSERT_NE(dot4, nullptr);
  EXPECT_THAT(dot4, op::Sharding("{devices=[2,2]0,2,1,3}"));
  auto* dot5 = FindInstruction(module.get(), "dot.5");
  ASSERT_NE(dot5, nullptr);
  EXPECT_THAT(dot5,
              op::Sharding("{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}"));
}

TEST_F(AutoShardingTest, ProcessCustomCallShardings) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[6,3] parameter(0)
  %copy = f32[6,3] copy(%param0)
  %annotate = f32[6,3] custom-call(%copy), custom_call_target="Sharding",
    backend_config="unspecified_dims=[1]",
    sharding={devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}
  %copy.2 = f32[6,3] copy(%annotate)
  ROOT %copy.3 = f32[6,3] copy(%copy.2)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  // %annotate's sharding is moved to %copy.
  auto* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(copy->has_sharding());
  EXPECT_THAT(copy,
              op::Sharding("{devices=[2,1,2]0,2,1,3 last_tile_dim_replicate}"));
}

TEST_F(AutoShardingTest, RemoveShardingAnnotationKeepAll) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  // Keep all user shardings
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, AutoShardingImplementation(option).RemoveShardingAnnotation(
                        module.get()));
  EXPECT_FALSE(changed);
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* ins : computation->instructions()) {
      EXPECT_TRUE(ins->has_sharding());
    }
  }
}

TEST_F(AutoShardingTest, RemoveShardingAnnotationRemoveIntermediate) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepInputOutputShardings;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, AutoShardingImplementation(option).RemoveShardingAnnotation(
                        module.get()));
  EXPECT_TRUE(changed);
  // Dot does not have shardings anymore.
  auto* dot = FindInstruction(module.get(), "dot");
  ASSERT_NE(dot, nullptr);
  EXPECT_FALSE(dot->has_sharding());
  // params and copy still have shardings.
  auto* param0 = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0, nullptr);
  EXPECT_TRUE(param0->has_sharding());
  EXPECT_THAT(
      param0,
      op::Sharding("{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* param1 = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1, nullptr);
  EXPECT_TRUE(param1->has_sharding());
  EXPECT_THAT(
      param1,
      op::Sharding("{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"));
  auto* copy = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(copy->has_sharding());
  EXPECT_THAT(copy, op::Sharding("{devices=[2,2]0,1,2,3}"));
}

TEST_F(AutoShardingTest, RemoveShardingAnnotationRemoveAll) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  // Remove all user shardings
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kRemoveAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, AutoShardingImplementation(option).RemoveShardingAnnotation(
                        module.get()));
  EXPECT_TRUE(changed);
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* ins : computation->instructions()) {
      EXPECT_FALSE(ins->has_sharding());
    }
  }
}

TEST_F(AutoShardingTest, MatmulMeshShape1DMeshShape) {
  AutoShardingOption option;
  option.enable = true;
  // Only provide device_mesh_shape
  option.device_mesh_shape = {4};
  RunMatMulAutoShardingWithOptions(option, 4);
  option.device_mesh_shape = {8};
  RunMatMulAutoShardingWithOptions(option, 8);
}

TEST_F(AutoShardingTest, MatmulMeshShape1DMeshShapeIds) {
  AutoShardingOption option;
  option.enable = true;

  // Add mesh_ids
  option.device_mesh_shape = {4};
  option.device_mesh_ids = {0, 1, 2, 3};
  RunMatMulAutoShardingWithOptions(option, 4);

  option.device_mesh_shape = {8};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  RunMatMulAutoShardingWithOptions(option, 8);
}

TEST_F(AutoShardingTest, MatmulMeshShape1DAllOptions) {
  AutoShardingOption option;
  option.enable = true;
  // Add alpha and beta
  option.device_mesh_shape = {4};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0};
  option.device_mesh_beta = {1.0};
  RunMatMulAutoShardingWithOptions(option, 4);

  option.device_mesh_shape = {8};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  option.device_mesh_alpha = {1.0};
  option.device_mesh_beta = {1.0};
  RunMatMulAutoShardingWithOptions(option, 8);
}

TEST_F(AutoShardingTest, MatmulMeshShape2DAllOptions) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.enable = true;
  option.device_mesh_shape = {1, 4};
  RunMatMulAutoShardingWithOptions(option, 4);

  option.enable = true;
  option.device_mesh_shape = {4, 1};
  RunMatMulAutoShardingWithOptions(option, 4);
}

TEST_F(AutoShardingTest, MatmulMeshShape2DNoAlphaBeta) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.enable = true;
  option.device_mesh_shape = {1, 4};
  RunMatMulAutoShardingWithOptions(option, 4);

  // Specifying all mesh_* options.
  option.enable = true;
  option.device_mesh_shape = {4, 1};
  RunMatMulAutoShardingWithOptions(option, 4);
}

TEST_F(AutoShardingTest, MatmulMeshShape2DNoAlphaBetaMeshIds) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.enable = true;
  option.device_mesh_shape = {1, 4};
  RunMatMulAutoShardingWithOptions(option, 4);

  // Specifying all mesh_* options.
  option.enable = true;
  option.device_mesh_shape = {4, 1};
  RunMatMulAutoShardingWithOptions(option, 4);
}

TEST_F(AutoShardingTest, MatmulMeshShape2DNoMeshIds) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.enable = true;
  option.device_mesh_shape = {1, 4};
  RunMatMulAutoShardingWithOptions(option, 4);

  // Specifying all mesh_* options.
  option.enable = true;
  option.device_mesh_shape = {4, 1};
  RunMatMulAutoShardingWithOptions(option, 4);
}

TEST_F(AutoShardingTest, DISABLED_MatmulMeshShape3DAllOptions) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2, 2};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5, 1.0};
  RunMatMulAutoShardingWithOptionsNoDeviceIds(option, {2, 2, 2}, true);
}

TEST_F(AutoShardingTest, Matmul3DMeshShape2DSharding) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 2};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.device_mesh_shape = {2, 1, 2};
  RunMatMulAutoShardingWithOptions(option, 4, 2);

  option.device_mesh_shape = {2, 2, 1};
  RunMatMulAutoShardingWithOptions(option, 4, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3DAllOptions) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5, 1.0};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {4, 1, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {1, 4, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3DNoAlphaBeta) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  option.device_mesh_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {4, 1, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {1, 4, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3DNoAlphaBetaMeshIds) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {4, 1, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {1, 4, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3DNoMeshIds) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5, 1.0};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {4, 1, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);

  option.device_mesh_shape = {1, 4, 2};
  RunAddAutoShardingWithOptions(option, 8, 2);
}

TEST_F(AutoShardingTest, DISABLED_MatMulMeshShape2D) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  RunMatMulAutoShardingWithOptions(option, 4, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape2D) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  RunAddAutoShardingWithOptions(option, 4, 2);
}

TEST_F(AutoShardingTest, DISABLED_AddMeshShape3D) {
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2, 2};
  option.device_mesh_alpha = {1.0, 1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5, 1.0};
  RunAddAutoShardingWithOptions(option, 2);
}

TEST_F(AutoShardingTest, InvalidOptions) {
  // Sizes do not match.
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 2, 4};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 0.5};
  EXPECT_FALSE(option.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option);

  // Size is too large.
  option.device_mesh_shape = {1, 2, 4, 7};
  option.device_mesh_alpha = {1.0, 1.0, 1.0, 1.0};
  option.device_mesh_beta = {1.0, 1.0, 1.0, 1.0};
  EXPECT_FALSE(option.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option);

  // device_mesh_shape is empty.
  AutoShardingOption empty_option;
  empty_option.enable = true;
  EXPECT_FALSE(empty_option.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(empty_option);

  // Non-positive values in device_mesh_shape.
  AutoShardingOption option_with_non_positive_mesh;
  option_with_non_positive_mesh.enable = true;
  option_with_non_positive_mesh.device_mesh_shape = {0, 4};
  EXPECT_FALSE(option_with_non_positive_mesh.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option_with_non_positive_mesh);
  option_with_non_positive_mesh.device_mesh_shape = {-1, 4};
  EXPECT_FALSE(option_with_non_positive_mesh.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option_with_non_positive_mesh);

  // device_mesh_shape and device_mesh_ids are not compatible.
  AutoShardingOption option_not_compatible;
  option_not_compatible.enable = true;
  option_not_compatible.device_mesh_shape = {4, 8};
  option_not_compatible.device_mesh_ids = {1, 2, 3, 4};
  EXPECT_FALSE(option_not_compatible.CheckAndSetup().ok());
  RunMatMulAutoShardingWithOptionsExpectFail(option_not_compatible);
}

TEST_F(AutoShardingTest, AutoShardingKeepUserShardingInputOutput) {
  // An HLO Module with sharding for all instructions.
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Remove the sharding in dot
  auto* dot = FindInstruction(module.get(), "dot");
  dot->clear_sharding();
  EXPECT_FALSE(dot->has_sharding());
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepInputOutputShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* dot_after = FindInstruction(module.get(), "dot");
  ASSERT_NE(dot_after, nullptr);
  EXPECT_THAT(dot_after, op::Sharding("{devices=[2,2]0,1,2,3}"));
  auto sharding = dot_after->sharding();
  TF_EXPECT_OK(sharding.Validate(dot_after->shape(), 4));
}

TEST_F(AutoShardingTest, AutoShardingKeepUserShardingAdd) {
  // An HLO Module with sharding for all instructions.
  const char* const hlo_string = R"(
HloModule module
ENTRY %elementwise {
  %param0 = f32[128,128]{0,1} parameter(0)
  %param1 = f32[128,128]{0,1} parameter(1)
  %add = f32[128,128]{0,1} add(%param0, %param1), sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  ROOT %copy = f32[128,128]{0,1} copy(%add)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Run AutoSharding
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* param0_after = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0_after, nullptr);
  EXPECT_THAT(param0_after,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* param1_after = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1_after, nullptr);
  EXPECT_THAT(param1_after,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* add_after = FindInstruction(module.get(), "add");
  ASSERT_NE(add_after, nullptr);
  EXPECT_THAT(add_after,
              op::Sharding("{devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}"));
}

TEST_F(AutoShardingTest, AutoShardingKeepUserShardingDot) {
  // An HLO Module with sharding for all instructions.
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry (param0: f32[4,256,64], param1: f32[4,256,32]) -> f32[64,32] {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[4,256,32]{2,1,0} parameter(1), sharding={devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}
  %dot = f32[64,32]{1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[4,256,32]{2,1,0} %param1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}, sharding={devices=[2,2]0,1,2,3}
  ROOT %copy = f32[64,32]{1,0} copy(f32[64,32]{1,0} %dot), sharding={devices=[2,2]0,1,2,3}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Remove the sharding in param0, param1 and copy
  auto* param0 = FindInstruction(module.get(), "param0");
  param0->clear_sharding();
  EXPECT_FALSE(param0->has_sharding());
  auto* param1 = FindInstruction(module.get(), "param1");
  param1->clear_sharding();
  EXPECT_FALSE(param1->has_sharding());
  auto* copy = FindInstruction(module.get(), "copy");
  copy->clear_sharding();
  EXPECT_FALSE(copy->has_sharding());
  // Run AutoSharding
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  EXPECT_TRUE(changed);
  auto* param0_after = FindInstruction(module.get(), "param0");
  ASSERT_NE(param0_after, nullptr);
  EXPECT_THAT(
      param0_after,
      op::Sharding("{devices=[1,1,2,2]0,1,2,3 last_tile_dim_replicate}"));
  auto* param1_after = FindInstruction(module.get(), "param1");
  ASSERT_NE(param1_after, nullptr);
  EXPECT_THAT(
      param1_after,
      op::Sharding("{devices=[1,1,2,2]0,2,1,3 last_tile_dim_replicate}"));
  auto* copy_after = FindInstruction(module.get(), "copy");
  ASSERT_NE(copy_after, nullptr);
  EXPECT_THAT(copy_after, op::Sharding("{devices=[2,2]0,1,2,3}"));
}

TEST_F(AutoShardingTest, DISABLED_TupleParameter) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %tupleparameter {
  %tuple_param = (f32[16,32,64]{2,1,0}, f32[16,32,64]{2,1,0}) parameter(0)
  %first = f32[16,32,64]{2,1,0} get-tuple-element((f32[16,32,64]{2,1,0}, f32[16,32,64]{2,1,0}) %tuple_param), index=0
  %second = f32[16,32,64]{2,1,0} get-tuple-element((f32[16,32,64]{2,1,0}, f32[16,32,64]{2,1,0}) %tuple_param), index=1
  ROOT root = f32[16,32,64]{2,1,0} add(%first, %second)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(10) << module->ToString();
  EXPECT_TRUE(changed);
  auto* tuple_param = FindInstruction(module.get(), "tuple_param");
  ASSERT_NE(tuple_param, nullptr);
  EXPECT_THAT(
      tuple_param,
      op::Sharding("{{devices=[2,2,1]0,2,1,3}, {devices=[2,2,1]0,2,1,3}}"));
  TF_EXPECT_OK(tuple_param->sharding().Validate(tuple_param->shape(), 4));
}

TEST_F(AutoShardingTest, Reshape) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param.0 = bf16[24,2048,2048]{2,1,0} parameter(0)
  %param.1 = s32[] parameter(1)
  %param.2 = bf16[512,1024,2048]{2,1,0} parameter(2)
  %constant = s32[] constant(0)
  %dynamic-slice = bf16[1,2048,2048]{2,1,0} dynamic-slice(bf16[24,2048,2048]{2,1,0} %param.0, s32[] %param.1, s32[] %constant, s32[] %constant), dynamic_slice_sizes={1,2048,2048}
  %reshape = bf16[2048,16,128]{2,1,0} reshape(bf16[1,2048,2048]{2,1,0} %dynamic-slice)
  %dot = bf16[512,1024,16,128]{3,2,1,0} dot(bf16[512,1024,2048]{2,1,0} %param.2, bf16[2048,16,128]{2,1,0} %reshape), lhs_contracting_dims={2}, rhs_contracting_dims={0}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {64, 1};
  option.device_mesh_ids.resize(64);
  std::iota(option.device_mesh_ids.begin(), option.device_mesh_ids.end(), 0);
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, AutoSharding(option).Run(module.get()));
  VLOG(1) << module->ToString();
  EXPECT_TRUE(changed);
}

TEST_F(AutoShardingTest, TestReshardingCostsForUserAnnotatedSharding) {
  const char* const hlo_string = R"(
HloModule module

ENTRY %entry {
  %param0 = f32[256,256] parameter(0)
  %param1 = f32[256,256] parameter(1)
  %dot = f32[256,256] dot(%param0, %param1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT %result = f32[256,256] tanh(%dot), sharding={devices=[1,4]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {2, 2};
  option.device_mesh_beta = {1, 1};
  option.device_mesh_alpha = {1, 1};
  option.preserve_shardings =
      AutoShardingOption::PreserveShardingsType::kKeepAllShardings;
  AutoSharding pass(option);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  LOG(INFO) << module->ToString();
  EXPECT_GT(pass.GetSolverOptimalObjectiveValue(), 0);
}

}  // namespace
}  // namespace spmd
}  // namespace xla
