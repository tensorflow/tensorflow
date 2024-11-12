/* Copyright 2020 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/builder/lib/sorting.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

using CpuTopKTest = CpuCodegenTest;

TEST_F(CpuTopKTest, CallRuntimeUnbatched) {
  XlaBuilder builder(TestName());
  XlaOp input =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {100}), "input");
  TopK(input, 10);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation xla_computation, builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(ProgramShape program_shape,
                          xla_computation.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, HloModule::CreateFromProto(xla_computation.proto(), config));

  constexpr char filecheck_pattern[] = R"(
    CHECK: call void @__xla_cpu_runtime_TopKF32(i64 1, i64 100, i64 10,
  )";

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  CompileAheadOfTimeAndVerifyIr(std::move(module), options, filecheck_pattern,
                                /*match_optimized_ir=*/true);
}

TEST_F(CpuTopKTest, CallRuntimeBatched) {
  XlaBuilder builder(TestName());
  XlaOp input =
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {5, 100}), "input");
  TopK(input, 10);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation xla_computation, builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(ProgramShape program_shape,
                          xla_computation.GetProgramShape());
  HloModuleConfig config(program_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, HloModule::CreateFromProto(xla_computation.proto(), config));

  constexpr char filecheck_pattern[] = R"(
    CHECK: call void @__xla_cpu_runtime_TopKF32(i64 5, i64 100, i64 10,
  )";

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  CompileAheadOfTimeAndVerifyIr(std::move(module), options, filecheck_pattern,
                                /*match_optimized_ir=*/true);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
