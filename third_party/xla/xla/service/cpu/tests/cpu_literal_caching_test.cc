/* Copyright 2018 The OpenXLA Authors.

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
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/cpu/tests/cpu_pjrt_codegen_test.h"

namespace xla {
namespace cpu {
namespace {
class CpuDuplicateConstantsTest : public CpuPjRtCodegenTest {};

TEST_F(CpuDuplicateConstantsTest, RepeatedArrayConstants) {
  // We use a while loop here to force the two constant HloInstructions to be in
  // different computations.  Otherwise the HLO optimizer itself CSEs them.
  const std::string hlo_text = R"(
HloModule RepeatedConstants

while_body {
  arg_body = f32[2,3,2] parameter(0)
  ROOT const = f32[2,3,2] constant(
    {{{1, 2}, {1001, 1002}, {2001, 2002}},
     {{2, 1}, {2001, 3002}, {2001, 2002}}})
}

while_cond {
  arg_cond = f32[2,3,2] parameter(0)
  slice0 = f32[1,1,1] slice(arg_cond), slice={[0:1], [0:1], [0:1]}
  lhs = f32[] reshape(slice0)
  slice1 = f32[1,1,1] slice(arg_cond), slice={[1:2], [0:1], [0:1]}
  rhs = f32[] reshape(slice1)
  ROOT cond = pred[] compare(lhs, rhs), direction=EQ
}

ENTRY main {
  param = f32[2,3,2] parameter(0)
  const_a = f32[2,3,2] constant(
    {{{1, 2}, {1001, 1002}, {2001, 2002}},
     {{2, 1}, {2001, 3002}, {2001, 2002}}})
  const_b = f32[2,3,2] while(f32[2,3,2] param), condition=while_cond, body=while_body
  ROOT out = (f32[2,3,2], f32[2,3,2]) tuple(const_a, const_b)
}
)";

  std::string filecheck_pattern = R"(
CHECK: private unnamed_addr constant [48 x i8]
CHECK-NOT: private unnamed_addr constant [48 x i8]
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  CompileAheadOfTimeAndVerifyIr(std::move(module), options, filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

TEST_F(CpuDuplicateConstantsTest, RepeatedTupleConstants) {
  // We use a while loop here to force the two constant HloInstructions to be in
  // different computations.  Otherwise the HLO optimizer itself CSEs them.
  const std::string hlo_text = R"(
HloModule RepeatedConstants

while_body {
  arg_body = (f32[2,1]{1,0}, f32[1]{0}) parameter(0)
  ROOT const = (f32[2,1]{1,0}, f32[1]{0}) constant(({ { 1 }, { 2 } }, {2} ))
}

while_cond {
  arg_cond = (f32[2,1]{1,0}, f32[1]{0}) parameter(0)
  gte0 = f32[2,1]{1,0} get-tuple-element(arg_cond), index=0
  slice0 = f32[1,1]{1,0} slice(gte0), slice={[0:1], [0:1]}
  lhs = f32[] reshape(slice0)
  gte1 = f32[1]{0} get-tuple-element(arg_cond), index=1
  rhs = f32[] reshape(gte1)
  ROOT cond = pred[] compare(lhs, rhs), direction=EQ
}

ENTRY main {
  param = (f32[2,1]{1,0}, f32[1]{0}) parameter(0)
  const_a = (f32[2,1]{1,0}, f32[1]{0}) constant(( { { 1 }, { 2 } }, {2} ))
  const_b = (f32[2,1]{1,0}, f32[1]{0}) while((f32[2,1]{1,0}, f32[1]{0}) param), condition=while_cond, body=while_body
  ROOT out = ((f32[2,1]{1,0}, f32[1]{0}), (f32[2,1]{1,0}, f32[1]{0})) tuple(const_a, const_b)
}
)";

  std::string filecheck_pattern = R"(
CHECK-DAG: private unnamed_addr constant [4 x i8]
CHECK-DAG: private unnamed_addr constant [8 x i8]
CHECK-NOT: private unnamed_addr constant [4 x i8]
CHECK-NOT: private unnamed_addr constant [8 x i8]
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  CompileAheadOfTimeAndVerifyIr(std::move(module), options, filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
