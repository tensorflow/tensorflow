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

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {
class CpuDuplicateConstantsTest : public CpuCodegenTest {};

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
  token0 = token[] after-all()
  infeed = (pred[], token[]) infeed(token0)
  ROOT cond = pred[] get-tuple-element((pred[], token[]) infeed), index=0
}

ENTRY main {
  param = f32[2,3,2] parameter(0)
  const_a = f32[2,3,2] constant(
    {{{1, 2}, {1001, 1002}, {2001, 2002}},
     {{2, 1}, {2001, 3002}, {2001, 2002}}})
  const_b = f32[2,3,2] while(f32[2,3,2] const_a), condition=while_cond, body=while_body

  token0 = token[] after-all()
  out0 = token[] outfeed(f32[2,3,2] const_a, token[] token0)
  ROOT out1 = token[] outfeed(f32[2,3,2] const_b, token[] token0)
}
)";

  std::string filecheck_pattern = R"(
CHECK: private unnamed_addr constant [48 x i8]
CHECK-NOT: private unnamed_addr constant [48 x i8]
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
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
  token0 = token[] after-all()
  infeed = (pred[], token[]) infeed(token0)
  ROOT cond = pred[] get-tuple-element((pred[], token[]) infeed), index=0
}

ENTRY main {
  param = f32[2,3,2] parameter(0)
  const_a = (f32[2,1]{1,0}, f32[1]{0}) constant(( { { 1 }, { 2 } }, {2} ))
  const_b = (f32[2,1]{1,0}, f32[1]{0}) while((f32[2,1]{1,0}, f32[1]{0}) const_a), condition=while_cond, body=while_body

  token0 = token[] after-all()
  out0 = () outfeed((f32[2,1]{1,0}, f32[1]{0}) const_a, token[] token0)
  ROOT out1 = () outfeed((f32[2,1]{1,0}, f32[1]{0}) const_b, token[] token0)
}
)";

  std::string filecheck_pattern = R"(
CHECK-DAG: private unnamed_addr constant [4 x i8]
CHECK-DAG: private unnamed_addr constant [8 x i8]
CHECK-NOT: private unnamed_addr constant [4 x i8]
CHECK-NOT: private unnamed_addr constant [8 x i8]
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
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
