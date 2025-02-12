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

#include <gtest/gtest.h>
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

using CpuOutfeedTest = CpuCodegenTest;

TEST_F(CpuOutfeedTest, OutfeedRoot) {
  const std::string hlo_text = R"(
HloModule Outfeed

ENTRY main {
  const_a = f32[2,3,2] constant(
    {{{1, 2}, {1001, 1002}, {2001, 2002}},
     {{2, 1}, {2001, 3002}, {2001, 2002}}})

  token0 = token[] after-all()
  outfeed = token[] outfeed(f32[2,3,2] const_a, token0)
  ROOT root = () tuple()
}
)";

  std::string filecheck_pattern = R"(
CHECK: private unnamed_addr constant [48 x i8]
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  CompileAheadOfTimeAndVerifyIr(std::move(module), options, filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

TEST_F(CpuOutfeedTest, OutfeedEmpty) {
  const std::string hlo_text = R"(
HloModule Outfeed

ENTRY main {
  const_a = f32[2,0] constant({{}, {}})
  token0 = token[] after-all()
  outfeed = token[] outfeed(f32[2,0] const_a, token0)
  ROOT root = () tuple()
}
)";

  std::string filecheck_pattern = R"(
CHECK: private unnamed_addr constant [0 x i8]
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  CompileAheadOfTimeAndVerifyIr(std::move(module), options, filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

TEST_F(CpuOutfeedTest, OutfeedTokenInTuple) {
  const std::string hlo_text = R"(
HloModule OutfeedTokenInTuple

ENTRY main {
  const = f32[] constant(42)
  epoch = token[] after-all()
  outfeed.tok = token[] outfeed(const, epoch)
  ROOT root = (token[], f32[]) tuple(outfeed.tok, const)
}
)";

  std::string filecheck_pattern = R"(
CHECK: Outfeed
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

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
