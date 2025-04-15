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

using CpuKeyValueSortTest = CpuCodegenTest;

TEST_F(CpuKeyValueSortTest, SortR1) {
  const std::string hlo_text = R"(
HloModule KeyValueSort

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY main {
  a = f32[10] parameter(0)

  ROOT result = f32[10] sort(f32[10] a), dimensions={0}, to_apply=compare
}
)";

  std::string filecheck_pattern = R"(
CHECK: call void @__xla_cpu_runtime_KeyValueSort
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_use_thunk_runtime(false);

  CompileAheadOfTimeAndVerifyIr(std::move(module), options, filecheck_pattern,
                                /*match_optimized_ir=*/true);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
