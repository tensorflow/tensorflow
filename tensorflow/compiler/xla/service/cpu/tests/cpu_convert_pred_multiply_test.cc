/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <cmath>
#include <iostream>
#include <vector>

#include "tensorflow/compiler/xla/service/cpu/tests/cpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace cpu {
namespace {

class CpuConvertPredMultiplyTest : public CpuCodegenTest {};

TEST_F(CpuConvertPredMultiplyTest, DoNotGenerateSelect) {
  const char* hlo_text = R"(
    HloModule Test1

    ENTRY Test1 {
      a = f32[] parameter(0)
      b = pred[] parameter(1)
      convert_0 = f32[] convert(b)
      ROOT multiply_0 = f32[] multiply(convert_0, a)
    }
  )";
  std::vector<Literal*> args;
  Literal param_a = LiteralUtil::CreateR0<float>(nanf(""));
  Literal param_b = LiteralUtil::CreateR0<bool>(false);
  args.push_back(&param_a);
  args.push_back(&param_b);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  const char* expected_ir = "//CHECK-NOT: select";
  CompileAndVerifyIr(hlo_text, expected_ir);

  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), args));
  auto result_value = result.GetFirstElement<float>();
  EXPECT_TRUE(isnan(result_value));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
