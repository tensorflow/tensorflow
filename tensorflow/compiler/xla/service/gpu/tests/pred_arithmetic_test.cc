/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"

namespace xla {
namespace gpu {

namespace {

class PredArithmeticTest : public GpuCodegenTest {
 protected:
  void RunAndCompareTruthTable(const char* const hlo_text) {
    auto false_literal = LiteralUtil::CreateR0(false);
    auto true_literal = LiteralUtil::CreateR0(true);

    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_text));

    // Each call to RunAndCompareNoHloPasses insists on taking ownership and
    // deleting the module, so just Clone it.

    EXPECT_TRUE(RunAndCompareNoHloPasses(
        module->Clone(), {&false_literal, &false_literal}, absl::nullopt));

    EXPECT_TRUE(RunAndCompareNoHloPasses(
        module->Clone(), {&false_literal, &true_literal}, absl::nullopt));

    EXPECT_TRUE(RunAndCompareNoHloPasses(
        module->Clone(), {&true_literal, &false_literal}, absl::nullopt));

    EXPECT_TRUE(RunAndCompareNoHloPasses(
        module->Clone(), {&true_literal, &true_literal}, absl::nullopt));
  }
};

TEST_F(PredArithmeticTest, AddPreds) {
  const char* hlo_text = R"(
HloModule TestModule

ENTRY compare {
  x = pred[] parameter(0)
  y = pred[] parameter(1)
  ROOT z = pred[] add(x, y)
}

)";

  RunAndCompareTruthTable(hlo_text);
}

TEST_F(PredArithmeticTest, MulPreds) {
  const char* hlo_text = R"(
HloModule TestModule

ENTRY compare {
  x = pred[] parameter(0)
  y = pred[] parameter(1)
  ROOT z = pred[] multiply(x, y)
}

)";

  RunAndCompareTruthTable(hlo_text);
}

TEST_F(PredArithmeticTest, MaxPreds) {
  const char* hlo_text = R"(
HloModule TestModule

ENTRY compare {
  x = pred[] parameter(0)
  y = pred[] parameter(1)
  ROOT z = pred[] maximum(x, y)
}

)";

  RunAndCompareTruthTable(hlo_text);
}

TEST_F(PredArithmeticTest, MinPreds) {
  const char* hlo_text = R"(
HloModule TestModule

ENTRY compare {
  x = pred[] parameter(0)
  y = pred[] parameter(1)
  ROOT z = pred[] minimum(x, y)
}

)";

  RunAndCompareTruthTable(hlo_text);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
