/* Copyright 2025 The OpenXLA Authors.

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

#include <utility>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using RegressionDotTest = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(RegressionDotTest, LargeBF16Gemm) {
  const char* hlo_text = R"(
HloModule test
sum {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT add = bf16[] add(a, b)
}
ENTRY main {
  X = bf16[24480,64] parameter(0)
  Y = bf16[64,3072] parameter(1)
  zero = bf16[] constant(0)
  prod = bf16[24480,3072] dot(X, Y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT R = bf16[3072] reduce(prod, zero), dimensions={0}, to_apply=sum
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fake_arguments,
      MakeFakeArguments(module.get(), /*pseudo_random=*/true,
                        /*use_large_range=*/false));

  EXPECT_TRUE(RunAndCompare(std::move(module),
                            LiteralUtil::MakePointers(fake_arguments),
                            ErrorSpec{1e-3, 1e-2}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
