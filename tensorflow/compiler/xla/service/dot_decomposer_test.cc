/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dot_decomposer.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using DotDecomposerTest = HloTestBase;

TEST_F(DotDecomposerTest, CanonicalizeMultipleNonContractingDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,63,512]{2,1,0} parameter(0)
    p1 = f32[512,512]{1,0} parameter(1)
    ROOT dot = f32[64,63,512]{2,1,0} dot(p0, p1), lhs_contracting_dims={2},
                                                  rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/1,
                                        /*rhs_contracting_dim=*/0),
                                op::Shape("f32[4032,512]"))));
}

}  // namespace
}  // namespace xla
