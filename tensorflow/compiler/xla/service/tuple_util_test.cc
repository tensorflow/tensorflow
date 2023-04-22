/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/tuple_util.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

StatusOr<std::unique_ptr<VerifiedHloModule>> GetParsedModule(
    HloComputation** entry_computation, HloInstruction** param0,
    HloInstruction** param1) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[32,32]{1,0},f32[32,32]{1,0},f32[32,32]{1,0}) parameter(0)
  ROOT p1 = f32[32,32]{1,0} parameter(1)
}
)";

  auto module = absl::make_unique<VerifiedHloModule>(
      "TupleUtilTest", HloModuleConfig(), /*verifier_layout_sensitive=*/true,
      /*allow_mixed_precision_in_hlo_verifier=*/false,
      ShapeUtil::ByteSizeOfElements);
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_string));

  *entry_computation = module->entry_computation();
  *param0 = (*entry_computation)->parameter_instruction(0);
  *param1 = (*entry_computation)->parameter_instruction(1);

  return std::move(module);
}

TEST(TupleUtilTest, ExtractPrefix) {
  HloInstruction *param0, *param1;
  HloComputation* entry_computation;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetParsedModule(&entry_computation, &param0, &param1));

  HloInstruction* prefix = TupleUtil::ExtractPrefix(param0, 2);

  EXPECT_THAT(prefix, op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                                op::GetTupleElement(op::Parameter(0), 1)));
}

TEST(TupleUtilTest, AppendSuffix) {
  HloInstruction *param0, *param1;
  HloComputation* entry_computation;

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetParsedModule(&entry_computation, &param0, &param1));

  HloInstruction* with_suffix =
      TupleUtil::AppendSuffix(param0, {param1, param1});

  EXPECT_THAT(with_suffix, op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                                     op::GetTupleElement(op::Parameter(0), 1),
                                     op::GetTupleElement(op::Parameter(0), 2),
                                     op::Parameter(1), op::Parameter(1)));
}

}  // namespace
}  // namespace xla
