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
#include <string>

#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

using TupleUtilTest = HloTestBase;

TEST_F(TupleUtilTest, ExtractPrefix) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[32,32]{1,0},f32[32,32]{1,0},f32[32,32]{1,0}) parameter(0)
  ROOT p1 = f32[32,32]{1,0} parameter(1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);
  HloInstruction* prefix = TupleUtil::ExtractPrefix(param0, 2);

  EXPECT_THAT(prefix, op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                                op::GetTupleElement(op::Parameter(0), 1)));
}

TEST_F(TupleUtilTest, AppendSuffix) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[32,32]{1,0},f32[32,32]{1,0},f32[32,32]{1,0}) parameter(0)
  ROOT p1 = f32[32,32]{1,0} parameter(1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* param0 =
      module->entry_computation()->parameter_instruction(0);
  HloInstruction* param1 =
      module->entry_computation()->parameter_instruction(1);

  HloInstruction* with_suffix =
      TupleUtil::AppendSuffix(param0, {param1, param1});

  EXPECT_THAT(with_suffix, op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                                     op::GetTupleElement(op::Parameter(0), 1),
                                     op::GetTupleElement(op::Parameter(0), 2),
                                     op::Parameter(1), op::Parameter(1)));
}

TEST_F(TupleUtilTest, ReplaceTupleWithTupleInst) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = f32[32,32]{1,0} parameter(0)
  p1 = f32[32,32]{1,0} parameter(1)
  ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(p0, p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * new_tuple,
                          TupleUtil::ReplaceTupleWith(p0, tuple, {1}));

  EXPECT_THAT(new_tuple, op::Tuple(op::Parameter(0), op::Parameter(0)));
}

TEST_F(TupleUtilTest, ReplaceTupleWithNonTupleInst) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  ROOT p0 = (f32[32,32]{1,0}, f32[32,32]{1,0}) parameter(0)
  p1 = f32[32,32]{1,0} parameter(1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * new_tuple,
                          TupleUtil::ReplaceTupleWith(p1, p0, {0}));

  EXPECT_THAT(new_tuple, op::Tuple(op::Parameter(1),
                                   op::GetTupleElement(op::Parameter(0), 1)));
}

TEST_F(TupleUtilTest, ReplaceTupleWithNonTupleInstNested) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  ROOT p0 = (f32[32,32]{1,0}, (f32[32,32]{1,0}, f32[32,32]{1,0})) parameter(0)
  p1 = f32[32,32]{1,0} parameter(1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* p1 = FindInstruction(module.get(), "p1");

  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * new_tuple,
                          TupleUtil::ReplaceTupleWith(p1, p0, {1, 0}));

  EXPECT_THAT(
      new_tuple,
      op::Tuple(op::GetTupleElement(op::Parameter(0), 0),
                op::Tuple(op::Parameter(1),
                          op::GetTupleElement(
                              op::GetTupleElement(op::Parameter(0), 1), 1))));
}

TEST_F(TupleUtilTest, AddGetTupleElements) {
  const std::string hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[32,32]{1,0}, (f32[32,32]{1,0}, f32[32,32]{1,0})) parameter(0)
  gte = (f32[32,32]{1,0}, f32[32,32]{1,0}) get-tuple-element(p0), index=1
  ROOT root = f32[32,32]{1,0} get-tuple-element(gte), index=1
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* p0 = FindInstruction(module.get(), "p0");
  HloInstruction* existing_gte = FindInstruction(module.get(), "gte");

  HloInstruction* new_gte = TupleUtil::AddGetTupleElements({p0, {1, 0}});

  EXPECT_THAT(new_gte, op::GetTupleElement(existing_gte, 0));
}

}  // namespace
}  // namespace xla
