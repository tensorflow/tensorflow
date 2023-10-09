/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/while_loop_unroller.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/verified_hlo_module.h"

namespace xla {
namespace {

class WhileLoopUnrollerTest : public HloTestBase {
 protected:
  [[nodiscard]] std::unique_ptr<VerifiedHloModule> MakeModuleWithSimpleLoop(
      int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithLoopBodyIndirectInc(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithNestedLoopBodyIndirectInc(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithLoopBodyNestedCopyIndVar(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithWhileFeedingAnotherWhile(int num_iters);

 public:
  void UnrollAndCompare(std::unique_ptr<HloModule> module,
                        absl::Span<Literal* const> arguments,
                        int64_t unroll_factor = -1) {
    Literal before_unroll = ExecuteAndTransfer(module->Clone(), arguments);
    VLOG(2) << "after unroll value: " << before_unroll.ToString();

    EXPECT_TRUE(WhileLoopUnroller(unroll_factor).Run(module.get()).value());

    Literal after_unroll = ExecuteAndTransfer(std::move(module), arguments);
    VLOG(2) << "after unroll value: " << after_unroll.ToString();

    ASSERT_TRUE(LiteralTestUtil::NearOrEqual(/*expected=*/before_unroll,
                                             /*actual=*/after_unroll,
                                             std::nullopt));
  }
};

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithSimpleLoop(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    idx = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithLoopBodyIndirectInc(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}) tuple(inc, get-tuple-element.2, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, constant.4)
    ROOT while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithLoopBodyNestedCopyIndVar(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    inner-copy = s32[] copy(get-tuple-element.1)
    outer-copy = s32[] reshape(inner-copy)
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(outer-copy, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}) tuple(inc, get-tuple-element.2, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, constant.4)
    ROOT while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithNestedLoopBodyIndirectInc(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}) tuple(inc, get-tuple-element.2, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, constant.4)
    ROOT while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  OuterLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.22 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[10]{0} get-tuple-element(loop_var.1), index=3
    output = s32[10]{0} add(get-tuple-element.3, get-tuple-element.3)
    /* inner loop call*/
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, get-tuple-element.22)
    inner-while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
        SimpleLoop.condition, body=SimpleLoop.body 
    get-tuple-element.6 = s32[3]{0} get-tuple-element(inner-while), index=2
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}, s32[10]{0}) tuple(inc, get-tuple-element.2, get-tuple-element.6, output)
  }
  OuterLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY OuterLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    constant.5 = s32[10]{0} constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    tuple.1 = (s32[], s32[], s32[3]{0}, s32[10]{0}) tuple(constant.3, constant.1, constant.4, constant.5)
    ROOT while = (s32[], s32[], s32[3]{0}, s32[10]{0}) while(tuple.1), condition=
        OuterLoop.condition, body=OuterLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithWhileFeedingAnotherWhile(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    const1 = s32[] constant(1)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, const1)
    ROOT tuple = (s32[], s32[3]{0}) tuple(inc, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  OuterLoop.body {
    loop_var.1 = (s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.22 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[10]{0} get-tuple-element(loop_var.1), index=2
    output1 = s32[3]{0} add(get-tuple-element.22, get-tuple-element.22)
    output2 = s32[10]{0} add(get-tuple-element.3, get-tuple-element.3)
    one = s32[] constant(1)
    inc = s32[] add(get-tuple-element.1, one)
    ROOT tuple = (s32[], s32[3]{0}, s32[10]{0}) tuple(inc, output1, output2)
  }
  OuterLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY entry.comp {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    constant.5 = s32[10]{0} constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    /* inner loop call*/
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    inner-while = (s32[], s32[3]{0}) while(tuple.1), condition=
        SimpleLoop.condition, body=SimpleLoop.body
    get-tuple-element.6 = s32[3]{0} get-tuple-element(inner-while), index=1
    tuple.2 = (s32[], s32[3]{0}, s32[10]{0}) tuple(constant.3, get-tuple-element.6, constant.5)
    ROOT while = (s32[], s32[3]{0}, s32[10]{0}) while(tuple.2), condition=
        OuterLoop.condition, body=OuterLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopUnroll) {
  UnrollAndCompare(MakeModuleWithSimpleLoop(/*num_iters=*/5), {});
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopNotRoot) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {});
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopNonZeroInit) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(4)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {});
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopS16IndVar) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s16[] get-tuple-element(loop_var.1), index=0
    constant.1 = s16[] constant(1)
    add = s16[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s16[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s16[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s16[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s16[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s16[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s16[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {});
}

TEST_F(WhileLoopUnrollerTest, LoopWithControlDep) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s16[] get-tuple-element(loop_var.1), index=0
    constant.1 = s16[] constant(1)
    add = s16[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s16[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s16[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s16[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s16[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s16[], s32[3]{0}) tuple(constant.3, constant.4)
    while1 = (s16[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    copy1 = copy(constant.3), control-predecessors={while1}
    ROOT add = add(copy1, constant.3)
  }
  )";
  EXPECT_FALSE(WhileLoopUnroller()
                   .Run(ParseAndReturnVerifiedModule(hlo_string).value().get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopPartialUnroll) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/5);
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/3).Run(m.get()).value());
}

TEST_F(WhileLoopUnrollerTest, IndirectBodyInc) {
  std::unique_ptr<HloModule> module =
      MakeModuleWithLoopBodyIndirectInc(/*num_iters=*/5);
  UnrollAndCompare(std::move(module), {});
}

TEST_F(WhileLoopUnrollerTest, NestedIndirectBodyInc) {
  std::unique_ptr<HloModule> module =
      MakeModuleWithNestedLoopBodyIndirectInc(/*num_iters=*/5);
  UnrollAndCompare(std::move(module), {});
}

TEST_F(WhileLoopUnrollerTest, WhileFeedingWhile) {
  UnrollAndCompare(MakeModuleWithWhileFeedingAnotherWhile(/*num_iters=*/5), {});
}

}  // namespace
}  // namespace xla
