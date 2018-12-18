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

#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Not;
using ::testing::ResultOf;

class HloElementTypeConverterTest : public HloTestBase {
 public:
  std::unique_ptr<HloModule> CreateModuleFromHloString(
      const string& hlo_string) {
    return HloRunner::CreateModuleFromString(hlo_string,
                                             GetDebugOptionsForTest())
        .ValueOrDie();
  }
};

TEST_F(HloElementTypeConverterTest, CustomCallsNotConverted) {
  const string& hlo_string = R"(
    HloModule custom_call
    ENTRY CustomCall {
      constant = bf16[1]{0} constant({12345})
      ROOT custom-call = bf16[1,2,3]{0,2,1} custom-call(constant),
           custom_call_target="foo"
    }
  )";
  auto module = CreateModuleFromHloString(hlo_string);
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_FALSE(converted);
}

TEST_F(HloElementTypeConverterTest, InfeedsOutfeedsNotConverted) {
  const string& hlo_string = R"(
    HloModule InfeedOutfeed
    ENTRY RoundTrip16MiBR1.v2 {
      token0 = token[] after-all()
      infeed = (bf16[4]{0}, token[]) infeed(token0)
      ROOT infeed.data = bf16[4]{0} get-tuple-element(infeed), index=0
      outfeed = token[] outfeed(infeed.data, token0)
    }
  )";
  auto module = CreateModuleFromHloString(hlo_string);
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_FALSE(converted);
}

TEST_F(HloElementTypeConverterTest, OperationsInNestedTuplesConverted) {
  const string& hlo_string = R"(
    HloModule NestedTuples
    ENTRY NestedTuples.v5 {
      constant.4 = bf16[] constant(42)
      constant.2 = f32[2]{0} constant({1, 2})
      constant.3 = bf16[] constant(42)
      add = bf16[] add(constant.2, constant.3)
      tuple = (f32[2]{0}, bf16[]) tuple(constant.2, add)
      constant.5 = bf16[2]{0} constant({22, 44})
      ROOT tuple.1 = ((f32[2]{0}, bf16[]), bf16[2]{0}) tuple(tuple, constant.5)
    }
  )";

  auto module = CreateModuleFromHloString(hlo_string);
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_TRUE(converted);
  const HloInstruction* bf16_op =
      module->entry_computation()->root_instruction()->operand(0)->operand(1);
  EXPECT_THAT(bf16_op, op::Convert(op::Add(op::Constant(), op::Convert())));
}

TEST_F(HloElementTypeConverterTest, BatchNormGradBF16Converted) {
  const string& hlo_string = R"(
    HloModule BatchNormGrad
    ENTRY BatchNormGrad.v6 {
      constant.4 = bf16[2,2,2,1]{3,2,1,0} constant({ { /*i0=0*/
      { /*i1=0*/ {0}, {0} }, { /*i1=1*/ {0}, {0} } }, { /*i0=1*/ { /*i1=0*/ {0},
      {0} }, { /*i1=1*/ {0}, {0} } } })
      constant.5 = bf16[2]{0} constant({1, 1})
      constant.6 = bf16[2]{0} constant({0, 0})
      constant.7 = bf16[2]{0} constant({1, 1})
      constant.8 = bf16[2,2,2,1]{3,2,1,0} constant({ { /*i0=0*/
      { /*i1=0*/ {1}, {2} }, { /*i1=1*/ {3}, {4} } }, { /*i0=1*/ { /*i1=0*/
      {5}, {6} }, { /*i1=1*/ {7}, {8} } } })
      ROOT batch-norm-grad = (bf16[2,2,2,1]{3,2,1,0}, bf16[2]{0}, bf16[2]{0})
      batch-norm-grad(constant.4, constant.5, constant.6, constant.7,
      constant.8), epsilon=0, feature_index=2
    }
  )";

  auto module = CreateModuleFromHloString(hlo_string);
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_TRUE(converted);
  const HloInstruction* tuple_instr =
      module->entry_computation()->root_instruction();
  ::testing::Matcher<const ::xla::HloInstruction*> batch_norm =
      op::BatchNormGrad();
  EXPECT_THAT(tuple_instr,
              op::Tuple(op::Convert(op::GetTupleElement(batch_norm, 0)),
                        op::Convert(op::GetTupleElement(batch_norm, 1)),
                        op::Convert(op::GetTupleElement(batch_norm, 2))));
}

TEST_F(HloElementTypeConverterTest, RngIsRemoved) {
  const string& hlo_string = R"(
HloModule RngIsRemoved

ENTRY main {
  constant.3 = bf16[] constant(0)
  constant.4 = bf16[] constant(1)
  ROOT rng = bf16[1,1000,20]{2,1,0} rng(constant.3, constant.4), distribution=rng_uniform
}
  )";
  auto module = CreateModuleFromHloString(hlo_string);
  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_TRUE(converted);

  std::function<bool(const HloInstruction*)> is_bf16_rng =
      [](const HloInstruction* inst) {
        return inst->shape().element_type() == BF16 &&
               inst->opcode() == HloOpcode::kRng;
      };

  EXPECT_THAT(module->entry_computation()->instructions(),
              Not(Contains(ResultOf(is_bf16_rng, Eq(true)))));
}

TEST_F(HloElementTypeConverterTest, RngCtrlDep) {
  const string& hlo_string = R"(
HloModule RngIsRemoved

ENTRY main {
  constant.3 = bf16[] constant(0)
  constant.4 = bf16[] constant(1)
  rng0 = bf16[1,2000,20]{2,1,0} rng(constant.3, constant.4), distribution=rng_uniform
  ROOT rng1 = bf16[1,1000,20]{2,1,0} rng(constant.3, constant.4), control-predecessors={%rng0}, distribution=rng_uniform
}
  )";
  auto module = CreateModuleFromHloString(hlo_string);

  HloElementTypeConverter type_converter(BF16, F32);
  TF_ASSERT_OK_AND_ASSIGN(bool converted, type_converter.Run(module.get()));
  EXPECT_TRUE(converted);

  HloInstruction *rng0, *rng1;
  for (auto* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kRng) {
      const Shape& shape = inst->shape();
      ASSERT_EQ(shape.dimensions_size(), 3);
      ASSERT_TRUE(shape.dimensions(1) == 2000 || shape.dimensions(1) == 1000);
      if (shape.dimensions(1) == 2000) {
        rng0 = inst;
      } else {
        rng1 = inst;
      }
    }
  }

  EXPECT_THAT(rng0->control_successors(), ElementsAre(rng1));
  EXPECT_THAT(rng1->control_predecessors(), ElementsAre(rng0));
}

}  // namespace
}  // namespace xla
