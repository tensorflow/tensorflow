/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {

namespace {

namespace op = ::xla::testing::opcode_matchers;

class HloModuleTest : public HloTestBase {
 protected:
  HloModuleTest() {}

  // Create a computation which returns a constant.
  std::unique_ptr<HloComputation> CreateConstantComputation() {
    auto builder = HloComputation::Builder("Constant");
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
    return builder.Build();
  }

  // Creates a computation which calls the given zero-parameter computations.
  std::unique_ptr<HloComputation> CreateCallComputation(
      absl::Span<HloComputation* const> computations) {
    auto builder = HloComputation::Builder("Call");
    for (auto computation : computations) {
      builder.AddInstruction(
          HloInstruction::CreateCall(r0f32_, {}, computation));
    }
    return builder.Build();
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
};

TEST_F(HloModuleTest, OneComputationPostOrder) {
  // Create a module with a single computation.
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(CreateConstantComputation());

  EXPECT_THAT(module->MakeComputationPostOrder(),
              ::testing::ElementsAre(computation));
}

TEST_F(HloModuleTest, TwoComputationsPostOrder) {
  // Create a module with two unconnected computations.
  auto module = CreateNewModule();
  auto computation1 = module->AddEntryComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateConstantComputation());

  EXPECT_THAT(module->MakeComputationPostOrder(),
              ::testing::UnorderedElementsAre(computation1, computation2));

  // We specified the same name for both computations, but the HloModule should
  // have made the names unique.
  EXPECT_EQ(computation1->name(), "Constant");
  EXPECT_EQ(computation2->name(), "Constant.1");
}

TEST_F(HloModuleTest, CloneTest) {
  // Create and copy a module with a diamond call graph of computations.
  auto module = CreateNewModule();
  auto computation1 =
      module->AddEmbeddedComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation3 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  module->AddEntryComputation(
      CreateCallComputation({computation2, computation3}));

  auto post_order = module->MakeComputationPostOrder();
  auto cloned_module = module->Clone("copy");
  auto post_order_copied = cloned_module->MakeComputationPostOrder();

  EXPECT_EQ(post_order.size(), post_order_copied.size());
  for (auto origin = post_order.begin(), copied = post_order_copied.begin();
       origin != post_order.end() && copied != post_order_copied.end();
       ++origin, ++copied) {
    EXPECT_EQ((*origin)->name() + ".copy", (*copied)->name());
  }
}

TEST_F(HloModuleTest, CloneHasFusion) {
  auto module = CreateNewModule();

  // Create the fused computation.
  HloComputation* fused_computation;
  {
    auto b = HloComputation::Builder("Fused");
    auto x = b.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "x"));
    b.AddInstruction(
        HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, x, x));
    fused_computation = module->AddEmbeddedComputation(b.Build());
  }

  // Create the entry computation.
  {
    auto b = HloComputation::Builder("Entry");
    auto input = b.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
    b.AddInstruction(
        HloInstruction::CreateFusion(r0f32_, HloInstruction::FusionKind::kInput,
                                     /*operands=*/{input}, fused_computation));
    module->AddEntryComputation(b.Build());
  }

  auto post_order = module->MakeComputationPostOrder();
  auto cloned_module = module->Clone("copy");
  auto post_order_copied = cloned_module->MakeComputationPostOrder();

  EXPECT_EQ(post_order.size(), post_order_copied.size());
  for (auto origin = post_order.begin(), copied = post_order_copied.begin();
       origin != post_order.end() && copied != post_order_copied.end();
       ++origin, ++copied) {
    if ((*origin)->name() == "Fused") {
      // Clone of the fused computation is handled when its fusion instruction
      // is cloned, which always use suffix ".clone".
      EXPECT_EQ((*origin)->name() + ".clone", (*copied)->name());
    } else {
      EXPECT_EQ((*origin)->name() + ".copy", (*copied)->name());
    }
  }
}

TEST_F(HloModuleTest, DiamondComputationsPostOrder) {
  // Create a module with a diamond call graph of computations.
  auto module = CreateNewModule();
  auto computation1 =
      module->AddEmbeddedComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation3 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation4 = module->AddEntryComputation(
      CreateCallComputation({computation2, computation3}));

  auto post_order = module->MakeComputationPostOrder();
  EXPECT_THAT(post_order,
              ::testing::UnorderedElementsAre(computation1, computation2,
                                              computation3, computation4));
  EXPECT_EQ(post_order.back(), computation4);
  EXPECT_EQ(post_order.front(), computation1);
}

TEST_F(HloModuleTest, LargeConstantToString) {
  // Create a module with a single computation.
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder("Constant");
  std::vector<float> values(16, 42.0);
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(values)));
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(
      "HloModule LargeConstantToString\n\nENTRY %Constant () -> f32[16] {\n  "
      "ROOT %constant = f32[16]{0} constant({...})\n}\n\n",
      module->ToString(HloPrintOptions().set_print_large_constants(false)));

  EXPECT_EQ(
      "HloModule LargeConstantToString\n\nENTRY %Constant () -> f32[16] {\n  "
      "ROOT %constant = f32[16]{0} constant({42, 42, 42, 42, 42, 42, 42, 42, "
      "42, 42, 42, 42, 42, 42, 42, 42})\n}\n\n",
      module->ToString(HloPrintOptions().set_print_large_constants(true)));
}

TEST_F(HloModuleTest, UniqueModuleId) {
  auto module_a = CreateNewModule();
  auto module_b = CreateNewModule();
  EXPECT_NE(module_a->unique_id(), module_b->unique_id());
}

TEST_F(HloModuleTest, ProtoSerializationWithoutSchedule) {
  const string text = R"(
HloModule axpy_module

ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %x = f32[2,4]{1,0} parameter(1)
  %y = f32[2,4]{1,0} parameter(2)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(text));
  ASSERT_FALSE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module_copy,
      HloModule::CreateFromProto(module->ToProto(), module->config()));
  ASSERT_FALSE(module_copy->has_schedule());
}

TEST_F(HloModuleTest, ProtoSerializationWithSchedule) {
  const string text = R"(
HloModule axpy_module, is_scheduled=true

ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %x = f32[2,4]{1,0} parameter(1)
  %y = f32[2,4]{1,0} parameter(2)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(text));
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module_copy,
      HloModule::CreateFromProto(module->ToProto(), module->config()));
  ASSERT_TRUE(module_copy->has_schedule());
  TF_ASSERT_OK(module_copy->schedule().Verify());
  EXPECT_EQ(module_copy->schedule().sequences().size(), 1);
  ASSERT_TRUE(module_copy->schedule().is_computation_scheduled(
      module_copy->entry_computation()));
  EXPECT_THAT(
      module_copy->schedule()
          .sequence(module_copy->entry_computation())
          .instructions(),
      ::testing::ElementsAre(op::Parameter(), op::Parameter(), op::Parameter(),
                             op::Broadcast(), op::Multiply(), op::Add()));
}

}  // namespace

}  // namespace xla
