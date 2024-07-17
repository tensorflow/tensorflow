/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_module.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/test_compilation_environment.pb.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/lib/strings/proto_serialization.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

// In order to use TestCompilationEnvironment* with CompilationEnvironments, we
// must define ProcessNewEnv for them.
std::unique_ptr<tsl::protobuf::Message> ProcessNewEnv(
    std::unique_ptr<tsl::protobuf::Message> msg) {
  std::unique_ptr<test::TestCompilationEnvironment1> env(
      tensorflow::down_cast<test::TestCompilationEnvironment1*>(msg.release()));
  if (!env) {
    env = std::make_unique<test::TestCompilationEnvironment1>();
    env->set_some_flag(100);
  }
  return env;
}

namespace {

namespace op = ::xla::testing::opcode_matchers;

class HloModuleTest : public HloTestBase {
 protected:
  static void SetUpTestSuite() {
    CompilationEnvironments::RegisterProcessNewEnvFn(
        test::TestCompilationEnvironment1::descriptor(), ProcessNewEnv);
  }

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
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(CreateConstantComputation());

  EXPECT_THAT(module->MakeComputationPostOrder(),
              ::testing::ElementsAre(computation));
}

TEST_F(HloModuleTest, TwoComputationsPostOrder) {
  // Create a module with two unconnected computations.
  auto module = CreateNewVerifiedModule();
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
  auto module = CreateNewVerifiedModule();
  auto computation1 =
      module->AddEmbeddedComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation3 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  module->AddEntryComputation(
      CreateCallComputation({computation2, computation3}));
  // Add a compilation environment to module
  auto env = std::make_unique<test::TestCompilationEnvironment1>();
  env->set_some_flag(10);
  TF_ASSERT_OK(module->comp_envs().AddEnv(std::move(env)));

  auto post_order = module->MakeComputationPostOrder();
  auto cloned_module = module->Clone("copy");
  auto post_order_copied = cloned_module->MakeComputationPostOrder();

  // Make sure module's CompilationEnvironments were copied to cloned_module
  EXPECT_EQ(cloned_module->comp_envs()
                .GetEnv<test::TestCompilationEnvironment1>()
                .some_flag(),
            10);

  EXPECT_EQ(post_order.size(), post_order_copied.size());
  for (auto origin = post_order.begin(), copied = post_order_copied.begin();
       origin != post_order.end() && copied != post_order_copied.end();
       ++origin, ++copied) {
    EXPECT_EQ(absl::StrCat((*origin)->name(), ".copy"), (*copied)->name());
  }
}

TEST_F(HloModuleTest, CloneFrontendAttributes) {
  auto module = CreateNewVerifiedModule();
  FrontendAttributes frontend_attributes;
  frontend_attributes.mutable_map()->emplace("attribute1", "attribute1_value");
  module->set_frontend_attributes(frontend_attributes);
  std::unique_ptr<HloModule> clone = module->Clone();
  bool areEqual = std::equal(
      frontend_attributes.map().begin(), frontend_attributes.map().end(),
      clone->frontend_attributes().map().begin(),
      [](const auto& kv1, const auto& kv2) {
        return kv1.first == kv2.first && kv1.second == kv2.second;
      });
  EXPECT_TRUE(areEqual);
}

TEST_F(HloModuleTest, CloneHasFusion) {
  auto module = CreateNewVerifiedModule();

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
      EXPECT_EQ(absl::StrCat((*origin)->name(), ".clone"), (*copied)->name());
    } else {
      EXPECT_EQ(absl::StrCat((*origin)->name(), ".copy"), (*copied)->name());
    }
  }
}

TEST_F(HloModuleTest, CloneCustomCallComputationToApply) {
  const char* const hlo_string = R"(
HloModule a_module

add_s32 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY entry () -> s32[] {
  %c1 = s32[] constant(1)
  %c2 = s32[] constant(2)
  ROOT %custom-call =
    s32[] custom-call(s32[] %c1, %c2),
    custom_call_target="foo",
    backend_config="this string is opaque",
    to_apply=add_s32
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  std::unique_ptr<HloModule> cloned_module = module->Clone();
  HloComputation* cloned_computation =
      cloned_module->GetComputationWithName("add_s32.clone");
  HloInstruction* cloned_custom_call =
      cloned_module->entry_computation()->GetInstructionWithName("custom-call");

  EXPECT_TRUE(cloned_computation->IsCustomCallComputation());
  EXPECT_EQ(cloned_computation->CustomCallInstruction(), cloned_custom_call);
}

TEST_F(HloModuleTest, CloneCustomCallComputationCalledComputations) {
  const char* const hlo_string = R"(
HloModule a_module

add_s32_0 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

add_s32_1 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY entry () -> s32[] {
  %c1 = s32[] constant(1)
  %c2 = s32[] constant(2)
  ROOT %custom-call =
    s32[] custom-call(s32[] %c1, %c2),
    custom_call_target="foo",
    backend_config="this string is opaque",
    called_computations={%add_s32_0, %add_s32_1}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  std::unique_ptr<HloModule> cloned_module = module->Clone();
  HloComputation* cloned_computation_0 =
      cloned_module->GetComputationWithName("add_s32_0.clone");
  HloComputation* cloned_computation_1 =
      cloned_module->GetComputationWithName("add_s32_1.clone");
  HloInstruction* cloned_custom_call =
      cloned_module->entry_computation()->GetInstructionWithName("custom-call");

  EXPECT_TRUE(cloned_computation_0->IsCustomCallComputation());
  EXPECT_EQ(cloned_computation_0->CustomCallInstruction(), cloned_custom_call);
  EXPECT_TRUE(cloned_computation_1->IsCustomCallComputation());
  EXPECT_EQ(cloned_computation_1->CustomCallInstruction(), cloned_custom_call);
}

TEST_F(HloModuleTest, CloneFusionComputation) {
  const char* const hlo_string = R"(
HloModule a_module

fused_computation () -> s32[] {
  ROOT %result = s32[] parameter(0)
}

ENTRY main {
  %c = s32[] constant(1)
  ROOT %fusion = s32[] fusion(%c), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  std::unique_ptr<HloModule> cloned_module = module->Clone();
  HloComputation* cloned_computation =
      cloned_module->GetComputationWithName("fused_computation.clone");
  HloInstruction* cloned_fusion_instr =
      cloned_module->entry_computation()->GetInstructionWithName("fusion");

  EXPECT_TRUE(cloned_computation->IsFusionComputation());
  EXPECT_EQ(cloned_computation->FusionInstruction(), cloned_fusion_instr);
}

TEST_F(HloModuleTest, DiamondComputationsPostOrder) {
  // Create a module with a diamond call graph of computations.
  auto module = CreateNewVerifiedModule();
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
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder("Constant");
  std::vector<float> values(16, 42.0);
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>(values)));
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(
      "HloModule LargeConstantToString, "
      "entry_computation_layout={()->f32[16]{0}}\n\nENTRY %Constant () -> "
      "f32[16] {\n  ROOT %constant = f32[16]{0} constant({...})\n}\n\n",
      module->ToString(HloPrintOptions().set_print_large_constants(false)));

  EXPECT_EQ(
      "HloModule LargeConstantToString, "
      "entry_computation_layout={()->f32[16]{0}}\n\nENTRY %Constant () -> "
      "f32[16] {\n  ROOT %constant = f32[16]{0} constant({42, 42, 42, 42, 42, "
      "42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42})\n}\n\n",
      module->ToString(HloPrintOptions().set_print_large_constants(true)));
}

TEST_F(HloModuleTest, UniqueModuleId) {
  auto module_a = CreateNewVerifiedModule();
  auto module_b = CreateNewVerifiedModule();
  EXPECT_NE(module_a->unique_id(), module_b->unique_id());
}

TEST_F(HloModuleTest, ProtoSerializationWithoutSchedule) {
  const std::string text = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  ASSERT_FALSE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(
      auto module_copy,
      HloModule::CreateFromProto(module->ToProto(), module->config()));
  ASSERT_FALSE(module_copy->has_schedule());
}

TEST_F(HloModuleTest, ProtoSerializationWithSchedule) {
  const std::string text = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK_AND_ASSIGN(
      auto module_copy,
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

TEST_F(HloModuleTest, ProtoSerializationPreservesIds) {
  // Verify that serializing then deserializing an HLO proto preserves the
  // unique IDs of the instruction and module.
  const std::string text =
      R"(HloModule ReduceR3ToR2_module

add_F32.v3 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY ReduceR3ToR2.v3 {
  input = f32[8,16,256]{2,1,0} parameter(0)
  constant = f32[] constant(0)
  ROOT reduce = f32[8,16]{1,0} reduce(input, constant), dimensions={2}, to_apply=add_F32.v3
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));

  // Perform various transformations on the graph:
  //
  //  * clone the reduction function
  //  * replace use of reduction function with the clone.
  //  * add a random instruction to the entry computation.
  //
  // This will create instruction and computation IDs which are interesting:
  // not consecutive and not densely packed.
  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();
  HloComputation* reduction = root->to_apply();
  HloComputation* reduction_clone =
      module->AddEmbeddedComputation(reduction->Clone());
  root->set_to_apply(reduction_clone);
  TF_ASSERT_OK(module->RemoveEmbeddedComputation(reduction));
  HloInstruction* negate = entry->AddInstruction(
      HloInstruction::CreateUnary(root->shape(), HloOpcode::kNegate, root));
  entry->set_root_instruction(negate);

  // Schedule the transformed module, this verifies that the serialized schedule
  // is robust against non-consecutive IDs as well (b/114712358).
  auto size_fn = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape());
  };
  HloMemoryScheduler scheduler(size_fn);
  TF_ASSERT_OK(scheduler.Run(module.get()).status());
  ASSERT_TRUE(module->has_schedule());

  // Serialize and deserialize and verify that the instruction and computations
  // unique ids are the same.
  TF_ASSERT_OK_AND_ASSIGN(
      auto module_copy,
      HloModule::CreateFromProto(module->ToProto(), module->config()));

  // The module IDs should *not* be the same because module ids must be globally
  // unique.
  EXPECT_NE(module->unique_id(), module_copy->unique_id());

  // Verify that the computations and instructions all have the same unique id.
  auto computation_copy = module_copy->computations();
  auto computation_copy_it = computation_copy.begin();
  for (const HloComputation* computation_orig : module->computations()) {
    const HloComputation* computation_copy = *computation_copy_it++;
    EXPECT_EQ(computation_orig->unique_id(), computation_copy->unique_id())
        << absl::StrFormat(
               "ID of original computation %s != ID of deserialized "
               "computation %s: %d != %d",
               computation_orig->name(), computation_copy->name(),
               computation_orig->unique_id(), computation_copy->unique_id());

    auto instruction_copy_it = computation_copy->instructions().begin();
    for (const HloInstruction* instruction_orig :
         computation_orig->instructions()) {
      const HloInstruction* instruction_copy = *instruction_copy_it++;
      EXPECT_EQ(instruction_orig->unique_id(), instruction_copy->unique_id())
          << absl::StrFormat(
                 "ID of original instruction %s != ID of deserialized "
                 "instruction %s: %d != %d",
                 instruction_orig->name(), instruction_copy->name(),
                 instruction_orig->unique_id(), instruction_copy->unique_id());
    }
  }

  // Verify that the next unique ID which the module would have handed out is
  // greater than the unique id of any instruction.
  int next_id = module_copy->NewUniqueInstructionId();
  for (const HloComputation* computation : module_copy->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      EXPECT_GT(next_id, instruction->unique_id());
    }
  }
}

TEST_F(HloModuleTest, VerifyReplaceComputationsWithReduceScatter) {
  const std::string text = R"(
  HloModule reduce-scatter
  %sum (a: f32[], b: f32[]) -> f32[] {
    %a = f32[] parameter(0)
    %b = f32[] parameter(1)
    ROOT %add = f32[] add(f32[] a, f32[] b)
  }
  ENTRY main {
    %param = f32[16,8,128]{2,1,0} parameter(0)
    ROOT %rs = f32[4,8,128]{2,1,0} reduce-scatter(f32[16,8,128]{2,1,0} %param), replica_groups={}, to_apply=%sum, dimensions={0}
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));

  // Create a replacement computation
  HloComputation* new_comp;
  {
    auto b = HloComputation::Builder("Fused");
    auto p0 =
        b.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "p0"));
    auto p1 =
        b.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "p1"));
    b.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(F32, {}), HloOpcode::kMultiply, p0, p1));
    new_comp = module->AddEmbeddedComputation(b.Build());
  }

  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();
  EXPECT_EQ(root->to_apply()->root_instruction()->opcode(), HloOpcode::kAdd);

  absl::flat_hash_map<HloComputation*, HloComputation*> replacement;
  replacement[root->to_apply()] = new_comp;
  module->ReplaceComputations(replacement);

  EXPECT_EQ(root->to_apply(), new_comp);
}

TEST_F(HloModuleTest, VerifyReplaceComputationsWithSortOp) {
  const std::string text = R"(
  HloModule sort

  compare {
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      p.1.lhs = f32[] parameter(2)
      p.1.rhs = f32[] parameter(3)
      ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  ENTRY top {
    p.0 = f32[32] parameter(0)
    p.1 = f32[32] parameter(1)
    ROOT %sort.148.1589 = (f32[32], f32[32]) sort(p.0, p.1), dimensions={0}, to_apply=compare
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));

  // Create a replacement computation
  HloComputation* new_comp;
  {
    auto b = HloComputation::Builder("Fused");
    auto p0 =
        b.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "p0"));
    auto p1 =
        b.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "p1"));
    b.AddInstruction(HloInstruction::CreateParameter(2, r0f32_, "p2"));
    b.AddInstruction(HloInstruction::CreateParameter(3, r0f32_, "p3"));
    b.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), p0, p1, ComparisonDirection::kGt));
    new_comp = module->AddEmbeddedComputation(b.Build());
  }

  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();
  EXPECT_EQ(root->to_apply()->root_instruction()->opcode(),
            HloOpcode::kCompare);
  EXPECT_EQ(root->to_apply()->root_instruction()->comparison_direction(),
            ComparisonDirection::kLt);

  absl::flat_hash_map<HloComputation*, HloComputation*> replacement;
  replacement[root->to_apply()] = new_comp;
  module->ReplaceComputations(replacement);

  EXPECT_EQ(root->to_apply(), new_comp);
}

TEST_F(HloModuleTest, OneComputationAllAllowed) {
  // Create a module with a single computation and
  // ensure it is available when placed in the allow-list
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(CreateConstantComputation());

  absl::flat_hash_set<HloComputation*> allowList = {computation};
  EXPECT_THAT(
      module->MakeComputationPostOrder(/*execution_threads=*/{}, allowList),
      ::testing::ElementsAre(computation));
}

TEST_F(HloModuleTest, OneComputationAllFiltered) {
  // Create a module with a single computation.
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(CreateConstantComputation());

  absl::flat_hash_set<HloComputation*> allowList = {};
  module->MakeComputationPostOrder(/*execution_threads=*/{}, allowList);
  EXPECT_THAT(
      module->MakeComputationPostOrder(/*execution_threads=*/{}, allowList),
      ::testing::IsEmpty());
}

TEST_F(HloModuleTest, DiamondComputationsPostOrderAllAllowed) {
  // Create a module with a diamond call graph of computations.
  auto module = CreateNewVerifiedModule();
  auto computation1 =
      module->AddEmbeddedComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation3 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation4 = module->AddEntryComputation(
      CreateCallComputation({computation2, computation3}));

  absl::flat_hash_set<HloComputation*> allowList = {computation1, computation2,
                                                    computation3, computation4};
  auto post_order =
      module->MakeComputationPostOrder(/*execution_threads=*/{}, allowList);
  EXPECT_THAT(post_order,
              ::testing::UnorderedElementsAre(computation1, computation2,
                                              computation3, computation4));
  EXPECT_EQ(post_order.back(), computation4);
  EXPECT_EQ(post_order.front(), computation1);
}

TEST_F(HloModuleTest, DiamondComputationsPostOrderMiddleFiltered) {
  // Create a module with a diamond call graph of computations.
  auto module = CreateNewVerifiedModule();
  auto computation1 =
      module->AddEmbeddedComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation3 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation4 = module->AddEntryComputation(
      CreateCallComputation({computation2, computation3}));

  absl::flat_hash_set<HloComputation*> allowList = {computation1, computation4};
  auto post_order =
      module->MakeComputationPostOrder(/*execution_threads=*/{}, allowList);
  EXPECT_THAT(post_order,
              ::testing::UnorderedElementsAre(computation1, computation4));
}

TEST_F(HloModuleTest, DiamondComputationsPostOrderAllFiltered) {
  // Create a module with a diamond call graph of computations.
  auto module = CreateNewVerifiedModule();
  auto computation1 =
      module->AddEmbeddedComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  auto computation3 =
      module->AddEmbeddedComputation(CreateCallComputation({computation1}));
  module->AddEntryComputation(
      CreateCallComputation({computation2, computation3}));

  absl::flat_hash_set<HloComputation*> allowList = {};
  auto post_order =
      module->MakeComputationPostOrder(/*execution_threads=*/{}, allowList);
  EXPECT_THAT(
      module->MakeComputationPostOrder(/*execution_threads=*/{}, allowList),
      ::testing::IsEmpty());
}

TEST_F(HloModuleTest, TwoComputationsFilterexecution_threads) {
  // Create a module with two computations with different execution_threads and
  // ensure thread name filtering can return proper computations.
  HloComputation::Builder builder(TestName());
  constexpr char kParallelThreadName[] = "parallel_thread";
  // Create a call instruction containing a single binary operation.
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.1f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));
  auto module = CreateNewVerifiedModule();
  auto* main_thread_computation = module->AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      auto* async_done,
      main_thread_computation->CreateAsyncInstructions(
          add, {ShapeUtil::MakeScalarShape(U32)}, kParallelThreadName));
  auto* parallel_thread_computation = async_done->async_wrapped_computation();

  EXPECT_THAT(
      module->MakeComputationPostOrder({HloInstruction::kMainExecutionThread}),
      ::testing::ElementsAre(main_thread_computation));
  EXPECT_THAT(module->MakeComputationPostOrder(),
              ::testing::ElementsAre(parallel_thread_computation,
                                     main_thread_computation));
  EXPECT_THAT(module->MakeComputationPostOrder({kParallelThreadName}),
              ::testing::ElementsAre(parallel_thread_computation));
  // Test that computations(execution_thread) return the expected values.
  int num_all_computations = 0;
  for ([[maybe_unused]] const HloComputation* comp :
       module->computations(/*execution_threads=*/{})) {
    ++num_all_computations;
  }
  EXPECT_EQ(num_all_computations, 2);
  int num_main_computations = 0;
  for (const HloComputation* comp :
       module->computations({HloInstruction::kMainExecutionThread})) {
    ++num_main_computations;
    EXPECT_EQ(comp->execution_thread(), HloInstruction::kMainExecutionThread);
  }
  EXPECT_EQ(num_main_computations, 1);
  int num_parallel_computations = 0;
  for (const HloComputation* comp :
       module->computations({kParallelThreadName})) {
    ++num_parallel_computations;
    EXPECT_EQ(comp->execution_thread(), kParallelThreadName);
  }
  EXPECT_EQ(num_parallel_computations, 1);
}

TEST_F(HloModuleTest, HloModuleWithConfigSerializationEquality) {
  const std::string computation_text =
      R"(HloModule ReduceR3ToR2_module

add_F32.v3 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY ReduceR3ToR2.v3 {
  input = f32[8,16,256]{2,1,0} parameter(0)
  constant = f32[] constant(0)
  ROOT reduce = f32[8,16]{1,0} reduce(input, constant), dimensions={2}, to_apply=add_F32.v3
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(computation_text));

  xla::HloModuleProtoWithConfig proto = module->ToProtoWithConfig();
  std::string serialized_module;
  ASSERT_TRUE(tsl::SerializeToStringDeterministic(proto, &serialized_module));
  std::string original_debug_str = proto.DebugString();
  RecordProperty("serialized_module", original_debug_str);

  // Verify that we can create a module from our parsed proto copy
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> reconstructed_module,
                          HloModule::CreateFromProtoWithConfig(proto));
  xla::HloModuleProtoWithConfig reconstructed_module_proto =
      reconstructed_module->ToProtoWithConfig();

  // The two protos should be equivalent except for the `id` field
  google::protobuf::util::MessageDifferencer diff;
  diff.set_message_field_comparison(
      google::protobuf::util::MessageDifferencer::EQUIVALENT);
  auto module_descriptor = HloModuleProto::GetDescriptor();
  auto unique_id_field = module_descriptor->FindFieldByName("id");
  diff.IgnoreField(unique_id_field);
  EXPECT_TRUE(diff.Compare(proto, reconstructed_module_proto));
}

static ShardableValueUpdatePairProto MakeShardPair(int offset) {
  ShardableValueUpdatePairProto pear;
  pear.set_input_parameter_number(offset + 1);
  for (int64_t i = 0; i < 5; ++i) {
    pear.add_parameter_shape_index(offset + i);
  }
  for (int64_t j = 0; j < 3; ++j) {
    pear.add_output_shape_index(offset + j);
  }
  return pear;
}

static HloModuleConfigProto::BoolList MakeOneHotBoolList(unsigned num_vals,
                                                         unsigned hot_idx) {
  HloModuleConfigProto::BoolList list;
  for (unsigned i = 0; i < num_vals; ++i) {
    list.add_vals(i == hot_idx);
  }
  return list;
}

static absl::StatusOr<HloModuleConfigProto> MakeTestModuleConfigProto() {
  HloModuleConfigProto proto;
  // entry_computation_layout_ is optional
  proto.set_seed(0xdeadbeef);
  proto.set_launch_id(0xfeed100);
  proto.set_replica_count(3);
  proto.set_num_partitions(2);
  for (int x = 0; x < 6; ++x) {
    proto.add_param_requires_broadcast_via_collectives(x & 1);
  }
  proto.set_use_spmd_partitioning(true);
  proto.set_use_auto_spmd_partitioning(true);
  for (unsigned x = 0; x < 4; ++x) {
    proto.add_auto_spmd_partitioning_mesh_ids(10 - x);
    proto.add_auto_spmd_partitioning_mesh_ids(x);
  }
  proto.set_deduplicate_hlo(true);
  proto.set_intra_op_parallelism_threads(42);
  proto.set_device_type("Google Test framework");
  // debug options
  *proto.mutable_debug_options() = DefaultDebugOptionsIgnoringFlags();
  // static device assignment
  {
    DeviceAssignmentProto device_assignment_proto;
    DeviceAssignment device_assignment(/*replica_count=*/3,
                                       /*computation_count=*/2);
    device_assignment.Serialize(&device_assignment_proto);
    proto.mutable_static_device_assignment()->Swap(&device_assignment_proto);
  }
  // Shardable Value Update Pairs
  for (int k = 0; k < 3; ++k) {
    *proto.add_shardable_value_update_pairs() = MakeShardPair(k);
  }
  proto.set_alias_passthrough_params(true);
  proto.set_content_aware_computation_sorting(true);
  proto.set_fusion_config_collection(HloModuleConfigProto::PER_NODE);
  // fusion config
  for (int idx = 0; idx < 4; ++idx) {
    bool reverse = (idx & 1) == 0;
    *proto.add_fusion_config() =
        MakeOneHotBoolList(6, (reverse) ? 6 - idx : idx);
  }
  // dot config
  for (int idx = 0; idx < 4; ++idx) {
    HloModuleConfigProto::Int64List int_list;
    for (int x = 1; x <= 3; ++x) {
      int_list.add_vals(x * x * idx);
    }
    proto.mutable_dot_config()->insert(
        {absl::StrCat("Node", idx, "dot"), std::move(int_list)});
  }

  // layout config
  for (int idx = 0; idx < 4; ++idx) {
    HloModuleConfigProto::Int64ListList list_of_lists;
    for (int x = 0; x < 4; ++x) {
      HloModuleConfigProto::Int64List int_list;
      for (int y = 0; y < 6; ++y) {
        int_list.add_vals(y * x + idx + y + 1);
      }
      list_of_lists.add_lists()->Swap(&int_list);
    }
    proto.mutable_layout_config()->Add(std::move(list_of_lists));
  }

  // memory space assignment config
  for (uint64_t mem_asgn = 42; mem_asgn < 50; ++mem_asgn) {
    proto.add_memory_space_assignment_config(mem_asgn);
  }

  // phase ordering config
  for (int n = 0; n < 4; ++n) {
    *proto.add_phase_ordering_config() = MakeOneHotBoolList(4, n);
  }
  proto.set_phase_index(2);

  proto.add_allow_spmd_sharding_propagation_to_output(true);
  for (int idx = 1; idx <= 3; ++idx) {
    int64_t allowance = 35 * idx;
    proto.mutable_analysis_allowance_map()->insert(
        {absl::StrCat("Key", idx), allowance});
  }
  proto.set_matrix_unit_operand_precision(PrecisionConfig::HIGH);
  return proto;
}

TEST_F(HloModuleTest, HloModuleConfigCreateFromProto) {
  TF_ASSERT_OK_AND_ASSIGN(HloModuleConfigProto input_proto,
                          MakeTestModuleConfigProto());
  TF_ASSERT_OK_AND_ASSIGN(auto good_config,
                          HloModuleConfig::CreateFromProto(input_proto));
  HloModuleConfigProto output_proto = good_config->ToProto();

  google::protobuf::util::MessageDifferencer diff;
  diff.set_message_field_comparison(
      google::protobuf::util::MessageDifferencer::EQUIVALENT);
  EXPECT_TRUE(diff.Compare(input_proto, output_proto));
}

TEST_F(HloModuleTest, HloModuleConfigToProto) {
  auto module = CreateNewVerifiedModule();
  const HloModuleConfig& good_config = module->config();
  HloModuleConfigProto first_proto = good_config.ToProto();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModuleConfig> remade_config,
                          HloModuleConfig::CreateFromProto(first_proto));
  ASSERT_NE(remade_config, nullptr);
  HloModuleConfigProto second_proto = remade_config->ToProto();

  google::protobuf::util::MessageDifferencer diff;
  diff.set_message_field_comparison(
      google::protobuf::util::MessageDifferencer::EQUIVALENT);
  EXPECT_TRUE(diff.Compare(first_proto, second_proto));
}

TEST_F(HloModuleTest, HloModuleStackFrames) {
  const std::string text = R"(
HloModule a_module

ENTRY main {
  %c = s32[] constant(1)
  ROOT %result = s32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  EXPECT_TRUE(module->get_stack_frame(1).empty());

  auto module_proto = module->ToProto();
  auto index = module_proto.mutable_stack_frame_index();
  index->add_file_names("main.py");
  index->add_function_names("main");
  auto location = index->add_file_locations();
  location->set_file_name_id(1);
  location->set_function_name_id(1);
  location->set_line(10);
  location->set_column(5);

  auto frame = index->add_stack_frames();
  frame->set_file_location_id(1);

  module_proto.mutable_computations(0)
      ->mutable_instructions(0)
      ->mutable_metadata()
      ->set_stack_frame_id(1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto module_with_stack_frames,
      HloModule::CreateFromProto(module_proto, module->config()));

  EXPECT_TRUE(module_with_stack_frames->get_stack_frame(0).empty());
  EXPECT_TRUE(module_with_stack_frames->get_stack_frame(2).empty());

  auto stack_frame = module_with_stack_frames->get_stack_frame(1);
  EXPECT_EQ(stack_frame.file_name, index->file_names(0));
  EXPECT_EQ(stack_frame.function_name, index->function_names(0));
  EXPECT_EQ(stack_frame.line, location->line());
  EXPECT_EQ(stack_frame.column, location->column());
}

}  // namespace

}  // namespace xla
