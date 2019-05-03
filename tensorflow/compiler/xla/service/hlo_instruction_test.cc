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

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

class HloInstructionTest : public HloTestBase {
 protected:
  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
};

// Simple visitor that collects the number of users and operands for certain HLO
// nodes. It also verifies some of the DFS visiting invariants (operands visited
// before their users, nodes not visited twice, etc.)
class OpAndUserCollectingVisitor : public DfsHloVisitorWithDefault {
 public:
  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Unimplemented("not implemented %s",
                         HloOpcodeString(hlo_instruction->opcode()));
  }

  Status HandleParameter(HloInstruction* parameter) override {
    EXPECT_FALSE(count_.contains(parameter));
    count_[parameter] = GetCountsForNode(parameter);
    return Status::OK();
  }

  Status HandleConstant(HloInstruction* constant) override {
    EXPECT_FALSE(count_.contains(constant));
    count_[constant] = GetCountsForNode(constant);
    return Status::OK();
  }

  Status HandleAdd(HloInstruction* add) override {
    auto lhs = add->operand(0);
    auto rhs = add->operand(1);
    EXPECT_FALSE(count_.contains(add));
    EXPECT_TRUE(count_.contains(lhs));
    EXPECT_TRUE(count_.contains(rhs));
    count_[add] = GetCountsForNode(add);
    return Status::OK();
  }

  Status HandleNegate(HloInstruction* negate) override {
    auto operand = negate->operand(0);
    EXPECT_FALSE(count_.contains(negate));
    EXPECT_TRUE(count_.contains(operand));
    count_[negate] = GetCountsForNode(negate);
    return Status::OK();
  }

  Status HandleMap(HloInstruction* map) override {
    EXPECT_FALSE(count_.contains(map));
    for (HloInstruction* arg : map->operands()) {
      EXPECT_TRUE(count_.contains(arg));
    }
    count_[map] = GetCountsForNode(map);
    return Status::OK();
  }

  Status HandleReduce(HloInstruction* reduce) override {
    auto arg = reduce->operand(0);
    auto init_value = reduce->operand(1);
    EXPECT_FALSE(count_.contains(reduce));
    EXPECT_TRUE(count_.contains(arg));
    EXPECT_TRUE(count_.contains(init_value));
    count_[reduce] = GetCountsForNode(reduce);
    return Status::OK();
  }

  int64 NumOperands(const HloInstruction* node) {
    auto count_iterator = count_.find(node);
    EXPECT_NE(count_.end(), count_iterator);
    return count_iterator->second.operand_count;
  }

  int64 NumUsers(const HloInstruction* node) {
    auto count_iterator = count_.find(node);
    EXPECT_NE(count_.end(), count_iterator);
    return count_iterator->second.user_count;
  }

 private:
  struct NumOpsAndUsers {
    int64 operand_count;
    int64 user_count;
  };

  // Helper function to count operands and users for the given HLO.
  NumOpsAndUsers GetCountsForNode(const HloInstruction* node) {
    NumOpsAndUsers counts{node->operand_count(), node->user_count()};
    return counts;
  }

  // Counters for HLOs. Maps HLO to a NumOpsAndUsers.
  absl::flat_hash_map<const HloInstruction*, NumOpsAndUsers> count_;
};

TEST_F(HloInstructionTest, BasicProperties) {
  auto parameter = HloInstruction::CreateParameter(1, r0f32_, "foo");

  EXPECT_EQ(HloOpcode::kParameter, parameter->opcode());
  EXPECT_TRUE(ShapeUtil::IsScalarWithElementType(parameter->shape(), F32));
  EXPECT_FALSE(ShapeUtil::IsScalarWithElementType(parameter->shape(), S32));
  EXPECT_FALSE(parameter->operand_count());
}

TEST_F(HloInstructionTest, UserWithTwoOperands) {
  // [Param foo]----->  |-----|
  //                    | Add |
  // [Param bar]----->  |-----|
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto bar =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "bar"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, bar));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_THAT(add->operands(), UnorderedElementsAre(foo, bar));
  EXPECT_THAT(foo->users(), UnorderedElementsAre(add));
  EXPECT_THAT(bar->users(), UnorderedElementsAre(add));

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(add->Accept(&visitor));

  EXPECT_EQ(2, visitor.NumOperands(add));
  EXPECT_EQ(0, visitor.NumUsers(add));
  EXPECT_EQ(1, visitor.NumUsers(foo));
  EXPECT_EQ(1, visitor.NumUsers(bar));
}

TEST_F(HloInstructionTest, MultipleUsers) {
  //        [Param foo]
  //       /     |     \
  //      /      |      \     [Param bar]
  //     /       |       \         |
  //     |       |       |         |
  //     V       V       V         V
  //  -------  -------   -----------
  //  | exp |  | exp |   |   add   |
  //  -------  -------   -----------
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto bar =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "bar"));
  auto exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo));
  auto exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, bar));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, foo->user_count());
  EXPECT_EQ(1, bar->user_count());
  EXPECT_EQ(0, exp1->user_count());
  EXPECT_EQ(0, exp2->user_count());
  EXPECT_EQ(0, add->user_count());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(add->Accept(&visitor));

  EXPECT_EQ(2, visitor.NumOperands(add));
  EXPECT_EQ(3, visitor.NumUsers(foo));
}

TEST_F(HloInstructionTest, RepeatedUser) {
  // Here we have a user 'add' nodes that uses the same HLO in both operands.
  // Make sure we don't count it as two distinct users.
  //
  //        [Param foo]
  //           |   |
  //           |   |
  //           |   |
  //           V   V
  //          -------
  //          | add |
  //          -------
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, foo));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(1, foo->user_count());

  // But 'add' still has two operands, even if both are the same HLO.
  EXPECT_EQ(2, add->operand_count());
}

TEST_F(HloInstructionTest, MultipleUsersAndOperands) {
  //        [param0]          [param1]
  //           |                 |
  //           |       [c0]      |
  //           |        |        |
  //           V        |        V
  //        -------     |     -------
  //        | add | <---^---> | add |
  //        -------           -------
  //           |                 |
  //           \     -------     /
  //            ---->| add |<----
  //                 -------
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32_, "param1"));
  auto c0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto addleft = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, param0, c0));
  auto addright = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, c0, param1));
  auto addtotal = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, addleft, addright));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(addtotal->Accept(&visitor));

  EXPECT_EQ(2, visitor.NumUsers(c0));
  EXPECT_EQ(2, visitor.NumOperands(addleft));
  EXPECT_EQ(2, visitor.NumOperands(addright));
  EXPECT_EQ(2, visitor.NumOperands(addtotal));
}

TEST_F(HloInstructionTest, MultipleUsersAndOperandsWithUnaryOps) {
  //        [param0]   [c0]   [param1]
  //           |        |        |
  //           |        V        |
  //           |     -------     |
  //           |     | neg |     |
  //           |     -------     |
  //           V        |        V
  //        -------     |     -------
  //        | add | <---^---> | add |
  //        -------           -------
  //           |                 |
  //           \     -------     /
  //            ---->| add |<----
  //                 -------
  //                    |
  //                    V
  //                 -------
  //                 | neg |
  //                 -------
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32_, "param1"));
  auto c0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto neg1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, c0));
  auto addleft = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, param0, neg1));
  auto addright = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, neg1, param1));
  auto addtotal = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, addleft, addright));
  auto neg2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, addtotal));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(neg2->Accept(&visitor));

  EXPECT_EQ(1, visitor.NumUsers(c0));
  EXPECT_EQ(2, visitor.NumUsers(neg1));
  EXPECT_EQ(2, visitor.NumOperands(addleft));
  EXPECT_EQ(2, visitor.NumOperands(addright));
  EXPECT_EQ(2, visitor.NumOperands(addtotal));
  EXPECT_EQ(1, visitor.NumOperands(neg2));
  EXPECT_EQ(0, visitor.NumUsers(neg2));
}

TEST_F(HloInstructionTest, TrivialMap) {
  // This tests creating a trivial x+1 map as the only operation.
  //
  // param0[100x10] ---> (map x+1)
  //
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  Shape f32a100x10 = ShapeUtil::MakeShape(F32, {100, 10});
  auto module = CreateNewVerifiedModule();

  // Builds an x+1.0 computation to use in a Map.
  auto embedded_builder = HloComputation::Builder("f32+1");
  auto param = embedded_builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "x"));
  auto value = embedded_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  embedded_builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param, value));
  auto add_f32 = module->AddEmbeddedComputation(embedded_builder.Build());

  // Builds a parameter and feeds it to the map.
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32a100x10, "p"));
  auto map = builder.AddInstruction(
      HloInstruction::CreateMap(f32a100x10, {param0}, add_f32));
  module->AddEntryComputation(builder.Build());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(map->Accept(&visitor));

  // Check counts.  We aren't walking the mapper computation yet.
  EXPECT_EQ(1, visitor.NumUsers(param0));
  EXPECT_EQ(0, visitor.NumUsers(map));
  EXPECT_EQ(1, visitor.NumOperands(map));

  // TODO(dehnert):  Add walking and counters for the wrapped computation.
}

TEST_F(HloInstructionTest, TrivialReduce) {
  // This tests creating a trivial x+y reduce as the only operation.
  //
  // param0[100x10] ---> (reduce x+y)
  //
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  Shape f32v100 = ShapeUtil::MakeShape(F32, {100});
  Shape f32a100x10 = ShapeUtil::MakeShape(F32, {100, 10});

  // Builds an x+y computation to use in a Reduce.
  auto embedded_builder = HloComputation::Builder("f32+f32");
  auto paramx = embedded_builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "x"));
  auto paramy = embedded_builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "y"));
  embedded_builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, paramx, paramy));
  auto module = CreateNewVerifiedModule();
  auto add_f32 = module->AddEmbeddedComputation(embedded_builder.Build());

  // Builds a parameter and an initial value and feeds them to the reduce.
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32a100x10, "p"));
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto reduce = builder.AddInstruction(
      HloInstruction::CreateReduce(f32v100, param0, const0,
                                   /*dimensions_to_reduce=*/{1}, add_f32));
  module->AddEntryComputation(builder.Build());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(reduce->Accept(&visitor));

  // Check counts.  We aren't walking the reducer computation.
  EXPECT_EQ(1, visitor.NumUsers(param0));
  EXPECT_EQ(1, visitor.NumUsers(const0));
  EXPECT_EQ(0, visitor.NumUsers(reduce));
  EXPECT_EQ(2, visitor.NumOperands(reduce));
}

TEST_F(HloInstructionTest, ReplaceUseInBinaryOps) {
  // Construct a graph of a few binary ops using two different
  // parameters. Replace one of the parameters with the other parameter in one
  // of the instructions.
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto bar =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "bar"));
  auto add_foobar = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, bar));
  auto add_foofoo = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, foo));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                      add_foobar, add_foofoo));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(2, foo->user_count());
  EXPECT_EQ(1, bar->user_count());

  // Replace the use of foo in add_foofoo with bar.
  ASSERT_IS_OK(foo->ReplaceUseWith(add_foofoo, bar));

  EXPECT_EQ(1, foo->user_count());
  EXPECT_EQ(2, bar->user_count());

  EXPECT_THAT(foo->users(), UnorderedElementsAre(add_foobar));
  EXPECT_THAT(add_foobar->operands(), ElementsAre(foo, bar));

  EXPECT_THAT(bar->users(), UnorderedElementsAre(add_foobar, add_foofoo));
  EXPECT_THAT(add_foobar->operands(), ElementsAre(foo, bar));
  EXPECT_THAT(add_foofoo->operands(), ElementsAre(bar, bar));
}

TEST_F(HloInstructionTest, ReplaceUseInVariadicOp) {
  // Construct a tuple containing several parameters. Replace one parameter with
  // another in the tuple.
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto bar =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "bar"));
  auto baz =
      builder.AddInstruction(HloInstruction::CreateParameter(2, r0f32_, "baz"));

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({foo, bar, baz, foo}));
  auto add_foobar = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, bar));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(2, foo->user_count());
  EXPECT_THAT(foo->users(), UnorderedElementsAre(tuple, add_foobar));

  // Replace the use of foo in tuple with bar.
  ASSERT_IS_OK(foo->ReplaceUseWith(tuple, bar));

  EXPECT_THAT(foo->users(), UnorderedElementsAre(add_foobar));

  // Both uses of foo in tuple should have been replaced with bar.
  EXPECT_THAT(tuple->operands(), ElementsAre(bar, bar, baz, bar));
}

TEST_F(HloInstructionTest, ReplaceUseInUnaryOp) {
  // Construct a couple unary instructions which use a parameter. Replace the
  // use of a parameter in one of the unary ops with the other parameter.
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto bar =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "bar"));

  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kLog, foo));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(2, foo->user_count());
  EXPECT_THAT(foo->users(), UnorderedElementsAre(exp, log));
  EXPECT_EQ(0, bar->user_count());

  // Replace the use of foo in exp with bar.
  ASSERT_IS_OK(foo->ReplaceUseWith(exp, bar));

  // The use of foo in log should not have been affected.
  EXPECT_EQ(1, foo->user_count());
  EXPECT_THAT(foo->users(), UnorderedElementsAre(log));
  EXPECT_THAT(log->operands(), ElementsAre(foo));

  // Bar should now be used in exp.
  EXPECT_EQ(1, bar->user_count());
  EXPECT_EQ(*bar->users().begin(), exp);
  EXPECT_EQ(1, exp->operands().size());
  EXPECT_EQ(*exp->operands().begin(), bar);
}

TEST_F(HloInstructionTest, ReplaceAllUsesWithInBinaryOps) {
  // Construct a simple graph of a few binary ops using two different
  // parameters. Replace all uses of one of the parameters with the other
  // parameter.
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto bar =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "bar"));
  auto add_foobar = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, bar));
  auto add_foofoo = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, foo));
  builder.AddInstruction(HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                      add_foobar, add_foofoo));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(2, foo->user_count());
  EXPECT_EQ(1, bar->user_count());

  // Replace all uses of foo with bar.
  ASSERT_IS_OK(foo->ReplaceAllUsesWith(bar));

  EXPECT_EQ(0, foo->user_count());
  EXPECT_EQ(2, bar->user_count());

  EXPECT_THAT(bar->users(), UnorderedElementsAre(add_foobar, add_foofoo));
}

TEST_F(HloInstructionTest, ReplaceAllUsesInMultipleOps) {
  // Construct a graph containing several ops (a unary, binary, and variadic)
  // which use two parameters. Replace all uses of one of the parameters with
  // the other parameter.
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto bar =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "bar"));

  auto add_foobar = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo, bar));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo));
  auto tuple = builder.AddInstruction(HloInstruction::CreateTuple({foo, bar}));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, foo->user_count());
  EXPECT_EQ(2, bar->user_count());

  // Replace all uses of foo with bar.
  ASSERT_IS_OK(foo->ReplaceAllUsesWith(bar));

  EXPECT_EQ(0, foo->user_count());
  EXPECT_EQ(3, bar->user_count());

  EXPECT_THAT(bar->users(), UnorderedElementsAre(add_foobar, exp, tuple));
}

// Simple visitor that collects and post-processes each node in the graph.
class NodeCollectorAndPostProcessor : public DfsHloVisitorWithDefault {
 public:
  NodeCollectorAndPostProcessor() {}

  Status Postprocess(HloInstruction* hlo) override {
    post_processed_nodes_.push_back(hlo);
    return Status::OK();
  }

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    visited_nodes_.push_back(hlo_instruction);
    return Status::OK();
  }

  const std::vector<const HloInstruction*>& visited_nodes() {
    return visited_nodes_;
  }

  const std::vector<const HloInstruction*>& post_processed_nodes() {
    return post_processed_nodes_;
  }

 private:
  std::vector<const HloInstruction*> visited_nodes_;
  std::vector<const HloInstruction*> post_processed_nodes_;
};

// Returns true if "vec" contains distinct nodes.
bool Distinct(const std::vector<const HloInstruction*>& vec) {
  std::set<const HloInstruction*> distinct_nodes(vec.begin(), vec.end());
  return distinct_nodes.size() == vec.size();
}

TEST_F(HloInstructionTest, PostProcessAllVisitedNodes) {
  // Verifies all the nodes are visited and post-processed in the same order,
  // and that each node is visited exactly once.
  //
  //    /--> exp --\
  // foo            add
  //    \--> log --/
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo));
  auto log = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kLog, foo));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, exp, log));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  NodeCollectorAndPostProcessor visitor;
  ASSERT_IS_OK(add->Accept(&visitor));
  // Verifies all the nodes are visited and post-processed in the same order.
  EXPECT_EQ(visitor.visited_nodes(), visitor.post_processed_nodes());
  // Verifies each node is visited exactly once.
  EXPECT_TRUE(Distinct(visitor.visited_nodes()));
}

TEST_F(HloInstructionTest, SingletonFusionOp) {
  HloComputation::Builder builder(TestName());
  // Create a fusion instruction containing a single unary operation.
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, constant));
  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {exp}, HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(fusion->operands(), ElementsAre(constant));
  EXPECT_THAT(constant->users(), ElementsAre(fusion));
}

TEST_F(HloInstructionTest, BinaryFusionOp) {
  HloComputation::Builder builder(TestName());
  // Create a fusion instruction containing a single binary operation.
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.1f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));
  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {add}, HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(fusion->operands(), ElementsAre(constant1, constant2));
  EXPECT_THAT(constant1->users(), ElementsAre(fusion));
  EXPECT_THAT(constant2->users(), ElementsAre(fusion));
}

TEST_F(HloInstructionTest, ChainFusionOp) {
  HloComputation::Builder builder(TestName());
  // Create a chain of fused unary ops.
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, constant));
  auto exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, exp1));
  auto exp3 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, exp2));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {exp3, exp2, exp1}, HloInstruction::FusionKind::kLoop);

  EXPECT_THAT(fusion->operands(), ElementsAre(constant));
  EXPECT_THAT(constant->users(), ElementsAre(fusion));
}

TEST_F(HloInstructionTest, PreserveMetadataInFusionAndClone) {
  HloComputation::Builder builder(TestName());
  // Create a chain of fused unary ops.
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, constant));
  auto exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, exp1));
  OpMetadata metadata;
  metadata.set_op_name("tf_op");
  exp1->set_metadata(metadata);
  exp2->set_metadata(metadata);

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {exp2, exp1}, HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(protobuf_util::ProtobufEquals(metadata, fusion->metadata()));
  EXPECT_TRUE(protobuf_util::ProtobufEquals(
      metadata, fusion->fused_expression_root()->metadata()));
  EXPECT_TRUE(protobuf_util::ProtobufEquals(
      metadata, fusion->fused_expression_root()->operand(0)->metadata()));

  auto cloned = fusion->CloneWithNewOperands(fusion->shape(), {});
  EXPECT_TRUE(protobuf_util::ProtobufEquals(metadata, fusion->metadata()));
}

TEST_F(HloInstructionTest, PreserveOutfeedShapeThroughClone) {
  HloComputation::Builder builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>({
          {1, 2},
          {3, 4},
      })));
  auto shape10 = ShapeUtil::MakeShapeWithLayout(F32, {2, 2}, {1, 0});
  auto shape01 = ShapeUtil::MakeShapeWithLayout(F32, {2, 2}, {0, 1});
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  auto outfeed10 = builder.AddInstruction(
      HloInstruction::CreateOutfeed(shape10, constant, token, ""));
  auto outfeed01 = builder.AddInstruction(
      HloInstruction::CreateOutfeed(shape01, constant, token, ""));

  auto clone01 = builder.AddInstruction(outfeed01->Clone());
  auto clone10 = builder.AddInstruction(outfeed10->Clone());

  EXPECT_TRUE(ShapeUtil::Equal(clone01->outfeed_shape(), shape01));
  EXPECT_TRUE(ShapeUtil::Equal(clone10->outfeed_shape(), shape10));
}

TEST_F(HloInstructionTest, PreserveTupleShapeThroughClone) {
  HloComputation::Builder builder(TestName());
  auto* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>({
          {1, 2},
          {3, 4},
      })));
  auto* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant, constant}));
  *ShapeUtil::GetMutableSubshape(tuple->mutable_shape(), {0})
       ->mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  *ShapeUtil::GetMutableSubshape(tuple->mutable_shape(), {1})
       ->mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  auto tuple_clone = tuple->Clone();
  EXPECT_TRUE(ShapeUtil::Equal(tuple_clone->shape(), tuple->shape()));
}

TEST_F(HloInstructionTest, FusionOpWithCalledComputations) {
  // Create a fusion instruction containing a single unary operation.
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  auto module = CreateNewVerifiedModule();

  auto make_map_computation = [&]() {
    auto builder = HloComputation::Builder("FusionMap");
    builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "param"));
    return module->AddEmbeddedComputation(builder.Build());
  };

  HloComputation* computation_x = make_map_computation();
  HloComputation* computation_y = make_map_computation();

  HloComputation::Builder builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto map_1_x = builder.AddInstruction(
      HloInstruction::CreateMap(scalar_shape, {constant}, computation_x));
  auto map_2_x = builder.AddInstruction(
      HloInstruction::CreateMap(scalar_shape, {map_1_x}, computation_x));
  auto map_3_y = builder.AddInstruction(
      HloInstruction::CreateMap(scalar_shape, {map_2_x}, computation_y));
  auto* computation = module->AddEntryComputation(builder.Build());

  auto* fusion = computation->CreateFusionInstruction(
      {map_3_y}, HloInstruction::FusionKind::kLoop);
  auto* fused_computation = fusion->fused_instructions_computation();
  EXPECT_THAT(fusion->called_computations(), ElementsAre(fused_computation));

  fusion->FuseInstruction(map_2_x);
  EXPECT_THAT(fusion->called_computations(), ElementsAre(fused_computation));

  fusion->FuseInstruction(map_1_x);
  EXPECT_THAT(fusion->called_computations(), ElementsAre(fused_computation));
}

TEST_F(HloInstructionTest, ComplexFusionOp) {
  HloComputation::Builder builder(TestName());
  // Fuse all instructions in complicated expression:
  //
  //   add = Add(C1, C2)
  //   clamp = Clamp(C2, add, add)
  //   exp = Exp(add)
  //   mul = Mul(exp, C3)
  //   sub = Sub(mul, clamp)
  //   tuple = Tuple({sub, sub, mul, C1})
  //
  // Notable complexities are repeated operands in the same instruction,
  // different shapes, use of value in different expressions.
  auto c1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto c2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.1f)));
  auto c3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(9.0f)));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, c1, c2));
  auto clamp = builder.AddInstruction(
      HloInstruction::CreateTernary(r0f32_, HloOpcode::kClamp, c2, add, add));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, add));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kMultiply, exp, c3));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kSubtract, mul, clamp));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({sub, sub, mul, c1}));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {tuple, sub, mul, exp, clamp, add}, HloInstruction::FusionKind::kLoop);

  // Operands in the fusion instruction's operands() vector should be in the
  // order in which their users were added fused.
  EXPECT_THAT(fusion->operands(), ElementsAre(c1, c3, c2));
  EXPECT_THAT(c1->users(), ElementsAre(fusion));
}

// Convenience function for comparing two HloInstructions.
static bool Identical(const HloInstruction& instruction1,
                      const HloInstruction& instruction2) {
  // Verify Identical is reflexive for both instructions.
  EXPECT_TRUE(instruction1.Identical(instruction1));
  EXPECT_TRUE(instruction2.Identical(instruction2));

  bool is_equal = instruction1.Identical(instruction2);
  // Verify Identical is symmetric.
  EXPECT_EQ(is_equal, instruction2.Identical(instruction1));
  return is_equal;
}

// Convenience function for comparing two HloInstructions for structural
// equality.
static bool StructuralEqual(const HloInstruction& instruction1,
                            const HloInstruction& instruction2) {
  auto eq_operand_shapes = [](const HloInstruction* a,
                              const HloInstruction* b) {
    return ShapeUtil::Equal(a->shape(), b->shape());
  };
  auto eq_computations = [](const HloComputation* a, const HloComputation* b) {
    return *a == *b;
  };

  // Verify Identical is reflexive for both instructions.
  EXPECT_TRUE(
      instruction1.Identical(instruction1, eq_operand_shapes, eq_computations));
  EXPECT_TRUE(
      instruction2.Identical(instruction2, eq_operand_shapes, eq_computations));

  bool is_equal =
      instruction1.Identical(instruction2, eq_operand_shapes, eq_computations);
  // Verify Identical is symmetric.
  EXPECT_EQ(is_equal, instruction2.Identical(instruction1, eq_operand_shapes,
                                             eq_computations));
  return is_equal;
}

TEST_F(HloInstructionTest, IdenticalInstructions) {
  // Test HloInstruction::Identical with some subset of instructions types.

  // Create a set of random constant operands to use below. Make them matrices
  // so dimensions are interesting.
  auto operand1 = HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}));
  auto operand2 = HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{10.0, 20.0}, {30.0, 40.0}}));
  auto vector_operand = HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({42.0, 123.0}));
  Shape shape = operand1->shape();

  // Convenient short names for the operands.
  HloInstruction* op1 = operand1.get();
  HloInstruction* op2 = operand2.get();

  // Operations which only depend on their operands and opcode.
  EXPECT_TRUE(
      Identical(*HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op1),
                *HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op1)));
  EXPECT_FALSE(
      Identical(*HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op1),
                *HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op2)));
  EXPECT_FALSE(
      Identical(*HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op1),
                *HloInstruction::CreateUnary(shape, HloOpcode::kNegate, op1)));

  // Tuples.
  EXPECT_TRUE(Identical(*HloInstruction::CreateTuple({op1, op2}),
                        *HloInstruction::CreateTuple({op1, op2})));
  EXPECT_FALSE(Identical(*HloInstruction::CreateTuple({op1, op2}),
                         *HloInstruction::CreateTuple({op2, op1})));

  // Broadcasts.
  EXPECT_TRUE(Identical(*HloInstruction::CreateBroadcast(shape, op1, {0, 1}),
                        *HloInstruction::CreateBroadcast(shape, op1, {0, 1})));
  EXPECT_FALSE(Identical(*HloInstruction::CreateBroadcast(shape, op1, {0, 1}),
                         *HloInstruction::CreateBroadcast(shape, op1, {1, 0})));
  Shape bcast_shape1 = ShapeUtil::MakeShape(F32, {2, 2, 42});
  Shape bcast_shape2 = ShapeUtil::MakeShape(F32, {2, 2, 123});
  EXPECT_FALSE(
      Identical(*HloInstruction::CreateBroadcast(bcast_shape1, op1, {0, 1}),
                *HloInstruction::CreateBroadcast(bcast_shape2, op1, {0, 1})));

  // Binary operands.
  EXPECT_TRUE(Identical(
      *HloInstruction::CreateBinary(shape, HloOpcode::kAdd, op1, op2),
      *HloInstruction::CreateBinary(shape, HloOpcode::kAdd, op1, op2)));
  EXPECT_FALSE(Identical(
      *HloInstruction::CreateBinary(shape, HloOpcode::kAdd, op1, op2),
      *HloInstruction::CreateBinary(shape, HloOpcode::kDivide, op2, op1)));
  EXPECT_FALSE(Identical(
      *HloInstruction::CreateBinary(shape, HloOpcode::kAdd, op1, op2),
      *HloInstruction::CreateBinary(shape, HloOpcode::kDivide, op1, op2)));
}

TEST_F(HloInstructionTest, IdenticalCallInstructions) {
  const char* const hlo_string = R"(
HloModule Module

subcomp1 (x: f32[]) -> f32[] {
  x = f32[] parameter(0)
  ROOT n = f32[] sine(x)
}

subcomp2 (x: f32[]) -> f32[] {
  x = f32[] parameter(0)
  ROOT n = f32[] cosine(x)
}

ENTRY entry (param: f32[]) -> (f32[], f32[], f32[]) {
  p = f32[] parameter(0)
  t1 = f32[] call(p), to_apply=subcomp1
  t2 = f32[] call(p), to_apply=subcomp1
  t3 = f32[] call(p), to_apply=subcomp2
  ROOT t = (f32[], f32[], f32[]) tuple(t1, t2, t3)
 }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));

  auto* root = module->entry_computation()->root_instruction();
  auto* t1 = root->operand(0);
  auto* t2 = root->operand(1);
  auto* t3 = root->operand(2);

  EXPECT_TRUE(StructuralEqual(*t1, *t2));
  EXPECT_FALSE(StructuralEqual(*t1, *t3));
}

TEST_F(HloInstructionTest, FunctionVisitor) {
  // Verify the function visitor HloInstruction::Accept visits all instructions
  // from a root properly given the following graph:
  //
  //        param
  //       /     \
  //    negate   exp
  //        \    /
  //         add
  const Shape f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, f32, "0"));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(f32, HloOpcode::kNegate, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(f32, HloOpcode::kExp, param));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(f32, HloOpcode::kAdd, negate, exp));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  int visit_num = 0;
  absl::flat_hash_map<HloInstruction*, int> visit_order;
  EXPECT_IS_OK(add->Accept([&visit_num, &visit_order](HloInstruction* inst) {
    EXPECT_FALSE(visit_order.contains(inst));
    visit_order[inst] = visit_num;
    visit_num++;
    return Status::OK();
  }));

  EXPECT_EQ(0, visit_order.at(param));
  // negate and exp can be visited in an arbitrary order.
  EXPECT_TRUE(visit_order.at(exp) == 1 || visit_order.at(exp) == 2);
  EXPECT_TRUE(visit_order.at(negate) == 1 || visit_order.at(negate) == 2);
  EXPECT_NE(visit_order.at(exp), visit_order.at(negate));
  EXPECT_EQ(3, visit_order.at(add));
}

TEST_F(HloInstructionTest, FullyElementwise) {
  const Shape r1f32 = ShapeUtil::MakeShape(F32, {5});
  HloComputation::Builder builder(TestName());
  auto x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r1f32, "x"));
  auto y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r1f32, "y"));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kAdd, x, y));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(add->IsElementwise());
  for (int i = 0; i < add->operand_count(); ++i) {
    EXPECT_TRUE(add->IsElementwiseOnOperand(i));
  }
}

TEST_F(HloInstructionTest, MapIsElementwise) {
  auto module = CreateNewVerifiedModule();
  const Shape r2f32 = ShapeUtil::MakeShapeWithLayout(F32, {10, 10}, {1, 0});
  HloComputation::Builder builder(TestName());
  HloComputation::Builder map_builder("id");
  map_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0"));
  auto map_computation = module->AddEmbeddedComputation(map_builder.Build());
  auto x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r2f32, "x"));
  auto map = builder.AddInstruction(
      HloInstruction::CreateMap(r2f32, {x}, map_computation));
  module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(map->IsElementwise());
}

TEST_F(HloInstructionTest, PartiallyElementwise) {
  const Shape r1f32 = ShapeUtil::MakeShape(F32, {5});
  const Shape r2f32 = ShapeUtil::MakeShape(F32, {3, 5});

  // Fused expression:
  //
  // p0     p1   p2   p3
  //   \   /    /     |
  //    mul    /      |
  //      \   /       |
  //       div     broadcast
  //          \    /
  //           max
  //
  // The fusion instruction is not elementwise on p3 because the broadcast is
  // not elementwise.
  HloComputation::Builder builder("PartiallyElementwise");
  HloInstruction* p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r2f32, "p0"));
  HloInstruction* p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r2f32, "p1"));
  HloInstruction* p2 =
      builder.AddInstruction(HloInstruction::CreateParameter(2, r2f32, "p2"));
  HloInstruction* p3 =
      builder.AddInstruction(HloInstruction::CreateParameter(3, r1f32, "p3"));
  HloInstruction* mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kMultiply, p0, p1));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kDivide, mul, p2));
  // Dimension 0 of shape [5] is mapped to dimension 1 of shape [3x5].
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r2f32, p3, {1}));
  HloInstruction* max = builder.AddInstruction(
      HloInstruction::CreateBinary(r2f32, HloOpcode::kMaximum, div, broadcast));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {max, broadcast, div, mul}, HloInstruction::FusionKind::kLoop);
  EXPECT_FALSE(fusion->IsElementwise());
  for (int64 operand_idx = 0; operand_idx < fusion->operand_count();
       ++operand_idx) {
    const HloInstruction* operand = fusion->operand(operand_idx);
    if (operand == p3) {
      EXPECT_FALSE(fusion->IsElementwiseOnOperand(operand_idx));
    } else {
      EXPECT_TRUE(fusion->IsElementwiseOnOperand(operand_idx));
    }
  }
}

TEST_F(HloInstructionTest, PartiallyElementwiseWithReuse) {
  // Fused expression:
  //         y
  //        /
  // x   broadcast
  //  \   /  |
  //   min   |
  //     \   /
  //      sub
  //
  const Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  const Shape r1f32 = ShapeUtil::MakeShape(F32, {5});

  HloComputation::Builder builder("PartiallyElementwiseWithReuse");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r1f32, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32, "y"));
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r1f32, y, {}));
  HloInstruction* min = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kMinimum, x, broadcast));
  HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
      r1f32, HloOpcode::kSubtract, min, broadcast));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {sub, broadcast, min}, HloInstruction::FusionKind::kLoop);
  EXPECT_FALSE(fusion->IsElementwise());
  for (int64 operand_idx = 0; operand_idx < fusion->operand_count();
       ++operand_idx) {
    if (fusion->operand(operand_idx) == y) {
      EXPECT_FALSE(fusion->IsElementwiseOnOperand(operand_idx));
    } else {
      EXPECT_TRUE(fusion->IsElementwiseOnOperand(operand_idx));
    }
  }
}

TEST_F(HloInstructionTest, CloneOfFusionPreservesShape) {
  // Fused expression:
  //
  // x     y
  // |     |
  // |  transpose
  //  \   /
  //   dot
  //
  // Tests that shapes aren't mangled by Clone().
  const Shape s1 = ShapeUtil::MakeShape(F32, {5, 10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20, 10});
  const Shape s2t = ShapeUtil::MakeShape(F32, {10, 20});
  const Shape sout = ShapeUtil::MakeShape(F32, {5, 20});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s2t, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      sout, x, reshape, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {dot, reshape}, HloInstruction::FusionKind::kLoop);

  auto fusion2 = fusion->Clone();
  const HloInstruction* root = fusion->fused_expression_root();
  const HloInstruction* root2 = fusion2->fused_expression_root();
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), root2->shape()));
  EXPECT_TRUE(
      ShapeUtil::Equal(root->operand(0)->shape(), root2->operand(0)->shape()));
  EXPECT_TRUE(
      ShapeUtil::Equal(root->operand(1)->shape(), root2->operand(1)->shape()));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(1)->operand(0)->shape(),
                               root2->operand(1)->operand(0)->shape()));
  EXPECT_TRUE(StructuralEqual(*fusion, *fusion2));
}

TEST_F(HloInstructionTest, NoRedundantFusionOperandsAfterReplacingUse) {
  // Fused expression:
  //
  // x     y
  // |     |
  // |  transpose
  //  \   /
  //   dot
  const Shape s = ShapeUtil::MakeShape(F32, {10, 10});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      s, x, reshape, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {dot, reshape}, HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(x->ReplaceAllUsesWith(y).ok());

  EXPECT_THAT(fusion->operands(), UnorderedElementsAre(y));
  EXPECT_EQ(fusion->fused_instructions_computation()->num_parameters(), 1);
}

TEST_F(HloInstructionTest, FusionEquality) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  // Create two fusion instructions containing a single unary operation.
  auto parameter =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "x"));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, parameter));
  auto neg = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, parameter));
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* fusion = computation->CreateFusionInstruction(
      {exp}, HloInstruction::FusionKind::kLoop);
  auto* fusion2 = computation->CreateFusionInstruction(
      {neg}, HloInstruction::FusionKind::kLoop);
  EXPECT_FALSE(StructuralEqual(*fusion, *fusion2));

  auto clone = fusion->Clone();
  EXPECT_TRUE(StructuralEqual(*fusion, *clone));
}

TEST_F(HloInstructionTest, NestedFusionEquality) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  // Build a nested fusion computation.
  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto a = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 0.0}, {0.0, 1.0}})));
  auto b = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{2.0, 2.0}, {2.0, 2.0}})));
  auto b_t = builder.AddInstruction(
      HloInstruction::CreateTranspose(data_shape, b, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = builder.AddInstruction(HloInstruction::CreateDot(
      data_shape, a, b_t, dot_dnums, DefaultPrecisionConfig(2)));
  auto one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  auto add_operand = builder.AddInstruction(
      HloInstruction::CreateBroadcast(data_shape, one, {}));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      data_shape, HloOpcode::kAdd, dot, add_operand));
  auto sub = builder.AddInstruction(HloInstruction::CreateBinary(
      data_shape, HloOpcode::kSubtract, dot, add_operand));
  builder.AddInstruction(
      HloInstruction::CreateBinary(data_shape, HloOpcode::kMultiply, add, sub));
  auto computation = module->AddEntryComputation(builder.Build());

  auto nested_fusion = computation->CreateFusionInstruction(
      {dot, b_t}, HloInstruction::FusionKind::kLoop);

  auto fusion = computation->CreateFusionInstruction(
      {add, nested_fusion}, HloInstruction::FusionKind::kOutput);
  auto fusion2 = computation->CreateFusionInstruction(
      {sub, nested_fusion}, HloInstruction::FusionKind::kOutput);
  auto clone = fusion->Clone();
  EXPECT_TRUE(StructuralEqual(*fusion, *clone));
  EXPECT_FALSE(StructuralEqual(*fusion, *fusion2));
}

TEST_F(HloInstructionTest, CloneSuffixNames) {
  // Test that the suffix string added to cloned instructions is not
  // duplicated. Rather a numeric incrementing value should be appended. That
  // is, we want "foo.clone2", not "foo.clone.clone".

  // Test cloning the same instruction multiple times.
  auto foo =
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "foo");
  EXPECT_EQ(foo->Clone()->name(), "foo.clone");
  EXPECT_EQ(foo->Clone()->Clone()->name(), "foo.clone2");
  EXPECT_EQ(foo->Clone()->Clone()->Clone()->name(), "foo.clone3");

  // Test custom suffixes.
  EXPECT_EQ(foo->Clone("bar")->name(), "foo.bar");
  EXPECT_EQ(foo->Clone("bar")->Clone("bar")->name(), "foo.bar2");
  EXPECT_EQ(foo->Clone("bar")->Clone("bar")->Clone()->name(), "foo.bar2.clone");

  // Test instruction name with a dot.
  auto foo_baz = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "foo.baz");
  EXPECT_EQ(foo_baz->Clone()->name(), "foo.baz.clone");

  // Test incrementing a large number after the suffix.
  auto foo_clone234 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "foo.clone234");
  EXPECT_EQ(foo_clone234->Clone()->name(), "foo.clone235");

  // Test a non-numeric string after the cloning suffix.
  auto foo_clonexyz = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "foo.clonexyz");
  EXPECT_EQ(foo_clonexyz->Clone()->name(), "foo.clonexyz.clone");

  // Test a name with multiple appearances of the suffix.
  auto foo_clone_clone3 = HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {}), "foo.clone.clone3");
  EXPECT_EQ(foo_clone_clone3->Clone()->name(), "foo.clone.clone4");
}

TEST_F(HloInstructionTest, Stringification) {
  // Tests stringification of a simple op, fusion, while, and conditional.
  const Shape s1 = ShapeUtil::MakeShape(F32, {5, 10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20, 10});
  const Shape s2t = ShapeUtil::MakeShape(F32, {10, 20});
  const Shape sout = ShapeUtil::MakeShape(F32, {5, 20});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s2t, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      sout, x, reshape, dot_dnums, DefaultPrecisionConfig(2)));

  auto options = HloPrintOptions().set_print_metadata(false);

  EXPECT_EQ(dot->ToString(options),
            "%dot = f32[5,20]{1,0} dot(f32[5,10]{1,0} %x, f32[10,20]{1,0} "
            "%transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}");

  auto options2 = HloPrintOptions()
                      .set_print_metadata(false)
                      .set_print_operand_shape(false)
                      .set_print_percent(false)
                      .set_include_layout_in_shapes(false);

  EXPECT_EQ(dot->ToString(options2),
            "dot = f32[5,20] dot(x, transpose), "
            "lhs_contracting_dims={1}, rhs_contracting_dims={0}");

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());

  HloInstruction* loop = builder.AddInstruction(
      HloInstruction::CreateWhile(sout, computation, computation, x));
  EXPECT_EQ(loop->ToString(options),
            "%while = f32[5,20]{1,0} while(f32[5,10]{1,0} %x), "
            "condition=%TransposeDot, body=%TransposeDot");

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloInstruction* conditional =
      builder.AddInstruction(HloInstruction::CreateConditional(
          sout, pred, x, computation, x, computation));
  EXPECT_EQ(conditional->ToString(options),
            "%conditional = f32[5,20]{1,0} conditional(pred[] %constant, "
            "f32[5,10]{1,0} %x, f32[5,10]{1,0} %x), "
            "true_computation=%TransposeDot, false_computation=%TransposeDot");
}

TEST_F(HloInstructionTest, StringifyGather_0) {
  Shape input_tensor_shape = ShapeUtil::MakeShape(F32, {50, 49, 48, 47, 46});
  Shape start_indices_tensor_shape =
      ShapeUtil::MakeShape(S64, {10, 9, 8, 7, 5});
  Shape gather_result_shape =
      ShapeUtil::MakeShape(F32, {10, 9, 8, 7, 30, 29, 28, 27, 26});

  HloComputation::Builder builder("Gather");
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_tensor_shape, "input_tensor"));
  HloInstruction* start_indices =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, start_indices_tensor_shape, "start_indices"));

  HloInstruction* gather_instruction = builder.AddInstruction(
      HloInstruction::CreateGather(gather_result_shape, input, start_indices,
                                   HloGatherInstruction::MakeGatherDimNumbers(
                                       /*offset_dims=*/{4, 5, 6, 7, 8},
                                       /*collapsed_slice_dims=*/{},
                                       /*start_index_map=*/{0, 1, 2, 3, 4},
                                       /*index_vector_dim=*/4),
                                   /*slice_sizes=*/{30, 29, 28, 27, 26}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(gather_instruction->ToString(),
            "%gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} "
            "gather(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, "
            "s64[10,9,8,7,5]{4,3,2,1,0} %start_indices), "
            "offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, "
            "start_index_map={0,1,2,3,4}, "
            "index_vector_dim=4, slice_sizes={30,29,28,27,26}");
}

TEST_F(HloInstructionTest, StringifyGather_1) {
  Shape input_tensor_shape = ShapeUtil::MakeShape(F32, {50, 49, 48, 47, 46});
  Shape start_indices_tensor_shape =
      ShapeUtil::MakeShape(S64, {10, 9, 5, 7, 6});
  Shape gather_result_shape =
      ShapeUtil::MakeShape(F32, {10, 9, 7, 6, 30, 29, 28, 27, 26});

  HloComputation::Builder builder("Gather");
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_tensor_shape, "input_tensor"));
  HloInstruction* start_indices =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, start_indices_tensor_shape, "start_indices"));

  HloInstruction* gather_instruction = builder.AddInstruction(
      HloInstruction::CreateGather(gather_result_shape, input, start_indices,
                                   HloGatherInstruction::MakeGatherDimNumbers(
                                       /*offset_dims=*/{4, 5, 6, 7, 8},
                                       /*collapsed_slice_dims=*/{},
                                       /*start_index_map=*/{0, 1, 2, 3, 4},
                                       /*index_vector_dim=*/2),
                                   /*slice_sizes=*/{30, 29, 28, 27, 26}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(gather_instruction->ToString(),
            "%gather = f32[10,9,7,6,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} "
            "gather(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, "
            "s64[10,9,5,7,6]{4,3,2,1,0} %start_indices), "
            "offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, "
            "start_index_map={0,1,2,3,4}, "
            "index_vector_dim=2, slice_sizes={30,29,28,27,26}");
}

TEST_F(HloInstructionTest, StringifyScatter) {
  Shape input_tensor_shape = ShapeUtil::MakeShape(F32, {50, 49, 48, 47, 46});
  Shape scatter_indices_tensor_shape =
      ShapeUtil::MakeShape(S64, {10, 9, 5, 7, 6});
  Shape scatter_updates_shape =
      ShapeUtil::MakeShape(F32, {10, 9, 7, 6, 30, 29, 28, 27, 26});

  HloComputation::Builder builder("Scatter");
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, input_tensor_shape, "input_tensor"));
  HloInstruction* scatter_indices =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, scatter_indices_tensor_shape, "scatter_indices"));
  HloInstruction* scatter_updates =
      builder.AddInstruction(HloInstruction::CreateParameter(
          2, scatter_updates_shape, "scatter_updates"));

  HloComputation::Builder update_builder("Scatter.update");
  update_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p1"));
  update_builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {}), "p2"));

  auto module = CreateNewVerifiedModule();
  auto* update_computation =
      module->AddEmbeddedComputation(update_builder.Build());

  HloInstruction* scatter_instruction =
      builder.AddInstruction(HloInstruction::CreateScatter(
          input_tensor_shape, input, scatter_indices, scatter_updates,
          update_computation,
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/{4, 5, 6, 7, 8},
              /*inserted_window_dims=*/{},
              /*scatter_dims_to_operand_dims=*/{0, 1, 2, 3, 4},
              /*index_vector_dim=*/2)));
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(
      scatter_instruction->ToString(),
      "%scatter = f32[50,49,48,47,46]{4,3,2,1,0} "
      "scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, "
      "s64[10,9,5,7,6]{4,3,2,1,0} %scatter_indices, "
      "f32[10,9,7,6,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %scatter_updates), "
      "update_window_dims={4,5,6,7,8}, inserted_window_dims={}, "
      "scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=2, "
      "to_apply=%Scatter.update");
}

TEST_F(HloInstructionTest, CanonnicalStringificationFusion) {
  // Tests stringification of a simple op, fusion, while, and conditional.
  const Shape s1 = ShapeUtil::MakeShape(F32, {5, 10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20, 10});
  const Shape s2t = ShapeUtil::MakeShape(F32, {10, 20});
  const Shape sout = ShapeUtil::MakeShape(F32, {5, 20});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s2t, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      sout, x, reshape, dot_dnums, DefaultPrecisionConfig(2)));

  auto options = HloPrintOptions().Canonical();

  EXPECT_EQ(dot->ToString(options),
            "f32[5,20]{1,0} dot(f32[5,10]{1,0}, f32[10,20]{1,0}), "
            "lhs_contracting_dims={1}, rhs_contracting_dims={0}");

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {dot, reshape}, HloInstruction::FusionKind::kLoop);

  const string expected_fusion =
      R"(f32[5,20]{1,0} fusion(f32[5,10]{1,0}, f32[20,10]{1,0}), kind=kLoop, calls=
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
  ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  EXPECT_EQ(fusion->ToString(options), expected_fusion);
}

TEST_F(HloInstructionTest, CanonnicalStringificationWhile) {
  // Tests stringification of a simple op, fusion, while, and conditional.
  const Shape s1 = ShapeUtil::MakeShape(F32, {5, 10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20, 10});
  const Shape s2t = ShapeUtil::MakeShape(F32, {10, 20});
  const Shape sout = ShapeUtil::MakeShape(F32, {5, 20});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s2t, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      sout, x, reshape, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({dot, reshape},
                                       HloInstruction::FusionKind::kLoop);

  HloInstruction* loop = builder.AddInstruction(
      HloInstruction::CreateWhile(sout, computation, computation, x));

  auto options = HloPrintOptions().Canonical();
  const string expected_loop =
      R"(f32[5,20]{1,0} while(f32[5,10]{1,0}), condition=
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  ROOT tmp_2 = f32[5,20]{1,0} fusion(f32[5,10]{1,0} tmp_0, f32[20,10]{1,0} tmp_1), kind=kLoop, calls=
  {
    tmp_0 = f32[5,10]{1,0} parameter(0)
    tmp_1 = f32[20,10]{1,0} parameter(1)
    tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
    ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
}, body=
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  ROOT tmp_2 = f32[5,20]{1,0} fusion(f32[5,10]{1,0} tmp_0, f32[20,10]{1,0} tmp_1), kind=kLoop, calls=
  {
    tmp_0 = f32[5,10]{1,0} parameter(0)
    tmp_1 = f32[20,10]{1,0} parameter(1)
    tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
    ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
})";
  EXPECT_EQ(loop->ToString(options), expected_loop);
}

TEST_F(HloInstructionTest, CanonnicalStringificationConditional) {
  // Tests stringification of a simple op, fusion, while, and conditional.
  const Shape s1 = ShapeUtil::MakeShape(F32, {5, 10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20, 10});
  const Shape s2t = ShapeUtil::MakeShape(F32, {10, 20});
  const Shape sout = ShapeUtil::MakeShape(F32, {5, 20});

  HloComputation::Builder builder("TransposeDot");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, s2, "y"));
  HloInstruction* reshape =
      builder.AddInstruction(HloInstruction::CreateTranspose(s2t, y, {1, 0}));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      sout, x, reshape, dot_dnums, DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  computation->CreateFusionInstruction({dot, reshape},
                                       HloInstruction::FusionKind::kLoop);

  builder.AddInstruction(
      HloInstruction::CreateWhile(sout, computation, computation, x));

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloInstruction* conditional =
      builder.AddInstruction(HloInstruction::CreateConditional(
          sout, pred, x, computation, x, computation));
  auto options = HloPrintOptions().Canonical();
  const string expected_conditional =
      R"(f32[5,20]{1,0} conditional(pred[], f32[5,10]{1,0}, f32[5,10]{1,0}), true_computation=
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  ROOT tmp_2 = f32[5,20]{1,0} fusion(f32[5,10]{1,0} tmp_0, f32[20,10]{1,0} tmp_1), kind=kLoop, calls=
  {
    tmp_0 = f32[5,10]{1,0} parameter(0)
    tmp_1 = f32[20,10]{1,0} parameter(1)
    tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
    ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
}, false_computation=
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  ROOT tmp_2 = f32[5,20]{1,0} fusion(f32[5,10]{1,0} tmp_0, f32[20,10]{1,0} tmp_1), kind=kLoop, calls=
  {
    tmp_0 = f32[5,10]{1,0} parameter(0)
    tmp_1 = f32[20,10]{1,0} parameter(1)
    tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
    ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
})";
  EXPECT_EQ(conditional->ToString(options), expected_conditional);
}

TEST_F(HloInstructionTest, CheckDeepClone) {
  const char* const hlo_string = R"(
HloModule Module

addy (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT zadd = s32[] add(lhs, rhs)
}

calla (x: s32[]) -> s32[] {
  x = s32[] parameter(0)
  reduce = s32[] reduce-window(x, x), to_apply=addy
  ROOT xadd = s32[] add(x, reduce)
}

body (bparam: s32[]) -> s32[] {
  constant = s32[] constant(1)
  bparam = s32[] parameter(0)
  v = s32[] call(bparam), to_apply=calla
  ROOT add = s32[] add(constant, bparam)
}

condition (cparam: s32[]) -> pred[] {
  xconstant = s32[] constant(5)
  cparam = s32[] parameter(0)
  ROOT greater-than = pred[] compare(xconstant, cparam), direction=GT
}

ENTRY entry (param: s32[]) -> s32[] {
  eparam = s32[] parameter(0)
  ROOT while = s32[] while(eparam), condition=condition, body=body
 }
)";
  // Check that deep clones really deep clones every instruction and
  // computations, without leaving dangling pointers to the old module.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(hlo_string));
  std::unique_ptr<HloModule> clone = module->Clone();
  for (HloComputation* computation : clone->computations()) {
    EXPECT_EQ(computation->parent(), clone.get());
    for (HloInstruction* instruction : computation->instructions()) {
      EXPECT_EQ(instruction->parent()->parent(), clone.get());
    }
  }
}

TEST_F(HloInstructionTest, IdenticalAccountsForBackendConfig) {
  const Shape shape = ShapeUtil::MakeShape(F32, {42});
  HloComputation::Builder builder("test");
  HloInstruction* p =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p"));

  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p, p));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p, p));

  EXPECT_TRUE(add1->Identical(*add2));
  add1->set_raw_backend_config_string("abc");
  EXPECT_FALSE(add1->Identical(*add2));
}

TEST_F(HloInstructionTest, IdenticalAccountsForCustomCallWindow) {
  auto instr1 = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                 /*operands=*/{},
                                                 /*custom_call_target=*/"foo");
  auto instr2 = instr1->Clone();
  EXPECT_TRUE(instr1->Identical(*instr2));

  Window w = window_util::MakeWindow({1, 2, 3});
  instr1->set_window(w);
  EXPECT_FALSE(instr1->Identical(*instr2));
}

TEST_F(HloInstructionTest, IdenticalAccountsForCustomCallDnums) {
  auto instr1 = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                 /*operands=*/{},
                                                 /*custom_call_target=*/"foo");
  auto instr2 = instr1->Clone();
  EXPECT_TRUE(instr1->Identical(*instr2));

  ConvolutionDimensionNumbers dnums;
  dnums.set_output_batch_dimension(42);
  instr1->set_convolution_dimension_numbers(dnums);
  EXPECT_FALSE(instr1->Identical(*instr2));
}

TEST_F(HloInstructionTest, CloneWindowOnCustomCall) {
  auto instr = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                /*operands=*/{},
                                                /*custom_call_target=*/"foo");
  Window w = window_util::MakeWindow({1, 2, 3});
  instr->set_window(w);
  auto clone = instr->Clone();
  EXPECT_TRUE(protobuf_util::ProtobufEquals(clone->window(), w))
      << clone->window().DebugString();
}

TEST_F(HloInstructionTest, CloneDnumsOnCustomCall) {
  auto instr = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                /*operands=*/{},
                                                /*custom_call_target=*/"foo");
  ConvolutionDimensionNumbers dnums;
  dnums.set_output_batch_dimension(42);
  instr->set_convolution_dimension_numbers(dnums);
  auto clone = instr->Clone();
  EXPECT_TRUE(protobuf_util::ProtobufEquals(
      clone->convolution_dimension_numbers(), dnums))
      << clone->convolution_dimension_numbers().DebugString();
}

TEST_F(HloInstructionTest, PreserveOperandPrecisionOnCloneConv) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  ENTRY test {
    arg0 = f32[1,2,1] parameter(0)
    arg1 = f32[1,1,1] parameter(1)
    ROOT conv = f32[1,2,1] convolution(arg0, arg1), window={size=1},
      dim_labels=b0f_0io->b0f, operand_precision={high,default}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(kHloString));
  auto* conv = module->entry_computation()->root_instruction();

  auto clone = conv->Clone();
  EXPECT_THAT(
      clone->precision_config().operand_precision(),
      ::testing::ElementsAre(PrecisionConfig::HIGH, PrecisionConfig::DEFAULT));
}

}  // namespace
}  // namespace xla
