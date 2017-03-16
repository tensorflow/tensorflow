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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace {

class HloInstructionTest : public HloTestBase {
 protected:
  HloInstructionTest() {}

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
};

// Simple visitor that collects the number of users and operands for certain HLO
// nodes. It also verifies some of the DFS visiting invariants (operands visited
// before their users, nodes not visited twice, etc.)
class OpAndUserCollectingVisitor : public DfsHloVisitorWithDefault {
 public:
  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Unimplemented("not implemented %s",
                         HloOpcodeString(hlo_instruction->opcode()).c_str());
  }

  Status HandleParameter(HloInstruction* parameter) override {
    EXPECT_EQ(0, count_.count(parameter));
    count_[parameter] = GetCountsForNode(parameter);
    return Status::OK();
  }

  Status HandleConstant(HloInstruction* constant,
                        const Literal& literal) override {
    EXPECT_EQ(0, count_.count(constant));
    count_[constant] = GetCountsForNode(constant);
    return Status::OK();
  }

  Status HandleAdd(HloInstruction* add, HloInstruction* lhs,
                   HloInstruction* rhs) override {
    EXPECT_EQ(0, count_.count(add));
    EXPECT_GT(count_.count(lhs), 0);
    EXPECT_GT(count_.count(rhs), 0);
    count_[add] = GetCountsForNode(add);
    return Status::OK();
  }

  Status HandleNegate(HloInstruction* negate,
                      HloInstruction* operand) override {
    EXPECT_EQ(0, count_.count(negate));
    EXPECT_GT(count_.count(operand), 0);
    count_[negate] = GetCountsForNode(negate);
    return Status::OK();
  }

  Status HandleMap(
      HloInstruction* map,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* /*function*/,
      tensorflow::gtl::ArraySlice<HloInstruction*> /*static_operands*/)
      override {
    EXPECT_EQ(0, count_.count(map));
    for (HloInstruction* arg : operands) {
      EXPECT_GT(count_.count(arg), 0);
    }
    count_[map] = GetCountsForNode(map);
    return Status::OK();
  }

  Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                      HloInstruction* init_value,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      HloComputation* function) override {
    EXPECT_EQ(0, count_.count(reduce));
    EXPECT_GT(count_.count(arg), 0);
    EXPECT_GT(count_.count(init_value), 0);
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
  std::unordered_map<const HloInstruction*, NumOpsAndUsers> count_;
};

TEST_F(HloInstructionTest, BasicProperties) {
  auto parameter = HloInstruction::CreateParameter(1, r0f32_, "foo");

  EXPECT_EQ(HloOpcode::kParameter, parameter->opcode());
  EXPECT_TRUE(ShapeUtil::IsScalarF32(parameter->shape()));
  EXPECT_EQ(0, parameter->operand_count());
}

TEST_F(HloInstructionTest, UserWithTwoOperands) {
  // [Param foo]----->  |-----|
  //                    | Add |
  // [Param bar]----->  |-----|
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto bar = HloInstruction::CreateParameter(1, r0f32_, "bar");
  auto add = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo.get(),
                                          bar.get());

  ExpectEqOrdered(add->operands(), {foo.get(), bar.get()});
  ExpectEqUnordered(foo->users(), {add.get()});
  ExpectEqUnordered(bar->users(), {add.get()});

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(add->Accept(&visitor));

  EXPECT_EQ(2, visitor.NumOperands(add.get()));
  EXPECT_EQ(0, visitor.NumUsers(add.get()));
  EXPECT_EQ(1, visitor.NumUsers(foo.get()));
  EXPECT_EQ(1, visitor.NumUsers(bar.get()));
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
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto bar = HloInstruction::CreateParameter(1, r0f32_, "bar");
  auto exp1 = HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo.get());
  auto exp2 = HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo.get());
  auto add = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo.get(),
                                          bar.get());

  EXPECT_EQ(3, foo->user_count());
  EXPECT_EQ(1, bar->user_count());
  EXPECT_EQ(0, exp1->user_count());
  EXPECT_EQ(0, exp2->user_count());
  EXPECT_EQ(0, add->user_count());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(add->Accept(&visitor));

  EXPECT_EQ(2, visitor.NumOperands(add.get()));
  EXPECT_EQ(3, visitor.NumUsers(foo.get()));
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
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto add = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, foo.get(),
                                          foo.get());
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
  auto param0 = HloInstruction::CreateParameter(0, r0f32_, "param0");
  auto param1 = HloInstruction::CreateParameter(1, r0f32_, "param1");
  auto c0 = HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f));
  auto addleft = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                              param0.get(), c0.get());
  auto addright = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                               c0.get(), param1.get());
  auto addtotal = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                               addleft.get(), addright.get());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(addtotal->Accept(&visitor));

  EXPECT_EQ(2, visitor.NumUsers(c0.get()));
  EXPECT_EQ(2, visitor.NumOperands(addleft.get()));
  EXPECT_EQ(2, visitor.NumOperands(addright.get()));
  EXPECT_EQ(2, visitor.NumOperands(addtotal.get()));
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
  auto param0 = HloInstruction::CreateParameter(0, r0f32_, "param0");
  auto param1 = HloInstruction::CreateParameter(1, r0f32_, "param1");
  auto c0 = HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f));
  auto neg1 = HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, c0.get());
  auto addleft = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                              param0.get(), neg1.get());
  auto addright = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                               neg1.get(), param1.get());
  auto addtotal = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                               addleft.get(), addright.get());
  auto neg2 =
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, addtotal.get());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(neg2->Accept(&visitor));

  EXPECT_EQ(1, visitor.NumUsers(c0.get()));
  EXPECT_EQ(2, visitor.NumUsers(neg1.get()));
  EXPECT_EQ(2, visitor.NumOperands(addleft.get()));
  EXPECT_EQ(2, visitor.NumOperands(addright.get()));
  EXPECT_EQ(2, visitor.NumOperands(addtotal.get()));
  EXPECT_EQ(1, visitor.NumOperands(neg2.get()));
  EXPECT_EQ(0, visitor.NumUsers(neg2.get()));
}

TEST_F(HloInstructionTest, TrivialMap) {
  // This tests creating a trivial x+1 map as the only operation.
  //
  // param0[100x10] ---> (map x+1)
  //
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  Shape f32a100x10 = ShapeUtil::MakeShape(F32, {100, 10});

  // Builds an x+1.0 computation to use in a Map.
  auto builder = HloComputation::Builder("f32+1");
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32, "x"));
  auto value = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, param, value));
  auto add_f32 = builder.Build();

  // Builds a parameter and feeds it to the map.
  auto param0 = HloInstruction::CreateParameter(1, f32a100x10, "");
  auto map =
      HloInstruction::CreateMap(f32a100x10, {param0.get()}, add_f32.get());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(map->Accept(&visitor));

  // Check counts.  We aren't walking the mapper computation yet.
  EXPECT_EQ(1, visitor.NumUsers(param0.get()));
  EXPECT_EQ(0, visitor.NumUsers(map.get()));
  EXPECT_EQ(1, visitor.NumOperands(map.get()));

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
  auto builder = HloComputation::Builder("f32+f32");
  auto paramx =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32, "x"));
  auto paramy =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32, "y"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, paramx, paramy));
  auto add_f32 = builder.Build();

  // Builds a parameter and an initial value and feeds them to the reduce.
  auto param0 = HloInstruction::CreateParameter(0, f32a100x10, "");
  auto const0 =
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f));
  auto c0 = HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f));
  auto reduce =
      HloInstruction::CreateReduce(f32v100, param0.get(), const0.get(),
                                   /*dimensions_to_reduce=*/{1}, add_f32.get());

  OpAndUserCollectingVisitor visitor;
  ASSERT_IS_OK(reduce->Accept(&visitor));

  // Check counts.  We aren't walking the reducer computation.
  EXPECT_EQ(1, visitor.NumUsers(param0.get()));
  EXPECT_EQ(1, visitor.NumUsers(const0.get()));
  EXPECT_EQ(0, visitor.NumUsers(reduce.get()));
  EXPECT_EQ(2, visitor.NumOperands(reduce.get()));
}

TEST_F(HloInstructionTest, ReplaceUseInBinaryOps) {
  // Construct a graph of a few binary ops using two different
  // parameters. Replace one of the parameters with the other parameter in one
  // of the instructions.
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto bar = HloInstruction::CreateParameter(1, r0f32_, "bar");
  auto add_foobar = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                 foo.get(), bar.get());
  auto add_foofoo = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                 foo.get(), foo.get());
  auto sum = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                          add_foobar.get(), add_foofoo.get());

  EXPECT_EQ(2, foo->user_count());
  EXPECT_EQ(1, bar->user_count());

  // Replace the use of foo in add_foofoo with bar.
  ASSERT_IS_OK(foo->ReplaceUseWith(add_foofoo.get(), bar.get()));

  EXPECT_EQ(1, foo->user_count());
  EXPECT_EQ(2, bar->user_count());

  ExpectEqUnordered(foo->users(), {add_foobar.get()});
  ExpectEqOrdered(add_foobar->operands(), {foo.get(), bar.get()});

  ExpectEqUnordered(bar->users(), {add_foobar.get(), add_foofoo.get()});
  ExpectEqOrdered(add_foobar->operands(), {foo.get(), bar.get()});
  ExpectEqOrdered(add_foofoo->operands(), {bar.get(), bar.get()});
}

TEST_F(HloInstructionTest, ReplaceUseInVariadicOp) {
  // Construct a tuple containing several parameters. Replace one parameter with
  // another in the tuple.
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto bar = HloInstruction::CreateParameter(1, r0f32_, "bar");
  auto baz = HloInstruction::CreateParameter(2, r0f32_, "baz");

  auto tuple =
      HloInstruction::CreateTuple({foo.get(), bar.get(), baz.get(), foo.get()});
  auto add_foobar = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                 foo.get(), bar.get());

  EXPECT_EQ(2, foo->user_count());
  ExpectEqUnordered(foo->users(), {tuple.get(), add_foobar.get()});

  // Replace the use of foo in tuple with bar.
  ASSERT_IS_OK(foo->ReplaceUseWith(tuple.get(), bar.get()));

  ExpectEqUnordered(foo->users(), {add_foobar.get()});

  // Both uses of foo in tuple should have been replaced with bar.
  ExpectEqOrdered(tuple->operands(),
                  {bar.get(), bar.get(), baz.get(), bar.get()});
}

TEST_F(HloInstructionTest, ReplaceUseInUnaryOp) {
  // Construct a couple unary instructions which use a parameter. Replace the
  // use of a parameter in one of the unary ops with the other parameter.
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto bar = HloInstruction::CreateParameter(1, r0f32_, "bar");

  auto exp = HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo.get());
  auto log = HloInstruction::CreateUnary(r0f32_, HloOpcode::kLog, foo.get());

  EXPECT_EQ(2, foo->user_count());
  ExpectEqUnordered(foo->users(), {exp.get(), log.get()});
  EXPECT_EQ(0, bar->user_count());

  // Replace the use of foo in exp with bar.
  ASSERT_IS_OK(foo->ReplaceUseWith(exp.get(), bar.get()));

  // The use of foo in log should not have been affected.
  EXPECT_EQ(1, foo->user_count());
  ExpectEqUnordered(foo->users(), {log.get()});
  ExpectEqOrdered(log->operands(), {foo.get()});

  // Bar should now be used in exp.
  EXPECT_EQ(1, bar->user_count());
  EXPECT_EQ(*bar->users().begin(), exp.get());
  EXPECT_EQ(1, exp->operands().size());
  EXPECT_EQ(*exp->operands().begin(), bar.get());
}

TEST_F(HloInstructionTest, ReplaceAllUsesWithInBinaryOps) {
  // Construct a simple graph of a few binary ops using two different
  // parameters. Replace all uses of one of the parameters with the other
  // parameter.
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto bar = HloInstruction::CreateParameter(1, r0f32_, "bar");
  auto add_foobar = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                 foo.get(), bar.get());
  auto add_foofoo = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                 foo.get(), foo.get());
  auto sum = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                          add_foobar.get(), add_foofoo.get());

  EXPECT_EQ(2, foo->user_count());
  EXPECT_EQ(1, bar->user_count());

  // Replace all uses of foo with bar.
  ASSERT_IS_OK(foo->ReplaceAllUsesWith(bar.get()));

  EXPECT_EQ(0, foo->user_count());
  EXPECT_EQ(2, bar->user_count());

  ExpectEqUnordered(bar->users(), {add_foobar.get(), add_foofoo.get()});
}

TEST_F(HloInstructionTest, ReplaceAllUsesInMultipleOps) {
  // Construct a graph containing several ops (a unary, binary, and variadic)
  // which use two parameters. Replace all uses of one of the parameters with
  // the other parameter.
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto bar = HloInstruction::CreateParameter(1, r0f32_, "bar");

  auto add_foobar = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                                 foo.get(), bar.get());
  auto exp = HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo.get());
  auto tuple = HloInstruction::CreateTuple({foo.get(), bar.get()});

  EXPECT_EQ(3, foo->user_count());
  EXPECT_EQ(2, bar->user_count());

  // Replace all uses of foo with bar.
  ASSERT_IS_OK(foo->ReplaceAllUsesWith(bar.get()));

  EXPECT_EQ(0, foo->user_count());
  EXPECT_EQ(3, bar->user_count());

  ExpectEqUnordered(bar->users(), {add_foobar.get(), exp.get(), tuple.get()});
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
  auto foo = HloInstruction::CreateParameter(0, r0f32_, "foo");
  auto exp = HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, foo.get());
  auto log = HloInstruction::CreateUnary(r0f32_, HloOpcode::kLog, foo.get());
  auto add = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, exp.get(),
                                          log.get());

  NodeCollectorAndPostProcessor visitor;
  ASSERT_IS_OK(add->Accept(&visitor));
  // Verifies all the nodes are visited and post-processed in the same order.
  EXPECT_EQ(visitor.visited_nodes(), visitor.post_processed_nodes());
  // Verifies each node is visited exactly once.
  EXPECT_TRUE(Distinct(visitor.visited_nodes()));
}

TEST_F(HloInstructionTest, SingletonFusionOp) {
  // Create a fusion instruction containing a single unary operation.
  auto constant =
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f));
  auto exp =
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, constant.get());

  auto fusion = HloInstruction::CreateFusion(
      r0f32_, HloInstruction::FusionKind::kLoop, exp.get());

  ExpectEqOrdered(fusion->operands(), {constant.get()});
  ExpectEqUnordered(constant->users(), {fusion.get(), exp.get()});
}

TEST_F(HloInstructionTest, BinaryFusionOp) {
  // Create a fusion instruction containing a single binary operation.
  auto constant1 =
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f));
  auto constant2 =
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.1f));
  auto add = HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd,
                                          constant1.get(), constant2.get());

  auto fusion = HloInstruction::CreateFusion(
      r0f32_, HloInstruction::FusionKind::kLoop, add.get());

  ExpectEqOrdered(fusion->operands(), {constant1.get(), constant2.get()});
  ExpectEqUnordered(constant1->users(), {fusion.get(), add.get()});
  ExpectEqUnordered(constant2->users(), {fusion.get(), add.get()});
}

TEST_F(HloInstructionTest, ChainFusionOp) {
  // Create a chain of fused unary ops.
  auto constant =
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f));
  auto exp1 =
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, constant.get());
  auto exp2 = HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, exp1.get());
  auto exp3 = HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, exp2.get());

  auto fusion = HloInstruction::CreateFusion(
      r0f32_, HloInstruction::FusionKind::kLoop, exp3.get());
  fusion->FuseInstruction(exp2.get());
  fusion->FuseInstruction(exp1.get());

  ExpectEqOrdered(fusion->operands(), {constant.get()});
  ExpectEqUnordered(constant->users(), {fusion.get(), exp1.get()});
}

TEST_F(HloInstructionTest, FusionOpWithCalledComputations) {
  // Create a fusion instruction containing a single unary operation.
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});

  auto make_map_computation = [&]() {
    auto builder = HloComputation::Builder("FusionMap");
    builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "param"));
    return builder.Build();
  };

  std::unique_ptr<HloComputation> computation_x = make_map_computation();
  std::unique_ptr<HloComputation> computation_y = make_map_computation();

  auto constant =
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f));
  auto map_1_x =
      HloInstruction::CreateMap(scalar_shape, {constant.get()},
                                computation_x.get(), /*static_operands=*/{});
  auto map_2_x =
      HloInstruction::CreateMap(scalar_shape, {map_1_x.get()},
                                computation_x.get(), /*static_operands=*/{});
  auto map_3_y =
      HloInstruction::CreateMap(scalar_shape, {map_2_x.get()},
                                computation_y.get(), /*static_operands=*/{});

  auto fusion = HloInstruction::CreateFusion(
      scalar_shape, HloInstruction::FusionKind::kLoop, map_3_y.get());

  ASSERT_EQ(fusion->called_computations().size(), 1);
  EXPECT_EQ(fusion->called_computations()[0], computation_y.get());

  fusion->FuseInstruction(map_2_x.get());
  ASSERT_EQ(fusion->called_computations().size(), 2);
  EXPECT_EQ(fusion->called_computations()[1], computation_x.get());

  fusion->FuseInstruction(map_1_x.get());
  ASSERT_EQ(fusion->called_computations().size(), 2);
}

TEST_F(HloInstructionTest, ComplexFusionOp) {
  // Fuse all instructions in complicated expression:
  //
  //   add = Add(C1, C2)
  //   clamp = Clamp(C2, add, add)
  //   exp = Exp(add)
  //   mul = Mul(exp, C3)
  //   sub = Sub(mul, clamp)
  //   tuple = Tuple({sub, sub, mul, C1})
  //
  // Notable complexities are repeated operands in a same instruction, different
  // shapes, use of value in different expressions.
  auto c1 = HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f));
  auto c2 = HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.1f));
  auto c3 = HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(9.0f));

  auto add =
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, c1.get(), c2.get());
  auto clamp = HloInstruction::CreateTernary(r0f32_, HloOpcode::kClamp,
                                             c2.get(), add.get(), add.get());
  auto exp = HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, add.get());
  auto mul = HloInstruction::CreateBinary(r0f32_, HloOpcode::kMultiply,
                                          exp.get(), c3.get());
  auto sub = HloInstruction::CreateBinary(r0f32_, HloOpcode::kSubtract,
                                          mul.get(), clamp.get());
  auto tuple =
      HloInstruction::CreateTuple({sub.get(), sub.get(), mul.get(), c1.get()});

  auto fusion = HloInstruction::CreateFusion(
      r0f32_, HloInstruction::FusionKind::kLoop, tuple.get());
  fusion->FuseInstruction(sub.get());
  fusion->FuseInstruction(mul.get());
  fusion->FuseInstruction(exp.get());
  fusion->FuseInstruction(clamp.get());
  fusion->FuseInstruction(add.get());

  // Operands in the fusion instruction's operands() vector should be in the
  // order in which their users were added fused.
  ExpectEqOrdered(fusion->operands(), {c1.get(), c3.get(), c2.get()});
  ExpectEqUnordered(c1->users(), {add.get(), tuple.get(), fusion.get()});
}

// Convenience function for comparing two HloInstructions inside of
// std::unique_ptrs.
static bool Identical(std::unique_ptr<HloInstruction> instruction1,
                      std::unique_ptr<HloInstruction> instruction2) {
  // Verify Identical is reflexive for both instructions.
  EXPECT_TRUE(instruction1->Identical(*instruction1));
  EXPECT_TRUE(instruction2->Identical(*instruction2));

  bool is_equal = instruction1->Identical(*instruction2);
  // Verify Identical is symmetric.
  EXPECT_EQ(is_equal, instruction2->Identical(*instruction1));
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
      Identical(HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op1),
                HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op1)));
  EXPECT_FALSE(
      Identical(HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op1),
                HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op2)));
  EXPECT_FALSE(
      Identical(HloInstruction::CreateUnary(shape, HloOpcode::kCopy, op1),
                HloInstruction::CreateUnary(shape, HloOpcode::kNegate, op1)));

  // Tuples.
  EXPECT_TRUE(Identical(HloInstruction::CreateTuple({op1, op2}),
                        HloInstruction::CreateTuple({op1, op2})));
  EXPECT_FALSE(Identical(HloInstruction::CreateTuple({op1, op2}),
                         HloInstruction::CreateTuple({op2, op1})));

  // Broadcasts.
  EXPECT_TRUE(Identical(HloInstruction::CreateBroadcast(shape, op1, {0, 1}),
                        HloInstruction::CreateBroadcast(shape, op1, {0, 1})));
  EXPECT_FALSE(Identical(HloInstruction::CreateBroadcast(shape, op1, {0, 1}),
                         HloInstruction::CreateBroadcast(shape, op1, {1, 0})));
  Shape bcast_shape1 = ShapeUtil::MakeShape(F32, {2, 2, 42});
  Shape bcast_shape2 = ShapeUtil::MakeShape(F32, {2, 2, 123});
  EXPECT_FALSE(
      Identical(HloInstruction::CreateBroadcast(bcast_shape1, op1, {0, 1}),
                HloInstruction::CreateBroadcast(bcast_shape2, op1, {0, 1})));

  // Binary operands.
  EXPECT_TRUE(Identical(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, op1, op2),
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, op1, op2)));
  EXPECT_FALSE(Identical(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, op1, op2),
      HloInstruction::CreateBinary(shape, HloOpcode::kDivide, op2, op1)));
  EXPECT_FALSE(Identical(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, op1, op2),
      HloInstruction::CreateBinary(shape, HloOpcode::kDivide, op1, op2)));
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
  auto param = HloInstruction::CreateParameter(0, f32, "0");
  auto negate =
      HloInstruction::CreateUnary(f32, HloOpcode::kNegate, param.get());
  auto exp = HloInstruction::CreateUnary(f32, HloOpcode::kExp, param.get());
  auto add = HloInstruction::CreateBinary(f32, HloOpcode::kAdd, negate.get(),
                                          exp.get());

  int visit_num = 0;
  std::unordered_map<HloInstruction*, int> visit_order;
  EXPECT_IS_OK(add->Accept([&visit_num, &visit_order](HloInstruction* inst) {
    EXPECT_EQ(0, visit_order.count(inst));
    visit_order[inst] = visit_num;
    visit_num++;
    return Status::OK();
  }));

  EXPECT_EQ(0, visit_order.at(param.get()));
  // negate and exp can be visited in an arbitrary order.
  EXPECT_TRUE(visit_order.at(exp.get()) == 1 || visit_order.at(exp.get()) == 2);
  EXPECT_TRUE(visit_order.at(negate.get()) == 1 ||
              visit_order.at(negate.get()) == 2);
  EXPECT_NE(visit_order.at(exp.get()), visit_order.at(negate.get()));
  EXPECT_EQ(3, visit_order.at(add.get()));
}

TEST_F(HloInstructionTest, FullyElementwise) {
  const Shape r1f32 = ShapeUtil::MakeShape(F32, {5});
  auto x = HloInstruction::CreateParameter(0, r1f32, "x");
  auto y = HloInstruction::CreateParameter(1, r1f32, "y");
  auto add =
      HloInstruction::CreateBinary(r1f32, HloOpcode::kAdd, x.get(), y.get());
  EXPECT_TRUE(add->IsElementwise());
  for (int i = 0; i < add->operand_count(); ++i) {
    EXPECT_TRUE(add->IsElementwiseOnOperand(i));
  }
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

  auto computation = builder.Build();
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
  //
  // x     y
  //  \   / \
  //   min   broadcast
  //     \   /
  //      sub
  //
  // The fusion instruction is elementwise on `x` because the only path from x
  // to sub contains only elementwise operations. It is not elementwise on `y`
  // because the path y->broadcast->sub is not all elementwise.
  const Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  const Shape r1f32 = ShapeUtil::MakeShape(F32, {5});

  HloComputation::Builder builder("PartiallyElementwiseWithReuse");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r1f32, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32, "y"));
  HloInstruction* min = builder.AddInstruction(
      HloInstruction::CreateBinary(r1f32, HloOpcode::kMinimum, x, y));
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r1f32, y, {0}));
  HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
      r1f32, HloOpcode::kSubtract, min, broadcast));

  auto computation = builder.Build();
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {sub, broadcast, min}, HloInstruction::FusionKind::kLoop);
  EXPECT_FALSE(fusion->IsElementwise());
  for (int64 operand_idx = 0; operand_idx < fusion->operand_count();
       ++operand_idx) {
    if (fusion->operand(operand_idx) == x) {
      EXPECT_TRUE(fusion->IsElementwiseOnOperand(operand_idx));
    } else {
      EXPECT_FALSE(fusion->IsElementwiseOnOperand(operand_idx));
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
  HloInstruction* dot = builder.AddInstruction(
      HloInstruction::CreateBinary(sout, HloOpcode::kDot, x, reshape));

  auto computation = builder.Build();
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {dot, reshape}, HloInstruction::FusionKind::kTransposeDot);

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
}

}  // namespace
}  // namespace xla
