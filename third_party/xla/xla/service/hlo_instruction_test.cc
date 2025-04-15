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

#include "xla/hlo/ir/hlo_instruction.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = ::xla::match;

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;

class HloInstructionTest : public HloTestBase {
 protected:
  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
};

// Simple visitor that collects the number of users and operands for certain HLO
// nodes. It also verifies some of the DFS visiting invariants (operands visited
// before their users, nodes not visited twice, etc.)
class OpAndUserCollectingVisitor : public DfsHloVisitorWithDefault {
 public:
  absl::Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Unimplemented("not implemented %s",
                         HloOpcodeString(hlo_instruction->opcode()));
  }

  absl::Status HandleParameter(HloInstruction* parameter) override {
    EXPECT_FALSE(count_.contains(parameter));
    count_[parameter] = GetCountsForNode(parameter);
    return absl::OkStatus();
  }

  absl::Status HandleConstant(HloInstruction* constant) override {
    EXPECT_FALSE(count_.contains(constant));
    count_[constant] = GetCountsForNode(constant);
    return absl::OkStatus();
  }

  absl::Status HandleAdd(HloInstruction* add) override {
    auto lhs = add->operand(0);
    auto rhs = add->operand(1);
    EXPECT_FALSE(count_.contains(add));
    EXPECT_TRUE(count_.contains(lhs));
    EXPECT_TRUE(count_.contains(rhs));
    count_[add] = GetCountsForNode(add);
    return absl::OkStatus();
  }

  absl::Status HandleNegate(HloInstruction* negate) override {
    auto operand = negate->operand(0);
    EXPECT_FALSE(count_.contains(negate));
    EXPECT_TRUE(count_.contains(operand));
    count_[negate] = GetCountsForNode(negate);
    return absl::OkStatus();
  }

  absl::Status HandleMap(HloInstruction* map) override {
    EXPECT_FALSE(count_.contains(map));
    for (HloInstruction* arg : map->operands()) {
      EXPECT_TRUE(count_.contains(arg));
    }
    count_[map] = GetCountsForNode(map);
    return absl::OkStatus();
  }

  absl::Status HandleReduce(HloInstruction* reduce) override {
    auto arg = reduce->operand(0);
    auto init_value = reduce->operand(1);
    EXPECT_FALSE(count_.contains(reduce));
    EXPECT_TRUE(count_.contains(arg));
    EXPECT_TRUE(count_.contains(init_value));
    count_[reduce] = GetCountsForNode(reduce);
    return absl::OkStatus();
  }

  int64_t NumOperands(const HloInstruction* node) {
    auto count_iterator = count_.find(node);
    EXPECT_NE(count_.end(), count_iterator);
    return count_iterator->second.operand_count;
  }

  int64_t NumUsers(const HloInstruction* node) {
    auto count_iterator = count_.find(node);
    EXPECT_NE(count_.end(), count_iterator);
    return count_iterator->second.user_count;
  }

 private:
  struct NumOpsAndUsers {
    int64_t operand_count;
    int64_t user_count;
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

  absl::Status Postprocess(HloInstruction* hlo) override {
    post_processed_nodes_.push_back(hlo);
    return absl::OkStatus();
  }

  absl::Status DefaultAction(HloInstruction* hlo_instruction) override {
    visited_nodes_.push_back(hlo_instruction);
    return absl::OkStatus();
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

TEST_F(HloInstructionTest, PostProcessAllVisitedNodesMultiComputation) {
  // Verifies all the nodes are visited and post-processed in the same order,
  // and that each node is visited exactly once.
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.1 = f32[] constant(1)
      c.2 = f32[] constant(2)
      c.3 = f32[] add(c.1, c.2)
      c.4 = f32[] constant(4)
      ROOT ret = f32[] multiply(c.4, c.3)
    }

    ENTRY axpy_computation {
      p.0 = f32[10] parameter(0)
      p.1 = f32[10] parameter(1)
      add.0 = f32[10] add(p.0, p.1)
      alpha = f32[] call(), to_apply=calculate_alpha
      broadcast = f32[10] broadcast(alpha), dimensions={}
      p.2 = f32[10] parameter(2)
      y = f32[10] multiply(broadcast, p.2)
      x = f32[10] subtract(y, add.0)
      p.3 = f32[10] parameter(3)
      ROOT add.1 = f32[10] add(x, p.3)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* add1 = FindInstruction(module.get(), "add.1");
  EXPECT_EQ(add1, module->entry_computation()->root_instruction());

  NodeCollectorAndPostProcessor visitor;
  ASSERT_IS_OK(add1->Accept(&visitor, /*call_finish_visit=*/true,
                            /*ignore_control_predecessors=*/false,
                            /*cross_computation=*/true));
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

  EXPECT_THAT(fusion->metadata(), EqualsProto(metadata));
  EXPECT_THAT(fusion->fused_expression_root()->metadata(),
              EqualsProto(metadata));
  EXPECT_THAT(fusion->fused_expression_root()->operand(0)->metadata(),
              EqualsProto(metadata));

  std::string new_name = "foobarfoo";
  auto cloned = fusion->CloneWithNewOperands(fusion->shape(), {}, new_name);
  EXPECT_THAT(fusion->metadata(), EqualsProto(metadata));

  size_t index = cloned->name().rfind(new_name);
  EXPECT_TRUE(index != std::string::npos);
}

TEST_F(HloInstructionTest, BinaryCallOp) {
  HloComputation::Builder builder(TestName());
  // Create a call instruction containing a single binary operation.
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.1f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));
  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* call = computation->CreateCallInstruction({add});

  EXPECT_THAT(call->operands(), ElementsAre(constant1, constant2));
  EXPECT_THAT(constant1->users(), ElementsAre(call));
  EXPECT_THAT(constant2->users(), ElementsAre(call));
}

TEST_F(HloInstructionTest, ChainCallOp) {
  HloComputation::Builder builder(TestName());
  // Create a chain of called unary ops.
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
  auto* call = computation->CreateCallInstruction({exp3, exp2, exp1});

  EXPECT_THAT(call->operands(), ElementsAre(constant));
  EXPECT_THAT(constant->users(), ElementsAre(call));
}

TEST_F(HloInstructionTest, MultiOutputCallOp) {
  HloComputation::Builder builder(TestName());
  // Create a chain of called unary ops.
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, constant));
  auto exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, exp1));
  auto exp3 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, exp2));
  auto exp4 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, constant));
  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, exp3, exp4));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  auto* call = computation->CreateCallInstruction({exp3, exp2, exp1});
  call->AppendInstructionIntoCalledComputation(exp4, /*add_output=*/true);

  EXPECT_THAT(call->operands(), ElementsAre(constant));
  EXPECT_EQ(add->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_THAT(add->operand(0)->operands(), ElementsAre(call));
  EXPECT_EQ(add->operand(1)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_THAT(add->operand(1)->operands(), ElementsAre(call));
}

TEST_F(HloInstructionTest, AsyncOp) {
  HloComputation::Builder builder(TestName());
  // Create a call instruction containing a single binary operation.
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.1f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));
  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      auto* async_done,
      computation->CreateAsyncInstructions(
          add, {ShapeUtil::MakeScalarShape(U32)}, "parallel_thread"));
  auto* async_start = async_done->operand(0);

  EXPECT_EQ(async_start->shape().tuple_shapes_size(), 3);
  EXPECT_EQ(async_start->async_execution_thread(), "parallel_thread");
  EXPECT_EQ(async_done->async_execution_thread(), "parallel_thread");
  EXPECT_TRUE(ShapeUtil::Equal(async_start->shape().tuple_shapes(2),
                               ShapeUtil::MakeScalarShape(U32)));
  EXPECT_EQ(async_start->async_wrapped_computation()->execution_thread(),
            "parallel_thread");
  EXPECT_EQ(async_done->async_wrapped_computation()->execution_thread(),
            "parallel_thread");
  EXPECT_THAT(async_start->operands(), ElementsAre(constant1, constant2));
  EXPECT_THAT(constant1->users(), ElementsAre(async_start));
  EXPECT_THAT(constant2->users(), ElementsAre(async_start));
  EXPECT_EQ(computation->root_instruction(), async_done);
}

TEST_F(HloInstructionTest, AsyncOpWithDeps) {
  HloComputation::Builder builder(TestName());
  // Create a call instruction containing a single binary operation.
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.1f)));

  auto constant3 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.1f)));
  auto constant4 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.1f)));

  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant3, constant4));

  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));

  auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32_, HloOpcode::kAdd, constant1, constant2));

  // control chain is add1 <- add <- add2
  TF_ASSERT_OK(add1->AddControlDependencyTo(add));

  TF_ASSERT_OK(add->AddControlDependencyTo(add2));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      auto* async_done,
      computation->CreateAsyncInstructions(
          add, {ShapeUtil::MakeScalarShape(U32)}, "parallel_thread"));
  auto* async_start = async_done->operand(0);
  // Verify that control chain is not broken.
  // New chain should be add1 <- asyncStart <- asyncDone <- add2
  EXPECT_EQ(async_start->control_predecessors().size(), 1);
  EXPECT_EQ(async_start->control_predecessors()[0], add1);

  EXPECT_EQ(async_done->control_successors().size(), 1);
  EXPECT_EQ(async_done->control_successors()[0], add2);

  EXPECT_EQ(async_start->shape().tuple_shapes_size(), 3);
  EXPECT_EQ(async_start->async_execution_thread(), "parallel_thread");
  EXPECT_EQ(async_done->async_execution_thread(), "parallel_thread");
  EXPECT_TRUE(ShapeUtil::Equal(async_start->shape().tuple_shapes(2),
                               ShapeUtil::MakeScalarShape(U32)));
  EXPECT_EQ(async_start->async_wrapped_computation()->execution_thread(),
            "parallel_thread");
  EXPECT_EQ(async_done->async_wrapped_computation()->execution_thread(),
            "parallel_thread");
  EXPECT_THAT(async_start->operands(), ElementsAre(constant1, constant2));
}

TEST_F(HloInstructionTest, PreserveOutfeedShapeThroughClone) {
  HloComputation::Builder builder(TestName());
  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>({
          {1, 2},
          {3, 4},
      })));
  auto shape10 = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2}, {1, 0});
  auto shape01 = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2}, {0, 1});
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

TEST_F(HloInstructionTest, PreserveShardingThroughCompatibleClone) {
  HloSharding sharding = HloSharding::AssignDevice(5);
  HloComputation::Builder builder(TestName());
  auto* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>({
          {1, 2},
          {3, 4},
      })));
  auto* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant, constant}));
  HloSharding tuple_sharding =
      HloSharding::SingleTuple(tuple->shape(), sharding);
  tuple->set_sharding(tuple_sharding);
  // Compatible with original shape as tuple tree structure and leaf ranks are
  // identical
  auto clone_shape = ShapeUtil::MakeShape(F32, {3, 3});
  clone_shape = ShapeUtil::MakeTupleShape({clone_shape, clone_shape});
  auto tuple_clone = tuple->CloneWithNewOperands(clone_shape, {});
  EXPECT_EQ(tuple_clone->sharding(), tuple_sharding);
}

TEST_F(HloInstructionTest,
       DoNotPreserveShardingThroughTupleTreeIncompatibleClone) {
  HloSharding sharding = HloSharding::AssignDevice(5);
  HloComputation::Builder builder(TestName());
  auto* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>({
          {1, 2},
          {3, 4},
      })));
  auto* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant, constant}));
  tuple->set_sharding(HloSharding::SingleTuple(tuple->shape(), sharding));
  // Incompatible with original shape as tuple tree structure is different
  auto clone_shape = ShapeUtil::MakeShape(F32, {2, 2});
  clone_shape =
      ShapeUtil::MakeTupleShape({clone_shape, clone_shape, clone_shape});
  auto tuple_clone = tuple->CloneWithNewOperands(clone_shape, {});
  EXPECT_FALSE(tuple_clone->has_sharding());
}

TEST_F(HloInstructionTest,
       DoNotPreserveShardingThroughLeafRankIncompatibleClone) {
  HloSharding sharding = HloSharding::AssignDevice(5);
  HloComputation::Builder builder(TestName());
  auto* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<float>({
          {1, 2},
          {3, 4},
      })));
  auto* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({constant, constant}));
  tuple->set_sharding(HloSharding::SingleTuple(tuple->shape(), sharding));
  // Incompatible with original shape as tuple tree structure is different
  auto clone_shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
  clone_shape = ShapeUtil::MakeTupleShape({clone_shape, clone_shape});
  auto tuple_clone = tuple->CloneWithNewOperands(clone_shape, {});
  EXPECT_FALSE(tuple_clone->has_sharding());
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
                          ParseAndReturnVerifiedModule(hlo_string));

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
  FunctionVisitor visitor([&visit_num, &visit_order](HloInstruction* inst) {
    EXPECT_FALSE(visit_order.contains(inst));
    visit_order[inst] = visit_num;
    visit_num++;
    return absl::OkStatus();
  });
  EXPECT_IS_OK(add->Accept(&visitor));

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
  const Shape r2f32 =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 10}, {1, 0});
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
  for (int64_t operand_idx = 0; operand_idx < fusion->operand_count();
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
  for (int64_t operand_idx = 0; operand_idx < fusion->operand_count();
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

TEST_F(HloInstructionTest, FuseInstructionKeepsInstruction) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  fused_add {
    p0 = f32[32,32]{1,0} parameter(0)
    p1 = f32[32,32]{1,0} parameter(1)
    ROOT add = f32[32,32]{1,0} add(p0, p1)
  }

  ENTRY reduce {
    p2 = f32[32,32]{1,0} parameter(0)
    p3 = f32[32,32]{1,0} parameter(1)
    c1 = f32[] constant(1)
    broadcast = f32[32,32]{1,0} broadcast(c1), dimensions={}
    mul = f32[32,32]{1,0} multiply(p2, p3)
    ROOT add = f32[32,32]{1,0} fusion(mul, broadcast), kind=kLoop, calls=fused_add
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloInstruction* fused_add = module->entry_computation()->root_instruction();
  HloInstruction* mul = fused_add->mutable_operand(0);
  EXPECT_EQ(1, mul->user_count());
  fused_add->FuseInstruction(mul);
  EXPECT_EQ(0, mul->user_count());
  // The fused instruction is still present in the computation.
  EXPECT_EQ(fused_add->parent(), mul->parent());
}

TEST_F(HloInstructionTest, FuseInstructionIntoMultiOutputKeepsInstruction) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  fused_add {
    p0 = f32[32,32]{1,0} parameter(0)
    p1 = f32[32,32]{1,0} parameter(1)
    ROOT add = f32[32,32]{1,0} add(p0, p1)
  }

  ENTRY reduce {
    p2 = f32[32,32]{1,0} parameter(0)
    p3 = f32[32,32]{1,0} parameter(1)
    c1 = f32[] constant(1)
    mul = f32[32,32]{1,0} multiply(p2, p3)
    broadcast = f32[32,32]{1,0} broadcast(c1), dimensions={}
    add = f32[32,32]{1,0} fusion(mul, broadcast), kind=kLoop, calls=fused_add
    ROOT root = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(mul, add)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloInstruction* root = module->entry_computation()->root_instruction();
  HloInstruction* mul = root->mutable_operand(0);
  HloInstruction* fused_add = root->mutable_operand(1);
  EXPECT_EQ(2, mul->user_count());
  fused_add->FuseInstructionIntoMultiOutput(mul);
  EXPECT_EQ(0, mul->user_count());
  // The fused instruction is still present in the computation.
  EXPECT_EQ(root->parent(), mul->parent());
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

TEST_F(HloInstructionTest, StringifyDot) {
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
            "%dot = f32[5,20]{1,0} dot(%x, %transpose), "
            "lhs_contracting_dims={1}, rhs_contracting_dims={0}");

  auto options2 = HloPrintOptions()
                      .set_print_metadata(false)
                      .set_print_operand_shape(false)
                      .set_print_percent(false)
                      .set_include_layout_in_shapes(false);

  EXPECT_EQ(dot->ToString(options2),
            "dot = f32[5,20] dot(x, transpose), "
            "lhs_contracting_dims={1}, rhs_contracting_dims={0}");
}

TEST_F(HloInstructionTest, StringifySparseDot) {
  HloComputation::Builder builder("SparseDot");
  HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {5, 16}), "x"));
  HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {32, 20}), "y"));
  HloInstruction* meta = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(U16, {5, 2}), "meta"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  SparsityDescriptor sparsity_descriptor;
  sparsity_descriptor.set_type(SparsityType::SPARSITY_STRUCTURED_N_M);
  sparsity_descriptor.set_n(2);
  sparsity_descriptor.set_m(4);
  sparsity_descriptor.set_index(0);
  sparsity_descriptor.set_dimension(1);
  std::vector<HloInstruction*> meta_operands = {meta};
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      ShapeUtil::MakeShape(F32, {5, 20}), x, y, dot_dnums,
      DefaultPrecisionConfig(2), {sparsity_descriptor}, meta_operands));

  EXPECT_EQ(
      dot->ToString(),
      "%dot = f32[5,20]{1,0} dot(%x, %y, %meta), lhs_contracting_dims={1}, "
      "rhs_contracting_dims={0}, sparsity=L.1@2:4");
}

TEST_F(HloInstructionTest, StringifyConditional) {
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
  builder.AddInstruction(HloInstruction::CreateDot(sout, x, reshape, dot_dnums,
                                                   DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());

  auto options = HloPrintOptions().set_print_metadata(false);
  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloInstruction* conditional =
      builder.AddInstruction(HloInstruction::CreateConditional(
          sout, pred, x, computation, x, computation));
  EXPECT_EQ(conditional->ToString(options),
            "%conditional = f32[5,20]{1,0} conditional(%constant, %x, %x), "
            "true_computation=%TransposeDot, false_computation=%TransposeDot");
}

TEST_F(HloInstructionTest, StringifyWhile) {
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
  builder.AddInstruction(HloInstruction::CreateDot(sout, x, reshape, dot_dnums,
                                                   DefaultPrecisionConfig(2)));

  auto module = CreateNewVerifiedModule();
  auto* computation = module->AddEntryComputation(builder.Build());

  auto options = HloPrintOptions().set_print_metadata(false);
  HloInstruction* loop = builder.AddInstruction(
      HloInstruction::CreateWhile(sout, computation, computation, x));
  EXPECT_EQ(loop->ToString(options),
            "%while = f32[5,20]{1,0} while(%x), condition=%TransposeDot, "
            "body=%TransposeDot");
}

TEST_F(HloInstructionTest, GetSetStatisticsViz) {
  const Shape shape = ShapeUtil::MakeShape(F32, {5, 10});

  HloComputation::Builder builder(TestName());
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));

  StatisticsViz statistics_viz;
  statistics_viz.set_stat_index_to_visualize(-1);

  x->set_statistics_viz(statistics_viz);

  EXPECT_FALSE(x->has_statistics());
  EXPECT_EQ(x->statistics_viz().stat_index_to_visualize(), -1);

  Statistic statistic;
  statistic.set_stat_name("stat-1");
  statistic.set_stat_val(30.0);

  x->add_single_statistic(statistic);
  x->set_stat_index_to_visualize(0);

  EXPECT_TRUE(x->has_statistics());
  EXPECT_THAT(x->statistic_to_visualize(), EqualsProto(statistic));

  statistic.set_stat_val(40.0);
  *statistics_viz.add_statistics() = statistic;

  x->set_statistics_viz(statistics_viz);

  EXPECT_THAT(x->statistics_viz(), EqualsProto(statistics_viz));
}

TEST_F(HloInstructionTest, StringifyStatisticsViz) {
  const Shape shape = ShapeUtil::MakeShape(F32, {5, 10});

  HloComputation::Builder builder(TestName());
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "y"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, x, y));

  // Empty statistics viz must not print "statistics={}"
  add->set_statistics_viz({});
  EXPECT_EQ(add->ToString(), "%add = f32[5,10]{1,0} add(%x, %y)");

  auto CreateStatisticsVizWithStatistics =
      [](int64_t stat_index_to_visualize,
         std::initializer_list<std::pair<absl::string_view, double>> statistics)
      -> StatisticsViz {
    StatisticsViz statistics_viz;
    statistics_viz.set_stat_index_to_visualize(stat_index_to_visualize);

    auto create_statistic = [](absl::string_view statistic_name,
                               double statistic_value) {
      Statistic statistic;
      statistic.set_stat_name(std::string(statistic_name));
      statistic.set_stat_val(statistic_value);
      return statistic;
    };

    for (const auto& [statistic_name, statistic_value] : statistics) {
      *statistics_viz.add_statistics() =
          create_statistic(statistic_name, statistic_value);
    }

    return statistics_viz;
  };

  add->set_statistics_viz(CreateStatisticsVizWithStatistics(
      1, {{"stat-1", 33.0}, {"stat-2", 44.0}}));

  EXPECT_EQ(add->ToString(),
            "%add = f32[5,10]{1,0} add(%x, %y), "
            "statistics={visualizing_index=1,stat-1=33,stat-2=44}");
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
                                   /*slice_sizes=*/{30, 29, 28, 27, 26},
                                   /*indices_are_sorted=*/false));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(gather_instruction->ToString(),
            "%gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} "
            "gather(%input_tensor, %start_indices), "
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
                                   /*slice_sizes=*/{30, 29, 28, 27, 26},
                                   /*indices_are_sorted=*/false));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(gather_instruction->ToString(),
            "%gather = f32[10,9,7,6,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} "
            "gather(%input_tensor, %start_indices), "
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
              /*index_vector_dim=*/2),
          /*indices_are_sorted=*/false,
          /*unique_indices=*/false));
  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(scatter_instruction->ToString(),
            "%scatter = f32[50,49,48,47,46]{4,3,2,1,0} "
            "scatter(%input_tensor, %scatter_indices, %scatter_updates), "
            "update_window_dims={4,5,6,7,8}, inserted_window_dims={}, "
            "scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=2, "
            "to_apply=%Scatter.update");
}

TEST_F(HloInstructionTest, StringifyAsyncOps) {
  const Shape s1 = ShapeUtil::MakeShape(F32, {10});
  const Shape s2 = ShapeUtil::MakeShape(F32, {20});
  const Shape s_tuple = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({s1}), s2, ShapeUtil::MakeShape(S32, {})});

  HloComputation::Builder async_builder("AsyncOp");
  HloInstruction* param = async_builder.AddInstruction(
      HloInstruction::CreateParameter(0, s1, "p0"));
  async_builder.AddInstruction(
      HloInstruction::CreateCustomCall(s2, {param},
                                       /*custom_call_target=*/"foo"));
  std::unique_ptr<HloComputation> async_computation = async_builder.Build();

  HloComputation::Builder entry_builder("Entry");
  HloInstruction* entry_param = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, s1, "p0"));
  HloInstruction* async_start =
      entry_builder.AddInstruction(HloInstruction::CreateAsyncStart(
          s_tuple, {entry_param}, async_computation.get(),
          /*async_execution_thread=*/"parallel_thread"));
  HloInstruction* async_update = entry_builder.AddInstruction(
      HloInstruction::CreateAsyncUpdate(s_tuple, async_start));
  entry_builder.AddInstruction(
      HloInstruction::CreateAsyncDone(s2, async_update));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(entry_builder.Build());
  module->AddEmbeddedComputation(std::move(async_computation));

  const std::string expected_with_syntax_sugar =
      R"(HloModule StringifyAsyncOps, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %custom-call-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(%p0), async_execution_thread="parallel_thread", custom_call_target="foo"
  %custom-call-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(%custom-call-start)
  ROOT %custom-call-done = f32[20]{0} custom-call-done(%custom-call-update)
}

)";
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_syntax_sugar_async_ops(true);
  EXPECT_EQ(module->ToString(), expected_with_syntax_sugar);
  const std::string expected_without_syntax_sugar =
      R"(HloModule StringifyAsyncOps, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

%AsyncOp (p0.1: f32[10]) -> f32[20] {
  %p0.1 = f32[10]{0} parameter(0)
  ROOT %custom-call = f32[20]{0} custom-call(%p0.1), custom_call_target="foo"
}, execution_thread="parallel_thread"

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %custom-call-start = ((f32[10]{0}), f32[20]{0}, s32[]) async-start(%p0), async_execution_thread="parallel_thread", calls=%AsyncOp
  %custom-call-update = ((f32[10]{0}), f32[20]{0}, s32[]) async-update(%custom-call-start)
  ROOT %custom-call-done = f32[20]{0} async-done(%custom-call-update)
}

)";
  auto options = HloPrintOptions().set_syntax_sugar_async_ops(false);
  EXPECT_EQ(module->ToString(options), expected_without_syntax_sugar);
}

TEST_F(HloInstructionTest, StringifyAsyncOpsWithReduceScatter) {
  const Shape rs_input_shape = ShapeUtil::MakeShape(F32, {20});
  const Shape rs_output_shape = ShapeUtil::MakeShape(F32, {10});

  std::unique_ptr<HloComputation> add_computation;
  {
    const Shape scalar_shape = ShapeUtil::MakeScalarShape(F32);
    HloComputation::Builder add_builder("add");
    HloInstruction* param0 = add_builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* param1 = add_builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    add_builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, param0, param1));
    add_computation = add_builder.Build();
  }

  std::unique_ptr<HloComputation> async_computation;
  {
    HloComputation::Builder async_builder("AsyncOp");
    HloInstruction* param = async_builder.AddInstruction(
        HloInstruction::CreateParameter(0, rs_input_shape, "pasync"));
    async_builder.AddInstruction(HloInstruction::CreateReduceScatter(
        rs_output_shape, {param}, add_computation.get(), CollectiveDeviceList(),
        false, std::nullopt, false, 0));
    async_computation = async_builder.Build();
  }

  const Shape async_start_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({rs_input_shape}), rs_output_shape});

  HloComputation::Builder entry_builder("Entry");
  HloInstruction* entry_param = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, rs_input_shape, "pentry"));
  HloInstruction* async_start =
      entry_builder.AddInstruction(HloInstruction::CreateAsyncStart(
          async_start_shape, {entry_param}, async_computation.get(),
          /*async_execution_thread=*/"parallel_thread"));
  HloInstruction* async_update = entry_builder.AddInstruction(
      HloInstruction::CreateAsyncUpdate(async_start_shape, async_start));
  entry_builder.AddInstruction(
      HloInstruction::CreateAsyncDone(rs_output_shape, async_update));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(entry_builder.Build());
  module->AddEmbeddedComputation(std::move(async_computation));
  module->AddEmbeddedComputation(std::move(add_computation));

  const std::string expected_with_syntax_sugar =
      R"(HloModule StringifyAsyncOpsWithReduceScatter, entry_computation_layout={(f32[20]{0})->f32[10]{0}}

%add (p0: f32[], p1: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = f32[] add(%p0, %p1)
}, execution_thread="parallel_thread"

ENTRY %Entry (pentry: f32[20]) -> f32[10] {
  %pentry = f32[20]{0} parameter(0)
  %reduce-scatter-start = ((f32[20]{0}), f32[10]{0}) reduce-scatter-start(%pentry), async_execution_thread="parallel_thread", replica_groups={}, dimensions={0}, to_apply=%add
  %reduce-scatter-update = ((f32[20]{0}), f32[10]{0}) reduce-scatter-update(%reduce-scatter-start)
  ROOT %reduce-scatter-done = f32[10]{0} reduce-scatter-done(%reduce-scatter-update)
}

)";
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_syntax_sugar_async_ops(true);
  EXPECT_EQ(module->ToString(), expected_with_syntax_sugar);

  const std::string expected_without_syntax_sugar =
      R"(HloModule StringifyAsyncOpsWithReduceScatter, entry_computation_layout={(f32[20]{0})->f32[10]{0}}

%add (p0: f32[], p1: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = f32[] add(%p0, %p1)
}, execution_thread="parallel_thread"

%AsyncOp (pasync: f32[20]) -> f32[10] {
  %pasync = f32[20]{0} parameter(0)
  ROOT %reduce-scatter = f32[10]{0} reduce-scatter(%pasync), replica_groups={}, dimensions={0}, to_apply=%add
}, execution_thread="parallel_thread"

ENTRY %Entry (pentry: f32[20]) -> f32[10] {
  %pentry = f32[20]{0} parameter(0)
  %reduce-scatter-start = ((f32[20]{0}), f32[10]{0}) async-start(%pentry), async_execution_thread="parallel_thread", calls=%AsyncOp
  %reduce-scatter-update = ((f32[20]{0}), f32[10]{0}) async-update(%reduce-scatter-start)
  ROOT %reduce-scatter-done = f32[10]{0} async-done(%reduce-scatter-update)
}

)";
  auto options = HloPrintOptions().set_syntax_sugar_async_ops(false);
  EXPECT_EQ(module->ToString(options), expected_without_syntax_sugar);
}

TEST_F(HloInstructionTest, CanonicalStringificationFusion) {
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
  constexpr char kParallelThreadName[] = "parallel_thread";
  computation->SetExecutionThread(kParallelThreadName);
  HloInstruction* fusion = computation->CreateFusionInstruction(
      {dot, reshape}, HloInstruction::FusionKind::kLoop);
  fusion->set_called_computations_execution_thread(
      kParallelThreadName,
      /*skip_async_execution_thread_overwrite*/ false);

  const std::string expected_fusion =
      R"(f32[5,20]{1,0} fusion(f32[5,10]{1,0}, f32[20,10]{1,0}), kind=kLoop, calls=
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
  ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}, execution_thread="parallel_thread")";
  EXPECT_EQ(fusion->ToString(options), expected_fusion);
}

TEST_F(HloInstructionTest, CanonicalStringificationWhile) {
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
  const std::string expected_loop =
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

TEST_F(HloInstructionTest, CanonicalStringificationConditional) {
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

  auto pred = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloInstruction* conditional =
      builder.AddInstruction(HloInstruction::CreateConditional(
          sout, pred, x, computation, x, computation));
  auto options = HloPrintOptions().Canonical();
  const std::string expected_conditional =
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
                          ParseAndReturnVerifiedModule(hlo_string));
  std::unique_ptr<HloModule> clone = module->Clone();
  for (HloComputation* computation : clone->computations()) {
    EXPECT_EQ(computation->parent(), clone.get());
    for (HloInstruction* instruction : computation->instructions()) {
      EXPECT_EQ(instruction->GetModule(), clone.get());
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

TEST_F(HloInstructionTest, IdenticalAccountsForCustomCallHasSideEffect) {
  auto instr1 = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                 /*operands=*/{},
                                                 /*custom_call_target=*/"foo");
  auto instr2 = instr1->Clone();
  EXPECT_TRUE(instr1->Identical(*instr2));

  auto custom_call_instr1 = Cast<HloCustomCallInstruction>(instr1.get());
  custom_call_instr1->set_custom_call_has_side_effect(true);
  EXPECT_FALSE(instr1->Identical(*instr2));
}

TEST_F(HloInstructionTest, CloneWindowOnCustomCall) {
  auto instr = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                /*operands=*/{},
                                                /*custom_call_target=*/"foo");
  Window w = window_util::MakeWindow({1, 2, 3});
  instr->set_window(w);
  auto clone = instr->Clone();
  EXPECT_THAT(clone->window(), EqualsProto(w));
}

TEST_F(HloInstructionTest, CloneDnumsOnCustomCall) {
  auto instr = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                /*operands=*/{},
                                                /*custom_call_target=*/"foo");
  ConvolutionDimensionNumbers dnums;
  dnums.set_output_batch_dimension(42);
  instr->set_convolution_dimension_numbers(dnums);
  auto clone = instr->Clone();
  EXPECT_THAT(clone->convolution_dimension_numbers(), EqualsProto(dnums));
}

TEST_F(HloInstructionTest, CloneHasSideEffectOnCustomCall) {
  auto instr = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                /*operands=*/{},
                                                /*custom_call_target=*/"foo");
  auto custom_call_instr = Cast<HloCustomCallInstruction>(instr.get());
  EXPECT_FALSE(custom_call_instr->custom_call_has_side_effect());
  custom_call_instr->set_custom_call_has_side_effect(true);
  EXPECT_TRUE(custom_call_instr->custom_call_has_side_effect());
  auto clone = instr->Clone();
  auto custom_call_clone = Cast<HloCustomCallInstruction>(clone.get());
  EXPECT_TRUE(custom_call_clone->custom_call_has_side_effect());
}

TEST_F(HloInstructionTest, CustomCallHasSideEffect) {
  auto instr = HloInstruction::CreateCustomCall(ShapeUtil::MakeShape(F32, {}),
                                                /*operands=*/{},
                                                /*custom_call_target=*/"foo");
  auto custom_call_instr = Cast<HloCustomCallInstruction>(instr.get());
  EXPECT_FALSE(instr->HasSideEffect());
  custom_call_instr->set_custom_call_has_side_effect(true);
  EXPECT_TRUE(instr->HasSideEffect());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  auto* conv = module->entry_computation()->root_instruction();

  auto clone = conv->Clone();
  EXPECT_THAT(clone->precision_config().operand_precision(),
              ElementsAre(PrecisionConfig::HIGH, PrecisionConfig::DEFAULT));
}

TEST_F(HloInstructionTest, ReuseReshapeOfFusionParameter) {
  // Create a fusion node which uses the reshape of a parameter twice.  Because
  // it's the same reshape, this counts as UseKind::kUsePermutingElements, which
  // is exposed publicly as "does not reuse this operand".
  constexpr char kHloString[] = R"(
  HloModule test_module
  f {
    p = f32[3,2] parameter(0)
    r = f32[2,3] reshape(p)
    x = f32[2,3] multiply(r, r)
    y = f32[2,3] add(r, r)
    ROOT sum = f32[2,3] add(x, y)
  }
  ENTRY test {
    p = f32[3,2] parameter(0)
    ROOT fusion = f32[2,3] fusion(p), calls=f, kind=kLoop
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_FALSE(root->ReusesOperandElements(0));
}

TEST_F(HloInstructionTest, ReuseMultipleReshapesOfFusionParameter) {
  // Create a fusion node which uses two different reshapes of a parameter
  // twice.  Because they're not the same reshapes, this counts as
  // UseKind::kUsePermutingElements, which is exposed publicly as "does reuse
  // this operand".
  constexpr char kHloString[] = R"(
  HloModule test_module
  f {
    p = f32[3,2] parameter(0)
    r1 = f32[2,3] reshape(p)
    r2 = f32[6,1] reshape(p)
    ROOT result = (f32[2,3], f32[6,1]) tuple(r1, r2)
  }
  ENTRY test {
    p = f32[3,2] parameter(0)
    ROOT fusion = (f32[2,3], f32[6,1]) fusion(p), calls=f, kind=kLoop
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->ReusesOperandElements(0));
}

TEST_F(HloInstructionTest, BitcastDoesNotReuseElements) {
  constexpr char kHloString[] = R"(
  HloModule test_module
  ENTRY test {
    p = f32[3,2]{0,1} parameter(0)
    ROOT bitcast = f32[6] bitcast(p)
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_FALSE(root->ReusesOperandElements(0));
}

TEST_F(HloInstructionTest, GatherDoesNotReuseElements) {
  constexpr char kHloString[] = R"(
  HloModule test_module

  ENTRY test {
    input = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
    idx = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
    ROOT gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0}
      gather(input, idx), offset_dims={4,5,6,7,8}, collapsed_slice_dims={},
      start_index_map={0,1,2,3,4}, index_vector_dim=4,
      slice_sizes={30,29,28,27,26}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_FALSE(root->ReusesOperandElements(0));
  EXPECT_FALSE(root->ReusesOperandElements(1));
}

TEST_F(HloInstructionTest, BackendConfigCanContainNonFiniteFloats) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot = b.AddInstruction(HloInstruction::CreateDot(
      shape, p0, p0, dot_dnums, DefaultPrecisionConfig(2)));
  gpu::GpuBackendConfig gpu_config;
  gpu::GemmBackendConfig& orig_config =
      *gpu_config.mutable_gemm_backend_config();
  orig_config.set_alpha_real(std::numeric_limits<double>::infinity());
  orig_config.set_alpha_imag(std::numeric_limits<double>::quiet_NaN());
  TF_ASSERT_OK(dot->set_backend_config(gpu_config));

  TF_ASSERT_OK_AND_ASSIGN(auto new_gpu_config,
                          dot->backend_config<gpu::GpuBackendConfig>());
  EXPECT_GT(new_gpu_config.gemm_backend_config().alpha_real(),
            std::numeric_limits<double>::max());
  EXPECT_NE(new_gpu_config.gemm_backend_config().alpha_imag(),
            new_gpu_config.gemm_backend_config().alpha_imag());
}

TEST_F(HloInstructionTest, VerifyToApplyRegionPointsToReduceScatter) {
  const Shape rs_input_shape = ShapeUtil::MakeShape(F32, {20});
  const Shape rs_output_shape = ShapeUtil::MakeShape(F32, {10});

  std::unique_ptr<HloComputation> add_computation;
  {
    const Shape scalar_shape = ShapeUtil::MakeScalarShape(F32);
    HloComputation::Builder add_builder("add");
    HloInstruction* param0 = add_builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* param1 = add_builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    add_builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, param0, param1));
    add_computation = add_builder.Build();
  }

  std::unique_ptr<HloComputation> main_computation;
  HloComputation::Builder main_builder("Entry");
  HloInstruction* param = main_builder.AddInstruction(
      HloInstruction::CreateParameter(0, rs_input_shape, "input"));
  main_builder.AddInstruction(HloInstruction::CreateReduceScatter(
      rs_output_shape, {param}, add_computation.get(), CollectiveDeviceList(),
      false, std::nullopt, false, 0));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(main_builder.Build());
  module->AddEmbeddedComputation(std::move(add_computation));
  // The only embedded computation in the graph should be pointing to
  // the reduce-scatter instruction.
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (!comp->IsEntryComputation()) {
      EXPECT_THAT(comp->caller_instructions(),
                  ElementsAre(module->entry_computation()->root_instruction()));
    }
  }
}

TEST_F(HloInstructionTest, VerifyToApplyRegionPointsToAllReduce) {
  const Shape ar_input_shape = ShapeUtil::MakeShape(F32, {20});

  std::unique_ptr<HloComputation> add_computation;
  {
    const Shape scalar_shape = ShapeUtil::MakeScalarShape(F32);
    HloComputation::Builder add_builder("add");
    HloInstruction* param0 = add_builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* param1 = add_builder.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "p1"));
    add_builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, param0, param1));
    add_computation = add_builder.Build();
  }

  std::unique_ptr<HloComputation> main_computation;
  HloComputation::Builder main_builder("Entry");
  HloInstruction* param = main_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ar_input_shape, "input"));
  main_builder.AddInstruction(HloInstruction::CreateAllReduce(
      ar_input_shape, {param}, add_computation.get(), CollectiveDeviceList(),
      false, std::nullopt, false));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(main_builder.Build());
  module->AddEmbeddedComputation(std::move(add_computation));
  // The only embedded computation in the graph should be pointing to
  // the all-reduce instruction.
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (!comp->IsEntryComputation()) {
      EXPECT_THAT(comp->caller_instructions(),
                  ElementsAre(module->entry_computation()->root_instruction()));
    }
  }
}

TEST_F(HloInstructionTest, PrintCycle) {
  constexpr char kHloString[] = R"(
  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=2,
      frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}}"
    }
    send = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all),
      channel_id=2, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}}"
    }, control-predecessors={recv}
    send-done = token[] send-done(send), channel_id=2
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=2
    ROOT recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  HloInstruction* recv = FindInstruction(module.get(), "recv");
  HloInstruction* send_done = FindInstruction(module.get(), "send-done");
  ASSERT_IS_OK(send_done->AddControlDependencyTo(recv));
  HloInstruction* root = FindInstruction(module.get(), "recv-data");
  NodeCollectorAndPostProcessor visitor;
  auto status = root->Accept(&visitor);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("recv\n send\n send-done\n recv"));
  // Remove the cycle to avoid error when destructing the verified module.
  ASSERT_IS_OK(send_done->DropAllControlDeps());
}

TEST_F(HloInstructionTest, VerifyBodyComputationPointsToWhile) {
  auto module = CreateNewVerifiedModule();
  const Shape scalar_shape = ShapeUtil::MakeScalarShape(F32);

  HloComputation::Builder cond_builder("cond");
  {
    HloInstruction* param = cond_builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* constant = cond_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1024.0)));
    cond_builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param,
                                      constant, ComparisonDirection::kLt));
  }
  auto cond_computation = module->AddEmbeddedComputation(cond_builder.Build());

  HloComputation::Builder body_builder("body");
  {
    HloInstruction* param = body_builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    body_builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kMultiply, param, param));
  }
  auto body_computation = module->AddEmbeddedComputation(body_builder.Build());

  std::unique_ptr<HloComputation> main_computation;
  HloComputation::Builder main_builder("Entry");
  HloInstruction* param = main_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "input"));
  main_builder.AddInstruction(HloInstruction::CreateWhile(
      scalar_shape, cond_computation, body_computation, param));

  module->AddEntryComputation(main_builder.Build());
  // Should find one while body computation in the graph and it should point to
  // the while instruction.
  int num_whiles = 0;
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      ++num_whiles;
      HloComputation* while_body = instruction->while_body();
      EXPECT_EQ(while_body->GetUniqueCaller(HloOpcode::kWhile).value(),
                instruction);
    }
  }
  EXPECT_EQ(num_whiles, 1);
}

TEST_F(HloInstructionTest,
       VerifyBranchComputationPointsToConditonal_TrueFalseConstructor) {
  auto module = CreateNewVerifiedModule();
  const Shape scalar_shape = ShapeUtil::MakeScalarShape(F32);

  HloComputation::Builder branch_0_builder("branch_0");
  {
    HloInstruction* param = branch_0_builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* constant = branch_0_builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1024.0)));
    branch_0_builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, param, constant));
  }
  auto branch_0_computation =
      module->AddEmbeddedComputation(branch_0_builder.Build());

  HloComputation::Builder branch_1_builder("branch_1");
  {
    HloInstruction* param = branch_1_builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    branch_1_builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kMultiply, param, param));
  }
  auto branch_1_computation =
      module->AddEmbeddedComputation(branch_1_builder.Build());

  std::unique_ptr<HloComputation> main_computation;
  HloComputation::Builder main_builder("Entry");

  HloInstruction* pred_param =
      main_builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(PRED, {}), "pred_param"));
  HloInstruction* param = main_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "input"));

  main_builder.AddInstruction(HloInstruction::CreateConditional(
      scalar_shape, pred_param, /*true_computation_arg=*/param,
      /*true_computation=*/branch_0_computation,
      /*false_computation_arg=*/param,
      /*false_computation=*/branch_1_computation));

  module->AddEntryComputation(main_builder.Build());
  // Should find conditional branch computations in the graph and it should
  // point to the conditional instruction.
  int num_conditional_branch_comp = 0;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    auto conditional_callers =
        comp->caller_instructions(HloOpcode::kConditional);
    if (!conditional_callers.empty()) {
      num_conditional_branch_comp += 1;
      EXPECT_THAT(conditional_callers,
                  ElementsAre(module->entry_computation()->root_instruction()));
    }
  }
  EXPECT_EQ(num_conditional_branch_comp, 2);
}

TEST_F(HloInstructionTest,
       VerifyBranchComputationPointsToConditonal_BranchIndexConstructor) {
  auto module = CreateNewVerifiedModule();
  const Shape scalar_shape = ShapeUtil::MakeScalarShape(F32);

  std::vector<HloComputation*> branch_computations;

  {
    HloComputation::Builder builder("branch_0");

    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    HloInstruction* constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1024.0)));
    builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, param, constant));

    branch_computations.push_back(
        module->AddEmbeddedComputation(builder.Build()));
  }

  {
    HloComputation::Builder builder("branch_1");

    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kMultiply, param, param));

    branch_computations.push_back(
        module->AddEmbeddedComputation(builder.Build()));
  }

  {
    HloComputation::Builder builder("branch_2");

    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "p0"));
    builder.AddInstruction(
        HloInstruction::CreateUnary(scalar_shape, HloOpcode::kLog, param));

    branch_computations.push_back(
        module->AddEmbeddedComputation(builder.Build()));
  }

  std::unique_ptr<HloComputation> main_computation;
  HloComputation::Builder main_builder("Entry");

  HloInstruction* branch_index =
      main_builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeScalarShape(S32), "branch_index_param"));
  HloInstruction* param = main_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "input"));

  std::vector<HloInstruction*> branch_computation_args(
      branch_computations.size(), param);

  main_builder.AddInstruction(HloInstruction::CreateConditional(
      scalar_shape, branch_index, branch_computations,
      branch_computation_args));

  module->AddEntryComputation(main_builder.Build());
  // Should find conditional branch computations in the graph and it should
  // point to the conditional instruction.
  int num_conditional_branch_comp = 0;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    auto conditional_callers =
        comp->caller_instructions(HloOpcode::kConditional);
    if (!conditional_callers.empty()) {
      num_conditional_branch_comp += 1;
      EXPECT_THAT(conditional_callers,
                  ElementsAre(module->entry_computation()->root_instruction()));
    }
  }
  EXPECT_EQ(num_conditional_branch_comp, branch_computations.size());
}

TEST_F(HloInstructionTest, BackendConfigCopiedToDerived) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p1"));
  auto add = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p1));

  gpu::GpuBackendConfig gpu_config;
  gpu_config.set_operation_queue_id(2);
  TF_ASSERT_OK(add->set_backend_config(gpu_config));
  auto add2 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p0));
  add->SetupDerivedInstruction(add2);
  auto backend_config = add2->backend_config<gpu::GpuBackendConfig>();
  EXPECT_TRUE(backend_config.ok());
  EXPECT_EQ(backend_config->operation_queue_id(), 2);
}

TEST_F(HloInstructionTest, BackendConfigNotCopiedToDerivedWithDiffOpcode) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p1"));
  auto or1 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kOr, p0, p1));

  gpu::GpuBackendConfig gpu_config;
  gpu_config.set_operation_queue_id(2);
  TF_ASSERT_OK(or1->set_backend_config(gpu_config));
  auto add2 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p1));
  or1->SetupDerivedInstruction(add2);
  EXPECT_FALSE(add2->has_backend_config());
}

TEST_F(HloInstructionTest, BackendConfigNotCopiedToDerivedWithConfig) {
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p1"));
  auto add = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p1));

  gpu::GpuBackendConfig gpu_config0;
  gpu::GpuBackendConfig gpu_config1;
  gpu_config0.set_operation_queue_id(2);
  gpu_config1.set_operation_queue_id(3);

  TF_ASSERT_OK(add->set_backend_config(gpu_config0));
  auto add2 = b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, p0, p0));
  TF_ASSERT_OK(add2->set_backend_config(gpu_config1));

  add->SetupDerivedInstruction(add2);
  auto backend_config = add2->backend_config<gpu::GpuBackendConfig>();
  EXPECT_TRUE(backend_config.ok());
  EXPECT_EQ(backend_config->operation_queue_id(), 3);
}

TEST_F(HloInstructionTest,
       MergeMultiOutputProducerFusionIntoMultiOutputFusion) {
  const std::string& hlo_string = R"(
    HloModule mof
    mof_producer {
      param0 = f32[10]{0} parameter(0)
      param1 = f32[10]{0} parameter(1)
      add = f32[10]{0} add(param0, param1)
      sub = f32[10]{0} subtract(param0, param1)
      ROOT res = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(param1, add, sub, param0)
    }

    mof_consumer {
      param0.0 = f32[10]{0} parameter(0)
      param1.0 = f32[10]{0} parameter(1)
      param2.0 = f32[10]{0} parameter(2)
      mul = f32[10]{0} multiply(param0.0, param1.0)
      div = f32[10]{0} divide(param0.0, param1.0)
      ROOT res = (f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(mul, div, param2.0)
    }

    ENTRY main {
      p0 = f32[10]{0} parameter(0)
      p1 = f32[10]{0} parameter(1)
      producer = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) fusion(p0, p1), kind=kLoop, calls=mof_producer
      gte0 = f32[10]{0} get-tuple-element(producer), index=0
      gte1 = f32[10]{0} get-tuple-element(producer), index=1
      gte2 = f32[10]{0} get-tuple-element(producer), index=2
      gte3 = f32[10]{0} get-tuple-element(producer), index=3
      consumer = (f32[10]{0}, f32[10]{0}, f32[10]{0}) fusion(gte1, gte2, gte3), kind=kLoop, calls=mof_consumer
      gte4 = f32[10]{0} get-tuple-element(consumer), index=0
      gte5 = f32[10]{0} get-tuple-element(consumer), index=1
      gte6 = f32[10]{0} get-tuple-element(consumer), index=2
      ROOT res = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(gte0, gte1, gte3, gte4, gte5, gte6)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* producer = FindInstruction(module.get(), "producer");
  HloInstruction* consumer = FindInstruction(module.get(), "consumer");
  consumer->MergeFusionInstructionIntoMultiOutput(producer);
  HloInstruction* fusion = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Parameter(1), m::GetTupleElement(m::Fusion(&fusion), 3),
                  m::Parameter(0), m::GetTupleElement(m::Fusion(), 0),
                  m::GetTupleElement(m::Fusion(), 1),
                  m::GetTupleElement(m::Fusion(), 2))));
  EXPECT_THAT(fusion->fused_instructions_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Multiply(m::Add(m::Parameter(0), m::Parameter(1)),
                              m::Subtract(m::Parameter(0), m::Parameter(1))),
                  m::Divide(m::Add(m::Parameter(0), m::Parameter(1)),
                            m::Subtract(m::Parameter(0), m::Parameter(1))),
                  m::Parameter(0), m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(HloInstructionTest,
       MergeMultiOutputProducerFusionIntoMultiOutputFusionAvoidDuplicateRoots) {
  const std::string& hlo_string = R"(
    HloModule mof
    mof_producer {
      param0 = f32[10]{0} parameter(0)
      param1 = f32[10]{0} parameter(1)
      add = f32[10]{0} add(param0, param1)
      sub = f32[10]{0} subtract(param0, param1)
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(add, sub)
    }

    mof_consumer {
      param0.0 = f32[10]{0} parameter(0)
      param1.0 = f32[10]{0} parameter(1)
      mul = f32[10]{0} multiply(param0.0, param1.0)
      div = f32[10]{0} divide(param0.0, param1.0)
      ROOT res = (f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(mul, div, param0.0)
    }

    ENTRY main {
      p0 = f32[10]{0} parameter(0)
      p1 = f32[10]{0} parameter(1)
      producer = (f32[10]{0}, f32[10]{0}) fusion(p0, p1), kind=kLoop, calls=mof_producer
      gte1 = f32[10]{0} get-tuple-element(producer), index=0
      gte2 = f32[10]{0} get-tuple-element(producer), index=1
      consumer = (f32[10]{0}, f32[10]{0}, f32[10]{0}) fusion(gte1, gte2), kind=kLoop, calls=mof_consumer
      gte3 = f32[10]{0} get-tuple-element(consumer), index=0
      gte4 = f32[10]{0} get-tuple-element(consumer), index=1
      gte5 = f32[10]{0} get-tuple-element(consumer), index=2
      ROOT res = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(gte1, gte3, gte4, gte5)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* producer = FindInstruction(module.get(), "producer");
  HloInstruction* consumer = FindInstruction(module.get(), "consumer");
  consumer->MergeFusionInstructionIntoMultiOutput(producer);
  HloInstruction* fusion = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion), 2),
                                  m::GetTupleElement(m::Fusion(), 0),
                                  m::GetTupleElement(m::Fusion(), 1),
                                  m::GetTupleElement(m::Fusion(), 2))));
  EXPECT_THAT(fusion->fused_instructions_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Multiply(m::Add(m::Parameter(0), m::Parameter(1)),
                              m::Subtract(m::Parameter(0), m::Parameter(1))),
                  m::Divide(m::Add(m::Parameter(0), m::Parameter(1)),
                            m::Subtract(m::Parameter(0), m::Parameter(1))),
                  m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(HloInstructionTest,
       MergeMultiOutputSiblingFusionsAvoidDuplicateFusionParameters) {
  const std::string& hlo_string = R"(
    HloModule mof
    mof_sibling1 {
      param0 = f32[10]{0} parameter(0)
      param1 = f32[10]{0} parameter(1)
      add = f32[10]{0} add(param0, param1)
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(param1, add)
    }

    mof_sibling2 {
      param0.0 = f32[10]{0} parameter(0)
      param1.0 = f32[10]{0} parameter(1)
      mul = f32[10]{0} multiply(param0.0, param1.0)
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(mul, param1.0)
    }

    ENTRY main {
      p0 = f32[10]{0} parameter(0)
      p1 = f32[10]{0} parameter(1)
      sibling1 = (f32[10]{0}, f32[10]{0}) fusion(p0, p1), kind=kLoop, calls=mof_sibling1
      gte0 = f32[10]{0} get-tuple-element(sibling1), index=0
      gte1 = f32[10]{0} get-tuple-element(sibling1), index=1
      sibling2 = (f32[10]{0}, f32[10]{0}) fusion(p0, p1), kind=kLoop, calls=mof_sibling2
      gte2 = f32[10]{0} get-tuple-element(sibling2), index=0
      gte3 = f32[10]{0} get-tuple-element(sibling2), index=1
      ROOT res = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(gte0, gte1, gte2, gte3)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* sibling1 = FindInstruction(module.get(), "sibling1");
  HloInstruction* sibling2 = FindInstruction(module.get(), "sibling2");
  sibling2->MergeFusionInstructionIntoMultiOutput(sibling1);
  HloInstruction* fusion = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Parameter(1),
                                  m::GetTupleElement(m::Fusion(&fusion), 2),
                                  m::GetTupleElement(m::Fusion(), 0),
                                  m::GetTupleElement(m::Fusion(), 1))));
  EXPECT_THAT(fusion->fused_instructions_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Multiply(m::Parameter(0), m::Parameter(1)),
                                  m::Parameter(1),
                                  m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(HloInstructionTest, UnfuseInstruction) {
  const std::string& hlo_string = R"(
    HloModule mof
    fusion_comp {
      param0 = f32[10]{0} parameter(0)
      param1 = f32[10]{0} parameter(1)
      add = f32[10]{0} add(param0, param1)
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(param1, add)
    }

    ENTRY main {
      p0 = f32[10]{0} parameter(0)
      p1 = f32[10]{0} parameter(1)
      fusion.1 = (f32[10]{0}, f32[10]{0}) fusion(p0, p1), kind=kLoop, calls=fusion_comp
      gte0 = f32[10]{0} get-tuple-element(fusion.1), index=0
      gte1 = f32[10]{0} get-tuple-element(fusion.1), index=1
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(gte0, gte1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* fusion = FindInstruction(module.get(), "fusion.1");
  HloInstruction* add = fusion->fused_instructions_computation()
                            ->root_instruction()
                            ->mutable_operand(1);
  TF_ASSERT_OK_AND_ASSIGN(auto unfused, fusion->UnfuseInstruction(add));
  EXPECT_THAT(unfused, GmockMatch(m::Add(m::Parameter(0), m::Parameter(1))));
}

TEST_F(HloInstructionTest, UnfuseInstruction2) {
  const std::string& hlo_string = R"(
    HloModule mof
    fusion_comp {
      param0 = f32[10]{0} parameter(0)
      param1 = f32[10]{0} parameter(1)
      add = f32[10]{0} add(param0, param1)
      add2 = f32[10]{0} add(add, param1)
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(param1, add2)
    }

    ENTRY main {
      p0 = f32[10]{0} parameter(0)
      p1 = f32[10]{0} parameter(1)
      fusion.1 = (f32[10]{0}, f32[10]{0}) fusion(p0, p1), kind=kLoop, calls=fusion_comp
      gte0 = f32[10]{0} get-tuple-element(fusion.1), index=0
      gte1 = f32[10]{0} get-tuple-element(fusion.1), index=1
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(gte0, gte1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* fusion = FindInstruction(module.get(), "fusion.1");
  HloInstruction* add2 = fusion->fused_instructions_computation()
                             ->root_instruction()
                             ->mutable_operand(1);
  HloInstruction* add = add2->mutable_operand(0);

  // add2 is not unfusable since it has non-const non-parameter operands.
  EXPECT_FALSE(fusion->UnfuseInstruction(add2).ok());

  TF_ASSERT_OK_AND_ASSIGN(auto unfused, fusion->UnfuseInstruction(add));
  EXPECT_THAT(unfused, GmockMatch(m::Add(m::Parameter(0), m::Parameter(1))));
}

TEST_F(HloInstructionTest, UnfuseInstructionWithConstantOperand) {
  const std::string& hlo_string = R"(
    HloModule mof
    fusion_comp {
      param0 = f32[10]{0} parameter(0)
      param1 = f32[10]{0} parameter(1)
      const = f32[] constant(1.0)
      broadcast = f32[10]{0} broadcast(const), dimensions={}
      add = f32[10]{0} add(param0, broadcast)
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(param1, add)
    }

    ENTRY main {
      p0 = f32[10]{0} parameter(0)
      p1 = f32[10]{0} parameter(1)
      fusion.1 = (f32[10]{0}, f32[10]{0}) fusion(p0, p1), kind=kLoop, calls=fusion_comp
      gte0 = f32[10]{0} get-tuple-element(fusion.1), index=0
      gte1 = f32[10]{0} get-tuple-element(fusion.1), index=1
      ROOT res = (f32[10]{0}, f32[10]{0}) tuple(gte0, gte1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* fusion = FindInstruction(module.get(), "fusion.1");
  HloInstruction* add = fusion->fused_instructions_computation()
                            ->root_instruction()
                            ->mutable_operand(1);
  TF_ASSERT_OK_AND_ASSIGN(auto unfused, fusion->UnfuseInstruction(add));
  EXPECT_THAT(unfused,
              GmockMatch(m::Add(m::Parameter(0), m::Broadcast(m::Constant()))));
}

TEST_F(HloInstructionTest, RaggedDotHasPrecisionConfig) {
  constexpr char kHloString[] = R"(
  HloModule module
  ENTRY entry_computation {
    a = f32[11,5] parameter(0)
    b = f32[3,5,7] parameter(1)
    c = u32[3] parameter(2)
    ROOT dot = f32[11,7] ragged-dot(a, b, c), lhs_contracting_dims={1}, rhs_contracting_dims={1}, lhs_ragged_dims={0}, rhs_group_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloString));
  auto* ragged_dot = module->entry_computation()->root_instruction();

  EXPECT_THAT(ragged_dot->precision_config().operand_precision(),
              ::testing::ElementsAre(PrecisionConfig::DEFAULT,
                                     PrecisionConfig::DEFAULT));
}

TEST_F(HloInstructionTest, ValidResultAccuracy) {
  ResultAccuracy result_accuracy_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        tolerance { rtol: 0.4 atol: 0.0 ulps: 1 }
      )pb",
      &result_accuracy_proto));
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  auto exp = builder.AddInstruction(HloInstruction::CreateUnary(
      r0f32_, HloOpcode::kExp, foo, result_accuracy_proto));
  // exp->set_result_accuracy(result_accuracy_proto);
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());
  EXPECT_THAT(exp->result_accuracy(), EqualsProto(result_accuracy_proto));

  // mode: HIGHEST
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        mode: HIGHEST
      )pb",
      &result_accuracy_proto));
  exp = builder.AddInstruction(HloInstruction::CreateUnary(
      r0f32_, HloOpcode::kExp, foo, result_accuracy_proto));
  EXPECT_THAT(exp->result_accuracy(), EqualsProto(result_accuracy_proto));
}

TEST_F(HloInstructionTest, InvalidResultAccuracy) {
  ResultAccuracy result_accuracy_proto;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        tolerance { rtol: -0.4 }
      )pb",
      &result_accuracy_proto));
  HloComputation::Builder builder(TestName());
  auto foo =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "foo"));
  ASSERT_DEATH(builder.AddInstruction(HloInstruction::CreateUnary(
                   r0f32_, HloOpcode::kExp, foo, result_accuracy_proto)),
               "Invalid result accuracy");
}

TEST_F(HloInstructionTest, CreateFromProtoExp) {
  HloInstructionProto proto_valid;
  proto_valid.set_opcode("exponential");
  proto_valid.set_name("exp");
  proto_valid.mutable_shape()->set_element_type(PrimitiveType::F32);
  proto_valid.mutable_result_accuracy()->mutable_tolerance()->set_rtol(0.4);
  proto_valid.mutable_result_accuracy()->mutable_tolerance()->set_atol(
      0.0);  // NOLINT
  proto_valid.mutable_result_accuracy()->mutable_tolerance()->set_ulps(1);
  proto_valid.add_operand_ids(0);
  ResultAccuracy r;
  r.mutable_tolerance()->set_rtol(0.4);
  r.mutable_tolerance()->set_atol(0.0);  // NOLINT
  r.mutable_tolerance()->set_ulps(1);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloInstruction> hlo,
      HloInstruction::CreateFromProto(
          proto_valid,
          {{0, HloInstruction::CreateParameter(0, r0f32_, "foo").get()}}));
  EXPECT_THAT(hlo->result_accuracy(), EqualsProto(r));
  HloInstructionProto proto_invalid;
  proto_invalid.set_opcode("exponential");
  proto_invalid.set_name("exp");
  proto_invalid.mutable_shape()->set_element_type(PrimitiveType::F32);
  proto_invalid.mutable_result_accuracy()->mutable_tolerance()->set_rtol(0.4);
  proto_invalid.mutable_result_accuracy()->mutable_tolerance()->set_atol(
      0.0);  // NOLINT
  proto_invalid.mutable_result_accuracy()->mutable_tolerance()->set_ulps(-1);
  proto_invalid.add_operand_ids(0);
  auto hlo_invalid = HloInstruction::CreateFromProto(
      proto_invalid,
      {{0, HloInstruction::CreateParameter(0, r0f32_, "foo").get()}});
  EXPECT_THAT(hlo_invalid.status().message(),
              ::testing::HasSubstr("Negative tolerance"));
}

TEST_F(HloInstructionTest, ExpInvalidResultAccuracy) {
  const char* const hlo_string = R"(
  HloModule exponential_hw

  ENTRY exponential_hw {
    %exponent = f32[] parameter(0)
    ROOT %exponential = f32[] exponential(f32[] %exponent), result_accuracy={mode=foo}
  }
  )";
  EXPECT_THAT(ParseAndReturnVerifiedModule(hlo_string).status().message(),
              ::testing::HasSubstr("Unknown accuracy mode"));
}

TEST_F(HloInstructionTest, NegativeResultAccuracy) {
  const char* const hlo_string = R"(
  HloModule negate

  ENTRY negate {
    %operand = f32[] parameter(0)
    ROOT %negate = f32[] negate(f32[] %operand), result_accuracy={tolerance={rtol=0.5, atol=1.0, ulps=2}}
  }
  )";
  // Negate is not a valid op for result accuracy.
  EXPECT_THAT(ParseAndReturnVerifiedModule(hlo_string).status().message(),
              ::testing::HasSubstr("unexpected attribute \"result_accuracy\""));
}

TEST_F(HloInstructionTest, ResultAccuracyString) {
  ResultAccuracy numeric_accuracy;
  numeric_accuracy.mutable_tolerance()->set_rtol(0.4);
  EXPECT_EQ(ResultAccuracyToleranceToString(numeric_accuracy.tolerance()),
            "tolerance={atol=0,rtol=0.4,ulps=0}");
  ResultAccuracy mode_accuracy;
  mode_accuracy.set_mode(ResultAccuracy::HIGHEST);
  EXPECT_EQ(ResultAccuracyToString(mode_accuracy.mode()), "highest");
}

TEST_F(HloInstructionTest, CreateUnaryWithResultAccuracy) {
  ResultAccuracy result_accuracy;
  result_accuracy.mutable_tolerance()->set_rtol(0.4);
  std::unique_ptr<HloInstruction> unary_inst = HloInstruction::CreateUnary(
      r0f32_, HloOpcode::kExp,
      HloInstruction::CreateParameter(0, r0f32_, "foo").get(), result_accuracy);
  EXPECT_THAT(unary_inst->result_accuracy(), EqualsProto(result_accuracy));
}

TEST_F(HloInstructionTest, PrintUnaryWithResultAccuracy) {
  ResultAccuracy result_accuracy;
  result_accuracy.mutable_tolerance()->set_rtol(0.4);
  HloComputation::Builder builder("Exp");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "x"));
  HloInstruction* exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, x, result_accuracy));
  EXPECT_EQ(exp->ToString(),
            "%exponential = f32[] exponential(%x), "
            "result_accuracy={tolerance={atol=0,rtol=0.4,ulps=0}}");
  EXPECT_TRUE(exp->has_result_accuracy());
  HloInstruction* exp_no_result_accuracy = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, x));
  EXPECT_EQ(exp_no_result_accuracy->ToString(),
            "%exponential = f32[] exponential(%x)");
  EXPECT_FALSE(exp_no_result_accuracy->has_result_accuracy());
  HloInstruction* exp_default_set = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, x));
  // Setting the mode to DEFAULT is the same as not setting it at all.
  exp_default_set->set_result_accuracy(ResultAccuracy());
  EXPECT_EQ(exp_default_set->ToString(),
            "%exponential = f32[] exponential(%x)");
  EXPECT_FALSE(exp_default_set->has_result_accuracy());
}

TEST_F(HloInstructionTest, EqualResultAccuracy) {
  ResultAccuracy result_accuracy_highest;
  result_accuracy_highest.set_mode(ResultAccuracy::HIGHEST);

  HloComputation::Builder builder("Exp");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "x"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, x));
  exp1->set_result_accuracy(result_accuracy_highest);
  HloInstruction* exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, x));
  exp2->set_result_accuracy(result_accuracy_highest);
  EXPECT_TRUE(exp1->equal_result_accuracy(exp2));
}

TEST_F(HloInstructionTest, DifferentResultAccuracy) {
  ResultAccuracy result_accuracy_highest;
  result_accuracy_highest.set_mode(ResultAccuracy::HIGHEST);

  HloComputation::Builder builder("Exp");
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "x"));
  HloInstruction* exp1 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, x));
  exp1->set_result_accuracy(result_accuracy_highest);
  HloInstruction* exp2 = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32_, HloOpcode::kExp, x));

  // Now set exp2 with different result accuracy.
  ResultAccuracy result_accuracy_rtol;
  result_accuracy_rtol.mutable_tolerance()->set_rtol(0.4);
  exp2->set_result_accuracy(result_accuracy_rtol);
  EXPECT_FALSE(exp1->equal_result_accuracy(exp2));
}

}  // namespace
}  // namespace xla
