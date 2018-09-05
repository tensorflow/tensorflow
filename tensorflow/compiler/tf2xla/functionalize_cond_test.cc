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

// Tests for the backward const analysis.

#include "tensorflow/compiler/tf2xla/functionalize_cond.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace functionalize_cond {

class FunctionalizeCondTest : public ::testing::Test {
 protected:
  FunctionalizeCondTest() {
    graph_.reset(new Graph(OpRegistry::Global()));
    flib_def_.reset(
        new FunctionLibraryDefinition(OpRegistry::Global(), fdef_lib_));
    fc_.reset(new functionalize_cond::FunctionalizeCond(graph_.get(),
                                                        flib_def_.get()));
  }

  CondStateMap::CondId GetUniqueId(
      const CondStateMap::CondStateMap::CondState& state) {
    return fc_->cond_state_map_.GetUniqueId(state);
  }

  xla::StatusOr<CondStateMap::CondId> JoinCondStatesNonMerge(
      CondStateMap::CondId src, CondStateMap::CondId dst) {
    return fc_->JoinCondStatesNonMerge(src, dst);
  }

  xla::StatusOr<CondStateMap::CondId> JoinCondStatesMerge(
      CondStateMap::CondId src, CondStateMap::CondId dst) {
    return fc_->JoinCondStatesMerge(src, dst);
  }

  bool ScopeIn(CondStateMap::CondId ff, CondStateMap::CondId* scope) {
    return fc_->cond_state_map_.ScopeIn(ff, scope);
  }

  CondStateMap::ContainsResult LhsHoldsWhereverRhsHolds(
      CondStateMap::CondId lhs, CondStateMap::CondId rhs) {
    return fc_->cond_state_map_.LhsHoldsWhereverRhsHolds(lhs, rhs);
  }

  FunctionDefLibrary fdef_lib_;
  std::unique_ptr<functionalize_cond::FunctionalizeCond> fc_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<Graph> graph_;
};

namespace {

TEST_F(FunctionalizeCondTest, ScopeIn) {
  Tensor pred_tensor(DT_BOOL, TensorShape());
  pred_tensor.flat<bool>().setZero();
  Node* pred = test::graph::Constant(graph_.get(), pred_tensor, "pred");
  Tensor val_tensor(DT_INT32, TensorShape());
  val_tensor.flat<int>().setZero();
  Node* val = test::graph::Constant(graph_.get(), val_tensor, "val");
  Node* s = test::graph::Switch(graph_.get(), val, pred);

  {
    CondStateMap::CondStateMap::CondState ss;
    ss.emplace_back(CondStateMap::CondNode(
        CondStateMap::CondNode::Type::kSwitch, s, BranchType::kThenBranch));
    CondStateMap::CondId id = GetUniqueId(ss);
    CondStateMap::CondId scope;
    ASSERT_TRUE(ScopeIn(id, &scope));
    ASSERT_TRUE(id == scope);
  }

  CondStateMap::CondState empty;
  {
    CondStateMap::CondState ss;
    ss.emplace_back(CondStateMap::CondNode(
        CondStateMap::CondNode::Type::kSwitch, s, BranchType::kBoth));
    ss.emplace_back(
        CondStateMap::CondNode(CondStateMap::CondNode::Type::kMerge));
    CondStateMap::CondId id = GetUniqueId(ss);
    CondStateMap::CondId scope_1;
    ASSERT_TRUE(ScopeIn(id, &scope_1));
    ASSERT_TRUE(scope_1 == GetUniqueId(empty));
    ASSERT_TRUE(id != scope_1);

    ss.clear();
    ss.emplace_back(CondStateMap::CondNode(
        CondStateMap::CondNode::Type::kSwitch, s, BranchType::kBoth));
    id = GetUniqueId(ss);
    CondStateMap::CondId scope_2;
    ASSERT_TRUE(ScopeIn(id, &scope_2));

    ASSERT_TRUE(LhsHoldsWhereverRhsHolds(scope_1, scope_2) ==
                CondStateMap::ContainsResult::kLhsContainsRhs);
  }
}

TEST_F(FunctionalizeCondTest, JoinCondStates) {
  Tensor pred_tensor(DT_BOOL, TensorShape());
  pred_tensor.flat<bool>().setZero();
  Node* pred = test::graph::Constant(graph_.get(), pred_tensor, "pred");
  Tensor val_tensor(DT_INT32, TensorShape());
  val_tensor.flat<int>().setZero();
  Node* val = test::graph::Constant(graph_.get(), val_tensor, "val");
  Node* s = test::graph::Switch(graph_.get(), val, pred);

  CondStateMap::CondId empty = GetUniqueId({});

  CondStateMap::CondId then_branch;
  {
    CondStateMap::CondState ss;
    ss.emplace_back(CondStateMap::CondNode(
        CondStateMap::CondNode::Type::kSwitch, s, BranchType::kThenBranch));
    then_branch = GetUniqueId(ss);
  }
  CondStateMap::CondId else_branch;
  {
    CondStateMap::CondState ss;
    ss.emplace_back(CondStateMap::CondNode(
        CondStateMap::CondNode::Type::kSwitch, s, BranchType::kElseBranch));
    else_branch = GetUniqueId(ss);
  }

  // An non-merge op with inputs from then and else branch.
  Status status = JoinCondStatesNonMerge(then_branch, else_branch).status();
  EXPECT_TRUE(errors::IsInvalidArgument(status));

  // Merge between then and else branch.
  auto joined_or = JoinCondStatesMerge(then_branch, else_branch);
  TF_EXPECT_OK(joined_or.status());
  CondStateMap::CondId joined = joined_or.ValueOrDie();

  // Merge between then branch and both branch.
  auto t = JoinCondStatesNonMerge(then_branch, joined);
  // Note: this is OK in terms of constraint predication, but
  TF_EXPECT_OK(t.status());

  // Post merge the propagated forward flow state has an additional merge.
  CondStateMap::CondId post_merge;
  {
    CondStateMap::CondState ss;
    ss = *joined;
    ss.emplace_back(
        CondStateMap::CondNode(CondStateMap::CondNode::Type::kMerge));
    post_merge = GetUniqueId(ss);
  }

  t = JoinCondStatesNonMerge(post_merge, joined);
  TF_EXPECT_OK(t.status());
  EXPECT_TRUE(joined == t.ValueOrDie());

  // No predicate that results in two paths predicated on different conditions
  // merge.
  t = JoinCondStatesMerge(post_merge, joined);
  EXPECT_FALSE(t.ok());

  // Post the merge we are effectively in the root scope and merging should
  // result in the more restrictive post merge state.
  t = JoinCondStatesNonMerge(post_merge, empty);
  TF_EXPECT_OK(t.status());
  EXPECT_TRUE(post_merge == t.ValueOrDie());
}

}  // namespace
}  // namespace functionalize_cond
}  // namespace tensorflow
