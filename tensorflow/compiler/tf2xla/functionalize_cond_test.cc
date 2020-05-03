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

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/types.pb.h"
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

  StateMap::CondId GetUniqueId(const StateMap::StateMap::CondState& state) {
    return fc_->state_map_.GetCondId(state);
  }

  string GetString(const StateMap::StateMap::CondId id) {
    return fc_->state_map_.CondStateToString(id);
  }

  xla::StatusOr<StateMap::CondId> JoinCondStatesNonMerge(StateMap::CondId src,
                                                         StateMap::CondId dst) {
    return fc_->JoinCondStatesNonMerge(src, dst);
  }

  xla::StatusOr<StateMap::CondId> JoinCondStatesMerge(Node* n,
                                                      StateMap::CondId src,
                                                      StateMap::CondId dst) {
    return fc_->JoinCondStatesMerge(n, src, dst);
  }

  FunctionDefLibrary fdef_lib_;
  std::unique_ptr<functionalize_cond::FunctionalizeCond> fc_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<Graph> graph_;
};

namespace {

TEST_F(FunctionalizeCondTest, JoinCondStates) {
  Tensor pred_tensor(DT_BOOL, TensorShape());
  pred_tensor.flat<bool>().setZero();
  Node* pred = test::graph::Constant(graph_.get(), pred_tensor, "pred");
  Tensor val_tensor(DT_INT32, TensorShape());
  val_tensor.flat<int>().setZero();
  Node* val = test::graph::Constant(graph_.get(), val_tensor, "val");
  Node* m = test::graph::Merge(graph_.get(), val, val);

  StateMap::CondId then_branch;
  {
    StateMap::CondState ss;
    ss.insert(std::make_pair(OutputTensor(pred, 0), BranchType::kThenBranch));
    then_branch = GetUniqueId(ss);
  }
  StateMap::CondId else_branch;
  {
    StateMap::CondState ss;
    ss.insert(std::make_pair(OutputTensor(pred, 0), BranchType::kElseBranch));
    else_branch = GetUniqueId(ss);
  }

  // An non-merge op with inputs from then and else branch.
  Status status = JoinCondStatesNonMerge(then_branch, else_branch).status();
  EXPECT_TRUE(errors::IsInvalidArgument(status));

  // Merge between then and else branch.
  auto joined_or = JoinCondStatesMerge(m, then_branch, else_branch);
  TF_EXPECT_OK(joined_or.status());
  StateMap::CondId joined = joined_or.ValueOrDie();

  // Merge between then branch and both branch.
  auto t = JoinCondStatesNonMerge(then_branch, joined);
  // Note: this is OK in terms of constraint predication, but
  TF_EXPECT_OK(t.status());
}

TEST_F(FunctionalizeCondTest, JoinCondStatesMergeWithInputNotInCondContext) {
  Tensor val_tensor(DT_INT32, TensorShape());
  val_tensor.flat<int>().setZero();
  Node* val = test::graph::Constant(graph_.get(), val_tensor, "val");
  Node* m = test::graph::Merge(graph_.get(), val, val);

  StateMap::CondState cond_state;
  auto joined_or = JoinCondStatesMerge(m, /*src=*/nullptr, &cond_state);
  EXPECT_FALSE(joined_or.ok());
}

TEST(FunctionalizeCond, DuplicateConstNodes) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto const_op = ops::Const(root.WithOpName("const"), 1);
  auto arg_0_op = ops::_Arg(root.WithOpName("arg_0"), DT_BOOL, 0);
  auto arg_1_op = ops::_Arg(root.WithOpName("arg_1"), DT_INT32, 1);
  auto switch_op = ops::Switch(root.WithOpName("switch"), arg_1_op, arg_0_op);
  auto identity_n_false_op =
      ops::IdentityN(root.WithOpName("identity_n_0"),
                     {switch_op.output_false, const_op, const_op});
  auto identity_n_true_op =
      ops::IdentityN(root.WithOpName("identity_n_1"),
                     {switch_op.output_true, const_op, const_op});
  auto merge_op = ops::Merge(
      root.WithOpName("merge"),
      {identity_n_false_op.output.front(), identity_n_true_op.output.front()});
  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));

  Graph graph(OpRegistry::Global());
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, &graph));

  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);

  auto status = tensorflow::FunctionalizeCond(&graph, &flib_def);
  TF_ASSERT_OK(status);

  FunctionDefLibrary flib_def_proto = flib_def.ToProto();
  for (const auto& fdef : flib_def_proto.function()) {
    absl::flat_hash_set<absl::string_view> node_names;
    for (const auto& node : fdef.node_def()) {
      EXPECT_TRUE(node_names.insert(node.name()).second)
          << node.op() << " with duplicate node name '" << node.name()
          << "' found.";
    }
  }
}

}  // namespace
}  // namespace functionalize_cond
}  // namespace tensorflow
