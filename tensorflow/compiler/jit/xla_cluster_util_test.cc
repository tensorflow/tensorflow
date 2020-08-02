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

#include "tensorflow/compiler/jit/xla_cluster_util.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

TEST(CreateCycleDetectionGraph, ConnectivityThroughEnterExitRegion) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter =
      ops::internal::Enter(root.WithOpName("enter"), a, "only_frame");
  Output exit = ops::internal::Exit(root.WithOpName("exit"), enter);
  Output b = ops::Add(root.WithOpName("b"), a, exit);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK(CreateCycleDetectionGraph(root.graph(), &cycles).status());
  EXPECT_FALSE(cycles.CanContractEdge(a.node()->id(), b.node()->id()));
}

TEST(CreateCycleDetectionGraph, ConnectivityThroughMultipleEnterExitRegions) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter_0 =
      ops::internal::Enter(root.WithOpName("enter_0"), a, "frame_0");
  Output exit_0 = ops::internal::Exit(root.WithOpName("exit_0"), enter_0);
  Output enter_1 =
      ops::internal::Enter(root.WithOpName("enter_1"), a, "frame_1");
  Output exit_1 = ops::internal::Exit(root.WithOpName("exit_1"), enter_1);
  Output b = ops::Add(root.WithOpName("b"), a, exit_1);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK(CreateCycleDetectionGraph(root.graph(), &cycles).status());
  EXPECT_FALSE(cycles.CanContractEdge(a.node()->id(), b.node()->id()));
}

TEST(CreateCycleDetectionGraph, ReachingEnterExit) {
  // TODO(b/127521408): We can lift this limitation with some work.
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output enter_0 =
      ops::internal::Enter(root.WithOpName("enter_0"), a, "frame_0");
  Output exit_0 = ops::internal::Exit(root.WithOpName("exit_0"), enter_0);

  Output add = ops::Add(root.WithOpName("add"), exit_0, exit_0);

  Output enter_1 =
      ops::internal::Enter(root.WithOpName("enter_1"), add, "frame_0");
  Output exit_1 = ops::internal::Exit(root.WithOpName("exit_1"), enter_1);

  FixupSourceAndSinkEdges(root.graph());

  GraphCycles cycles;
  TF_ASSERT_OK_AND_ASSIGN(bool ok,
                          CreateCycleDetectionGraph(root.graph(), &cycles));
  EXPECT_FALSE(ok);
}

const char* kCPU0 = "/job:localhost/replica:0/task:0/device:CPU:0";
const char* kGPU0 = "/job:localhost/replica:0/task:0/device:GPU:0";
const char* kGPU1 = "/job:localhost/replica:0/task:0/device:GPU:1";

TEST(IsSingleGpuGraph, ReturnsTrue) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kGPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b"), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_TRUE(IsSingleGpuGraph(*root.graph()));
}

TEST(IsSingleGpuGraph, ReturnsFalseForCpuGraph) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kCPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b"), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_FALSE(IsSingleGpuGraph(*root.graph()));
}

TEST(IsSingleGpuGraph, ReturnsFalseForMultiGpuGraph) {
  Scope root = Scope::NewRootScope().WithAssignedDevice(kGPU0).ExitOnError();

  Output a = ops::Const(root.WithOpName("a"), Input::Initializer(0.0));
  Output b = ops::Add(root.WithOpName("b").WithAssignedDevice(kGPU1), a, a);
  Output c = ops::Add(root.WithOpName("c"), b, b);

  FixupSourceAndSinkEdges(root.graph());

  EXPECT_FALSE(IsSingleGpuGraph(*root.graph()));
}

xla::StatusOr<std::vector<string>> GetNodesRelatedToRefVarsSorted(
    const Scope& scope, FunctionLibraryDefinition* flib_def = nullptr) {
  FunctionDefLibrary flib;
  FunctionLibraryDefinition flib_def_local(OpRegistry::Global(), flib);
  if (flib_def == nullptr) {
    flib_def = &flib_def_local;
  }

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));

  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(
          nullptr, Env::Default(), /*config=*/nullptr, TF_GRAPH_DEF_VERSION,
          flib_def, OptimizerOptions{}));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  TF_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> nodes_related_to_ref_vars,
                      GetNodesRelatedToRefVariables(*graph, lib_runtime));

  std::vector<string> names;
  absl::c_transform(nodes_related_to_ref_vars, std::back_inserter(names),
                    [](Node* n) { return n->name(); });
  absl::c_sort(names);
  return names;
}

void CreateSubgraphTouchingRefVar(const Scope& s) {
  Output variable =
      ops::Variable(s.WithOpName("variable"), PartialTensorShape{}, DT_FLOAT);
  Output read = ops::Identity(s.WithOpName("read_ref_var"), variable);
  Output neg = ops::Negate(s.WithOpName("negate_ref"), read);
  Output add = ops::Add(s.WithOpName("add_ref"), neg, neg);

  Output constant =
      ops::Const(s.WithOpName("constant_ref"), Input::Initializer(0.0));
  s.graph()->AddControlEdge(constant.node(), variable.node());
}

void CreateSubgraphNotTouchingRefVar(const Scope& s) {
  Output constant =
      ops::Const(s.WithOpName("constant_normal"), Input::Initializer(0.0));
  Output neg = ops::Negate(s.WithOpName("negate_normal"), constant);
  Output add = ops::Add(s.WithOpName("add_normal"), neg, neg);
}

void CreateSubgraphCallingFunctionWithRefVar(const Scope& s) {
  NameAttrList ref_float_function;
  ref_float_function.set_name("RefFloatFn");
  ops::PartitionedCall call(s.WithOpName("RefFloat"), {absl::Span<Input>{}},
                            {DT_FLOAT}, ref_float_function);
  Output constant =
      ops::Const(s.WithOpName("constant_ref_pco"), Input::Initializer(0.0));
  s.graph()->AddControlEdge(call.operation.node(), constant.node());
}

void CreateSubgraphCallingFunctionWithoutRefVar(const Scope& s) {
  NameAttrList regular_float_function;
  regular_float_function.set_name("RegularFloatFn");
  ops::PartitionedCall call(s.WithOpName("RegularFloat"), {absl::Span<Input>{}},
                            {DT_FLOAT}, regular_float_function);
  Output constant =
      ops::Const(s.WithOpName("constant_normal_pco"), Input::Initializer(0.0));
  s.graph()->AddControlEdge(call.operation.node(), constant.node());
}

void AddRefFunctionFunctionDef(FunctionDefLibrary* fdef_lib) {
  FunctionDef make_ref_float = FunctionDefHelper::Define(
      "RefFloatFn", {}, {"r:float"}, {},
      {{{"var"},
        "VariableV2",
        {},
        {{"dtype", DT_FLOAT}, {"shape", TensorShape({})}}},
       {{"r"}, "Identity", {"var"}, {{"T", DT_FLOAT}}}});
  *fdef_lib->add_function() = make_ref_float;
}

void AddRegularFunctionFunctionDef(FunctionDefLibrary* fdef_lib) {
  Tensor seven(DT_FLOAT, {});
  seven.scalar<float>()() = 7;
  FunctionDef make_regular_float = FunctionDefHelper::Define(
      "RegularFloatFn", {}, {"r:float"}, {},
      {{{"r"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", seven}}}});
  *fdef_lib->add_function() = make_regular_float;
}

TEST(NodesRelatedToRefVariables, Basic) {
  Scope root = Scope::NewRootScope().ExitOnError();

  FunctionDefLibrary fdef_lib;

  CreateSubgraphTouchingRefVar(root);
  CreateSubgraphNotTouchingRefVar(root);

  AddRefFunctionFunctionDef(&fdef_lib);
  CreateSubgraphCallingFunctionWithRefVar(root);

  AddRegularFunctionFunctionDef(&fdef_lib);
  CreateSubgraphCallingFunctionWithoutRefVar(root);

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<string> names,
                          GetNodesRelatedToRefVarsSorted(root, &flib_def));

  std::vector<string> expected({
      "RefFloat",
      "add_ref",
      "constant_ref",
      "constant_ref_pco",
      "negate_ref",
      "read_ref_var",
      "variable",
  });

  EXPECT_EQ(names, expected);
}

Status MakeLoop(Scope s, Output init_value, absl::string_view loop_name) {
  s = s.NewSubScope(std::string(loop_name));
  ops::internal::Enter enter(s.WithOpName("init_value"), init_value, loop_name);
  ops::Merge merge(s.WithOpName("merge"), {init_value, init_value});
  Output next_iteration =
      ops::NextIteration(s.WithOpName("next_itr"), merge.output);
  return s.graph()->UpdateEdge(next_iteration.node(), 0, merge.output.node(),
                               1);
}

TEST(NodesRelatedToRefVariables, Cycles) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output variable = ops::Variable(root.WithOpName("variable"),
                                  PartialTensorShape{}, DT_FLOAT);
  TF_ASSERT_OK(
      MakeLoop(root, ops::Identity(root.WithOpName("read_ref_var"), variable),
               "ref_loop"));
  TF_ASSERT_OK(MakeLoop(
      root, ops::Const(root.WithOpName("constant"), Input::Initializer(0.0)),
      "normal_loop"));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<string> names,
                          GetNodesRelatedToRefVarsSorted(root));
  std::vector<string> expected({"read_ref_var", "ref_loop/init_value",
                                "ref_loop/merge", "ref_loop/next_itr",
                                "variable"});

  EXPECT_EQ(names, expected);
}
}  // namespace
}  // namespace tensorflow
