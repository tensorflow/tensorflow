/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/graph_rewrite/variable_merger_pass.h"

#include <map>
#include <vector>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

TEST(VarHandleMerger, SimpleMergesWork) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto v = ops::VarHandleOp(
        scope.WithOpName("V"), DT_FLOAT, TensorShape({}),
        ops::VarHandleOp::Attrs().Container("c1").SharedName("n1"));
    auto w = ops::VarHandleOp(
        scope.WithOpName("W"), DT_INT32, TensorShape({77}),
        ops::VarHandleOp::Attrs().Container("c2").SharedName("n2"));
    auto v_read = ops::ReadVariableOp(scope.WithOpName("VRead"), v, DT_FLOAT);
    auto w_read = ops::ReadVariableOp(scope.WithOpName("WRead"), w, DT_INT32);
    auto w_cast = ops::Cast(scope.WithOpName("Cast"), w_read, DT_FLOAT);
    ops::Sub(scope.WithOpName("Sub"), v_read, w_cast);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  VariableMergerPass merger;
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  TF_ASSERT_OK(merger.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  GraphDef expected;
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto handles = ops::_VarHandlesOp(
        scope.WithOpName("VarHandles_10315266686041849873/_0"),
        /*containers=*/{"c1", "c2"},
        /*shared_names=*/{"n1", "n2"}, /*N=*/2, /*dtypes=*/{DT_FLOAT, DT_INT32},
        /*shapes=*/{TensorShape({}), TensorShape({77})});
    auto read = ops::_ReadVariablesOp(
        scope.WithOpName("ReadVariables_13269360303885824085/_1"),
        /*resources=*/{handles[0], handles[1]},
        /*dtypes=*/{DT_FLOAT, DT_INT32});
    auto w_cast = ops::Cast(scope.WithOpName("Cast"), read[1], DT_FLOAT);
    ops::Sub(scope.WithOpName("Sub"), read[0], w_cast);
    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  TF_EXPECT_GRAPH_EQ(expected, actual);
}

TEST(VarHandleMerger, VarHandlesWithControlDepsDontMerge) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto v = ops::VarHandleOp(
        scope.WithOpName("V"), DT_FLOAT, TensorShape({}),
        ops::VarHandleOp::Attrs().Container("c1").SharedName("n1"));
    auto w = ops::VarHandleOp(
        scope.WithOpName("W").WithControlDependencies(v), DT_INT32,
        TensorShape({77}),
        ops::VarHandleOp::Attrs().Container("c2").SharedName("n2"));
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  GraphDef expected;
  graph->ToGraphDef(&expected);

  VariableMergerPass merger;
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  TF_ASSERT_OK(merger.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  TF_EXPECT_GRAPH_EQ(expected, actual);
}

TEST(VarHandleMerger, ReadVariableOpsWithDifferentControlDepsDontMerge) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto w = ops::VarHandleOp(
        scope.WithOpName("W"), DT_INT32, TensorShape({77}),
        ops::VarHandleOp::Attrs().Container("c2").SharedName("n2"));
    auto v = ops::VarHandleOp(
        scope.WithOpName("V"), DT_FLOAT, TensorShape({}),
        ops::VarHandleOp::Attrs().Container("c1").SharedName("n1"));
    auto w_read = ops::ReadVariableOp(scope.WithOpName("WRead"), w, DT_INT32);
    auto w_cast = ops::Cast(scope.WithOpName("Cast"), w_read, DT_FLOAT);
    auto v_read = ops::ReadVariableOp(
        scope.WithOpName("VRead").WithControlDependencies(w_cast), v, DT_FLOAT);
    ops::Sub(scope.WithOpName("Sub"), v_read, w_cast);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  VariableMergerPass merger;
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  TF_ASSERT_OK(merger.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  GraphDef expected;
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto handles = ops::_VarHandlesOp(
        scope.WithOpName("VarHandles_10315266686041849873/_0"),
        /*containers=*/{"c1", "c2"},
        /*shared_names=*/{"n1", "n2"}, /*N=*/2, /*dtypes=*/{DT_FLOAT, DT_INT32},
        /*shapes=*/{TensorShape({}), TensorShape({77})});
    auto w_read =
        ops::ReadVariableOp(scope.WithOpName("WRead"), handles[1], DT_INT32);
    auto w_cast = ops::Cast(scope.WithOpName("Cast"), w_read, DT_FLOAT);
    auto v_read = ops::ReadVariableOp(
        scope.WithOpName("VRead").WithControlDependencies(w_cast), handles[0],
        DT_FLOAT);
    ops::Sub(scope.WithOpName("Sub"), v_read, w_cast);
    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  TF_EXPECT_GRAPH_EQ(expected, actual);
}

TEST(VarHandleMerger, ReadVariableOpsWithSameControlDepsMerge) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto u = ops::VarHandleOp(
        scope.WithOpName("U"), DT_FLOAT, TensorShape({}),
        ops::VarHandleOp::Attrs().Container("c1").SharedName("n1"));
    auto v = ops::VarHandleOp(
        scope.WithOpName("V"), DT_FLOAT, TensorShape({}),
        ops::VarHandleOp::Attrs().Container("c2").SharedName("n2"));
    auto w = ops::VarHandleOp(
        scope.WithOpName("W"), DT_INT32, TensorShape({77}),
        ops::VarHandleOp::Attrs().Container("c3").SharedName("n3"));

    auto w_read = ops::ReadVariableOp(scope.WithOpName("WRead"), w, DT_INT32);
    auto w_cast = ops::Cast(scope.WithOpName("Cast"), w_read, DT_FLOAT);
    auto v_read = ops::ReadVariableOp(
        scope.WithOpName("VRead").WithControlDependencies(w_cast), v, DT_FLOAT);
    auto u_read = ops::ReadVariableOp(
        scope.WithOpName("URead").WithControlDependencies(w_cast), u, DT_FLOAT);
    auto d = ops::Sub(scope.WithOpName("Sub"), v_read, w_cast);
    ops::Sub(scope.WithOpName("Add"), d, u_read);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  VariableMergerPass merger;
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  TF_ASSERT_OK(merger.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  GraphDef expected;
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto handles = ops::_VarHandlesOp(
        scope.WithOpName("VarHandles_15520412301618992443/_0"),
        /*containers=*/{"c1", "c2", "c3"},
        /*shared_names=*/{"n1", "n2", "n3"}, /*N=*/3,
        /*dtypes=*/{DT_FLOAT, DT_FLOAT, DT_INT32},
        /*shapes=*/{TensorShape({}), TensorShape({}), TensorShape({77})});
    auto w_read =
        ops::ReadVariableOp(scope.WithOpName("WRead"), handles[2], DT_INT32);
    auto w_cast = ops::Cast(scope.WithOpName("Cast"), w_read, DT_FLOAT);
    auto read = ops::_ReadVariablesOp(
        scope.WithOpName("ReadVariables_8281595736094071329/_1")
            .WithControlDependencies(w_cast),
        /*resources=*/{handles[0], handles[1]},
        /*dtypes=*/{DT_FLOAT, DT_FLOAT});
    auto d = ops::Sub(scope.WithOpName("Sub"), read[1], w_cast);
    ops::Sub(scope.WithOpName("Add"), d, read[0]);
    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  TF_EXPECT_GRAPH_EQ(expected, actual);
}
}  // namespace
}  // namespace tensorflow
