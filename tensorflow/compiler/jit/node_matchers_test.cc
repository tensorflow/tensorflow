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

#include "tensorflow/compiler/jit/node_matchers.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"

namespace tensorflow {
namespace testing {
namespace {

using ::testing::_;

using testing::matchers::AssignedDevice;
using testing::matchers::ConstantValue;
using testing::matchers::CtrlDeps;
using testing::matchers::Inputs;
using testing::matchers::Name;
using testing::matchers::NodeWith;
using testing::matchers::Op;

template <typename M, typename T>
string Explain(const T& t, const M& m) {
  ::testing::StringMatchResultListener listener;
  EXPECT_THAT(t, ::testing::Not(m));  // For the error message.
  EXPECT_FALSE(m.MatchAndExplain(t, &listener));
  return listener.str();
}

TEST(NodeMatchers, CheckAgainstConstant) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output placeholder =
      ops::Placeholder(root.WithOpName("placeholder"), DT_FLOAT);

  EXPECT_THAT(placeholder.node(), NodeWith(Op("Placeholder")));
  EXPECT_THAT(placeholder.node(), NodeWith(Name("placeholder")));
  EXPECT_THAT(placeholder.node(),
              NodeWith(Op("Placeholder"), Name("placeholder")));
  EXPECT_THAT(placeholder.node(),
              NodeWith(Name("placeholder"), Op("Placeholder")));
  EXPECT_THAT(placeholder.node(), NodeWith(Inputs()));
  EXPECT_THAT(placeholder.node(),
              NodeWith(Op("Placeholder"), Name("placeholder"), Inputs()));

  EXPECT_EQ(Explain(placeholder.node(), NodeWith(Op("Add"))),
            "\nexpected op Add but found Placeholder");
  EXPECT_EQ(Explain(placeholder.node(), NodeWith(Name("add"))),
            "\nexpected name add but found placeholder");
  EXPECT_EQ(Explain(placeholder.node(), NodeWith(Inputs(NodeWith()))),
            "\nexpected 1 inputs but node has 0");
}

TEST(NodeMatchers, CheckAgainstBinary) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output placeholder_a =
      ops::Placeholder(root.WithOpName("placeholder_a"), DT_FLOAT);
  Output placeholder_b =
      ops::Placeholder(root.WithOpName("placeholder_b"), DT_FLOAT);
  Output add = ops::Add(root.WithOpName("add"), placeholder_a, placeholder_b);

  EXPECT_THAT(add.node(), NodeWith(Op("Add"), Name("add"),
                                   Inputs(NodeWith(Name("placeholder_a")),
                                          NodeWith(Name("placeholder_b")))));

  EXPECT_EQ(Explain(add.node(), NodeWith(Inputs())),
            "\nexpected 0 inputs but node has 2");
  EXPECT_EQ(
      Explain(add.node(), NodeWith(Inputs(NodeWith(Name("blah")), _))),
      "\ninput 0 does not match expected:\nname: blah, \nsource does not match "
      "expected name: blah\n\t\nexpected name blah but found placeholder_a");
  EXPECT_EQ(
      Explain(add.node(), NodeWith(Inputs(_, NodeWith(Name("blah"))))),
      "\ninput 1 does not match expected:\nname: blah, \nsource does not match "
      "expected name: blah\n\t\nexpected name blah but found placeholder_b");
}

TEST(NodeMatchers, CheckControlDependence) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output placeholder_a =
      ops::Placeholder(root.WithOpName("placeholder_a"), DT_FLOAT);
  Output placeholder_b =
      ops::Placeholder(root.WithOpName("placeholder_b"), DT_FLOAT);
  Output placeholder_c =
      ops::Placeholder(root.WithOpName("placeholder_c"), DT_FLOAT);
  Output placeholder_d =
      ops::Placeholder(root.WithOpName("placeholder_d"), DT_FLOAT);

  root.graph()->AddControlEdge(placeholder_a.node(), placeholder_c.node());
  root.graph()->AddControlEdge(placeholder_b.node(), placeholder_c.node());

  EXPECT_THAT(placeholder_c.node(),
              NodeWith(Name("placeholder_c"),
                       CtrlDeps(NodeWith(Name("placeholder_a")),
                                NodeWith(Name("placeholder_b")))));
  EXPECT_THAT(placeholder_d.node(),
              NodeWith(Name("placeholder_d"), CtrlDeps()));

  EXPECT_EQ(
      Explain(placeholder_c.node(), NodeWith(CtrlDeps())),
      "ctrl_deps, which has 2 elements, does not match expected: is empty");
  EXPECT_EQ(Explain(placeholder_d.node(), NodeWith(CtrlDeps(NodeWith()))),
            "ctrl_deps does not match expected: has 1 element and that element "
            "is any node");
}

TEST(NodeMatchers, ConstVaulue) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Output placeholder =
      ops::Placeholder(root.WithOpName("placeholder"), DT_FLOAT);
  Output const_0d = ops::Const(root.WithOpName("const_0d"), 42);

  Output const_2d = ops::Const(root.WithOpName("const_2d"), {{1, 2}, {4, 3}});

  EXPECT_THAT(const_0d.node(), NodeWith(ConstantValue(42)));
  EXPECT_THAT(const_0d.node(), NodeWith(ConstantValue(42), Name("const_0d")));

  EXPECT_THAT(const_2d.node(), NodeWith(ConstantValue({{1, 2}, {4, 3}})));

  EXPECT_EQ(Explain(placeholder.node(), NodeWith(ConstantValue(42))),
            "\nexpected op Const but found Placeholder");
  EXPECT_EQ(
      Explain(const_0d.node(), NodeWith(ConstantValue(43))),
      "\nmismatch in constant tensor at index 0 expected = 43 actual = 42");
  EXPECT_EQ(
      Explain(const_0d.node(), NodeWith(ConstantValue({{1, 2}, {4, 3}}))),
      "\nwas looking for tensor with 4 elements, found tensor with 1 elements");
  EXPECT_EQ(
      Explain(const_2d.node(), NodeWith(ConstantValue(42))),
      "\nwas looking for tensor with 1 elements, found tensor with 4 elements");
}

TEST(NodeMatchers, AssignedDevice) {
  Scope root = Scope::NewRootScope().ExitOnError();

  Output placeholder_a =
      ops::Placeholder(root.WithOpName("placeholder_a"), DT_FLOAT);
  Output placeholder_b =
      ops::Placeholder(root.WithOpName("placeholder_b"), DT_FLOAT);

  Output assigned_add =
      ops::Add(root.WithOpName("assigned_add"), placeholder_a, placeholder_b);
  assigned_add.node()->set_assigned_device_name(
      "/job:localhost/replica:0/task:0/device:CPU:0");

  Output unassigned_add =
      ops::Add(root.WithOpName("unassigned_add"), placeholder_a, placeholder_b);

  EXPECT_THAT(
      assigned_add.node(),
      NodeWith(AssignedDevice("/job:localhost/replica:0/task:0/device:CPU:0")));
  EXPECT_THAT(unassigned_add.node(), NodeWith(AssignedDevice("")));

  EXPECT_EQ(Explain(unassigned_add.node(),
                    NodeWith(AssignedDevice(
                        "/job:localhost/replica:0/task:0/device:CPU:0"))),
            "\nexpected assigned_device "
            "/job:localhost/replica:0/task:0/device:CPU:0 but found \"\"");
}

}  // namespace
}  // namespace testing
}  // namespace tensorflow
