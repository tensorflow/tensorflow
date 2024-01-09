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

#include "tensorflow/core/grappler/optimizers/data/function_utils.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace grappler {
namespace function_utils {
namespace {

TEST(FunctionDefTensorDesc, Parsing) {
  FunctionDefTensorDesc f("Cast:y:0");
  EXPECT_EQ(f.full_str, "Cast:y:0");
  EXPECT_EQ(f.node_name, "Cast");
  EXPECT_EQ(f.node_output, "y");
  EXPECT_EQ(f.position, 0);

  FunctionDefTensorDesc f2("Arg0");
  EXPECT_EQ(f2.full_str, "Arg0");
  EXPECT_EQ(f2.node_name, "Arg0");
  EXPECT_EQ(f2.node_output, "");
  EXPECT_EQ(f2.position, -1);
}

TEST(ReplaceReferencesTest, ReplaceReferencesTest) {
  FunctionDef outer = FunctionDefHelper::Create(
      "outer", {"arg0: int32"}, {"out: int32", "out2: int64"}, {}, {},
      {{"out", "MapDefun:output:0"}, {"out2", "Cast:y:0"}});
  NodeDef* derive_node =
      AddNode("X", "Some_Op", {"MapDefun:output:0"}, {}, &outer);
  // Check that both the input to "X" and retval of "outer" are replaced.
  ReplaceReferences("MapDefun:output:0", "arg0", &outer);
  EXPECT_EQ(outer.ret().at("out"), "arg0");
  EXPECT_EQ(derive_node->input(0), "arg0");
}

TEST(FunctionUtilsTest, AddFunctionOutputWithUniqueName) {
  FunctionDef function = test::function::XTimesTwo();
  AddFunctionOutputWithUniqueName("y", "two", &function, DT_INT64);
  EXPECT_TRUE(ContainsFunctionOutputWithName("y/_1", function));
  EXPECT_EQ(function.ret().at("y/_1"), "two");
}

TEST(FunctionUtilsTest, AddFunctionInput) {
  FunctionDef fdef;
  auto arg0 = AddFunctionInput("arg0", &fdef, DT_INT32);
  auto arg1 = AddFunctionInput("arg1", &fdef, DT_BOOL);
  EXPECT_EQ(fdef.signature().input_arg().data()[0], arg0);
  EXPECT_EQ(arg0->name(), "arg0");
  EXPECT_EQ(arg0->type(), DT_INT32);
  EXPECT_EQ(fdef.signature().input_arg().data()[1], arg1);
  EXPECT_EQ(arg1->name(), "arg1");
  EXPECT_EQ(arg1->type(), DT_BOOL);
}

TEST(FunctionUtilsTest, ContainsFunctionNodeWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_FALSE(ContainsFunctionNodeWithName(
      "weird_name_that_should_not_be_there", function));
  EXPECT_TRUE(ContainsFunctionNodeWithName("two", function));
}

TEST(FunctionUtilsTest, ContainsFunctionNodeWithOp) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_FALSE(ContainsFunctionNodeWithOp("weird_op_that_should_not_be_there",
                                          function));
  EXPECT_TRUE(ContainsFunctionNodeWithOp("Mul", function));
}

TEST(FunctionUtilsTest, ContainsFunctionOutputWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_TRUE(ContainsFunctionOutputWithName("y", function));
  EXPECT_FALSE(ContainsFunctionOutputWithName("Add:z:0", function));
}

TEST(FunctionUtilsTest, FindFunctionNodeWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(
      FindFunctionNodeWithName("weird_name_that_should_not_be_there", function),
      -1);
  EXPECT_NE(FindFunctionNodeWithName("two", function), -1);
}

TEST(FunctionUtilsTest, FindFunctionNodeWithOp) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(
      FindFunctionNodeWithOp("weird_op_that_should_not_be_there", function),
      -1);
  EXPECT_NE(FindFunctionNodeWithOp("Mul", function), -1);
}

TEST(FunctionUtilsTest, FindFunctionInputWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(FindFunctionInputWithName("x", function), 0);
  EXPECT_EQ(FindFunctionInputWithName("not_a_name", function), -1);
}

TEST(FunctionUtilsTest, FindFunctionOutputWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(FindFunctionOutputWithName("y", function), 0);
  EXPECT_EQ(FindFunctionOutputWithName("Add:z:0", function), -1);
}

TEST(FunctionUtilsTest, SetUniqueFunctionNodeName) {
  FunctionDef function = test::function::XTimesTwo();
  NodeDef node;
  SetUniqueFunctionNodeName("abc", &function, &node);
  for (const NodeDef& function_node : function.node_def()) {
    EXPECT_NE(node.name(), function_node.name());
  }
  auto* new_node = function.add_node_def();
  *new_node = node;

  NodeDef other;
  SetUniqueFunctionNodeName("abc", &function, &other);
  EXPECT_NE(other.name(), new_node->name());
}

TEST(FunctionUtilsTest, AddNodeToFunctionDef) {
  FunctionDef func;
  const char* op_name = "xxx";
  AddNode(op_name, op_name, {}, {}, &func);

  const NodeDef& node1 = func.node_def(FindFunctionNodeWithName("xxx", func));
  EXPECT_EQ(node1.op(), op_name);
  EXPECT_EQ(node1.input_size(), 0);
  EXPECT_EQ(node1.attr_size(), 0);

  const std::vector<string> inputs({"input1", "input2"});
  AddNode("", op_name, inputs, {}, &func);
  const NodeDef& node2 =
      func.node_def(FindFunctionNodeWithName("xxx/_2", func));
  EXPECT_EQ(node2.op(), op_name);
  EXPECT_EQ(node2.attr_size(), 0);
  EXPECT_EQ(node2.input_size(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(node2.input(i), inputs[i]);
  }

  AttrValue a1, a2;
  a1.set_type(DT_INT32);
  a2.set_type(DT_INT64);
  const std::vector<std::pair<string, AttrValue>> attrs(
      {{"attr1", a1}, {"attr2", a2}});
  AddNode("", op_name, {}, attrs, &func);
  const NodeDef& node3 =
      func.node_def(FindFunctionNodeWithName("xxx/_3", func));
  EXPECT_EQ(node3.op(), op_name);
  EXPECT_EQ(node3.input_size(), 0);
  EXPECT_EQ(node3.attr_size(), attrs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    EXPECT_EQ(attrs[i].second.type(), node3.attr().at(attrs[i].first).type());
  }
}

// Graph containing function with "If" and "Assert" Op.
/*
  @eager_function.defun
  def test_function():
    pred = constant_op.constant(True)

    def fn1():
      return control_flow_ops.no_op()

    def fn2():
      return control_flow_ops.Assert(False, ["Wrong branch!!!"])

    return cond.cond(pred, fn1, fn2)

  r = test_function()
*/
// Following proto is generated in python using the above code block, to
// regenerate get the graph_def from the default graph/specified graph for the
// code block (e.g ops.get_default_graph.as_graph_def()).
constexpr char kCondGraphProto[] = R"proto(
  node {
    name: "StatefulPartitionedCall"
    op: "StatefulPartitionedCall"
    attr {
      key: "Tin"
      value { list {} }
    }
    attr {
      key: "Tout"
      value { list { type: DT_BOOL } }
    }
    attr {
      key: "_gradient_op_type"
      value { s: "PartitionedCall-20" }
    }
    attr {
      key: "config"
      value { s: "" }
    }
    attr {
      key: "config_proto"
      value { s: "" }
    }
    attr {
      key: "executor_type"
      value { s: "" }
    }
    attr {
      key: "f"
      value { func { name: "__inference_test_function_19" } }
    }
  }
  library {
    function {
      signature {
        name: "cond_true_3"
        input_arg { name: "identity_const" type: DT_BOOL }
        output_arg { name: "identity_1" type: DT_BOOL }
      }
      node_def { name: "NoOp" op: "NoOp" }
      node_def {
        name: "Identity"
        op: "Identity"
        input: "identity_const"
        input: "^NoOp"
        attr {
          key: "T"
          value { type: DT_BOOL }
        }
      }
      node_def {
        name: "Identity_1"
        op: "Identity"
        input: "Identity:output:0"
        attr {
          key: "T"
          value { type: DT_BOOL }
        }
      }
      ret { key: "identity_1" value: "Identity_1:output:0" }
    }
    function {
      signature {
        name: "cond_false_4"
        input_arg { name: "identity_const" type: DT_BOOL }
        output_arg { name: "identity_1" type: DT_BOOL }
        is_stateful: true
      }
      node_def {
        name: "Assert/Const"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_STRING }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_STRING
              tensor_shape {}
              string_val: "Wrong branch!!!"
            }
          }
        }
      }
      node_def {
        name: "Assert/Assert/condition"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_BOOL }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_BOOL
              tensor_shape {}
              bool_val: false
            }
          }
        }
      }
      node_def {
        name: "Assert/Assert/data_0"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_STRING }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_STRING
              tensor_shape {}
              string_val: "Wrong branch!!!"
            }
          }
        }
      }
      node_def {
        name: "Assert/Assert"
        op: "Assert"
        input: "Assert/Assert/condition:output:0"
        input: "Assert/Assert/data_0:output:0"
        attr {
          key: "T"
          value { list { type: DT_STRING } }
        }
        attr {
          key: "summarize"
          value { i: 3 }
        }
      }
      node_def {
        name: "Identity"
        op: "Identity"
        input: "identity_const"
        input: "^Assert/Assert"
        attr {
          key: "T"
          value { type: DT_BOOL }
        }
      }
      node_def {
        name: "Identity_1"
        op: "Identity"
        input: "Identity:output:0"
        input: "^Assert/Assert"
        attr {
          key: "T"
          value { type: DT_BOOL }
        }
      }
      ret { key: "identity_1" value: "Identity_1:output:0" }
    }
    function {
      signature {
        name: "__inference_test_function_19"
        output_arg { name: "identity" type: DT_BOOL }
        is_stateful: true
      }
      node_def {
        name: "Const"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_BOOL }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_BOOL
              tensor_shape {}
              bool_val: true
            }
          }
        }
      }
      node_def {
        name: "cond"
        op: "If"
        input: "Const:output:0"
        input: "Const:output:0"
        attr {
          key: "Tcond"
          value { type: DT_BOOL }
        }
        attr {
          key: "Tin"
          value { list { type: DT_BOOL } }
        }
        attr {
          key: "Tout"
          value { list { type: DT_BOOL } }
        }
        attr {
          key: "_lower_using_switch_merge"
          value { b: true }
        }
        attr {
          key: "else_branch"
          value { func { name: "cond_false_4" } }
        }
        attr {
          key: "output_shapes"
          value { list { shape {} } }
        }
        attr {
          key: "then_branch"
          value { func { name: "cond_true_3" } }
        }
      }
      node_def {
        name: "cond/Identity"
        op: "Identity"
        input: "cond:output:0"
        attr {
          key: "T"
          value { type: DT_BOOL }
        }
      }
      node_def {
        name: "Identity"
        op: "Identity"
        input: "cond/Identity:output:0"
        input: "^cond"
        attr {
          key: "T"
          value { type: DT_BOOL }
        }
      }
      ret { key: "identity" value: "Identity:output:0" }
    }
  }
  versions { producer: 27 min_consumer: 12 })proto";

// Graph containing function with "While" Op in python.
/*
  @eager_function.defun
  def test_function():
    return control_flow_ops.while_loop(
        lambda i: i < 3, lambda i: i + 1, [0], maximum_iterations=1)

  r = test_function()
*/
// Following proto is generated in python using the above code block, to
// regenerate get the graph_def from the default graph/specified graph for the
// code block (e.g ops.get_default_graph.as_graph_def()).
constexpr char kWhileGraphProto[] = R"proto(
  node {
    name: "StatefulPartitionedCall"
    op: "StatefulPartitionedCall"
    attr {
      key: "Tin"
      value { list {} }
    }
    attr {
      key: "Tout"
      value { list { type: DT_INT32 } }
    }
    attr {
      key: "_gradient_op_type"
      value { s: "PartitionedCall-35" }
    }
    attr {
      key: "config"
      value { s: "" }
    }
    attr {
      key: "config_proto"
      value { s: "" }
    }
    attr {
      key: "executor_type"
      value { s: "" }
    }
    attr {
      key: "f"
      value { func { name: "__inference_test_function_34" } }
    }
  }
  library {
    function {
      signature {
        name: "while_body_5"
        input_arg { name: "while_loop_counter" type: DT_INT32 }
        input_arg { name: "const" type: DT_INT32 }
        input_arg { name: "maximum_iterations" type: DT_INT32 }
        output_arg { name: "identity" type: DT_INT32 }
        output_arg { name: "identity_1" type: DT_INT32 }
        output_arg { name: "identity_2" type: DT_INT32 }
      }
      node_def {
        name: "add/y"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_INT32 }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_INT32
              tensor_shape {}
              int_val: 1
            }
          }
        }
      }
      node_def {
        name: "add"
        op: "Add"
        input: "const"
        input: "add/y:output:0"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "add_1/y"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_INT32 }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_INT32
              tensor_shape {}
              int_val: 1
            }
          }
        }
      }
      node_def {
        name: "add_1"
        op: "Add"
        input: "while_loop_counter"
        input: "add_1/y:output:0"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "Identity"
        op: "Identity"
        input: "add_1:z:0"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "Identity_1"
        op: "Identity"
        input: "add:z:0"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "Identity_2"
        op: "Identity"
        input: "maximum_iterations"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      ret { key: "identity" value: "Identity:output:0" }
      ret { key: "identity_1" value: "Identity_1:output:0" }
      ret { key: "identity_2" value: "Identity_2:output:0" }
    }
    function {
      signature {
        name: "__inference_test_function_34"
        output_arg { name: "identity" type: DT_INT32 }
        is_stateful: true
      }
      node_def {
        name: "maximum_iterations"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_INT32 }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_INT32
              tensor_shape {}
              int_val: 1
            }
          }
        }
      }
      node_def {
        name: "Const"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_INT32 }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_INT32
              tensor_shape {}
              int_val: 0
            }
          }
        }
      }
      node_def {
        name: "while/loop_counter"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_INT32 }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_INT32
              tensor_shape {}
              int_val: 0
            }
          }
        }
      }
      node_def {
        name: "while"
        op: "While"
        input: "while/loop_counter:output:0"
        input: "Const:output:0"
        input: "maximum_iterations:output:0"
        attr {
          key: "T"
          value { list { type: DT_INT32 type: DT_INT32 type: DT_INT32 } }
        }
        attr {
          key: "_lower_using_switch_merge"
          value { b: true }
        }
        attr {
          key: "body"
          value { func { name: "while_body_5" } }
        }
        attr {
          key: "cond"
          value { func { name: "while_cond_4" } }
        }
        attr {
          key: "output_shapes"
          value {
            list {
              shape {}
              shape {}
              shape {}
            }
          }
        }
      }
      node_def {
        name: "while/Identity"
        op: "Identity"
        input: "while:output:0"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "while/Identity_1"
        op: "Identity"
        input: "while:output:1"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "while/Identity_2"
        op: "Identity"
        input: "while:output:2"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "Identity"
        op: "Identity"
        input: "while/Identity_1:output:0"
        input: "^while"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      ret { key: "identity" value: "Identity:output:0" }
    }
    function {
      signature {
        name: "while_cond_4"
        input_arg { name: "while_loop_counter" type: DT_INT32 }
        input_arg { name: "const" type: DT_INT32 }
        input_arg { name: "less_maximum_iterations" type: DT_INT32 }
        output_arg { name: "identity" type: DT_BOOL }
      }
      node_def {
        name: "Less"
        op: "Less"
        input: "while_loop_counter"
        input: "less_maximum_iterations"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "Less_1/y"
        op: "Const"
        attr {
          key: "dtype"
          value { type: DT_INT32 }
        }
        attr {
          key: "value"
          value {
            tensor {
              dtype: DT_INT32
              tensor_shape {}
              int_val: 3
            }
          }
        }
      }
      node_def {
        name: "Less_1"
        op: "Less"
        input: "const"
        input: "Less_1/y:output:0"
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
      }
      node_def {
        name: "LogicalAnd"
        op: "LogicalAnd"
        input: "Less:z:0"
        input: "Less_1:z:0"
      }
      node_def {
        name: "Identity"
        op: "Identity"
        input: "LogicalAnd:z:0"
        attr {
          key: "T"
          value { type: DT_BOOL }
        }
      }
      ret { key: "identity" value: "Identity:output:0" }
    }
  }
  versions { producer: 27 min_consumer: 12 })proto";

// TODO(shivaniagrawal): split the test into multiple tests for better
// readability and add full coverage i.e. add/separate out the tests for all
// branches of IsNodeStateful and IsFunctionStateful:
// - test for IsNodeStateful for Cond that has a stateful branch
// - test for IsNodeStateful for Cond that does not have a stateful branches
// - test for IsNodeStateful for While that has a stateful branch
// - test for IsNodeStateful for While that does not have a stateful branches
// - test for IsNodeStateful for Assert
// - test for IsNodeStateful for a stateful op
// - test for IsNodeStateful for a stateless op
//
// - test for IsFunctionStateful for a function that contains a Cond
// - test for IsFunctionStateful for a function that contains a While
// - test for IsFunctionStateful for a function that contains an Assert (and no
//   other stateful op)
// - test for IsFunctionStateful for a function that contains a stateful op
//   other than Assert
// - test for IsFunctionStateful for a function that does not contain a stateful
//   op

TEST(FunctionUtilsTest, IsFunctionStateful) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);

  NodeDef* nodeA = graph_utils::AddNode("", "A", {}, {}, &graph);
  FunctionDef* function = graph_def.mutable_library()->add_function();
  *function = test::function::XTimesTwo();

  FunctionLibraryDefinition lib_def(OpRegistry::Global(),
                                    *graph_def.mutable_library());

  EXPECT_FALSE(IsFunctionStateful(lib_def, *function));

  // Op "A" is not a registered Op.
  EXPECT_TRUE(IsNodeStateful(lib_def, *nodeA));

  // Get graph_def for the graph `kCondGraphProto`, graph with function
  // containing "If" and "Assert" Op.

  GraphDef graph_def_cond;
  protobuf::TextFormat::ParseFromString(kCondGraphProto, &graph_def_cond);
  FunctionLibraryDefinition cond_lib(OpRegistry::Global(),
                                     graph_def_cond.library());

  const FunctionDef* no_op_fnc = cond_lib.Find("cond_true_3");

  EXPECT_FALSE(IsFunctionStateful(cond_lib, *no_op_fnc));
  EXPECT_FALSE(IsFunctionStateful(cond_lib, *no_op_fnc, true));

  const FunctionDef* assert_func = cond_lib.Find("cond_false_4");

  EXPECT_TRUE(IsFunctionStateful(cond_lib, *assert_func));
  EXPECT_FALSE(IsFunctionStateful(cond_lib, *assert_func, true));

  EXPECT_TRUE(ContainsFunctionNodeWithOp("Const", *assert_func));
  EXPECT_TRUE(ContainsFunctionNodeWithOp("Assert", *assert_func));

  for (auto node : assert_func->node_def()) {
    if (node.op() == "Const") {
      EXPECT_FALSE(IsNodeStateful(lib_def, node));
    }
    if (node.op() == "Assert") {
      EXPECT_TRUE(IsNodeStateful(lib_def, node));
      EXPECT_FALSE(IsNodeStateful(lib_def, node, true));
    }
  }

  const FunctionDef* cond_func = cond_lib.Find("__inference_test_function_19");

  EXPECT_TRUE(IsFunctionStateful(cond_lib, *cond_func));
  EXPECT_FALSE(IsFunctionStateful(cond_lib, *cond_func, true));

  // Get graph def for the graph `kWhileGraphProto`, graph with function
  // containing "While" Op.

  GraphDef graph_def_while;
  protobuf::TextFormat::ParseFromString(kWhileGraphProto, &graph_def_while);

  FunctionLibraryDefinition while_lib(OpRegistry::Global(),
                                      graph_def_while.library());
  const FunctionDef* while_function =
      while_lib.Find("__inference_test_function_34");
  EXPECT_FALSE(IsFunctionStateful(while_lib, *while_function));
  EXPECT_FALSE(IsFunctionStateful(while_lib, *while_function, true));
}
}  // namespace
}  // namespace function_utils
}  // namespace grappler
}  // namespace tensorflow
