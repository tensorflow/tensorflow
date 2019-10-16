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

#include "tensorflow/core/grappler/optimizers/data/vectorization_utils.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace grappler {
namespace vectorization_utils {
namespace {

// Wraps a function in another function with a MapDefun node
Status WrapFunctionWithMapDefun(const FunctionDef& inner, FunctionDef* result) {
  Graph graph(OpRegistry::Global());
  std::vector<NodeBuilder::NodeOut> inputs;
  inputs.reserve(inner.signature().input_arg_size());
  for (int i = 0; i < inner.signature().input_arg_size(); ++i) {
    Node* arg;
    TF_RETURN_IF_ERROR(
        NodeBuilder(strings::StrCat("arg", i), /*op_name=*/"_Arg")
            .Attr("T", inner.signature().input_arg(i).type())
            .Attr("index", i)
            .Finalize(&graph, &arg));
    inputs.push_back(arg);
  }

  DataTypeVector output_types;
  output_types.reserve(inner.signature().output_arg_size());
  for (const auto& output_arg : inner.signature().output_arg()) {
    output_types.push_back(output_arg.type());
  }

  Node* map_defun_node;
  NameAttrList func_attr;
  func_attr.set_name(inner.signature().name());
  TF_RETURN_IF_ERROR(
      NodeBuilder("map_defun", "MapDefun")
          .Input(inputs)                               // arguments
          .Input(std::vector<NodeBuilder::NodeOut>())  // captured_inputs
          .Attr("f", func_attr)
          .Attr("output_types", output_types)
          .Attr("output_shapes", std::vector<PartialTensorShape>(
                                     inner.signature().output_arg_size()))
          .Finalize(&graph, &map_defun_node));

  for (size_t i = 0; i < map_defun_node->num_outputs(); ++i) {
    Node* ret;
    TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat("ret", i), "_Retval")
                           .Input(map_defun_node, i)
                           .Attr("index", static_cast<int>(i))
                           .Finalize(&graph, &ret));
  }

  return GraphToFunctionDef(graph, "outer_function", result);
}

// Wraps the function `fn` in another function with a MapDefun node, then
// vectorizes the wrapper function with VectorizeMapDefun.
Status WrapAndVectorize(const FunctionDef& fn, FunctionDefLibrary* lib,
                        FunctionDef** result) {
  FunctionDef outer;
  TF_RETURN_IF_ERROR(WrapFunctionWithMapDefun(fn, &outer));
  const NodeDef& map_defun_node = outer.node_def(0);

  *lib->add_function() = outer;
  *lib->add_function() = fn;

  TF_RETURN_IF_ERROR(VectorizeMapDefun(outer, map_defun_node, lib, result));

  return Status::OK();
}

FunctionDefHelper::Node Cast(string&& name, std::vector<string>&& inputs,
                             DataType src, DataType dst) {
  return {{name},
          "Cast",
          inputs,
          {{"SrcT", src}, {"DstT", dst}, {"Truncate", false}}};
}

string GetRetval(const FunctionDef& function_def, int index) {
  return function_def.ret().at(
      function_def.signature().output_arg(index).name());
}

///==================================//
// Tests for vectorization framework //
///==================================//

// Before:
//
//                 +------+   +------+
// +---------------+ Arg0 +---+ Arg1 +--------+
// |               +---+--+   +---+--+        |
// |                   |          |           |
// |               +---v--+   +---v--+        |
// |   +-----------+ Arg0 +---+ Arg1 +----+   |
// |   |           +---+--+   +---+--+    |   |
// |   |               |          |       |   |
// |   | MapDefun  +---v--+   +---v--+    |   |
// |   +-----------+ Ret0 +---+ Ret1 +----+   |
// |               +---+--+   +---+--+        |
// |                   |          |           |
// |               +---v--+   +---v--+        |
// +---------------+ Ret0 +---+ Ret1 +--------+
//                 +------+   +------+
//
//
//  After:
//
//                 +------+   +------+
// +---------------+ Arg0 +---+ Arg1 +--------+
// |               +---+--+   +---+--+        |
// |                   |          |           |
// |                   |          |           |
// |                   |          |           |
// |               +---v--+   +---v--+        |
// +---------------+ Ret0 +---+ Ret1 +--------+
//                 +------+   +------+
//
TEST(VectorizeMapDefunTest, VectorizeWithNoOps) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32", "arg1: int32"},
      /*out_def=*/{"ret0: int32", "ret1: int32"},
      /*attr_def=*/{},
      /*node_def=*/{},
      /*ret_def=*/{{"ret0", "arg0"}, {"ret1", "arg1"}});
  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));

  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  EXPECT_EQ(GetRetval(*vectorized, 0),
            vectorized->signature().input_arg(0).name());
  EXPECT_EQ(GetRetval(*vectorized, 1),
            vectorized->signature().input_arg(1).name());
}

// Before:
//
//                 +------+   +------+
// +---------------+ Arg0 +---+ Arg1 +--------+
// |               +---+--+   +---+--+        |
// |                   |          |           |
// |               +---v--+   +---v--+        |
// |   +-----------+ Arg0 +---+ Arg1 +----+   |
// |   |           +---+--+   +---+--+    |   |
// |   |               |          |       |   |
// |   |   +------+    |          |       |   |
// |   |   |Const |    |          |       |   |
// |   |   +---v--+    |          |       |   |
// |   |       |       |          |       |   |
// |   |       |   +---v--+   +---v--+    |   |
// |   |       +---| XOp1 |   | Cast |    |   |
// |   |           +---+--+   +---+--+    |   |
// |   |               |          |       |   |
// |   | MapDefun  +---v--+   +---v--+    |   |
// |   +-----------+ Ret0 +---+ Ret1 +----+   |
// |               +---+--+   +---+--+        |
// |                   |          |           |
// |               +---v--+   +---v--+        |
// +---------------+ Ret0 +---+ Ret1 +--------+
//                 +------+   +------+
//
//   where XOp1 does not have a vectorizer defined.
//
// After:
//
//
//                 +------+   +------+
// +---------------+ Arg0 +---+ Arg1 +--------+
// |               +---+--+   +---+--+        |
// |                   |          |           |
// |               +---v--+       |           |
// |   +-----------+ Arg0 +-+     |           |
// |   |           +---+--+ |     |           |
// |   |               |    |     |           |
// |   |   +------+    |    |     |           |
// |   |   |Const |    |    |     |           |
// |   |   +---v--+    |    |     |           |
// |   |       |       |    |     |           |
// |   |       |   +---v--+ | +---v--+        |
// |   |       +---| XOp1 | | | Cast |        |
// |   |           +---+--+ | +---+--+        |
// |   |               |    |     |           |
// |   | MapDefun  +---v--+ |     |           |
// |   +-----------+ Ret0 +-+     |           |
// |               +---+--+       |           |
// |                   |          |           |
// |               +---v--+   +---v--+        |
// +---------------+ Ret0 +---+ Ret1 +--------+
//                 +------+   +------+
//
TEST(VectorizeMapDefunTest, VectorizeWithUnvectorizableOp) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32", "arg1: int32"},
      /*out_def=*/{"ret0: int32", "ret1: int32"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"MatMul"}, "MatMul", {"arg0", "arg0"}, {{"T", DT_INT32}}},
       Cast("Cast", {"arg1"}, DT_INT32, DT_INT32)},  //
      /*ret_def=*/{{"ret0", "MatMul:product:0"}, {"ret1", "Cast:y:0"}});
  // TODO(rachelim): If we ever write a converter for MatMul, we have to
  // change this test.

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));

  ASSERT_TRUE(
      function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  auto map_defun_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("MapDefun", *vectorized));

  // The Cast node should be converted just fine.
  ASSERT_TRUE(function_utils::ContainsFunctionNodeWithOp("Cast", *vectorized));
  auto cast = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Cast", *vectorized));
  EXPECT_EQ(GetRetval(*vectorized, 1), strings::StrCat(cast.name(), ":y:0"));

  // The inner function should only have one retval.
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), lib);
  const FunctionDef* map_defun_fn =
      lib_def.Find(map_defun_node.attr().at("f").func().name());
  EXPECT_EQ(map_defun_fn->signature().output_arg_size(), 1);
}
// Before:
//
//                 +------+
// +---------------+ Arg0 +-------------------+
// |               +---+--+                   |
// |                   |                      |
// |               +---v--+                   |
// |   +-----------+ Arg0 +---------------+   |
// |   |           +---+--+               |   |
// |   |               |                  |   |
// |   |               |                  |   |
// |   |           +---v--+               |   |
// |   |           | Cast |               |   |
// |   |           +---+--+               |   |
// |   |               |                  |   |
// |   |               +----------+       |   |
// |   |               |          |       |   |
// |   | MapDefun  +---v--+   +---v--+    |   |
// |   +-----------+ Ret0 +---+ Ret1 +----+   |
// |               +---+--+   +---+--+        |
// |                   |          |           |
// |               +---v--+   +---v--+        |
// +---------------+ Ret0 +---+ Ret1 +--------+
//                 +------+   +------+
//
//
//  After:
//
//                 +------+
// +---------------+ Arg0 +-------------------+
// |               +---+--+                   |
// |                   |                      |
// |                   |                      |
// |               +---v--+                   |
// |               | Cast |                   |
// |               +---+--+                   |
// |                   |                      |
// |                   +----------+           |
// |                   |          |           |
// |               +---v--+   +---v--+        |
// +---------------+ Ret0 +---+ Ret1 +--------+
//                 +------+   +------+
//
TEST(VectorizeMapDefunTest, VectorizeWithOutputUsedTwice) {
  // Tests that behavior is correct when an output is used more than once.
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int64", "ret1: int64"},
      /*attr_def=*/{},
      /*node_def=*/{Cast("Cast", {"arg0"}, DT_INT32, DT_INT64)},
      /*ret_def=*/{{"ret0", "Cast:y:0"}, {"ret1", "Cast:y:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));

  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  const NodeDef& cast_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Cast", *vectorized));
  EXPECT_EQ(cast_node.input(0), vectorized->signature().input_arg(0).name());
  EXPECT_EQ(GetRetval(*vectorized, 0),
            strings::StrCat(cast_node.name(), ":y:0"));
  EXPECT_EQ(GetRetval(*vectorized, 1),
            strings::StrCat(cast_node.name(), ":y:0"));
  EXPECT_EQ(vectorized->node_def_size(), 1);
}
// Before:
//
//                        +------+
// +----------------------+ Arg0 +----------------------+
// |                      +---+--+                      |
// |                          |                         |
// |                      +---v--+                      |
// |   +------------------+ Arg0 +------------------+   |
// |   |                  +---+--+                  |   |
// |   |                      |                     |   |
// |   |                  +---+--+                  |   |
// |   |                  | Cast |                  |   |
// |   |                  +---+--+                  |   |
// |   |                      |                     |   |
// |   |                  +---v---+ num=3           |   |
// |   |                  |Unstack| axis=0          |   |
// |   |                  ++--+--++                 |   |
// |   |                   |  |  |                  |   |
// |   |              +----+  |  +-------+          |   |
// |   |              |       |          |          |   |
// |   | MapDefun +---v--+  +-v----+  +--v---+      |   |
// |   +----------+ Ret0 +--+ Ret1 +--+ Ret2 +------+   |
// |              +---+--+  +--+---+  +--+---+          |
// |                  |        |         |              |
// |              +---v--+  +--v---+  +--v---+          |
// +--------------+ Ret0 +--+ Ret1 +--+ Ret2 +----------+
//                +------+  +------+  +------+
//
//
//  After:
//
//                        +------+
// +----------------------+ Arg0 +----------------------+
// |                      +---+--+                      |
// |                          |                         |
// |                      +---+--+                      |
// |                      | Cast |                      |
// |                      +---+--+                      |
// |                          |                         |
// |                      +---v---+ num=3               |
// |                      |Unstack| axis=1              |
// |                      ++--+--++                     |
// |                       |  |  |                      |
// |                  +----+  |  +-------+              |
// |                  |       |          |              |
// |                  |       |          |              |
// |              +---v--+  +-v----+  +--v---+          |
// +--------------+ Ret0 +--+ Ret1 +--+ Ret2 +----------+
//                +------+  +------+  +------+
//
TEST(VectorizeMapDefunTest, VectorizeWithChainedConvertibleOps) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int32", "ret1: int32", "ret2: int32"},
      /*attr_def=*/{},
      /*node_def=*/
      {Cast("Cast", {"arg0"}, DT_INT32, DT_INT32),
       {{"MyUnstack"},
        "Unpack",
        {"Cast:y:0"},
        {{"T", DT_INT32}, {"axis", 0}, {"num", 3}}}},
      /*ret_def=*/
      {{"ret0", "MyUnstack:output:0"},
       {"ret1", "MyUnstack:output:1"},
       {"ret2", "MyUnstack:output:2"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  const NodeDef& cast_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Cast", *vectorized));
  EXPECT_EQ(cast_node.input(0), vectorized->signature().input_arg(0).name());
  const NodeDef& unpack_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Unpack", *vectorized));
  EXPECT_EQ(unpack_node.input(0), strings::StrCat(cast_node.name(), ":y:0"));
  EXPECT_EQ(unpack_node.attr().at("axis").i(), 1);
  EXPECT_EQ(unpack_node.attr().at("T").type(), DT_INT32);
  EXPECT_EQ(unpack_node.attr().at("num").i(), 3);

  EXPECT_EQ(GetRetval(*vectorized, 0),
            strings::StrCat(unpack_node.name(), ":output:0"));
  EXPECT_EQ(GetRetval(*vectorized, 1),
            strings::StrCat(unpack_node.name(), ":output:1"));
  EXPECT_EQ(GetRetval(*vectorized, 2),
            strings::StrCat(unpack_node.name(), ":output:2"));
  EXPECT_EQ(vectorized->node_def_size(), 2);
}

// Before:
//
//
//                 +------+
// +---------------+ Arg0 +---------+
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |   +-----------+ Arg0 +-----+   |
// |   |           +---+--+     |   |
// |   |     +---------+        |   |
// |   | +---v--+      |        |   |
// |   | |Print |      |        |   |
// |   | +---+--+      |        |   |
// |   |     :     +---v--+     |   |
// |   |     ::::::> Cast |     |   |
// |   |           +---+--+     |   |
// |   |               |        |   |
// |   | MapDefun  +---v--+     |   |
// |   +-----------+ Ret0 +-----+   |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
//
//
//  After:
//
//  No change because we don't deal with control inputs for now.
//
TEST(VectorizeMapDefunTest, VectorizeWithControlInputs) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int64"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"Print"},
        "Print",
        {"arg0", "arg0"},
        {{"T", DT_INT32}, {"U", gtl::ArraySlice<DataType>({DT_INT32})}}},
       Cast("Cast", {"arg0", "^Print"}, DT_INT32, DT_INT64)},
      /*ret_def=*/{{"ret0", "Cast:y:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  // They should be unchanged
  // We check this somewhat manually as the names of nodes may have changed
  EXPECT_EQ(vectorized->node_def_size(), 1);
  const NodeDef& map_defun_node = vectorized->node_def(0);
  EXPECT_EQ(map_defun_node.op(), "MapDefun");
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), lib);
  const FunctionDef* map_defun_fn =
      lib_def.Find(map_defun_node.attr().at("f").func().name());

  const NodeDef& print_node = map_defun_fn->node_def(
      function_utils::FindFunctionNodeWithOp("Print", *map_defun_fn));
  const NodeDef& cast_node = map_defun_fn->node_def(
      function_utils::FindFunctionNodeWithOp("Cast", *map_defun_fn));
  string control_input = strings::StrCat("^", print_node.name());
  EXPECT_TRUE(cast_node.input(0) == control_input ||
              cast_node.input(1) == control_input);
}

// Before:
//
//
//                 +------+
// +---------------+ Arg0 +---------+
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |   +-----------+ Arg0 +-----+   |
// |   |           +------+     |   |
// |   |                        |   |
// |   |                        |   |
// |   |           +------+     |   |
// |   |           |Const |     |   |
// |   |           +---+--+     |   |
// |   |               |        |   |
// |   |           +---v--+     |   |
// |   |           | Cast |     |   |
// |   |           +---+--+     |   |
// |   |               |        |   |
// |   | MapDefun  +---v--+     |   |
// |   +-----------+ Ret0 +-----+   |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
//
//
//  After:
//
//                 +------+
// +---------------+ Arg0 +---------+
// |               +------+         |
// |                                |
// |               +------+         |
// |               |Const |         |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |               | Cast |         |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |               |Stack*|         |
// |               +---+--+         |
// |                   |            |
// |                   |            |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
// *Not actually a Stack node, but does the equivalent.
//
TEST(VectorizeMapDefunTest, VectorizeWithUnstackedOutput) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int64"},
      /*attr_def=*/{},
      /*node_def=*/
      {FunctionDefHelper::Const("Const", 2),
       Cast("Cast", {"Const:output:0"}, DT_INT32, DT_INT64)},
      /*ret_def=*/{{"ret0", "Cast:y:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  auto const_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Const", *vectorized));
  auto cast_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Cast", *vectorized));
  EXPECT_EQ(cast_node.input(0).substr(0, cast_node.input(0).find(':')),
            const_node.name());
}

// Before:
//
//
//                 +------+
// +---------------+ Arg0 +---------+
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |   +-----------+ Arg0 +-----+   |
// |   |           +------+     |   |
// |   |                        |   |
// |   | +------+  +------+     |   |
// |   | |Const |  |Const |     |   |
// |   | +---+--+  +---+--+     |   |
// |   |     :     +---v--+     |   |
// |   |     ::::::> Cast |     |   |
// |   |           +---+--+     |   |
// |   |               |        |   |
// |   | MapDefun  +---v--+     |   |
// |   +-----------+ Ret0 +-----+   |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
//
//
//  After:
//
//
//                 +------+
// +---------------+ Arg0 +---------+
// |               +------+         |
// |                                |
// |                                |
// |               +------+         |
// |     +------+  |Const |         |
// |     |Const |  +---+--+         |
// |     +---+--+      |            |
// |         :     +---v--+         |
// |         ::::::> Cast |         |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |               +Stack*+         |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
// *Not actually a Stack node, but does the equivalent.
//
TEST(VectorizeMapDefunTest, VectorizeWithUnstackedControl) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int64"},
      /*attr_def=*/{},
      /*node_def=*/
      {FunctionDefHelper::Const("Const", 2),
       FunctionDefHelper::Const("ConstDep", 3),
       Cast("Cast", {"Const:output:0", "^ConstDep"}, DT_INT32, DT_INT64)},
      /*ret_def=*/{{"ret0", "Cast:y:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));

  auto find_const = [vectorized](int val) -> const NodeDef* {
    for (const auto& n : vectorized->node_def()) {
      if (n.attr().at("value").tensor().int_val(0) == val) {
        return &n;
      }
    }
    return nullptr;
  };

  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  auto const_node = find_const(2);
  auto const_dep_node = find_const(3);
  auto cast_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Cast", *vectorized));
  EXPECT_EQ(cast_node.input(0).substr(0, cast_node.input(0).find(':')),
            const_node->name());
  EXPECT_EQ(cast_node.input(1), strings::StrCat("^", const_dep_node->name()));
}

///==================================//
// Tests for specific op vectorizers //
///==================================//

// Before:
//
//                        +------+
// +----------------------+ Arg0 +----------------------+
// |                      +---+--+                      |
// |                          |                         |
// |                      +---v--+                      |
// |   +------------------+ Arg0 +------------------+   |
// |   |                  +---+--+                  |   |
// |   |                      |                     |   |
// |   |                      |                     |   |
// |   |                  +---v---+ num=3           |   |
// |   |                  |Unstack| axis=0          |   |
// |   |                  ++--+--++                 |   |
// |   |                   |  |  |                  |   |
// |   |              +----+  |  +-------+          |   |
// |   |              |       |          |          |   |
// |   | MapDefun +---v--+  +-v----+  +--v---+      |   |
// |   +----------+ Ret0 +--+ Ret1 +--+ Ret2 +------+   |
// |              +---+--+  +--+---+  +--+---+          |
// |                  |        |         |              |
// |              +---v--+  +--v---+  +--v---+          |
// +--------------+ Ret0 +--+ Ret1 +--+ Ret2 +----------+
//                +------+  +------+  +------+
//
//
//  After:
//
//                        +------+
// +----------------------+ Arg0 +----------------------+
// |                      +---+--+                      |
// |                          |                         |
// |                          |                         |
// |                          |                         |
// |                      +---v---+ num=3               |
// |                      |Unstack| axis=1              |
// |                      ++--+--++                     |
// |                       |  |  |                      |
// |                  +----+  |  +-------+              |
// |                  |       |          |              |
// |                  |       |          |              |
// |              +---v--+  +-v----+  +--v---+          |
// +--------------+ Ret0 +--+ Ret1 +--+ Ret2 +----------+
//                +------+  +------+  +------+
//
TEST(VectorizerTest, VectorizeUnstack) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int32", "ret1: int32", "ret2: int32"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"MyUnstack"},
        "Unpack",
        {"arg0"},
        {{"T", DT_INT32}, {"axis", 0}, {"num", 3}}}},
      /*ret_def=*/
      {{"ret0", "MyUnstack:output:0"},
       {"ret1", "MyUnstack:output:1"},
       {"ret2", "MyUnstack:output:2"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  const NodeDef& unpack_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Unpack", *vectorized));
  EXPECT_EQ(unpack_node.input(0), vectorized->signature().input_arg(0).name());
  EXPECT_EQ(unpack_node.attr().at("axis").i(), 1);
  EXPECT_EQ(unpack_node.attr().at("T").type(), DT_INT32);
  EXPECT_EQ(unpack_node.attr().at("num").i(), 3);
  EXPECT_EQ(GetRetval(*vectorized, 0),
            strings::StrCat(unpack_node.name(), ":output:0"));
  EXPECT_EQ(GetRetval(*vectorized, 1),
            strings::StrCat(unpack_node.name(), ":output:1"));
  EXPECT_EQ(GetRetval(*vectorized, 2),
            strings::StrCat(unpack_node.name(), ":output:2"));
  EXPECT_EQ(vectorized->node_def_size(), 1);
}

// Before:
//
//
//                 +------+
// +---------------+ Arg0 +---------+
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |   +-----------+ Arg0 +-----+   |
// |   |           +---+--+     |   |
// |   |               |        |   |
// |   |               |        |   |
// |   |           +---v--+     |   |
// |   |           | Cast |     |   |
// |   |           +---+--+     |   |
// |   |               |        |   |
// |   | MapDefun  +---v--+     |   |
// |   +-----------+ Ret0 +-----+   |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
//
//
//  After:
//
//                 +------+
// +---------------+ Arg0 +---------+
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |               | Cast |         |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
//
TEST(VectorizerTest, VectorizeCast) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int64"},
      /*attr_def=*/{},
      /*node_def=*/{Cast("Cast", {"arg0"}, DT_INT32, DT_INT64)},
      /*ret_def=*/{{"ret0", "Cast:y:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  const NodeDef& cast_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Cast", *vectorized));
  EXPECT_EQ(cast_node.input(0), vectorized->signature().input_arg(0).name());
  EXPECT_EQ(GetRetval(*vectorized, 0),
            strings::StrCat(cast_node.name(), ":y:0"));
  EXPECT_EQ(vectorized->node_def_size(), 1);
}

// Before:
//
//                   +------+
// +-----------------+ Arg0 +----------------------+
// |                 +---+--+                      |
// |                     |                         |
// |                 +---v--+                      |
// |   +-------------+ Arg0 +------------------+   |
// |   |             +---+--+                  |   |
// |   |                 |                     |   |
// |   |                 |          +-----+    |   |
// |   |                 |          |Const|    |   |
// |   |                 |          +-+---+    |   |
// |   |                 |            |        |   |
// |   |                 |   +--------+        |   |
// |   |                 |   |                 |   |
// |   |               +-v---v-+               |   |
// |   |               |  Add  |               |   |
// |   |               +-+-----+               |   |
// |   |                 |                     |   |
// |   |                 |                     |   |
// |   | MapDefun      +-v----+                |   |
// |   +---------------| Ret  |----------------+   |
// |                   +--v---+                    |
// |                      |                        |
// |                      |                        |
// |                   +--v----                    |
// +-------------------| Ret  |--------------------+
//                     +------+
//
//
//  After:
//
//              +------+
// +------------+ Arg0 +----------------------+
// |            +---+--+                      |
// |                |                         |
// |                |              +-----+    |
// |                |              |Const|    |
// |              +-v---------+    +--+--+    |
// |              |ExpandDims*|       |       |
// |              +-----+-----+       |       |
// |                    |             |       |
// |                    +-----+ +-----+       |
// |                          | |             |
// |                        +-v-v-+           |
// |                        | Add |           |
// |                        +--+--+           |
// |                           |              |
// |                       +---v--+           |
// +-----------------------+ Ret  +-----------+
//                         +------+
//
TEST(VectorizerTest, VectorizeAdd) {
  // Note that this checks that the "Add" vectorizer is successful, but does not
  // check that the transformed function is correct (i.e. produces the same
  // output as the unvectorized map defun). For the latter, the tests are in
  // tensorflow/python/data/experimental/kernel_tests/optimization/
  // map_vectorization_test.py
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int32"},
      /*attr_def=*/{},
      /*node_def=*/
      {FunctionDefHelper::Const("Const", 2),
       {{"Add"}, "Add", {"arg0", "Const:output:0"}, {{"T", DT_INT32}}}},
      /*ret_def=*/{{"ret0", "Add:z:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
}

// Tests that a function which applies a cwise op can be vectorized completely.
Status CwiseTestHelper(DataType input_type, const string& op_type,
                       size_t arity) {
  // Note that this checks that the cwise op vectorizer is successful, but does
  // not check that the transformed function is correct (i.e. produces the same
  // output as the unvectorized map defun). For the latter, the tests are in
  // tensorflow/python/data/experimental/kernel_tests/optimization/
  // map_vectorization_test.py

  FunctionDef inner;
  // Create inner function with a single operation of type op_type. The output
  // type attr of the function is inferred by NodeBuilder.
  Node *op, *retval;
  Graph graph(OpRegistry::Global());

  auto node_builder = NodeBuilder("op", op_type);
  for (size_t i = 0; i < arity; ++i) {
    Node* arg;
    TF_RETURN_IF_ERROR(NodeBuilder(strings::StrCat("arg", i), "_Arg")
                           .Attr("T", input_type)
                           .Attr("index", static_cast<int>(i))
                           .Finalize(&graph, &arg));

    node_builder = node_builder.Input(arg);
  }
  TF_RETURN_IF_ERROR(node_builder.Finalize(&graph, &op));

  TF_RETURN_IF_ERROR(NodeBuilder("ret", "_Retval")
                         .Input(op)
                         .Attr("index", 0)
                         .Finalize(&graph, &retval));

  TF_RETURN_IF_ERROR(GraphToFunctionDef(graph, "inner_function", &inner));

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_RETURN_IF_ERROR(WrapAndVectorize(inner, &lib, &vectorized));

  return function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized)
             ? errors::Internal(
                   "Test for cwise vectorizer for op \"", op_type,
                   "\" failed. The function was not fully vectorized.")
             : Status::OK();
}

class BitwiseUnaryTest : public ::testing::TestWithParam<const char*> {};

TEST_P(BitwiseUnaryTest, VectorizeCwiseBitwiseUnary) {
  TF_EXPECT_OK(CwiseTestHelper(DT_INT32, GetParam(), 1));
}

INSTANTIATE_TEST_CASE_P(Test, BitwiseUnaryTest, ::testing::Values("Invert"));

class LogicalUnaryTest : public ::testing::TestWithParam<const char*> {};

TEST_P(LogicalUnaryTest, VectorizeCwiseLogicalUnary) {
  TF_EXPECT_OK(CwiseTestHelper(DT_BOOL, GetParam(), 1));
}

INSTANTIATE_TEST_CASE_P(Test, LogicalUnaryTest,
                        ::testing::Values("LogicalNot"));

class ComplexUnaryTest : public ::testing::TestWithParam<const char*> {};

TEST_P(ComplexUnaryTest, VectorizeCwiseComplexUnary) {
  TF_EXPECT_OK(CwiseTestHelper(DT_COMPLEX64, GetParam(), 1));
}

INSTANTIATE_TEST_CASE_P(Test, ComplexUnaryTest,
                        ::testing::Values("Angle", "ComplexAbs", "Conj", "Imag",
                                          "Real"));

class RealUnaryTest : public ::testing::TestWithParam<const char*> {};

TEST_P(RealUnaryTest, VectorizeCwiseRealUnary) {
  TF_EXPECT_OK(CwiseTestHelper(DT_FLOAT, GetParam(), 1));
}

INSTANTIATE_TEST_CASE_P(
    Test, RealUnaryTest,
    ::testing::Values("Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh",
                      "BesselI0e", "BesselI1e", "Ceil", "Cos", "Cosh",
                      "Digamma", "Elu", "Erf", "Erfc", "Exp", "Expm1", "Floor",
                      "Inv", "IsFinite", "IsInf", "Lgamma", "Log", "Log1p",
                      "Neg", "Reciprocal", "Relu", "Relu6", "Rint", "Round",
                      "Rsqrt", "Selu", "Sigmoid", "Sign", "Sin", "Sinh",
                      "Softplus", "Softsign", "Sqrt", "Square", "Tanh", "Tan"));

class BitwiseBinaryTest : public ::testing::TestWithParam<const char*> {};

TEST_P(BitwiseBinaryTest, VectorizeCwiseBitwiseBinary) {
  TF_EXPECT_OK(CwiseTestHelper(DT_INT32, GetParam(), 2));
}

INSTANTIATE_TEST_CASE_P(Test, BitwiseBinaryTest,
                        ::testing::Values("BitwiseAnd", "BitwiseOr",
                                          "BitwiseXor", "LeftShift",
                                          "RightShift"));

class LogicalBinaryTest : public ::testing::TestWithParam<const char*> {};

TEST_P(LogicalBinaryTest, VectorizeCwiseLogicalBinary) {
  TF_EXPECT_OK(CwiseTestHelper(DT_BOOL, GetParam(), 2));
}

INSTANTIATE_TEST_CASE_P(Test, LogicalBinaryTest,
                        ::testing::Values("LogicalAnd", "LogicalOr"));

class RealBinaryTest : public ::testing::TestWithParam<const char*> {};

TEST_P(RealBinaryTest, VectorizeCwiseRealBinary) {
  TF_EXPECT_OK(CwiseTestHelper(DT_FLOAT, GetParam(), 2));
}

INSTANTIATE_TEST_CASE_P(
    Test, RealBinaryTest,
    ::testing::Values("Add", "AddV2", "Atan2", "Complex", "Div", "DivNoNan",
                      "Equal", "FloorDiv", "FloorMod", "Greater",
                      "GreaterEqual", "Igamma", "Igammac", "IgammaGradA",
                      "Less", "LessEqual", "Maximum", "Minimum", "Mod", "Mul",
                      "NotEqual", "Polygamma", "Pow", "RealDiv",
                      "SquaredDifference", "Sub", "TruncateDiv", "TruncateMod",
                      "Zeta"));

// Before:
//
//
//                 +------+
// +---------------+ Arg0 +---------------------+
// |               +---+--+                     |
// |                   |                        |
// |               +---v--+                     |
// |   +-----------+ Arg0 +-----------------+   |
// |   |           +---+--+                 |   |
// |   |               |                    |   |
// |   |               |                    |   |
// |   |               |   (3,3,3)          |   |
// |   |               |   +-----+          |   |
// |   |               |   |Const|          |   |
// |   |               |   +--+--+          |   |
// |   |               |      |             |   |
// |   |               | +----+             |   |
// |   |           +---v-v-+                |   |
// |   |           |Reshape|                |   |
// |   |           +---+---+                |   |
// |   |               |                    |   |
// |   | MapDefun  +---v--+                 |   |
// |   +-----------+ Ret0 +-----------------+   |
// |               +---+--+                     |
// |                   |                        |
// |               +---v--+                     |
// +---------------+ Ret0 +---------------------+
//                 +------+
//
//
//  After:
//
//           +------+
// +---------+ Arg0 +------------------------+
// |         +---+--+                        |
// |             |                           |
// |             |                           |
// |             |     +-----+               |
// |             |     |Const|               |
// |             |     +--+--+               |
// |             |        |                  |
// |             |    +---v---+              |
// |             |    |Concat*|              |
// |             |    +---+---+              |
// |             |        |                  |
// |             | +------+                  |
// |             | |                         |
// |         +---v-v-+                       |
// |         |Reshape|                       |
// |         +---+---+                       |
// |             |                           |
// |         +---v--+                        |
// +---------+ Ret0 +------------------------+
//           +------+
//
// (Where Concat* appends the 0th dim of the input to the new shape)
//
TEST(VectorizerTest, VectorizeReshape) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int32"},
      /*attr_def=*/{},
      /*node_def=*/
      {FunctionDefHelper::Const("Const", gtl::ArraySlice<int>({3, 3, 3})),
       {{"Reshape"},
        "Reshape",
        {"arg0", "Const:output:0"},
        {{"T", DT_INT32}, {"Tshape", DT_INT32}}}},
      /*ret_def=*/{{"ret0", "Reshape:output:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  EXPECT_TRUE(
      function_utils::ContainsFunctionNodeWithOp("Reshape", *vectorized));
  auto reshape_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Reshape", *vectorized));
  EXPECT_EQ(GetRetval(*vectorized, 0),
            strings::StrCat(reshape_node.name(), ":output:0"));
}

// Before:
//
//
//                 +------+
// +---------------+ Arg0 +---------------------+
// |               +---+--+                     |
// |                   |                        |
// |               +---v--+                     |
// |   +-----------+ Arg0 +-----------------+   |
// |   |           +---+--+                 |   |
// |   |               |                    |   |
// |   |               |   record_defaults  |   |
// |   |               |   +-----+  +-----+ |   |
// |   |               |   |Const|  |Const| |   |
// |   |               |   +--+--+  +--+--+ |   |
// |   |               |      |        |    |   |
// |   |               | +----+        |    |   |
// |   |               | |             |    |   |
// |   |               | | +-----------+    |   |
// |   |               | | |                |   |
// |   |           +---v-v-v-+              |   |
// |   |           |DecodeCSV|              |   |
// |   |           +---+---+-+              |   |
// |   |               |   |                |   |
// |   |               |   +------+         |   |
// |   |               |          |         |   |
// |   | MapDefun  +---v--+   +---v--+      |   |
// |   +-----------+ Ret0 +---+ Ret1 +------+   |
// |               +---+--+   +---+--+          |
// |                   |          |             |
// |               +---v--+   +---v--+          |
// +---------------+ Ret0 +---+ Ret1 +----------+
//                 +------+   +------+
//
//  After:
//
//           +------+
// +---------+ Arg0 +------------------------+
// |         +---+--+                        |
// |             |                           |
// |             |                           |
// |             |     +-----+ +-----+       |
// |             |     |Const| |Const|       |
// |             |     +--+--+ +--+--+       |
// |             |        |       |          |
// |             |        |       |          |
// |             | +------+       |          |
// |             | | +------------+          |
// |             | | |                       |
// |             | | |                       |
// |         +---v-v-v-+                     |
// |         |DecodeCSV|                     |
// |         +---+---+-+                     |
// |             |   |                       |
// |             |   +-------+               |
// |             |           |               |
// |           +-v----+   +--v---+           |
// +-----------+ Ret0 +---+ Ret1 +-----------+
//             +------+   +------+
//
//
TEST(VectorizerTest, VectorizeDecodeCSV) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: string"},
      /*out_def=*/{"ret0: int32", "ret1: string"},
      /*attr_def=*/{},
      /*node_def=*/
      {FunctionDefHelper::Const("Default0", gtl::ArraySlice<int>({2})),
       FunctionDefHelper::Const("Default1", gtl::ArraySlice<tstring>({})),
       {{"DecodeCSV"},
        "DecodeCSV",
        {"arg0", "Default0:output:0", "Default1:output:0"},
        {{"OUT_TYPE", DataTypeVector({DT_INT32, DT_STRING})}}}},
      /*ret_def=*/
      {{"ret0", "DecodeCSV:output:0"}, {"ret1", "DecodeCSV:output:1"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
}

TEST(VectorizerTest, VectorizeDecodeCSVWithStackedDefaults) {
  // When the `record_defaults` input to DecodeCSV are stacked,
  // the node should not be vectorized.
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: string", "arg1: int32", "arg2: string"},
      /*out_def=*/{"ret0: int32", "ret1: string"},
      /*attr_def=*/{},
      /*node_def=*/
      {{{"DecodeCSV"},
        "DecodeCSV",
        {"arg0", "arg1", "arg2"},  // Inputs come from args, which are "stacked"
        {{"OUT_TYPE", DataTypeVector({DT_INT32, DT_STRING})}}}},
      /*ret_def=*/
      {{"ret0", "DecodeCSV:output:0"}, {"ret1", "DecodeCSV:output:1"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
}

// Before:
//
//
//                 +------+
// +---------------+ Arg0 +---------------------+
// |               +---+--+                     |
// |                   |                        |
// |               +---v--+                     |
// |   +-----------+ Arg0 +-----------------+   |
// |   |           +---+--+                 |   |
// |   |               |                    |   |
// |   |               |   dense_defaults   |   |
// |   |               |   +-----+  +-----+ |   |
// |   |               |   |Const|  |Const| |   |
// |   |               |   +--+--+  +--+--+ |   |
// |   |               |      |        |    |   |
// |   |               | +----+        |    |   |
// |   |               | |             |    |   |
// |   |               | | +-----------+    |   |
// |   |               | | |                |   |
// |   |           +---v-v-v----------+     |   |
// |   |           |ParseSingleExample|     |   |
// |   |           +---+---+----------+     |   |
// |   |               |                    |   |
// |   |             (...)                  |   |
// |   |               |                    |   |
// |   | MapDefun  +---v--+                 |   |
// |   +-----------+ Rets*+-----------------+   |
// |               +---+--+                     |
// |                   |                        |
// |               +---v--+                     |
// +---------------+ Rets*+---------------------+
//                 +------+
//
//  After:
//
//           +------+
// +---------+ Arg0 +------------------------------------+
// |         +---+--+                                    |
// |             |                                       |
// |             |   names                               |
// |             |   sparse_types                        |
// |             |   dense_types   dense_defaults        |
// |             |  +============+ +-----+ +-----+       |
// |             |  |  Consts*   | |Const| |Const|       |
// |             |  +============+ +--+--+ +--+--+       |
// |             |       |            |       |          |
// |             |     (...)          |       |          |
// |             |       |     +------+       |          |
// |             |       |     | +------------+          |
// |             |       |     | |                       |
// |             |       |     | |                       |
// |         +---v-------v-----v-v-+                     |
// |         |  ParseExample       |                     |
// |         +---+-----------------+                     |
// |             |                                       |
// |           (...)                                     |
// |             |                                       |
// |           +-v----+                                  |
// +-----------+ Rets*+----------------------------------+
//             +------+
//
// *Multiple nodes. Only one drawn for brevity.
//
TEST(VectorizerTest, VectorizeParseSingleExample) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: string"},
      /*out_def=*/
      {"si0: int64", "si1: int64", "sv0: int64", "sv1: string", "ss0: int64",
       "ss1: int64", "dv0: int64", "dv1: string"},
      /*attr_def=*/{},
      /*node_def=*/
      {FunctionDefHelper::Const("DenseIntDefault", static_cast<int64>(0)),
       FunctionDefHelper::Const("DenseStrDefault", tstring("")),
       {{"Parse"},
        "ParseSingleExample",
        {"arg0", "DenseIntDefault:output:0", "DenseStrDefault:output:0"},
        {
            {"Tdense", DataTypeVector({DT_INT64, DT_STRING})},
            {"dense_keys", gtl::ArraySlice<string>({"dense_int", "dense_str"})},
            {"dense_shapes", gtl::ArraySlice<TensorShape>({}, {})},
            {"num_sparse", 2},
            {"sparse_keys", gtl::ArraySlice<string>({"spar_int", "spar_str"})},
            {"sparse_types", DataTypeVector({DT_INT64, DT_STRING})},
        }}},
      /*ret_def=*/
      {
          {"si0", "Parse:sparse_indices:0"},
          {"si1", "Parse:sparse_indices:1"},
          {"sv0", "Parse:sparse_values:0"},
          {"sv1", "Parse:sparse_values:1"},
          {"ss0", "Parse:sparse_shapes:0"},
          {"ss1", "Parse:sparse_shapes:1"},
          {"dv0", "Parse:dense_values:0"},
          {"dv1", "Parse:dense_values:1"},
      });

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      !function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  EXPECT_TRUE(
      function_utils::ContainsFunctionNodeWithOp("ParseExample", *vectorized));
}

TEST(VectorizerTest, VectorizeParseSingleExampleWithStackedDefaults) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: string", "arg1: string"},
      /*out_def=*/{"dv0: int64", "dv1: string"},
      /*attr_def=*/{},
      /*node_def=*/
      {FunctionDefHelper::Const("DenseIntDefault", static_cast<int64>(0)),
       {{"Parse"},
        "ParseSingleExample",
        {"arg0", "DenseIntDefault:output:0", "arg1"},
        {
            {"Tdense", DataTypeVector({DT_INT64, DT_STRING})},
            {"dense_keys", gtl::ArraySlice<string>({"dense_int", "dense_str"})},
            {"dense_shapes", gtl::ArraySlice<TensorShape>({}, {})},
            {"num_sparse", 0},
            {"sparse_keys", gtl::ArraySlice<string>({})},
            {"sparse_types", DataTypeVector({})},
        }}},
      /*ret_def=*/
      {
          {"dv0", "Parse:dense_values:0"},
          {"dv1", "Parse:dense_values:1"},
      });

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_TRUE(
      function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
}

TEST(VectorizerTest, VectorizeTranspose) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"out: int32"},
      /*attr_def=*/{},
      /*node_def=*/
      {FunctionDefHelper::Const("Perm", gtl::ArraySlice<int>({1, 0})),
       {{"Transpose"},
        "Transpose",
        {"arg0", "Perm:output:0"},
        {{"T", DT_INT32}, {"Tperm", DT_INT32}}}},
      /*ret_def=*/{{"out", "Transpose:y:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));
  EXPECT_FALSE(
      function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
}

TEST(VectorizerTest, VectorizeIdentity) {
  FunctionDef inner = FunctionDefHelper::Create(
      /*function_name=*/"inner_function",
      /*in_def=*/{"arg0: int32"},
      /*out_def=*/{"ret0: int32"},
      /*attr_def=*/{},
      /*node_def=*/{{{"Identity"}, "Identity", {"arg0"}, {{"T", DT_INT32}}}},
      /*ret_def=*/{{"ret0", "Identity:output:0"}});

  FunctionDefLibrary lib;
  FunctionDef* vectorized;
  TF_ASSERT_OK(WrapAndVectorize(inner, &lib, &vectorized));

  EXPECT_FALSE(
      function_utils::ContainsFunctionNodeWithOp("MapDefun", *vectorized));
  ASSERT_TRUE(
      function_utils::ContainsFunctionNodeWithOp("Identity", *vectorized));
  const NodeDef& identity_node = vectorized->node_def(
      function_utils::FindFunctionNodeWithOp("Identity", *vectorized));

  EXPECT_EQ(identity_node.input(0),
            vectorized->signature().input_arg(0).name());
  EXPECT_EQ(GetRetval(*vectorized, 0),
            strings::StrCat(identity_node.name(), ":output:0"));
  EXPECT_EQ(vectorized->node_def_size(), 1);
}

}  // namespace
}  // namespace vectorization_utils
}  // namespace grappler
}  // namespace tensorflow
