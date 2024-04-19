/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/inline_function_utils.h"

#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;

TEST(InlineFunctionBody, ColocationConstraintPropagation) {
  // A test that ensures that when there is a colocation constraint on one of
  // inputs of the function to be inlined, the constraint is propagated with the
  // new name.

  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  // Add a simple function that takes 2 inputs and the op is colocated with
  // one of the inputs.
  FunctionDef fdef = FunctionDefHelper::Define(
      "f",                       // Name
      {"x: float", "y: float"},  // Args
      {"z: float"},              // Returns
      {},                        // Attr def
      // Nodes
      {
          {{"z"},
           "AddV2",
           {"x", "y"},
           {{"T", DT_FLOAT}, {"_class", std::vector<string>({"loc:@x"})}}},
      });
  TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));

  auto g = std::make_unique<Graph>(OpRegistry::Global());
  GraphConstructorOptions opts;
  const Tensor kZero = test::AsScalar<int64_t>(0);
  const Tensor kOne = test::AsScalar<int64_t>(1);
  GraphDef gdef = GDef(
      {
          NDef("inp0", "Const", {}, {{"dtype", DT_FLOAT}, {"value", kZero}}),
          NDef("inp1", "Const", {}, {{"dtype", DT_FLOAT}, {"value", kOne}}),
          NDef("call", "StatefulPartitionedCall", {"inp0", "inp1"},
               {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
                {"Tout", DataTypeSlice{DT_FLOAT}},
                {"f", FunctionDefHelper::FunctionRef("f", {})}}),
          NDef("out0", "_Retval", {"call:0"}, {{"T", DT_FLOAT}}),
      },
      {});
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, gdef, g.get()));

  // The 'caller' node is the one with the name "call".
  Node* caller = nullptr;
  for (Node* node : g->nodes()) {
    if (node->name() == "call") {
      caller = node;
    }
  }

  std::unique_ptr<FunctionBody> fbody;
  TF_ASSERT_OK(FunctionDefToBodyHelper(fdef, {}, &flib_def, &fbody));

  InlineFunctionBodyOptions inline_options;
  TF_ASSERT_OK(InlineFunctionBody(flib_def, g.get(), caller, fbody.get(),
                                  inline_options));

  GraphDef expected_gdef = GDef(
      {
          NDef("inp0", "Const", {}, {{"dtype", DT_FLOAT}, {"value", kZero}}),
          NDef("inp1", "Const", {}, {{"dtype", DT_FLOAT}, {"value", kOne}}),
          NDef("out0", "_Retval", {"Func/call/output/_2"}, {{"T", DT_FLOAT}}),
          NDef("Func/call/input/_0", "Identity", {"inp0"}, {{"T", DT_FLOAT}}),
          NDef("Func/call/input/_1", "Identity", {"inp1"}, {{"T", DT_FLOAT}}),
          // The important assertion here is that the _class colocation
          // constraint gets updated from loc:@x to the "new" input which is
          // Func/call/input/_0.
          NDef("call/z", "AddV2", {"Func/call/input/_0", "Func/call/input/_1"},
               {{"T", DT_FLOAT},
                {"_class", std::vector<string>({"loc:@Func/call/input/_0"})}}),
          NDef("Func/call/output/_2", "Identity", {"call/z"},
               {{"T", DT_FLOAT}}),
      },
      {});

  GraphDef output_gdef;
  g->ToGraphDef(&output_gdef);
  TF_EXPECT_GRAPH_EQ(expected_gdef, output_gdef);
}

}  // namespace
}  // namespace tensorflow
