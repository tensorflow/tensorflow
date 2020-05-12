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

#include "tensorflow/compiler/jit/force_xla_constants_on_host_pass.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

Status ForceXlaConstantsOnHost(const Scope& s,
                               FunctionLibraryDefinition* flib_def,
                               std::unique_ptr<Graph>* result) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  GraphOptimizationPassOptions options;
  SessionOptions session_options;
  session_options.env = Env::Default();
  options.graph = &graph;
  options.session_options = &session_options;
  options.flib_def = flib_def;
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get()));
  ForceXlaConstantsOnHostPass rewriter;
  TF_RETURN_IF_ERROR(rewriter.Run(options));
  *result = std::move(graph);
  return Status::OK();
}

TEST(ForceXlaConstantsOnHostPassTest, Simple) {
  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Scope root = Scope::NewRootScope().ExitOnError();
  FunctionDefLibrary library;

  FunctionDef called_func =
      FunctionDefHelper::Create("TransposeCall",
                                /*in_def=*/{"a:float", "b:int32"},
                                /*out_def=*/{"c:float"}, {},
                                {{{"t0"},
                                  "Transpose",
                                  {"a", "b"},
                                  {
                                      {"T", DT_FLOAT},
                                      {"Tperm", DT_INT32},
                                  }}},
                                {{"c", "t0:y:0"}});

  AttrValue true_attribute;
  true_attribute.set_b(true);
  (*called_func.mutable_attr())[kXlaMustCompileAttr] = true_attribute;
  *library.add_function() = called_func;
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(library));
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), library);
  Output in = ops::Placeholder(root, DT_FLOAT);
  Output perm = ops::Const(root, {3, 1, 2, 0});

  NameAttrList b_name_attr;
  b_name_attr.set_name("TransposeCall");
  ops::PartitionedCall call(root.WithOpName("call"), {in, perm}, {DT_FLOAT},
                            b_name_attr);
  call.output.front().node()->AddAttr(kXlaMustCompileAttr, true);

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(ForceXlaConstantsOnHost(root, &flib_def, &graph));

  bool found = false;
  for (Node* node : graph->nodes()) {
    if (CanCreateXlaKernel(node->def())) {
      EXPECT_FALSE(found);
      found = true;
      std::vector<int32> hostmem_attr;
      EXPECT_TRUE(TryGetNodeAttr(node->def(), "_input_hostmem", &hostmem_attr));
      EXPECT_EQ(hostmem_attr.size(), 1);
      EXPECT_EQ(hostmem_attr[0], 1);
    }
  }
  EXPECT_TRUE(found);
}

}  // namespace
}  // namespace tensorflow
