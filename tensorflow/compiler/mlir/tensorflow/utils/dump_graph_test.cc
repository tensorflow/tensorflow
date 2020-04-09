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

#include "tensorflow/compiler/mlir/tensorflow/utils/dump_graph.h"

#include "llvm/Support/InitLLVM.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

void ExpectHasSubstr(const string& s, const string& expected) {
  EXPECT_TRUE(absl::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

TEST(Dump, TexualIrToFileSuccess) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  string ret = DumpTextualIRToFile(MlirDumpConfig().with_name("tir"), graph);
  ASSERT_EQ(ret, io::JoinPath(testing::TmpDir(), "tir.mlir"));

  string actual;
  TF_CHECK_OK(ReadFileToString(Env::Default(), ret, &actual));
  string expected = R"(
  func @main() {
    tf_executor.graph {
      %control = tf_executor.island wraps "tf.NoOp"() {device = ""} : () -> ()
      tf_executor.fetch
    }
    return
  })";
  ExpectHasSubstr(actual, expected);
}

TEST(Dump, TexualIrToFileWithStdPipelineSuccess) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  string ret = DumpTextualIRToFile(
      MlirDumpConfig().with_name("tir_std").with_standard_pipeline(), graph);
  ASSERT_EQ(ret, io::JoinPath(testing::TmpDir(), "tir_std.mlir"));

  string actual;
  TF_CHECK_OK(ReadFileToString(Env::Default(), ret, &actual));
  string expected = R"(
  func @main() {
    return
  })";
  ExpectHasSubstr(actual, expected);
}

}  // namespace
}  // namespace tensorflow

namespace {
void RegisterDialects() {
  static bool init_once = []() {
    mlir::registerDialect<mlir::tf_executor::TensorFlowExecutorDialect>();
    mlir::registerDialect<mlir::TF::TensorFlowDialect>();
    mlir::registerDialect<mlir::StandardOpsDialect>();
    return true;
  }();
  (void)init_once;
}
}  // namespace

int main(int argc, char** argv) {
  RegisterDialects();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
