/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/graph_to_tf_executor.h"

#include <stdlib.h>

#include <string>

#include <gtest/gtest.h>
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/bytes/read_all.h"  // from @riegeli
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {
namespace {

using mlir::DialectRegistry;
using mlir::MLIRContext;

std::string TestDataPath() {
  return tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tf2xla/api/v2/testdata/");
}

class GraphToTfExecutorTest : public ::testing::Test {
 public:
  GraphToTfExecutorTest() {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
  }

  GraphDef CreateGraphDef(std::string graphdef_filename) {
    std::string file_path = TestDataPath() + graphdef_filename;
    std::string contents;
    GraphDef graph_def;
    auto status = riegeli::ReadAll(riegeli::FdReader(file_path), contents);
    if (!status.ok()) {
      return graph_def;
    }
    tsl::protobuf::TextFormat::ParseFromString(contents, &graph_def);
    return graph_def;
  }

  DialectRegistry registry_;
  MLIRContext context_;
};

TEST_F(GraphToTfExecutorTest, BasicConvertGraphToTfExecutorPasses) {
  Graph graph(OpRegistry::Global());
  GraphDebugInfo debug_info;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(),
                                     FunctionDefLibrary());
  GraphImportConfig specs;
  GraphDef graph_def = CreateGraphDef("valid_graph.txt");
  GraphConstructorOptions opts;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));

  TF_ASSERT_OK(
      ConvertGraphToTfExecutor(graph, debug_info, flib_def, specs, &context_));
}

}  // namespace
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
