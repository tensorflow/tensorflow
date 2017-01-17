/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class XlaCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    client_ = xla::ClientLibrary::LocalClientOrDie();

    XlaCompiler::Options options;
    options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
    options.client = client_;
    compiler_.reset(new XlaCompiler(options));

    XlaOpRegistry::RegisterJitKernels();

    FunctionDefLibrary flib;
    flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));
    flr_.reset(NewFunctionLibraryRuntime(
        compiler_->device_mgr(), /*env=*/nullptr, compiler_->device(),
        TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions(),
        /*custom_kernel_creator=*/nullptr));
  }

  xla::Client* client_;
  std::unique_ptr<XlaCompiler> compiler_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<FunctionLibraryRuntime> flr_;
};

TEST_F(XlaCompilerTest, Simple) {
  // Builds a graph that adds two Tensors.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Add(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[0].parameter = 0;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});
  args[1].parameter = 1;

  // Compiles the graph.
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler_->CompileGraph("add", std::move(graph), flr_.get(),
                                       args, /*use_tuple_arg=*/false, &result));

  // Tests that the generated computation works.
  std::unique_ptr<xla::Literal> param0_literal =
      xla::LiteralUtil::CreateR1<int32>({7, 42});
  std::unique_ptr<xla::Literal> param1_literal =
      xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(result.computation, {param0_data.get(), param1_data.get()})
          .ConsumeValueOrDie();
  std::unique_ptr<xla::Literal> actual_literal =
      client_->Transfer(*actual).ConsumeValueOrDie();

  std::unique_ptr<xla::Literal> expected_literal =
      xla::LiteralUtil::CreateR1<int32>({4, 143});
  xla::LiteralTestUtil::ExpectEqual(*expected_literal, *actual_literal);
}

}  // namespace
}  // namespace tensorflow
