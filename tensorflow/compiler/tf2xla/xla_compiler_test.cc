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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class XlaCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    client_ = xla::ClientLibrary::LocalClientOrDie();

    XlaOpRegistry::RegisterJitKernels();

    FunctionDefLibrary flib;
    flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));
  }

  XlaCompiler::Options DefaultOptions() {
    XlaCompiler::Options options;
    options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
    options.client = client_;
    return options;
  }

  std::unique_ptr<FunctionLibraryRuntime> BuildFunctionLibraryRuntime(
      const XlaCompiler& compiler) {
    return std::unique_ptr<FunctionLibraryRuntime>(NewFunctionLibraryRuntime(
        compiler.device_mgr(), /*env=*/nullptr, compiler.device(),
        TF_GRAPH_DEF_VERSION, flib_def_.get(), OptimizerOptions(),
        /*custom_kernel_creator=*/nullptr));
  }

  xla::Client* client_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
};

// Tests compilation of an empty graph.
TEST_F(XlaCompilerTest, EmptyReturnValues) {
  XlaCompiler compiler(DefaultOptions());
  auto flr = BuildFunctionLibraryRuntime(compiler);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph("add", std::move(graph), flr.get(),
                                     /*args=*/{}, /*use_tuple_arg=*/false,
                                     &result));

  // No computation should be generated.
  EXPECT_EQ(0, result.computation.handle().handle());
}

// Tests compilation and execution of a graph that adds two tensors.
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
  XlaCompiler compiler(DefaultOptions());
  auto flr = BuildFunctionLibraryRuntime(compiler);

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph("add", std::move(graph), flr.get(), args,
                                     /*use_tuple_arg=*/false, &result));

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

// Tests handling of compile-time constant outputs.
TEST_F(XlaCompilerTest, ConstantOutputs) {
  // Builds a graph with one compile-time constant output and one data-dependent
  // output, i.e.,
  // func(a) { b=7; c=-a; return b, c; }
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::Const<int32>(scope.WithOpName("B"), 7);
  auto c = ops::Neg(scope.WithOpName("C"), a);
  auto d = ops::_Retval(scope.WithOpName("D"), b, 0);
  auto e = ops::_Retval(scope.WithOpName("E"), c, 1);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[0].parameter = 0;

  {
    // Compiles the graph, with resolve_compile_time_constants enabled.
    XlaCompiler::Options options = DefaultOptions();
    options.resolve_compile_time_constants = true;
    XlaCompiler compiler(options);
    auto flr = BuildFunctionLibraryRuntime(compiler);

    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());

    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph("constants", std::move(graph_copy),
                                       flr.get(), args, /*use_tuple_arg=*/false,
                                       &result));

    ASSERT_EQ(2, result.outputs.size());
    EXPECT_TRUE(result.outputs[0].is_constant);
    test::ExpectTensorEqual<int32>(result.outputs[0].constant_value,
                                   test::AsScalar(7));
    EXPECT_FALSE(result.outputs[1].is_constant);

    // Tests that the generated computation works.
    std::unique_ptr<xla::Literal> param0_literal =
        xla::LiteralUtil::CreateR1<int32>({7, 42});
    std::unique_ptr<xla::GlobalData> param0_data =
        client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

    std::unique_ptr<xla::GlobalData> actual =
        client_->Execute(result.computation, {param0_data.get()})
            .ConsumeValueOrDie();
    std::unique_ptr<xla::Literal> actual_literal =
        client_->Transfer(*actual).ConsumeValueOrDie();

    std::unique_ptr<xla::Literal> expected_literal =
        xla::LiteralUtil::CreateR1<int32>({-7, -42});
    xla::LiteralTestUtil::ExpectEqual(*expected_literal, *actual_literal);
  }

  {
    // Compiles the graph, with resolve_compile_time_constants disabled.
    XlaCompiler::Options options = DefaultOptions();
    options.resolve_compile_time_constants = false;
    XlaCompiler compiler(options);
    auto flr = BuildFunctionLibraryRuntime(compiler);

    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());

    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph("constants", std::move(graph_copy),
                                       flr.get(), args, /*use_tuple_arg=*/false,
                                       &result));

    ASSERT_EQ(2, result.outputs.size());
    EXPECT_FALSE(result.outputs[0].is_constant);
    EXPECT_FALSE(result.outputs[1].is_constant);

    // Tests that the generated computation works.
    std::unique_ptr<xla::Literal> param0_literal =
        xla::LiteralUtil::CreateR1<int32>({7, 42});
    std::unique_ptr<xla::GlobalData> param0_data =
        client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

    std::unique_ptr<xla::GlobalData> actual =
        client_->Execute(result.computation, {param0_data.get()})
            .ConsumeValueOrDie();
    std::unique_ptr<xla::Literal> actual_literal =
        client_->Transfer(*actual).ConsumeValueOrDie();

    std::unique_ptr<xla::Literal> expected0 =
        xla::LiteralUtil::CreateR0<int32>(7);
    std::unique_ptr<xla::Literal> expected1 =
        xla::LiteralUtil::CreateR1<int32>({-7, -42});
    std::unique_ptr<xla::Literal> expected =
        xla::LiteralUtil::MakeTuple({expected0.get(), expected1.get()});
    xla::LiteralTestUtil::ExpectEqual(*expected, *actual_literal);
  }
}

}  // namespace
}  // namespace tensorflow
