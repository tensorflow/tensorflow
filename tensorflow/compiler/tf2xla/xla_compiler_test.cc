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
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

// Helper class to test the ability to pass resources through to XLA
// compiled kernels.
class DummyResourceForTest : public ResourceBase {
 public:
  string DebugString() override { return "dummy"; }
  void Increment() { ++value_; }
  int Get() { return value_; }

 private:
  int value_ = 0;
};

class DummyReadResourceOp : public XlaOpKernel {
 public:
  explicit DummyReadResourceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    ResourceMgr* rm = ctx->op_kernel_context()->resource_manager();
    OP_REQUIRES(ctx, rm, errors::Internal("No resource manager."));
    DummyResourceForTest* dummy;
    OP_REQUIRES_OK(ctx, rm->Lookup<DummyResourceForTest>(
                            rm->default_container(), "dummy", &dummy));
    dummy->Increment();
    dummy->Unref();

    ctx->SetOutput(0, ctx->Input(0));
  }
};

class DummyReadResourceCC {
 public:
  DummyReadResourceCC(const Scope& scope, const Input& value) {
    if (!scope.ok()) return;
    auto _value = ops::AsNodeOut(scope, value);
    if (!scope.ok()) return;
    Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("DummyReadResource");
    auto builder = NodeBuilder(unique_name, "DummyReadResource").Input(_value);
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    if (!scope.ok()) return;
    scope.UpdateStatus(scope.DoShapeInference(ret));
    if (!scope.ok()) return;
    this->output_ = Output(ret, 0);
  }
  Node* node() const { return output_.node(); }

  Output output_;
};

REGISTER_OP("DummyReadResource")
    .Input("input: int32")
    .Output("output: int32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
A dummy Op.

input: dummy input.
output: dummy output.
)doc");

REGISTER_XLA_OP(Name("DummyReadResource"), DummyReadResourceOp);

// DummyDuplicateOp is present purely to test multiple REGISTER_XLA_OP calls
// on the same Op name below.
class DummyDuplicateOp : public XlaOpKernel {
 public:
  explicit DummyDuplicateOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    ctx->SetOutput(0, ctx->Input(0));
  }
};

REGISTER_OP("DummyDuplicateOp")
    .Input("input: int32")
    .Output("output: int32")
    .Doc(R"doc(
A dummy Op.

input: dummy input.
output: dummy output.
)doc");

REGISTER_XLA_OP(Name("DummyDuplicateOp").Device(DEVICE_CPU_XLA_JIT),
                DummyDuplicateOp);
REGISTER_XLA_OP(Name("DummyDuplicateOp").Device(DEVICE_GPU_XLA_JIT),
                DummyDuplicateOp);

class XlaCompilerTest : public ::testing::Test {
 protected:
  XlaCompilerTest() : cpu_device_type_(DEVICE_CPU_XLA_JIT) {}

  void SetUp() override {
    client_ = xla::ClientLibrary::LocalClientOrDie();

    XlaOpRegistry::RegisterCompilationKernels();

    FunctionDefLibrary flib;
    flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));
  }

  XlaCompiler::Options DefaultOptions() {
    XlaCompiler::Options options;
    options.device_type = &cpu_device_type_;
    options.client = client_;
    options.flib_def = flib_def_.get();
    return options;
  }

  DeviceType cpu_device_type_;
  xla::Client* client_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
};

// Tests compilation of an empty graph.
TEST_F(XlaCompilerTest, EmptyReturnValues) {
  XlaCompiler compiler(DefaultOptions());

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph),
                                     /*args=*/{}, &result));

  // No computation should be generated.
  EXPECT_EQ(0, result.computation->handle().handle());
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
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(xla::S32, {2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(xla::S32, {2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  // Tests that the generated computation works.
  std::unique_ptr<xla::Literal> param0_literal =
      xla::Literal::CreateR1<int32>({7, 42});
  std::unique_ptr<xla::Literal> param1_literal =
      xla::Literal::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(*param0_literal).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(*param1_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .ConsumeValueOrDie();
  std::unique_ptr<xla::Literal> actual_literal =
      client_->Transfer(*actual).ConsumeValueOrDie();

  std::unique_ptr<xla::Literal> expected_literal =
      xla::Literal::CreateR1<int32>({4, 143});
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
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(xla::S32, {2});

  XlaCompiler::Options options = DefaultOptions();
  XlaCompiler compiler(options);
  {
    // Compiles the graph, with resolve_compile_time_constants enabled.

    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());

    XlaCompiler::CompileOptions compile_options;
    compile_options.resolve_compile_time_constants = true;
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(compile_options, "constants",
                                       std::move(graph_copy), args, &result));

    ASSERT_EQ(2, result.outputs.size());
    EXPECT_TRUE(result.outputs[0].is_constant);
    test::ExpectTensorEqual<int32>(result.outputs[0].constant_value,
                                   test::AsScalar(7));
    EXPECT_FALSE(result.outputs[1].is_constant);

    // Tests that the generated computation works.
    std::unique_ptr<xla::Literal> param0_literal =
        xla::Literal::CreateR1<int32>({7, 42});
    std::unique_ptr<xla::GlobalData> param0_data =
        client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

    std::unique_ptr<xla::GlobalData> actual =
        client_->Execute(*result.computation, {param0_data.get()})
            .ConsumeValueOrDie();
    std::unique_ptr<xla::Literal> actual_literal =
        client_->Transfer(*actual).ConsumeValueOrDie();

    std::unique_ptr<xla::Literal> expected_literal =
        xla::Literal::CreateR1<int32>({-7, -42});
    xla::LiteralTestUtil::ExpectEqual(*expected_literal, *actual_literal);
  }

  {
    // Compiles the graph, with resolve_compile_time_constants disabled.
    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());

    XlaCompiler::CompileOptions compile_options;
    compile_options.resolve_compile_time_constants = false;
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(compile_options, "constants",
                                       std::move(graph_copy), args, &result));

    ASSERT_EQ(2, result.outputs.size());
    EXPECT_FALSE(result.outputs[0].is_constant);
    EXPECT_FALSE(result.outputs[1].is_constant);

    // Tests that the generated computation works.
    std::unique_ptr<xla::Literal> param0_literal =
        xla::Literal::CreateR1<int32>({7, 42});
    std::unique_ptr<xla::GlobalData> param0_data =
        client_->TransferToServer(*param0_literal).ConsumeValueOrDie();

    std::unique_ptr<xla::GlobalData> actual =
        client_->Execute(*result.computation, {param0_data.get()})
            .ConsumeValueOrDie();
    std::unique_ptr<xla::Literal> actual_literal =
        client_->Transfer(*actual).ConsumeValueOrDie();

    std::unique_ptr<xla::Literal> expected0 = xla::Literal::CreateR0<int32>(7);
    std::unique_ptr<xla::Literal> expected1 =
        xla::Literal::CreateR1<int32>({-7, -42});
    std::unique_ptr<xla::Literal> expected =
        xla::Literal::MakeTuple({expected0.get(), expected1.get()});
    xla::LiteralTestUtil::ExpectEqual(*expected, *actual_literal);
  }
}

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, ResourceManager) {
  // Builds a graph that calls the dummy resource Op.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = DummyReadResourceCC(scope.WithOpName("B"), a);
  auto c = ops::_Retval(scope.WithOpName("C"), b.output_, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the argument.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(xla::S32, {2});

  DummyResourceForTest* resource = new DummyResourceForTest();

  // Compiles the graph.
  auto options = DefaultOptions();
  std::function<Status(ResourceMgr*)> populate_function =
      [resource](ResourceMgr* rm) {
        resource->Ref();
        return rm->Create(rm->default_container(), "dummy", resource);
      };
  options.populate_resource_manager = &populate_function;
  XlaCompiler compiler(options);

  EXPECT_EQ(0, resource->Get());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "dummy",
                                     std::move(graph), args, &result));

  EXPECT_EQ(1, resource->Get());

  resource->Unref();
}

}  // namespace
}  // namespace tensorflow
