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
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
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
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

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

  FunctionLibraryDefinition* LocalFlibDef(XlaCompiler* compiler) {
    return compiler->local_flib_def_.get();
  }

  DeviceType cpu_device_type_;
  xla::Client* client_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
};

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
    ctx->SetOutput(1, ctx->Input(0));
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
    this->output1_ = Output(ret, 0);
    this->output2_ = Output(ret, 1);
  }

  Output output1_;
  Output output2_;
};

REGISTER_OP("DummyReadResource")
    .Input("input: int32")
    .Output("output1: int32")
    .Output("output2: int32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
A dummy Op.

input: dummy input.
output1: dummy output.
output2: dummy output.
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


// Tests compilation and execution of an empty graph.
TEST_F(XlaCompilerTest, EmptyReturnValues) {
  XlaCompiler compiler(DefaultOptions());

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph),
                                     /*args=*/{}, &result));

  TF_ASSERT_OK(client_->Execute(*result.computation, {}).status());
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
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

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

  std::unique_ptr<xla::Literal> expected0 =
      xla::Literal::CreateR1<int32>({4, 143});
  std::unique_ptr<xla::Literal> expected_literal =
      xla::Literal::MakeTuple({expected0.get()});
  xla::LiteralTestUtil::ExpectEqual(*expected_literal, *actual_literal);
}

TEST_F(XlaCompilerTest, HasSaneErrorOnNonCompileTimeConstantInputToReshape) {
  // Builds a graph that adds reshapes a tensor, but with the shape not
  // statically known.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Reshape(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  Status status =
      compiler.CompileGraph(XlaCompiler::CompileOptions(), "reshape",
                            std::move(graph), args, &result);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(
      str_util::StrContains(status.error_message(), "depends on a parameter"))
      << status.error_message();
  EXPECT_TRUE(
      str_util::StrContains(status.error_message(), "[[Node: C = Reshape"))
      << status.error_message();
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
  args[0].shape = TensorShape({2});

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

    std::unique_ptr<xla::Literal> expected0 =
        xla::Literal::CreateR1<int32>({-7, -42});
    std::unique_ptr<xla::Literal> expected_literal =
        xla::Literal::MakeTuple({expected0.get()});
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
  auto c = ops::Add(scope.WithOpName("C"), b.output2_, b.output1_);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the argument.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});

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

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, DeterministicCompilation) {
  // Builds a graph that contains a node with two output edges. The compiler
  // should always traverse them in the same order.
  const int64 test_count = 2;

  std::vector<XlaCompiler::CompilationResult> results(test_count);

  for (int64 i = 0; i < test_count; ++i) {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
    auto b = ops::Neg(scope.WithOpName("B"), a);
    auto c = ops::Neg(scope.WithOpName("C"), a);
    auto d = ops::Add(scope.WithOpName("D"), b, c);
    auto e = ops::_Retval(scope.WithOpName("E"), d, 0);
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    TF_ASSERT_OK(scope.ToGraph(graph.get()));

    // Builds a description of the argument.
    std::vector<XlaCompiler::Argument> args(1);
    args[0].kind = XlaCompiler::Argument::kParameter;
    args[0].type = DT_INT32;
    args[0].shape = TensorShape({2});

    // Compiles the graph.
    auto options = DefaultOptions();
    XlaCompiler compiler(options);

    TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "dummy",
                                       std::move(graph), args, &results[i]));
  }

  for (int64 i = 1; i < test_count; ++i) {
    auto m1 =
        results[i - 1].computation->Snapshot().ValueOrDie()->entry().requests();
    auto m2 =
        results[i].computation->Snapshot().ValueOrDie()->entry().requests();
    // Check if every entry is the same.
    for (auto& entry1 : m1) {
      int64 key = entry1.first;
      auto value1 = entry1.second;
      auto entry2 = m2.find(key);
      auto value2 = entry2->second;
      EXPECT_TRUE(entry2 != m2.end());
      string str1, str2;
      value1.AppendToString(&str1);
      value2.AppendToString(&str2);
      EXPECT_EQ(str1, str2);
    }
  }
}

// Tests a computation that receives a TensorArray resource as input and
// updates it.
TEST_F(XlaCompilerTest, CanPassTensorArraysToAndFromComputation) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto flow = ops::Const<float>(scope, {});
  auto grad1 = ops::TensorArrayGrad(scope, arg, flow, "grad1");
  auto grad2 = ops::TensorArrayGrad(scope, arg, grad1.flow_out, "grad2");
  auto index = ops::Const<int32>(scope, 1);
  auto write = ops::TensorArrayWrite(scope, grad1.grad_handle, index, index,
                                     grad2.flow_out);
  auto read = ops::TensorArrayRead(scope, arg, index, write.flow_out, DT_INT32);
  auto retval = ops::_Retval(scope.WithOpName("retval"), read, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kTensorArray;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({});
  args[0].tensor_array_size = 2;
  args[0].tensor_array_gradients = {"grad2"};

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  ASSERT_EQ(1, result.resource_updates.size());
  const XlaCompiler::ResourceUpdate& update = result.resource_updates[0];
  EXPECT_EQ(0, update.input_index);
  EXPECT_EQ(DT_INT32, update.type);
  EXPECT_EQ((std::set<string>{"grad1", "grad2"}),
            update.tensor_array_gradients_accessed);

  // Tests that the generated computation works.
  std::unique_ptr<xla::Literal> input_base =
      xla::Literal::CreateR1<int32>({7, 42});
  std::unique_ptr<xla::Literal> input_grad2 =
      xla::Literal::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::Literal> input =
      xla::Literal::MakeTuple({input_base.get(), input_grad2.get()});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(*input).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_->Execute(*result.computation, {param0_data.get()})
          .ConsumeValueOrDie();
  std::unique_ptr<xla::Literal> actual_literal =
      client_->Transfer(*actual).ConsumeValueOrDie();

  std::unique_ptr<xla::Literal> output_read = xla::Literal::CreateR0<int32>(42);
  std::unique_ptr<xla::Literal> output_base =
      xla::Literal::CreateR1<int32>({7, 42});
  std::unique_ptr<xla::Literal> output_grad1 =
      xla::Literal::CreateR1<int32>({0, 1});
  std::unique_ptr<xla::Literal> output_grad2 =
      xla::Literal::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::Literal> output_resource = xla::Literal::MakeTuple(
      {output_base.get(), output_grad1.get(), output_grad2.get()});
  std::unique_ptr<xla::Literal> expected_literal =
      xla::Literal::MakeTuple({output_read.get(), output_resource.get()});
  xla::LiteralTestUtil::ExpectEqual(*expected_literal, *actual_literal);
}

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, UnwrittenTensorArrayGradientsAreNotComputationOutputs) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto flow = ops::Const<float>(scope, {});
  auto grad1 = ops::TensorArrayGrad(scope, arg, flow, "grad1");
  auto index = ops::Const<int32>(scope, 1);
  auto read = ops::TensorArrayRead(scope, arg, index, grad1.flow_out, DT_INT32);
  auto retval = ops::_Retval(scope.WithOpName("retval"), read, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kTensorArray;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({});
  args[0].tensor_array_size = 2;
  args[0].tensor_array_gradients = {"grad1"};

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  EXPECT_EQ(0, result.resource_updates.size());
}

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, NewTensorArrayGradientsAreComputationOutputs) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto flow = ops::Const<float>(scope, {});
  auto grad1 = ops::TensorArrayGrad(scope, arg, flow, "grad2");
  auto index = ops::Const<int32>(scope, 1);
  auto read = ops::TensorArrayRead(scope, arg, index, grad1.flow_out, DT_INT32);
  auto retval = ops::_Retval(scope.WithOpName("retval"), read, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kTensorArray;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({});
  args[0].tensor_array_size = 2;
  args[0].tensor_array_gradients = {"grad1"};

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  EXPECT_EQ(1, result.resource_updates.size());
}

// Tests CompileFunction with undefined function fails.
TEST_F(XlaCompilerTest, UndefinedFunctionFails) {
  XlaCompiler compiler(DefaultOptions());

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  XlaCompiler::CompilationResult result;
  NameAttrList name_attr;
  name_attr.set_name("Function_NotDefined_");
  Status status =
      compiler.CompileFunction(XlaCompiler::CompileOptions(), name_attr,
                               /*args=*/{}, &result);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(str_util::StrContains(StringPiece(status.error_message()),
                                    "is not defined."))
      << status.error_message();
}

FunctionDef FillFn() {
  return FunctionDefHelper::Define(
      // Name
      "FillFn",
      // Args
      {"x: T", "dims: int32"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {{{"y"}, "Fill", {"dims", "x"}, {{"T", "$T"}}}});
}

TEST_F(XlaCompilerTest, FunctionCallWithConstants) {
  // Certain operations in a function, "Fill" for example, requires the
  // operator's argument to be a compile-time constant instead of a parameter.
  // This testcase tests if XlaCompiler can handle such operators inside
  // function calls.
  XlaCompiler compiler(DefaultOptions());

  FunctionDefLibrary flib;
  *flib.add_function() = FillFn();

  TF_ASSERT_OK(flib_def_->AddFunctionDef(FillFn()));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto value = ops::Const<int32>(scope.WithOpName("value"), 1, {});
  auto shape = ops::Const<int32>(scope.WithOpName("shape"), {5}, {1});
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib));

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("fill", "FillFn", flib_def_.get())
                   .Input(value.name(), 0, DT_INT32)
                   .Input(shape.name(), 1, DT_INT32)
                   .Finalize(&def));
  Status status;
  Node* fill = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(fill));
  scope.graph()->AddEdge(value.node(), 0, fill, 0);
  scope.graph()->AddEdge(shape.node(), 0, fill, 1);

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(fill), 0);

  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the argument.
  std::vector<XlaCompiler::Argument> args;

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "fill",
                                     std::move(graph), args, &result));
}

// Tests CompileFunction with a local function lookup failing, fails with
// informative error about both lookups.
TEST_F(XlaCompilerTest, LocalFunctionWithWrongArgumentsFail) {
  XlaCompiler compiler(DefaultOptions());

  auto local_flib_def = LocalFlibDef(&compiler);
  TF_ASSERT_OK(local_flib_def->AddFunctionDef(test::function::XTimesTwo()));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  XlaCompiler::CompilationResult result;
  NameAttrList name_attr;
  name_attr.set_name("XTimesTwo");
  Status status =
      compiler.CompileFunction(XlaCompiler::CompileOptions(), name_attr,
                               /*args=*/{}, &result);

  ASSERT_FALSE(status.ok());
  // Flib lookup failure.
  EXPECT_TRUE(str_util::StrContains(StringPiece(status.error_message()),
                                    "is not defined."))
      << status.error_message();
  // Local flib lookup failure.
  EXPECT_TRUE(str_util::StrContains(StringPiece(status.error_message()),
                                    "Attr T is not found"))
      << status.error_message();
}

// Tests a simple graph that reads and writes a variable.
TEST_F(XlaCompilerTest, Variables) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  auto write = ops::AssignAddVariableOp(scope, var, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto read_plus_one = ops::Add(scope, read, ops::Const<int32>(scope, 1));
  auto d = ops::_Retval(scope.WithOpName("D"), read_plus_one, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

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

  std::unique_ptr<xla::Literal> expected0 =
      xla::Literal::CreateR1<int32>({5, 144});
  std::unique_ptr<xla::Literal> expected1 =
      xla::Literal::CreateR1<int32>({4, 143});
  std::unique_ptr<xla::Literal> expected_literal =
      xla::Literal::MakeTuple({expected0.get(), expected1.get()});
  xla::LiteralTestUtil::ExpectEqual(*expected_literal, *actual_literal);
}

// Tests a simple graph that reads and writes a variable, with a
// variable_representation_shape_fn passed to the compiler that flattens all
// variable tensors to vectors.
TEST_F(XlaCompilerTest, VariableRepresentationShapeFunction) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  auto write = ops::AssignAddVariableOp(scope, var, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto read_plus_one = ops::Add(scope, read, ops::Const<int32>(scope, 1));
  auto d = ops::_Retval(scope.WithOpName("D"), read_plus_one, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 2});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2, 2});

  // Compiles the graph.
  XlaCompiler::Options options = DefaultOptions();
  options.variable_representation_shape_fn = [](const TensorShape& shape,
                                                DataType type) {
    return TensorShape({shape.num_elements()});
  };
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  // Tests that the generated computation works.
  std::unique_ptr<xla::Literal> param0_literal =
      xla::Literal::CreateR2<int32>({{4, 55}, {1, -3}});
  std::unique_ptr<xla::Literal> param1_literal =
      xla::Literal::CreateR1<int32>({22, 11, 33, 404});
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

  std::unique_ptr<xla::Literal> expected0 =
      xla::Literal::CreateR2<int32>({{27, 67}, {35, 402}});
  std::unique_ptr<xla::Literal> expected1 =
      xla::Literal::CreateR1<int32>({26, 66, 34, 401});
  std::unique_ptr<xla::Literal> expected_literal =
      xla::Literal::MakeTuple({expected0.get(), expected1.get()});
  xla::LiteralTestUtil::ExpectEqual(*expected_literal, *actual_literal);
}

}  // namespace
}  // namespace tensorflow
