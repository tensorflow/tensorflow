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

#include "absl/strings/match.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class XlaCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    client_ = xla::ClientLibrary::LocalClientOrDie();

    XlaOpRegistry::RegisterCompilationKernels();

    FunctionDefLibrary flib;
    flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));
  }

  XlaCompiler::Options DefaultOptions() {
    XlaCompiler::Options options;
    options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
    options.client = client_;
    options.flib_def = flib_def_.get();
    return options;
  }

  FunctionLibraryDefinition* LocalFlibDef(XlaCompiler* compiler) {
    return compiler->local_flib_def_.get();
  }

  xla::Client* client_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
};

namespace {

// Helper class to test the ability to pass resources through to XLA
// compiled kernels.
class DummyResourceForTest : public ResourceBase {
 public:
  string DebugString() const override { return "dummy"; }
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
  xla::Literal param0_literal = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal param1_literal = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .ConsumeValueOrDie();
  xla::Literal actual_literal = client_->Transfer(*actual).ConsumeValueOrDie();

  xla::Literal expected0 = xla::LiteralUtil::CreateR1<int32>({4, 143});
  xla::Literal expected_literal = xla::LiteralUtil::MakeTuple({&expected0});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

// Tests compilation of a graph where the _Retval node is not necessarily last
// amongst the graph nodes in construction order, and always_return_tuple is
// false. Regression test for bug where the wrong value was returned.
TEST_F(XlaCompilerTest, OutOfOrderGraph) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  // The _Retval node is not last in construction order.
  auto d = ops::_Retval(scope.WithOpName("D"), a, 0);
  auto c = ops::Add(scope.WithOpName("C"), a, b);

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

  XlaCompiler::CompileOptions compile_options;
  compile_options.always_return_tuple = false;
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));

  // Tests that the generated computation works.
  xla::Literal param0_literal = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal param1_literal = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .ConsumeValueOrDie();
  xla::Literal actual_literal = client_->Transfer(*actual).ConsumeValueOrDie();

  EXPECT_TRUE(xla::LiteralTestUtil::Equal(param0_literal, actual_literal));
}

// Tests that the compiler can correctly propagate the layout assigned by
// shape_representation_fn_ to resource returns that have not been written to.
TEST_F(XlaCompilerTest, HonorShapeRepresentationFnForUnwrittenResource) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 0);
  auto d = ops::_Retval(scope.WithOpName("D"), var, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kVariable;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});

  auto options = DefaultOptions();
  options.shape_representation_fn =
      [](const TensorShape& shape, DataType dt,
         bool use_fast_memory) -> StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dt, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    return xla_shape;
  };
  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  XlaCompiler::CompileOptions compile_options;
  compile_options.return_updated_values_for_all_resources = true;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));
  xla::Shape transposed =
      xla::ShapeUtil::MakeShapeWithLayout(xla::S32, {2, 3}, {0, 1});
  // Check that the return shapes are correctly tranposed.
  EXPECT_EQ(result.xla_output_shape,
            xla::ShapeUtil::MakeTupleShape({transposed}));
}

// Tests that the compiler can correctly propagate fast mem attribute for input
// resource variable.
TEST_F(XlaCompilerTest, HonorShapeRepresentationFnForFastMemVar) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 0);
  auto d = ops::_Retval(scope.WithOpName("D"), var, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kVariable;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});
  args[0].fast_mem = true;

  auto options = DefaultOptions();
  int fast_mem_arg_count = 0;
  options.shape_representation_fn =
      [&fast_mem_arg_count](const TensorShape& shape, DataType dt,
                            bool use_fast_memory) -> StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dt, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    if (use_fast_memory) {
      fast_mem_arg_count++;
    }
    return xla_shape;
  };
  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  XlaCompiler::CompileOptions compile_options;
  compile_options.return_updated_values_for_all_resources = true;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));
  // Count 2: one for argument, one for the return value.
  EXPECT_EQ(fast_mem_arg_count, 2);
}

// Tests that the compiler can correctly propagate the layout assigned by
// shape_representation_fn_ to return types.
TEST_F(XlaCompilerTest, HonorShapeRepresentationFnForRetVal) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  // Adds an identity op around the resource to make sure identity ops propagate
  // resources correctly.
  auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
  auto write = ops::AssignAddVariableOp(scope, identity, a);
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
  args[0].shape = TensorShape({2, 3});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2, 3});

  auto options = DefaultOptions();
  options.shape_representation_fn =
      [](const TensorShape& shape, DataType dt,
         bool use_fast_memory) -> StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dt, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    return xla_shape;
  };
  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));
  xla::Shape transposed =
      xla::ShapeUtil::MakeShapeWithLayout(xla::S32, {2, 3}, {0, 1});
  // Check that the return shapes are correctly tranposed.
  EXPECT_EQ(result.xla_output_shape,
            xla::ShapeUtil::MakeTupleShape({transposed, transposed}));
  EXPECT_EQ(result.computation->GetProgramShape().ConsumeValueOrDie().result(),
            xla::ShapeUtil::MakeTupleShape({transposed, transposed}));
}

// The layout of resource variable shouldn't change after transpose
TEST_F(XlaCompilerTest, TransposeVariables) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  // Adds an identity op around the resource to make sure identity ops propagate
  // resources correctly.
  auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
  auto write = ops::AssignAddVariableOp(scope, identity, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto transposed_read = ops::Transpose(scope, read, {1, 0});
  auto reshape = ops::Reshape(scope, transposed_read, {2, 3});
  auto d = ops::_Retval(scope.WithOpName("D"), reshape, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2, 3});
  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "transpose",
                                     std::move(graph), args, &result));
  xla::Shape transposed =
      xla::ShapeUtil::MakeShapeWithLayout(xla::S32, {2, 3}, {1, 0});
  // Check that the return shapes are correctly tranposed.
  EXPECT_EQ(result.xla_output_shape,
            xla::ShapeUtil::MakeTupleShape({transposed, transposed}));
}

// Unranked fake param returns a 0 shaped tensor.
TEST_F(XlaCompilerTest, UnrankedFakeParam) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  PartialTensorShape shape;
  auto a = ops::FakeParam(scope, DT_INT32, shape);
  auto ret = ops::_Retval(scope.WithOpName("D"), a, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "compile",
                                     std::move(graph), {}, &result));
  // Check that the return shapes are correctly tranposed.
  EXPECT_EQ(result.xla_output_shape,
            xla::ShapeUtil::MakeTupleShape(
                {xla::ShapeUtil::MakeShape(xla::S32, {0})}));
}

// Tests that the compiler doesn't reorder the parameters.
TEST_F(XlaCompilerTest, MixedOrderArguments) {
  for (bool swap_order : {false, true}) {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto var =
        ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, swap_order ? 0 : 1);
    auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, swap_order ? 1 : 0);
    // Adds an identity op around the resource to make sure identity ops
    // propagate resources correctly.
    auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
    auto write = ops::AssignAddVariableOp(scope, identity, a);
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

    if (swap_order) {
      // Even after swapping arguments, the compiler should maintain the new
      // ordering of parameters.
      std::swap(args[0], args[1]);
    }
    // Compiles the graph.
    XlaCompiler compiler(DefaultOptions());

    XlaCompiler::CompileOptions compile_options;
    compile_options.always_return_tuple = false;
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                       args, &result));

    EXPECT_THAT(result.input_mapping, ::testing::ElementsAre(0, 1));
  }
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
      absl::StrContains(status.error_message(), "depends on a parameter"))
      << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.error_message(), "{{node C}}"))
      << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "must be a compile-time constant"))
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
    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());

    XlaCompiler::CompileOptions compile_options;
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(compile_options, "constants",
                                       std::move(graph_copy), args, &result));

    ASSERT_EQ(2, result.outputs.size());
    EXPECT_FALSE(result.outputs[0].is_constant);
    EXPECT_FALSE(result.outputs[1].is_constant);

    // Tests that the generated computation works.
    xla::Literal param0_literal = xla::LiteralUtil::CreateR1<int32>({7, 42});
    std::unique_ptr<xla::GlobalData> param0_data =
        client_->TransferToServer(param0_literal).ConsumeValueOrDie();

    std::unique_ptr<xla::GlobalData> actual =
        client_->Execute(*result.computation, {param0_data.get()})
            .ConsumeValueOrDie();
    xla::Literal actual_literal =
        client_->Transfer(*actual).ConsumeValueOrDie();

    xla::Literal expected0 = xla::LiteralUtil::CreateR0<int32>(7);
    xla::Literal expected1 = xla::LiteralUtil::CreateR1<int32>({-7, -42});
    xla::Literal expected =
        xla::LiteralUtil::MakeTuple({&expected0, &expected1});
    EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected, actual_literal));
  }
}

TEST_F(XlaCompilerTest, ConstantOutputsOfFunctionalNode) {
  // Define a function with one compile-time constant output and one
  // data-dependent output.
  // @function.Defun(noinline=True)
  // foo(a) {b=7; return b, a; }
  const Tensor seven = test::AsScalar<int>(7);
  FunctionDef fdef = FunctionDefHelper::Create(
      "foo", {"a_0:int32"}, {"const:int32", "a:int32"}, {},
      {
          {{"Const"}, "Const", {}, {{"dtype", DT_INT32}, {"value", seven}}},
      },
      {{"a", "a_0"}, {"const", "Const:output:0"}});
  (*fdef.mutable_attr())["_noinline"].set_b(true);
  FunctionDefLibrary fdef_lib;
  *(fdef_lib.add_function()) = fdef;
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));
    auto arg = ops::_Arg(scope.WithOpName("input_arg"), DT_INT32, 0);
    NodeDef foo;
    foo.set_name("foo");
    foo.set_op("foo");
    *foo.add_input() = "input_arg";
    Status status;
    scope.graph()->AddNode(foo, &status);
    TF_ASSERT_OK(status);
    NodeDef retval_1;
    retval_1.set_name("retval_0");
    retval_1.set_op(FunctionLibraryDefinition::kRetOp);
    *retval_1.add_input() = "foo";
    (*retval_1.mutable_attr())["T"].set_type(DT_INT32);
    (*retval_1.mutable_attr())["index"].set_i(0);
    scope.graph()->AddNode(retval_1, &status);
    TF_ASSERT_OK(status);
    NodeDef retval_2;
    retval_2.set_name("retval_1");
    retval_2.set_op(FunctionLibraryDefinition::kRetOp);
    *retval_2.add_input() = "foo:1";
    (*retval_2.mutable_attr())["T"].set_type(DT_INT32);
    (*retval_2.mutable_attr())["index"].set_i(1);
    scope.graph()->AddNode(retval_2, &status);
    TF_ASSERT_OK(status);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({1});

  XlaCompiler::Options options = DefaultOptions();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  options.flib_def = &flib_def;
  XlaCompiler compiler(options);

  XlaCompiler::CompileOptions compile_options;
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "constants",
                                     std::move(graph), args, &result));

  ASSERT_EQ(2, result.outputs.size());
  EXPECT_FALSE(result.outputs[1].is_constant);
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
  const int64_t test_count = 2;

  std::vector<XlaCompiler::CompilationResult> results(test_count);

  for (int64_t i = 0; i < test_count; ++i) {
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

  for (int64_t i = 1; i < test_count; ++i) {
    const auto& m1 = results[i - 1].computation->proto();
    const auto& m2 = results[i].computation->proto();
    ASSERT_EQ(m1.computations_size(), m2.computations_size());
    // Check if every hlo computation is the same.
    for (int k = 0; k < m1.computations_size(); k++) {
      const auto& c1 = m1.computations(k);
      const auto& c2 = m2.computations(k);
      ASSERT_EQ(c1.instructions_size(), c2.instructions_size());
      for (int j = 0; j < c1.instructions_size(); j++) {
        auto instr1 = c1.instructions(j);
        auto instr2 = c2.instructions(j);
        instr1.clear_name();
        instr1.clear_id();
        instr1.clear_operand_ids();
        instr2.clear_name();
        instr2.clear_id();
        instr2.clear_operand_ids();
        // The names of instructions were uniquified by the XlaBuilder and the
        // unique ids may be different, the rest of the fields should be
        // identical.
        string str1, str2;
        LOG(INFO) << "instr1 = " << instr1.DebugString();
        LOG(INFO) << "instr2 = " << instr2.DebugString();
        instr1.AppendPartialToString(&str1);
        instr2.AppendPartialToString(&str2);
        EXPECT_EQ(str1, str2);
      }
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
  args[0].max_array_size = 2;
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
  xla::Literal input_base = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal input_grad2 = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  xla::Literal input = xla::LiteralUtil::MakeTuple({&input_base, &input_grad2});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(input).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_->Execute(*result.computation, {param0_data.get()})
          .ConsumeValueOrDie();
  xla::Literal actual_literal = client_->Transfer(*actual).ConsumeValueOrDie();

  xla::Literal output_read = xla::LiteralUtil::CreateR0<int32>(42);
  xla::Literal output_base = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal output_grad1 = xla::LiteralUtil::CreateR1<int32>({0, 1});
  xla::Literal output_grad2 = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  xla::Literal output_resource =
      xla::LiteralUtil::MakeTuple({&output_base, &output_grad1, &output_grad2});
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&output_read, &output_resource});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
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
  args[0].max_array_size = 2;
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
  args[0].max_array_size = 2;
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
  EXPECT_TRUE(absl::StrContains(status.error_message(), "is not defined."))
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
  EXPECT_TRUE(absl::StrContains(status.error_message(), "is not defined."))
      << status.error_message();
  // Local flib lookup failure.
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Attr T is not found"))
      << status.error_message();
}

FunctionDef SliceFn() {
  return FunctionDefHelper::Define(
      // Name
      "SliceFn",
      // Args
      {"x: T", "begin: Index", "size: Index"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}", "Index: {int32,int64}"},
      // Nodes
      {{{"y"},
        "Slice",
        {"x", "begin", "size"},
        {{"T", "$T"}, {"Index", "$Index"}}}});
}

TEST_F(XlaCompilerTest, SliceWithDynamicBegins) {
  // Certain operations in a function, "Slice" for example, support both dynamic
  // inputs and static inputs. This test checks that dynamic inputs can also
  // be supported in a function call.
  XlaCompiler compiler(DefaultOptions());

  FunctionDefLibrary flib;
  *flib.add_function() = SliceFn();

  TF_ASSERT_OK(flib_def_->AddFunctionDef(SliceFn()));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto value = ops::Const<int32>(scope.WithOpName("shape"), {5}, {1});
  auto begin = ops::_Arg(scope.WithOpName("arg"), DT_INT32, 0);
  auto size = ops::Const<int32>(scope.WithOpName("value"), {1}, {1});

  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib));

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("slice", "SliceFn", flib_def_.get())
                   .Input(value.name(), 0, DT_INT32)
                   .Input(begin.node()->name(), 1, DT_INT32)
                   .Input(size.name(), 2, DT_INT32)
                   .Finalize(&def));
  Status status;
  Node* slice = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(slice));
  scope.graph()->AddEdge(value.node(), 0, slice, 0);
  scope.graph()->AddEdge(begin.node(), 0, slice, 1);
  scope.graph()->AddEdge(size.node(), 0, slice, 2);

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(slice), 0);

  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the argument.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({1});

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "slice",
                                     std::move(graph), args, &result));
}

void RunAndCheckVariablesComputation(
    xla::Client* client, const XlaCompiler::CompilationResult& result) {
  xla::Literal param0_literal = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal param1_literal = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param0_data =
      client->TransferToServer(param0_literal).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> param1_data =
      client->TransferToServer(param1_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .ConsumeValueOrDie();
  xla::Literal actual_literal = client->Transfer(*actual).ConsumeValueOrDie();

  xla::Literal expected0 = xla::LiteralUtil::CreateR1<int32>({5, 144});
  xla::Literal expected1 = xla::LiteralUtil::CreateR1<int32>({4, 143});
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&expected0, &expected1});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

// Tests a simple graph that reads and writes a variable.
TEST_F(XlaCompilerTest, Variables) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  // Adds an identity op around the resource to make sure identity ops propagate
  // resources correctly.
  auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
  auto write = ops::AssignAddVariableOp(scope, identity, a);
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
  RunAndCheckVariablesComputation(client_, result);
}

TEST_F(XlaCompilerTest, ResultLayoutSingle) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Retval(scope.WithOpName("RET"), a, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});

  auto options = DefaultOptions();
  // Sets the representation function to return a non-default layout.
  options.shape_representation_fn =
      [](const TensorShape& shape, DataType type,
         bool use_fast_memory) -> StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(type, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    return xla_shape;
  };

  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  auto compile_options = XlaCompiler::CompileOptions();
  compile_options.always_return_tuple = false;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "id", std::move(graph),
                                     args, &result));
  EXPECT_TRUE(xla::ShapeUtil::Equal(
      result.xla_output_shape,
      xla::ShapeUtil::MakeShapeWithLayout(xla::S32, {2, 3}, {0, 1})));
  EXPECT_EQ(result.computation->GetProgramShape().ConsumeValueOrDie().result(),
            result.xla_output_shape);
}

TEST_F(XlaCompilerTest, ResultLayoutMultiple) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Retval(scope.WithOpName("RET1"), a, 0);
  auto c = ops::_Retval(scope.WithOpName("RET2"), a, 1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});

  auto options = DefaultOptions();
  // Sets the representation function to return a non-default layout.
  options.shape_representation_fn =
      [](const TensorShape& shape, DataType type,
         bool use_fast_memory) -> StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(type, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    return xla_shape;
  };

  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "id",
                                     std::move(graph), args, &result));
  xla::Shape result_shape =
      xla::ShapeUtil::MakeShapeWithLayout(xla::S32, {2, 3}, {0, 1});

  EXPECT_TRUE(xla::ShapeUtil::Equal(
      result.xla_output_shape,
      xla::ShapeUtil::MakeTupleShape({result_shape, result_shape})));
  EXPECT_EQ(result.computation->GetProgramShape().ConsumeValueOrDie().result(),
            result.xla_output_shape);
}

// Tests a simple graph that reads and writes a variable.
TEST_F(XlaCompilerTest, ReturnResourceHandleOnly) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 0);
  auto d = ops::_Retval(scope.WithOpName("D"), var, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kVariable;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  // Tests that the generated computation works.
  xla::Literal param1_literal = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_->Execute(*result.computation, {param1_data.get()})
          .ConsumeValueOrDie();
  xla::Literal actual_literal = client_->Transfer(*actual).ConsumeValueOrDie();

  xla::Literal expected_literal = xla::LiteralUtil::MakeTuple({});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

TEST_F(XlaCompilerTest, ReturnResourceHandle) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  // Adds an identity op around the resource to make sure identity ops propagate
  // resources correctly.
  auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
  auto write = ops::AssignAddVariableOp(scope, identity, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto read_plus_one = ops::Add(scope, read, ops::Const<int32>(scope, 1));
  auto r = ops::_Retval(scope.WithOpName("R"), var, 0);
  auto d = ops::_Retval(scope.WithOpName("D"), read_plus_one, 1);

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
  RunAndCheckVariablesComputation(client_, result);
}

StatusOr<std::unique_ptr<Graph>> BuildTestGraph() {
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
  TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));
  return std::move(graph);
}

// Tests a simple graph that reads and writes a variable, with a
// shape_representation_fn passed to the compiler that flattens all
// variable tensors to vectors.
TEST_F(XlaCompilerTest, VariableRepresentationShapeFunction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Graph> graph, BuildTestGraph());

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
  options.shape_representation_fn =
      [](const TensorShape& shape, DataType type,
         bool use_fast_memory) -> StatusOr<xla::Shape> {
    xla::PrimitiveType ptype;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(type, &ptype));
    return xla::ShapeUtil::MakeShape(ptype, {shape.num_elements()});
  };
  XlaCompiler compiler(options);

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = false;  // Only reshape variables.

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::ProgramShape> program_shape,
                          client_->GetComputationShape(*result.computation));

  ASSERT_EQ(program_shape->parameters_size(), 2);
  EXPECT_TRUE(
      xla::ShapeUtil::Compatible(program_shape->parameters(0),
                                 xla::ShapeUtil::MakeShape(xla::S32, {2, 2})));
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->parameters(1), xla::ShapeUtil::MakeShape(xla::S32, {4})));
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->result(),
      xla::ShapeUtil::MakeTupleShape(
          {xla::ShapeUtil::MakeShape(xla::S32, {2, 2}),
           xla::ShapeUtil::MakeShape(xla::S32, {4})})));

  // Tests that the generated computation works.
  xla::Literal param0_literal =
      xla::LiteralUtil::CreateR2<int32>({{4, 55}, {1, -3}});
  xla::Literal param1_literal =
      xla::LiteralUtil::CreateR1<int32>({22, 11, 33, 404});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .ConsumeValueOrDie();
  xla::Literal actual_literal = client_->Transfer(*actual).ConsumeValueOrDie();

  xla::Literal expected0 =
      xla::LiteralUtil::CreateR2<int32>({{27, 67}, {35, 402}});
  xla::Literal expected1 = xla::LiteralUtil::CreateR1<int32>({26, 66, 34, 401});
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&expected0, &expected1});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

TEST_F(XlaCompilerTest, ArgRetvalShapeRepresentationFunction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Graph> graph, BuildTestGraph());

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
  options.shape_representation_fn =
      [](const TensorShape& shape, DataType type,
         bool use_fast_memory) -> StatusOr<xla::Shape> {
    xla::PrimitiveType ptype;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(type, &ptype));
    return xla::ShapeUtil::MakeShape(ptype, {shape.num_elements()});
  };
  XlaCompiler compiler(options);

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;  // Reshape args and retvals.

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::ProgramShape> program_shape,
                          client_->GetComputationShape(*result.computation));

  ASSERT_EQ(program_shape->parameters_size(), 2);
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->parameters(0), xla::ShapeUtil::MakeShape(xla::S32, {4})));
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->parameters(1), xla::ShapeUtil::MakeShape(xla::S32, {4})));
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->result(),
      xla::ShapeUtil::MakeTupleShape(
          {xla::ShapeUtil::MakeShape(xla::S32, {4}),
           xla::ShapeUtil::MakeShape(xla::S32, {4})})));

  // Tests that the generated computation works.
  xla::Literal param0_literal =
      xla::LiteralUtil::CreateR1<int32>({4, 55, 1, -3});
  xla::Literal param1_literal =
      xla::LiteralUtil::CreateR1<int32>({22, 11, 33, 404});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(param0_literal).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .ConsumeValueOrDie();
  xla::Literal actual_literal = client_->Transfer(*actual).ConsumeValueOrDie();

  xla::Literal expected0 = xla::LiteralUtil::CreateR1<int32>({27, 67, 35, 402});
  xla::Literal expected1 = xla::LiteralUtil::CreateR1<int32>({26, 66, 34, 401});
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&expected0, &expected1});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

// Tests a graph which has a function with an invalid op.
TEST_F(XlaCompilerTest, FunctionWithInvalidOp) {
  XlaCompiler compiler(DefaultOptions());

  FunctionDefLibrary flib;
  FunctionDef fn = FillFn();
  NodeDef* node = fn.add_node_def();
  node->set_name("Invalid");
  node->set_op("InvalidOp"); /* unsupported op */
  node = fn.add_node_def();
  node->set_name("Switch");
  node->set_op("Switch"); /* control flow node */
  *flib.add_function() = fn;

  TF_ASSERT_OK(flib_def_->AddFunctionDef(fn));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto value = ops::Const<int32>(scope.WithOpName("value"), 1, {});
  auto shape = ops::Const<int32>(scope.WithOpName("shape"), {5}, {1});
  TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(flib));

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("fill_fn", "FillFn", flib_def_.get())
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

  std::vector<XlaCompiler::Argument> args;
  XlaCompiler::CompilationResult result;
  status = compiler.CompileGraph(XlaCompiler::CompileOptions(), "fill",
                                 std::move(graph), args, &result);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.error_message(), "InvalidOp"))
      << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.error_message(), "{{node fill_fn}}"))
      << status.error_message();
}

// Tests a graph which has a node with invalid data type.
TEST_F(XlaCompilerTest, NodeWithInvalidDataType) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  NodeDef shape;
  shape.set_name("Shape");
  shape.set_op("Shape");
  (*shape.mutable_attr())["T"].set_type(DT_INT32);
  (*shape.mutable_attr())["out_type"].set_type(DT_BOOL); /* invalid type */
  Status status;
  Node* shape_node = graph->AddNode(shape, &status);
  TF_ASSERT_OK(status);
  graph->AddControlEdge(graph->source_node(), shape_node);

  std::vector<XlaCompiler::Argument> args;
  XlaCompiler::CompilationResult result;
  XlaCompiler compiler(DefaultOptions());
  status = compiler.CompileGraph(XlaCompiler::CompileOptions(), "invalid_type",
                                 std::move(graph), args, &result);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.error_message(),
                                "is not in the list of allowed values"))
      << status.error_message();
  EXPECT_TRUE(absl::StrContains(status.error_message(), "{{node Shape}}"))
      << status.error_message();
}

TEST_F(XlaCompilerTest, SingleOpWithoutInputs) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  NodeDef no_op;
  no_op.set_name("NoOp");
  no_op.set_op("NoOp");
  Status status;
  graph->AddNode(no_op, &status);
  TF_ASSERT_OK(status);

  std::vector<XlaCompiler::Argument> args;
  XlaCompiler compiler(DefaultOptions());
  // No control edge linking NoOp with source/sink.
  {
    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "NoOp",
                                       std::move(graph_copy), args, &result));
  }
}

class DummySideEffectingOp : public XlaOpKernel {
 public:
  explicit DummySideEffectingOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, ctx->compiler()->SetNodeToken(
                            name(), xla::CreateToken(ctx->builder())));
  }
};

REGISTER_OP("DummySideEffectingOp");

REGISTER_XLA_OP(Name("DummySideEffectingOp"), DummySideEffectingOp);

TEST_F(XlaCompilerTest, TokenInputAndOutput) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  NodeDef side_effecting_op;
  side_effecting_op.set_name("DummySideEffectingOp");
  side_effecting_op.set_op("DummySideEffectingOp");
  AddNodeAttr(kXlaTokenInputNodesAttrName,
              std::vector<string>{kXlaTokenArgNodeName}, &side_effecting_op);
  AddNodeAttr(kXlaOriginalOutsideCompilationNodeName, side_effecting_op.name(),
              &side_effecting_op);
  Status status;
  graph->AddNode(side_effecting_op, &status);
  TF_ASSERT_OK(status);
  EXPECT_TRUE(FixupSourceAndSinkEdges(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kVariable;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 2});

  {
    // The case for entry computation: we don't add token input/output. Instead,
    // we use CreateToken HLO to create the entry token.
    XlaCompiler::CompileOptions options;
    options.is_entry_computation = true;
    options.add_token_input_output = false;
    options.return_updated_values_for_all_resources = true;
    XlaCompiler compiler(DefaultOptions());

    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(options, "NoOp", std::move(graph_copy),
                                       args, &result));
    EXPECT_EQ(result.xla_input_shapes.size(), 1);
    EXPECT_TRUE(result.xla_output_shape.IsTuple());
    EXPECT_EQ(xla::ShapeUtil::TupleElementCount(result.xla_output_shape), 1);
  }
  {
    // The case for non-entry computation (e.g. while loop body). We add token
    // input/output.
    XlaCompiler::CompileOptions options;
    options.is_entry_computation = false;
    options.add_token_input_output = true;
    options.return_updated_values_for_all_resources = true;
    XlaCompiler compiler(DefaultOptions());

    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(options, "NoOp", std::move(graph_copy),
                                       args, &result));
    EXPECT_EQ(result.xla_input_shapes.size(), 2);
    EXPECT_TRUE(result.xla_input_shapes[1].IsToken());
    EXPECT_TRUE(result.xla_output_shape.IsTuple());
    EXPECT_EQ(xla::ShapeUtil::TupleElementCount(result.xla_output_shape), 2);
    EXPECT_TRUE(xla::ShapeUtil::GetTupleElementShape(result.xla_output_shape, 1)
                    .IsToken());
  }
}

TEST_F(XlaCompilerTest, OpsWithTensorListInput) {
  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  // Build cond fn for While.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
    auto result = ops::Const<bool>(scope, {true}, {});
    ops::_Retval(scope.WithOpName("ret"), result, 0);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(*graph, "cond", &fdef));
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }
  // Build body fn for While.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    auto arg = ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
    ops::_Retval(scope.WithOpName("ret"), arg, 0);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(*graph, "body", &fdef));
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto element_shape = ops::Const<int32>(scope, {1}, {1});
  auto max_elements = ops::Const<int32>(scope, {10}, {});
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
  std::initializer_list<Output> out = {arg, arg};
  auto add_n = ops::AddN(scope, out);
  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("cond");
  body_fn.set_name("body");
  auto while_op =
      ops::While(scope, std::initializer_list<Input>{arg}, cond_fn, body_fn);
  auto ret0 = ops::_Retval(scope.WithOpName("ret0"), add_n, 0);
  auto ret1 = ops::_Retval(scope.WithOpName("ret1"), while_op.output[0], 1);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kTensorList;
  xla::Shape tensor_list_element_shape;
  TF_ASSERT_OK(TensorShapeToXLAShape(DT_INT32, TensorShape{1},
                                     &tensor_list_element_shape));
  xla::Shape index_shape;
  TF_ASSERT_OK(TensorShapeToXLAShape(DT_INT32, TensorShape{}, &index_shape));
  std::vector<xla::Shape> shapes{tensor_list_element_shape, index_shape};
  xla::Shape arg_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  args[0].shape = arg_shape;

  // Compiles the graph.
  XlaCompiler::Options options = DefaultOptions();
  options.flib_def = &flib_def;
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));
  ASSERT_EQ(result.outputs.size(), 2);
  const XlaCompiler::OutputDescription& output0 = result.outputs[0];
  ASSERT_TRUE(output0.is_tensor_list);
  const XlaCompiler::OutputDescription& output1 = result.outputs[1];
  ASSERT_TRUE(output1.is_tensor_list);
}

// Test the compiler supports WhileOp with a loop body where DT_RESOURCE
// variables are both inputs and outputs.
TEST_F(XlaCompilerTest, WhileWithResources) {
  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  // Build cond fn for While.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_RESOURCE, 1);
    auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_RESOURCE, 2);
    auto less = ops::Less(scope, arg0, ops::Const<int32>(scope, 10));
    (void)ops::_Retval(scope.WithOpName("ret"), less, 0);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(*graph, "cond", &fdef));
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }
  // Build body fn for While.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_RESOURCE, 1);
    auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_RESOURCE, 2);
    auto read1 = ops::ReadVariableOp(scope.WithOpName("read1"), arg1, DT_INT32);
    auto plus_read1 = ops::Add(scope, arg0, read1);
    auto read2 = ops::ReadVariableOp(scope.WithOpName("read2"), arg2, DT_INT32);
    auto minus_read2 = ops::Sub(scope, plus_read1, read2);
    (void)ops::_Retval(scope.WithOpName("ret0"), minus_read2, 0);
    (void)ops::_Retval(scope.WithOpName("ret1"), arg1, 1);
    (void)ops::_Retval(scope.WithOpName("ret2"), arg2, 2);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(*graph, "body", &fdef));
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
  auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_RESOURCE, 1);
  auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_RESOURCE, 2);

  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("cond");
  body_fn.set_name("body");
  auto while_op = ops::While(
      scope, std::initializer_list<Input>{arg0, arg1, arg2}, cond_fn, body_fn);

  (void)ops::_Retval(scope.WithOpName("ret0"), while_op.output[0], 0);
  (void)ops::_Retval(scope.WithOpName("ret1"), while_op.output[1], 1);
  (void)ops::_Retval(scope.WithOpName("ret2"), while_op.output[2], 2);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(3);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({});
  args[2].kind = XlaCompiler::Argument::kResource;
  args[2].resource_kind = XlaResource::kVariable;
  args[2].initialized = true;
  args[2].type = DT_INT32;
  args[2].shape = TensorShape({});

  // Compiles the graph.
  XlaCompiler::Options options = DefaultOptions();
  options.flib_def = &flib_def;
  XlaCompiler compiler(options);

  XlaCompiler::CompileOptions compile_options = XlaCompiler::CompileOptions();
  compile_options.return_updated_values_for_all_resources = true;
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "tested_while_with_vars",
                                     std::move(graph), args, &result));
  ASSERT_EQ(result.outputs.size(), 3);
  const XlaCompiler::OutputDescription& output1 = result.outputs[1];
  ASSERT_EQ(output1.input_index, 1);
  const XlaCompiler::OutputDescription& output2 = result.outputs[2];
  ASSERT_EQ(output2.input_index, 2);

  // Tests that the generated computation works.
  xla::Literal literal0 = xla::LiteralUtil::CreateR0<int32>(0);
  xla::Literal literal1 = xla::LiteralUtil::CreateR0<int32>(2);
  xla::Literal literal2 = xla::LiteralUtil::CreateR0<int32>(1);
  std::unique_ptr<xla::GlobalData> data0 =
      client_->TransferToServer(literal0).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> data1 =
      client_->TransferToServer(literal1).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> data2 =
      client_->TransferToServer(literal2).ConsumeValueOrDie();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation,
                    {data0.get(), data1.get(), data2.get()})
          .ConsumeValueOrDie();
  xla::Literal actual_literal = client_->Transfer(*actual).ConsumeValueOrDie();

  xla::Literal expected0 = xla::LiteralUtil::CreateR0<int32>(10);
  xla::Literal expected1 = xla::LiteralUtil::CreateR0<int32>(2);
  xla::Literal expected2 = xla::LiteralUtil::CreateR0<int32>(1);
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&expected0, &expected1, &expected2});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

TEST_F(XlaCompilerTest, SetShardingForReturnedTuple) {
  // Builds a graph that returns its only argument.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Retval(scope.WithOpName("B"), a, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Sets _XlaSharding attribute for the _Retval node.
  auto node_name_index = graph->BuildNodeNameIndex();
  Node* ret_node = node_name_index["B"];
  ASSERT_NE(ret_node, nullptr);
  xla::Array<int64> tile_assignment({2});
  tile_assignment.FillIota(0);
  xla::HloSharding sharding = xla::HloSharding::Tile(tile_assignment);
  ret_node->AddAttr("_XlaSharding", sharding.ToProto().SerializeAsString());

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "test",
                                     std::move(graph), args, &result));

  // Tests that we set sharding on the root TUPLE instruction.
  const auto& hlo_module_proto = result.computation->proto();
  ASSERT_EQ(hlo_module_proto.computations_size(), 1);
  const auto& hlo_computation_proto = hlo_module_proto.computations(0);
  absl::optional<xla::HloInstructionProto> root_instruction_proto;
  for (const auto& inst : hlo_computation_proto.instructions()) {
    if (inst.id() == hlo_computation_proto.root_id()) {
      root_instruction_proto = inst;
      break;
    }
  }
  ASSERT_TRUE(root_instruction_proto);
  xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(
      {xla::ShapeUtil::MakeShape(xla::S32, {2})});
  xla::HloSharding tuple_sharding = xla::HloSharding::Tuple(
      tuple_shape, std::vector<xla::HloSharding>{sharding});
  EXPECT_EQ(root_instruction_proto->sharding().SerializeAsString(),
            tuple_sharding.ToProto().SerializeAsString());
}

TEST_F(XlaCompilerTest, AliasResourceUpdates) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Const<int32>(scope.WithOpName("A"), {1, 2});
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  auto write = ops::AssignAddVariableOp(scope, var, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto d = ops::_Retval(scope.WithOpName("D"), read, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[0].constant_value = Tensor(DT_INT32, {1, 1});
  args[0].initialized = true;

  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompileOptions compile_options;
  compile_options.alias_resource_update = true;

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));

  const xla::HloInputOutputAliasProto& alias =
      result.computation->proto().input_output_alias();
  EXPECT_EQ(alias.entries_size(), 1);
  EXPECT_EQ(alias.entries(0).parameter_number(), 0);
}

// Tests that passing in an exact duplicate input to SetDeviceToHostMeatadata
// is not an error.
TEST_F(XlaCompilerTest, SetDeviceToHostMetadataExactDuplicate) {
  XlaCompiler compiler(DefaultOptions());

  const string& key = "comm_key";
  std::vector<DataType> types{DT_INT32};
  std::vector<TensorShape> shapes{TensorShape({2})};

  TF_ASSERT_OK(compiler.SetDeviceToHostMetadata(key, types, shapes));
  TF_ASSERT_OK(compiler.SetDeviceToHostMetadata(key, types, shapes));
}

// Tests that passing in a mismatched duplicate input to
// SetDeviceToHostMeatadata is not an error.
TEST_F(XlaCompilerTest, SetDeviceToHostMetadataMismatchedDuplicate) {
  XlaCompiler compiler(DefaultOptions());

  const string& key = "comm_key";
  std::vector<DataType> types{DT_INT32};
  std::vector<TensorShape> shapes{TensorShape({2})};
  std::vector<DataType> types2{DT_FLOAT};
  std::vector<TensorShape> shapes2{TensorShape({1})};

  TF_ASSERT_OK(compiler.SetDeviceToHostMetadata(key, types, shapes));
  Status status = compiler.SetDeviceToHostMetadata(key, types2, shapes2);
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
}

// Tests that passing in an exact duplicate input to SetHostToDeviceMeatadata
// is not an error.
TEST_F(XlaCompilerTest, SetHostToDeviceMetadataExactDuplicate) {
  XlaCompiler compiler(DefaultOptions());

  const string& key = "comm_key";
  std::vector<DataType> types{DT_INT32};
  std::vector<TensorShape> shapes{TensorShape({2})};

  TF_ASSERT_OK(compiler.SetHostToDeviceMetadata(key, types, shapes));
  TF_ASSERT_OK(compiler.SetHostToDeviceMetadata(key, types, shapes));
}

// Tests that passing in a mismatched duplicate input to
// SetHostToDeviceMeatadata is not an error.
TEST_F(XlaCompilerTest, SetHostToDeviceMetadataMismatchedDuplicate) {
  XlaCompiler compiler(DefaultOptions());

  const string& key = "comm_key";
  std::vector<DataType> types{DT_INT32};
  std::vector<TensorShape> shapes{TensorShape({2})};
  std::vector<DataType> types2{DT_FLOAT};
  std::vector<TensorShape> shapes2{TensorShape({1})};

  TF_ASSERT_OK(compiler.SetHostToDeviceMetadata(key, types, shapes));
  Status status = compiler.SetHostToDeviceMetadata(key, types2, shapes2);
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace tensorflow
