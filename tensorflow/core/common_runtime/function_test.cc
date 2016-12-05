/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/function.h"

#include <atomic>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

Status GetOpSig(const string& op, const OpDef** sig) {
  return OpRegistry::Global()->LookUpOpDef(op, sig);
}

void FunctionTestSchedClosure(std::function<void()> fn) {
  static thread::ThreadPool* w =
      new thread::ThreadPool(Env::Default(), "Test", 8);
  w->Schedule(fn);
}

void HasError(const Status& s, const string& substr) {
  EXPECT_TRUE(StringPiece(s.ToString()).contains(substr))
      << s << ", expected substring " << substr;
}

class FunctionTest : public ::testing::Test {
 protected:
  FunctionTest()
      : device_(DeviceFactory::NewDevice("CPU", {},
                                         "/job:localhost/replica:0/task:0")) {}

  ~FunctionTest() override {
    delete exec_;
    delete device_;
  }

  void Create(const FunctionDef& fdef, InstantiateAttrValueSlice attrs) {
    delete exec_;
    InstantiationResult result;
    TF_CHECK_OK(InstantiateFunction(fdef, attrs, GetOpSig, &result));

    arg_types_ = result.arg_types;
    ret_types_ = result.ret_types;

    Graph* g = new Graph(OpRegistry::Global());
    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    opts.expect_device_spec = false;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, result.gdef, g));

    const int version = g->versions().producer();
    LocalExecutorParams params;
    params.device = device_;
    params.create_kernel = [this, version](const NodeDef& ndef,
                                           OpKernel** kernel) {
      return CreateNonCachedKernel(device_, nullptr, ndef, version, kernel);
    };
    params.delete_kernel = [](OpKernel* kernel) {
      DeleteNonCachedKernel(kernel);
    };
    TF_CHECK_OK(NewLocalExecutor(params, g, &exec_));
  }

  void Run(const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    FunctionCallFrame frame(arg_types_, ret_types_);
    TF_CHECK_OK(frame.SetArgs(args));
    Executor::Args exec_args;
    exec_args.call_frame = &frame;
    exec_args.runner = FunctionTestSchedClosure;
    TF_CHECK_OK(exec_->Run(exec_args));
    std::vector<Tensor> computed;
    TF_CHECK_OK(frame.GetRetvals(&computed));
    CHECK_EQ(computed.size(), rets.size());
    for (int i = 0; i < rets.size(); ++i) {
      *(rets[i]) = computed[i];
    }
  }

  Device* device_ = nullptr;
  Executor* exec_ = nullptr;
  DataTypeVector arg_types_;
  DataTypeVector ret_types_;
};

TEST_F(FunctionTest, XTimesTwo) {
  Create(test::function::XTimesTwo(), {{"T", DT_FLOAT}});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  Run({x}, {&y});
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
}

TEST_F(FunctionTest, WXPlusB) {
  Create(test::function::WXPlusB(), {{"T", DT_FLOAT}});
  auto w = test::AsTensor<float>({1., 2., 3., 4.}, {2, 2});
  auto x = test::AsTensor<float>({1., 3., 2., 4.}, {2, 2});
  auto b = test::AsTensor<float>({0.5, 2.5}, {2});
  Tensor y;
  Run({w, x, b}, {&y});
  test::ExpectTensorEqual<float>(
      y, test::AsTensor<float>({5.5, 13.5, 11.5, 27.5}, {2, 2}));
}

class FunctionLibraryRuntimeTest : public ::testing::Test {
 protected:
  FunctionLibraryRuntimeTest()
      : device_(DeviceFactory::NewDevice("CPU", {},
                                         "/job:localhost/replica:0/task:0")) {}

  ~FunctionLibraryRuntimeTest() override {
    delete lib_;
    delete lib_def_;
    delete device_;
  }

  void Init(const std::vector<FunctionDef>& flib) {
    FunctionDefLibrary proto;
    for (const auto& fdef : flib) *(proto.add_function()) = fdef;
    delete lib_def_;
    lib_def_ = new FunctionLibraryDefinition(OpRegistry::Global(), proto);
    delete lib_;
    OptimizerOptions opts;
    lib_ = NewFunctionLibraryRuntime(nullptr, Env::Default(), device_,
                                     TF_GRAPH_DEF_VERSION, lib_def_, opts);
  }

  Status Run(const string& name, InstantiateAttrValueSlice attrs,
             const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    FunctionLibraryRuntime::Handle handle;
    Status status = lib_->Instantiate(name, attrs, &handle);
    if (!status.ok()) {
      return status;
    }

    std::atomic<int32> call_count(0);
    std::function<void(std::function<void()>)> runner =
        [&call_count](std::function<void()> fn) {
          ++call_count;
          FunctionTestSchedClosure(fn);
        };

    Notification done;
    FunctionLibraryRuntime::Options opts;
    opts.runner = &runner;
    std::vector<Tensor> out;
    lib_->Run(opts, handle, args, &out, [&status, &done](const Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }
    CHECK_EQ(rets.size(), out.size());
    for (size_t i = 0; i < rets.size(); ++i) {
      *rets[i] = out[i];
    }

    EXPECT_GE(call_count, 1);  // Test runner is used.

    return Status::OK();
  }

  Graph* GetFuncBody(const string& name, InstantiateAttrValueSlice attrs) {
    FunctionLibraryRuntime::Handle handle;
    Status status = lib_->Instantiate(name, attrs, &handle);
    if (!status.ok()) {
      LOG(ERROR) << status;
      return nullptr;
    }
    const FunctionBody* fbody = lib_->GetFunctionBody(handle);
    CHECK_NOTNULL(fbody);
    Graph* ret = new Graph(lib_def_);
    CopyGraph(*fbody->graph, ret);
    return ret;
  }

  Graph* GetGradBody(const string& func, InstantiateAttrValueSlice attrs) {
    FunctionLibraryRuntime::Handle handle;
    Status status = lib_->Instantiate(func, attrs, &handle);
    if (!status.ok()) {
      LOG(ERROR) << status;
      return nullptr;
    }
    const FunctionBody* fbody = lib_->GetFunctionBody(handle);
    CHECK_NOTNULL(fbody);
    FunctionBody* gbody = SymbolicGradient(*fbody);
    CHECK_NOTNULL(gbody);
    Graph* ret = new Graph(lib_def_);
    CopyGraph(*gbody->graph, ret);
    delete gbody;
    return ret;
  }

  Device* device_ = nullptr;
  FunctionLibraryDefinition* lib_def_ = nullptr;
  FunctionLibraryRuntime* lib_ = nullptr;
};

TEST_F(FunctionLibraryRuntimeTest, IsStateful) {
  Init({});
  EXPECT_TRUE(lib_->IsStateful("Variable"));
  EXPECT_TRUE(lib_->IsStateful("VariableV2"));
  EXPECT_FALSE(lib_->IsStateful("Matmul"));
}

TEST_F(FunctionLibraryRuntimeTest, XTimesTwo) {
  Init({test::function::XTimesTwo()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  TF_CHECK_OK(Run("XTimesTwo", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
}

TEST_F(FunctionLibraryRuntimeTest, XTimesN) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour(),
        test::function::XTimes16()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  TF_CHECK_OK(Run("XTimesTwo", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
  TF_CHECK_OK(Run("XTimesFour", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({4, 8, 12, 16}));
  TF_CHECK_OK(Run("XTimes16", {{"T", DT_FLOAT}}, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({16, 32, 48, 64}));
}

TEST_F(FunctionLibraryRuntimeTest, ExpandInlineFunctions) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour(),
        test::function::XTimes16()});
  Graph* g = GetFuncBody("XTimes16", {{"T", DT_FLOAT}});
  ASSERT_TRUE(g != nullptr);
  const char* e0 = R"P(
(n2:float) -> (n4:float) {
  n3 = XTimesFour[T=float](n2)
  n4 = XTimesFour[T=float](n3)
}
)P";
  EXPECT_EQ(e0, DebugString(g));

  ExpandInlineFunctions(lib_, g);
  const char* e1 = R"P(
(n2:float) -> (n17:float) {
  n10 = Identity[T=float](n2)
  n7 = XTimesTwo[T=float](n10)
  n8 = XTimesTwo[T=float](n7)
  n11 = Identity[T=float](n8)
  n16 = Identity[T=float](n11)
  n13 = XTimesTwo[T=float](n16)
  n14 = XTimesTwo[T=float](n13)
  n17 = Identity[T=float](n14)
}
)P";
  EXPECT_EQ(e1, DebugString(g));

  ExpandInlineFunctions(lib_, g);
  const char* e2 = R"P(
(n2:float) -> (n17:float) {
  n18 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n25 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n32 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n39 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n19 = Cast[DstT=float, SrcT=int64](n18)
  n26 = Cast[DstT=float, SrcT=int64](n25)
  n33 = Cast[DstT=float, SrcT=int64](n32)
  n40 = Cast[DstT=float, SrcT=int64](n39)
  n10 = Identity[T=float](n2)
  n23 = Identity[T=float](n10)
  n21 = Mul[T=float](n23, n19)
  n24 = Identity[T=float](n21)
  n30 = Identity[T=float](n24)
  n28 = Mul[T=float](n30, n26)
  n31 = Identity[T=float](n28)
  n11 = Identity[T=float](n31)
  n16 = Identity[T=float](n11)
  n37 = Identity[T=float](n16)
  n35 = Mul[T=float](n37, n33)
  n38 = Identity[T=float](n35)
  n44 = Identity[T=float](n38)
  n42 = Mul[T=float](n44, n40)
  n45 = Identity[T=float](n42)
  n17 = Identity[T=float](n45)
}
)P";
  EXPECT_EQ(e2, DebugString(g));

  // No further inlining.
  ExpandInlineFunctions(lib_, g);
  EXPECT_EQ(e2, DebugString(g));

  // Get rid of redundant Identity nodes.
  RemoveIdentityNodes(g);
  const char* e3 = R"P(
(n2:float) -> (n42:float) {
  n18 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n25 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n32 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n39 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n19 = Cast[DstT=float, SrcT=int64](n18)
  n26 = Cast[DstT=float, SrcT=int64](n25)
  n33 = Cast[DstT=float, SrcT=int64](n32)
  n40 = Cast[DstT=float, SrcT=int64](n39)
  n21 = Mul[T=float](n2, n19)
  n28 = Mul[T=float](n21, n26)
  n35 = Mul[T=float](n28, n33)
  n42 = Mul[T=float](n35, n40)
}
)P";
  EXPECT_EQ(e3, DebugString(g));
  delete g;
}

TEST_F(FunctionLibraryRuntimeTest, OptimizeGraph) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour(),
        test::function::XTimes16()});
  Graph* g = GetFuncBody("XTimes16", {{"T", DT_FLOAT}});
  ASSERT_TRUE(g != nullptr);
  ExpandInlineFunctions(lib_, g);
  OptimizeGraph(lib_, &g);
  const char* e0 = R"P(
(n2:float) -> (n7:float) {
  n8 = Const[dtype=float, value=Tensor<type: float shape: [] values: 2>]()
  n4 = Mul[T=float](n2, n8)
  n5 = Mul[T=float](n4, n8)
  n6 = Mul[T=float](n5, n8)
  n7 = Mul[T=float](n6, n8)
}
)P";
  EXPECT_EQ(e0, DebugString(g));
  delete g;
}

TEST_F(FunctionLibraryRuntimeTest, ManySwapsOld) {
  auto func = FDH::Define(  // Creates a FunctionDef using FunctionDef::Nodes
      // Name
      "ManySwapsFirst",
      // Args
      {"x: float", "y: float"},
      // Return values
      {"o: float"},
      // attr def
      {},
      // Nodes
      {{{"a0", "b0"}, "Swap", {"x", "y"}, {{"T", DT_FLOAT}}},
       {{"a1", "b1"}, "Swap", {"a0", "b0"}, {{"T", DT_FLOAT}}},
       {{"a2", "b2"}, "Swap", {"a1", "b1"}, {{"T", DT_FLOAT}}},
       {{"a3", "b3"}, "Swap", {"a2", "b2"}, {{"T", DT_FLOAT}}},
       {{"a4", "b4"}, "Swap", {"a3", "b3"}, {{"T", DT_FLOAT}}},
       {{"a5", "b5"}, "Swap", {"a4", "b4"}, {{"T", DT_FLOAT}}},
       {{"o"}, "Identity", {"a5"}, {{"T", DT_FLOAT}}}});
  Init({test::function::Swap(), func});
  Graph* g = GetFuncBody("ManySwapsFirst", {});
  ASSERT_TRUE(g != nullptr);
  OptimizeGraph(lib_, &g);
  const char* e0 = R"P(
(n3:float, n2:float) -> (n3:float) {
}
)P";
  EXPECT_EQ(e0, DebugString(g));
  delete g;
}

// Like the above test, but using NodeDefs in the FunctionDef.
TEST_F(FunctionLibraryRuntimeTest, DISABLED_ManySwapsNodeDef) {
  auto func = FDH::Create(  // Creates a FunctionDef using NodeDefs
      // Name
      "ManySwapsNodeDef",
      // Input
      {"x: float", "y: float"},
      // Output
      {"o: float"},
      // Attr
      {},
      // Nodes
      {{{"a"}, "Swap", {"x", "y"}, {{"T", DT_FLOAT}}},
       {{"b"}, "Swap", {"a:o0", "a:o1"}, {{"T", DT_FLOAT}}},
       {{"c"}, "Swap", {"b:o0", "b:o1"}, {{"T", DT_FLOAT}}},
       {{"d"}, "Swap", {"c:o0", "c:o1"}, {{"T", DT_FLOAT}}},
       {{"e"}, "Swap", {"d:o0", "d:o1"}, {{"T", DT_FLOAT}}},
       {{"f"}, "Swap", {"e:o0", "e:o1"}, {{"T", DT_FLOAT}}},
       {{"g"}, "Identity", {"f:o0"}, {{"T", DT_FLOAT}}}},
      // Return
      {{"o", "g:output"}});
  Init({test::function::Swap(), func});
  Graph* g = GetFuncBody("ManySwapsNodeDef", {});
  ASSERT_TRUE(g != nullptr);
  OptimizeGraph(lib_, &g);
  const char* e0 = R"P(
(n3:float, n2:float) -> (n3:float) {
}
)P";
  EXPECT_EQ(e0, DebugString(g));
  delete g;
}

TEST_F(FunctionLibraryRuntimeTest, ControlDeps) {
  auto func = FDH::Define(
      // Name
      "ManySwapsFirst",
      // Args
      {"x: float", "y: float"},
      // Return values
      {"o: float"},
      // attr def
      {},
      // Nodes
      //
      // o = x*x + y*y.  Furthermore, The 1st swap depends on x2, and
      // y2 depends on the 2nd swap.  The 2nd swap has data dependency
      // on the 1st swap. The optimization should maintain the control
      // dependencies.
      {{{"a0", "b0"}, "Swap", {"x", "y"}, {{"T", DT_FLOAT}}, {"x2"}},
       {{"a1", "b1"}, "Swap", {"a0", "b0"}, {{"T", DT_FLOAT}}},
       {{"x2"}, "Mul", {"x", "x"}, {{"T", DT_FLOAT}}},
       {{"y2"}, "Mul", {"y", "y"}, {{"T", DT_FLOAT}}, {"a1"}},
       {{"o"}, "Add", {"x2", "y2"}, {{"T", DT_FLOAT}}}});
  Init({test::function::Swap(), func});
  Graph* g = GetFuncBody("ManySwapsFirst", {});
  ASSERT_TRUE(g != nullptr);
  OptimizeGraph(lib_, &g);

  // NOTE: We can remove n8, n9, n10, n11 with a control edge n8->n5.
  // But we don't have a pass doing that.
  const char* e0 = R"P(
(n3:float, n2:float) -> (n6:float) {
  n4 = Mul[T=float](n3, n3)
  n8 = NoOp() @ n4
  n9 = Identity[T=float](n3) @ n8
  n10 = Identity[T=float](n2) @ n8
  n11 = NoOp() @ n10, n9
  n5 = Mul[T=float](n2, n2) @ n11
  n6 = Add[T=float](n4, n5)
}
)P";
  EXPECT_EQ(e0, DebugString(g));
  delete g;
}

TEST_F(FunctionLibraryRuntimeTest, Error_NotFound) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  HasError(Run("Foo", {{"T", DT_FLOAT}}, {x}, {&y}),
           "Not found: Function Foo is not defined.");
}

TEST_F(FunctionLibraryRuntimeTest, Error_InstantiaionError) {
  auto bad_x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"y"}, "Add", {"x", "x"}, {{"no_T", "$T"}}},
      });
  Init({bad_x_times_two, test::function::XTimesFour(),
        test::function::XTimes16()});

  // Instantiating "XTimesTwo" should fail.
  FunctionLibraryRuntime::Handle handle;
  HasError(lib_->Instantiate("XTimesTwo", {{"T", DT_FLOAT}}, &handle),
           "Not found: type attr not found");

  // But XTimesFour and XTimes16 instantiation should succeed. Only
  // when they run, they fail because XTimesTwo is bad.
  TF_CHECK_OK(lib_->Instantiate("XTimesFour", {{"T", DT_FLOAT}}, &handle));
  TF_CHECK_OK(lib_->Instantiate("XTimes16", {{"T", DT_FLOAT}}, &handle));

  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  HasError(Run("XTimes16", {{"T", DT_FLOAT}}, {x}, {&y}),
           "type attr not found");
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_XTimesTwo) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour(),
        test::function::XTimes16()});
  auto f = GetFuncBody("XTimesTwo", {{"T", DT_FLOAT}});
  const char* e0 = R"P(
(n4:float) -> (n5:float) {
  n2 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n3 = Cast[DstT=float, SrcT=int64](n2)
  n5 = Mul[T=float](n4, n3)
}
)P";
  EXPECT_EQ(e0, DebugString(f));
  delete f;
  auto g = GetGradBody("XTimesTwo", {{"T", DT_FLOAT}});
  const char* e1 = R"P(
(n4:float, n6:float) -> (n7:float) {
  n2 = Const[dtype=int64, value=Tensor<type: int64 shape: [] values: 2>]()
  n3 = Cast[DstT=float, SrcT=int64](n2)
  n5 = Mul[T=float](n4, n3)
  n7 = SymbolicGradient[Tin={float, float, float}, Tout={float, float}, f=Mul[T=float]](n4, n3, n6)
}
)P";
  EXPECT_EQ(e1, DebugString(g));

  OptimizeGraph(lib_, &g);
  const char* e2 = R"P(
(n2:float, n3:float) -> (n9:float) {
  n11 = Const[dtype=int32, value=Tensor<type: int32 shape: [0] values: >]()
  n10 = Const[dtype=float, value=Tensor<type: float shape: [] values: 2>]()
  n6 = Shape[T=float, out_type=int32](n2)
  n5 = Mul[T=float](n3, n10)
  n7 = BroadcastGradientArgs[T=int32](n6, n11)
  n8 = Sum[T=float, Tidx=int32, keep_dims=false](n5, n7)
  n9 = Reshape[T=float, Tshape=int32](n8, n6)
}
)P";
  EXPECT_EQ(e2, DebugString(g));

  delete g;
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_Add) {
  Init({});
  auto T = DT_FLOAT;
  auto g = GetFuncBody("SymbolicGradient",
                       {{"f", FDH::FunctionRef("Add", {{"T", T}})}});
  const char* e0 = R"P(
(n7:float, n5:float, n2:float) -> (n14:float, n11:float) {
  n3 = Identity[T=float](n2)
  n4 = Identity[T=float](n2)
  n6 = Shape[T=float, out_type=int32](n5)
  n8 = Shape[T=float, out_type=int32](n7)
  n9 = BroadcastGradientArgs[T=int32](n8, n6)
  n10 = Sum[T=float, Tidx=int32, keep_dims=false](n3, n9:1)
  n13 = Sum[T=float, Tidx=int32, keep_dims=false](n4, n9)
  n11 = Reshape[T=float, Tshape=int32](n10, n6)
  n14 = Reshape[T=float, Tshape=int32](n13, n8)
}
)P";
  EXPECT_EQ(e0, DebugString(g));
  delete g;
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_Mul) {
  Init({});
  auto T = DT_FLOAT;
  auto g = GetFuncBody("SymbolicGradient",
                       {{"f", FDH::FunctionRef("Mul", {{"T", T}})}});
  const char* e0 = R"P(
(n6:float, n3:float, n2:float) -> (n14:float, n11:float) {
  n4 = Mul[T=float](n2, n3)
  n5 = Shape[T=float, out_type=int32](n3)
  n7 = Mul[T=float](n6, n2)
  n8 = Shape[T=float, out_type=int32](n6)
  n9 = BroadcastGradientArgs[T=int32](n8, n5)
  n10 = Sum[T=float, Tidx=int32, keep_dims=false](n7, n9:1)
  n13 = Sum[T=float, Tidx=int32, keep_dims=false](n4, n9)
  n11 = Reshape[T=float, Tshape=int32](n10, n5)
  n14 = Reshape[T=float, Tshape=int32](n13, n8)
}
)P";
  EXPECT_EQ(e0, DebugString(g));
  delete g;
}

TEST_F(FunctionLibraryRuntimeTest, Gradient_AddSum) {
  // Sum(Add(x, y))
  auto T = DT_FLOAT;
  auto test = FDH::Define("Test", {"x:float", "y:float"}, {"l:float"}, {},
                          {
                              {{"z"}, "Add", {"x", "y"}, {{"T", T}}},
                              FDH::Const("zero", 0),
                              FDH::Const("one", 1),
                              {{"r"}, "Rank", {"z"}, {{"T", T}}},
                              {{"indices"}, "Range", {"zero", "r", "one"}},
                              {{"l"}, "Sum", {"z", "indices"}, {{"T", T}}},
                          });

  // TestGrad = Test'(x, y)
  auto grad =
      FDH::Define("TestGrad", {"x:float", "y:float"}, {"dx:float", "dy:float"},
                  {}, {FDH::Const<float>("dz", 1),
                       {{"grad"},
                        "SymbolicGradient",
                        {"x", "y", "dz"},
                        {
                            {"f", FDH::FunctionRef("Test")},
                            {"Tin", DataTypeSlice{T, T, T}},
                            {"Tout", DataTypeSlice{T, T}},
                        }},
                       {{"dx"}, "Identity", {"grad:0"}, {{"T", DT_FLOAT}}},
                       {{"dy"}, "Identity", {"grad:1"}, {{"T", DT_FLOAT}}}});

  Init({test, grad});

  Graph* g = GetFuncBody("TestGrad", {});
  ASSERT_TRUE(g != nullptr);
  const char* e0 = R"P(
(n4:float, n3:float) -> (n8:float, n6:float) {
  n2 = Const[dtype=float, value=Tensor<type: float shape: [] values: 1>]()
  n5 = SymbolicGradient[Tin={float, float, float}, Tout={float, float}, f=Test](n4, n3, n2)
  n6 = Identity[T=float](n5:1)
  n8 = Identity[T=float](n5)
}
)P";
  EXPECT_EQ(e0, DebugString(g));

  ExpandInlineFunctions(lib_, g);
  const char* e1 = R"P(
(n4:float, n3:float) -> (n8:float, n6:float) {
  n10 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 1>]()
  n11 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n2 = Const[dtype=float, value=Tensor<type: float shape: [] values: 1>]()
  n26 = Identity[T=float](n2)
  n25 = Identity[T=float](n3)
  n24 = Identity[T=float](n4)
  n14 = Add[T=float](n24, n25)
  n15 = Rank[T=float](n14)
  n16 = Range[Tidx=int32](n11, n15, n10)
  n20 = ZerosLike[T=int32](n15)
  n17 = Sum[T=float, Tidx=int32, keep_dims=false](n14, n16)
  n19 = SymbolicGradient[Tin={float, int32, float}, Tout={float, int32}, f=Sum[T=float, Tidx=int32, keep_dims=false]](n14, n16, n26)
  n21 = SymbolicGradient[Tin={float, float, float}, Tout={float, float}, f=Add[T=float]](n24, n25, n19)
  n28 = Identity[T=float](n21:1)
  n27 = Identity[T=float](n21)
  n6 = Identity[T=float](n28)
  n8 = Identity[T=float](n27)
}
)P";
  EXPECT_EQ(e1, DebugString(g));

  OptimizeGraph(lib_, &g);
  const char* e2 = R"P(
(n4:float, n3:float) -> (n25:float, n23:float) {
  n11 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 1>]()
  n2 = Const[dtype=float, value=Tensor<type: float shape: [] values: 1>]()
  n7 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n19 = Shape[T=float, out_type=int32](n3)
  n8 = Add[T=float](n4, n3)
  n20 = Shape[T=float, out_type=int32](n4)
  n9 = Rank[T=float](n8)
  n14 = Shape[T=float, out_type=int32](n8)
  n21 = BroadcastGradientArgs[T=int32](n20, n19)
  n10 = Range[Tidx=int32](n7, n9, n11)
  n12 = Shape[T=int32, out_type=int32](n10)
  n13 = Fill[T=int32](n12, n11)
  n15 = DynamicStitch[N=2, T=int32](n10, n10, n14, n13)
  n16 = Reshape[T=float, Tshape=int32](n2, n15)
  n17 = Div[T=int32](n14, n15)
  n18 = Tile[T=float, Tmultiples=int32](n16, n17)
  n24 = Sum[T=float, Tidx=int32, keep_dims=false](n18, n21)
  n22 = Sum[T=float, Tidx=int32, keep_dims=false](n18, n21:1)
  n25 = Reshape[T=float, Tshape=int32](n24, n20)
  n23 = Reshape[T=float, Tshape=int32](n22, n19)
}
)P";
  EXPECT_EQ(e2, DebugString(g));
  delete g;
}

namespace {

bool DoNothing(Graph* g) { return false; }

string Optimize(std::function<bool(Graph* g)> pass, const FunctionDef& fdef) {
  InstantiationResult result;
  InstantiateAttrValueMap empty;
  TF_CHECK_OK(InstantiateFunction(fdef, empty, GetOpSig, &result));
  Graph* g = new Graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, result.gdef, g));
  pass(g);
  Graph* g1 = new Graph(OpRegistry::Global());
  CopyGraph(*g, g1);
  delete g;
  GraphDef gdef;
  g1->ToGraphDef(&gdef);
  delete g1;
  return DebugString(gdef);
}

}  // end namespace

TEST(OptimizationTest, RemoveDeadNodes) {
  auto T = DT_INT32;
  auto func = FDH::Define(
      // Name
      "F",
      // Args
      {"x: int32"},
      // Return values
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {// a = Square<T>(x)
       {{"a"}, "Square", {"x"}, {{"T", T}}},
       // 1
       FDH::Const("o", 1),
       // A bunch of extra arithmetic that y doesn't depend on
       {{"x1"}, "Add", {"o", "o"}, {{"T", T}}},
       {{"x2"}, "Mul", {"a", "x1"}, {{"T", T}}},
       {{"x3"}, "Mul", {"x1", "x2"}, {{"T", T}}},
       // A stateful node.
       {{"keep_me"}, "RandomUniform", {"o"}, {{"T", T}, {"dtype", DT_FLOAT}}},
       // y = Add<T>(a, o)
       {{"y"}, "Add", {"a", "o"}, {{"T", T}}}});
  const char* e0 = R"S(
(n0:int32) -> (n7:int32) {
  n2 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 1>]()
  n6 = RandomUniform[T=int32, dtype=float, seed2=0, seed=0](n2)
  n3 = Add[T=int32](n2, n2)
  n1 = Square[T=int32](n0)
  n7 = Add[T=int32](n1, n2)
  n4 = Mul[T=int32](n1, n3)
  n5 = Mul[T=int32](n3, n4)
}
)S";
  EXPECT_EQ(Optimize(DoNothing, func), e0);

  // TODO(zhifengc): Comes up another test case.
  EXPECT_EQ(Optimize(::tensorflow::RemoveDeadNodes, func), e0);
}

TEST(OptimizationTest, RemoveIdentityNodes_Ref) {
  auto T = DT_FLOAT;
  auto func = FDH::Define(
      // Name
      "F",
      // Args
      {},
      // Return values
      {"ret: float"},
      // Attrs
      {},
      // Nodes
      {// variable
       {{"v"}, "VariableV2", {}, {{"dtype", T}, {"shape", TensorShape({})}}},
       // read the variable. Shouldn't be removed.
       {{"v_read"}, "Identity", {"v"}, {{"T", T}}},
       // returns v + v
       {{"ret"}, "Add", {"v_read", "v_read"}, {{"T", T}}}});
  const char* e0 = R"S(
() -> (n2:float) {
  n0 = VariableV2[container="", dtype=float, shape=[], shared_name=""]()
  n1 = Identity[T=float](n0)
  n2 = Add[T=float](n1, n1)
}
)S";
  EXPECT_EQ(Optimize(DoNothing, func), e0);

  const char* e1 = R"S(
() -> (n2:float) {
  n0 = VariableV2[container="", dtype=float, shape=[], shared_name=""]()
  n1 = Identity[T=float](n0)
  n2 = Add[T=float](n1, n1)
}
)S";
  EXPECT_EQ(Optimize(::tensorflow::RemoveIdentityNodes, func), e1);
}

TEST(OptimizationTest, RemoveIdentityNodes) {
  auto T = DT_INT32;
  auto func = FDH::Define(
      // Name
      "F",
      // Args
      {"x: int32"},
      // Return values
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {// a = Square<T>(x)
       {{"a"}, "Square", {"x"}, {{"T", T}}},
       // 1
       FDH::Const("o", 1),
       // A bunch of extra arithmetic that y doesn't depend on
       {{"x1"}, "Identity", {"a"}, {{"T", T}}},
       {{"x2"}, "Identity", {"x1"}, {{"T", T}}},
       {{"x3"}, "Identity", {"x2"}, {{"T", T}}},
       // A stateful node.
       {{"keep_me"},
        "RandomUniform",
        {"o"},
        {{"T", T}, {"dtype", DT_FLOAT}},
        {"x3"}},
       // y = Add<T>(a, o)
       {{"y"}, "Add", {"a", "o"}, {{"T", T}}}});
  const char* e0 = R"S(
(n0:int32) -> (n7:int32) {
  n2 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 1>]()
  n1 = Square[T=int32](n0)
  n7 = Add[T=int32](n1, n2)
  n3 = Identity[T=int32](n1)
  n4 = Identity[T=int32](n3)
  n5 = Identity[T=int32](n4)
  n6 = RandomUniform[T=int32, dtype=float, seed2=0, seed=0](n2) @ n5
}
)S";
  EXPECT_EQ(Optimize(DoNothing, func), e0);

  const char* e1 = R"S(
(n0:int32) -> (n7:int32) {
  n2 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 1>]()
  n1 = Square[T=int32](n0)
  n7 = Add[T=int32](n1, n2)
  n6 = RandomUniform[T=int32, dtype=float, seed2=0, seed=0](n2) @ n1
}
)S";
  EXPECT_EQ(Optimize(::tensorflow::RemoveIdentityNodes, func), e1);
}

TEST(OptimizationTest, RemoveListArrayConverter) {
  auto func = FDH::Define(
      // Name
      "Test",
      // Args
      {"i: float"},
      // Return values
      {"o: float"},
      // Attrs
      {},
      // Nodes
      {FDH::Const("zero", 0),
       {{"s"}, "Split", {"zero", "i"}, {{"num_split", 4}, {"T", DT_FLOAT}}},
       {{"a"},
        "_ArrayToList",
        {"s"},
        {{"N", 4},
         {"T", DT_FLOAT},
         {"out_types", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}}}},
       {{"l"}, "Mul", {"a:0", "a:1"}, {{"T", DT_FLOAT}}},
       {{"r"}, "Mul", {"a:2", "a:3"}, {{"T", DT_FLOAT}}},
       {{"x"},
        "_ListToArray",
        {"l", "r"},
        {{"N", 2},
         {"T", DT_FLOAT},
         {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}}},
       {{"o"}, "AddN", {"x"}, {{"N", 2}, {"T", DT_FLOAT}}}});

  const char* e0 = R"P(
(n0:float) -> (n7:float) {
  n1 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n2 = Split[T=float, num_split=4](n1, n0)
  n3 = _ArrayToList[N=4, T=float, out_types={float, float, float, float}](n2, n2:1, n2:2, n2:3)
  n5 = Mul[T=float](n3:2, n3:3)
  n4 = Mul[T=float](n3, n3:1)
  n6 = _ListToArray[N=2, T=float, Tin={float, float}](n4, n5)
  n7 = AddN[N=2, T=float](n6, n6:1)
}
)P";
  EXPECT_EQ(Optimize(DoNothing, func), e0);

  const char* e1 = R"P(
(n0:float) -> (n7:float) {
  n1 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n2 = Split[T=float, num_split=4](n1, n0)
  n5 = Mul[T=float](Func/_2, Func/_3)
  n4 = Mul[T=float](Func/_0, Func/_1)
  n7 = AddN[N=2, T=float](Func/_4, Func/_5)
  Func/_0 = Identity[T=float](n2)
  Func/_1 = Identity[T=float](n2:1)
  Func/_2 = Identity[T=float](n2:2)
  Func/_3 = Identity[T=float](n2:3)
  Func/_4 = Identity[T=float](n4)
  Func/_5 = Identity[T=float](n5)
}
)P";
  EXPECT_EQ(Optimize(RemoveListArrayConverter, func), e1);

  const char* e2 = R"P(
(n0:float) -> (n7:float) {
  n1 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n2 = Split[T=float, num_split=4](n1, n0)
  n5 = Mul[T=float](n2:2, n2:3)
  n4 = Mul[T=float](n2, n2:1)
  n7 = AddN[N=2, T=float](n4, n5)
}
)P";
  auto remove_listarray_and_identity = [](Graph* g) {
    return RemoveListArrayConverter(g) && RemoveIdentityNodes(g);
  };
  EXPECT_EQ(Optimize(remove_listarray_and_identity, func), e2);
}

TEST(OptimizationTest, RemoveListArrayConverter_WithContolDeps) {
  auto func = FDH::Define(
      // Name
      "Test",
      // Args
      {"i: float"},
      // Return values
      {"o: float"},
      // Attrs
      {},
      // Nodes
      {FDH::Const("dummy", 0),
       {{"x"},
        "_ListToArray",
        {"i", "i"},
        {{"N", 2}, {"T", DT_FLOAT}, {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}}},
        // Control dep
        {"dummy"}},
       {{"o"},
        "AddN",
        {"x"},
        {{"N", 2}, {"T", DT_FLOAT}},
        // Control dep
        {"x"}}});

  const char* e0 = R"P(
(n0:float) -> (n3:float) {
  n1 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n2 = _ListToArray[N=2, T=float, Tin={float, float}](n0, n0) @ n1
  n3 = AddN[N=2, T=float](n2, n2:1) @ n2
}
)P";
  EXPECT_EQ(Optimize(DoNothing, func), e0);

  const char* e1 = R"P(
(n0:float) -> (n3:float) {
  n1 = Const[dtype=int32, value=Tensor<type: int32 shape: [] values: 0>]()
  n3 = AddN[N=2, T=float](Func/_0, Func/_1) @ Func/_3
  Func/_0 = Identity[T=float](n0) @ Func/_2
  Func/_1 = Identity[T=float](n0) @ Func/_2
  Func/_2 = NoOp() @ n1
  Func/_3 = NoOp() @ Func/_0, Func/_1
}
)P";
  EXPECT_EQ(Optimize(RemoveListArrayConverter, func), e1);

  auto remove_listarray_and_identity = [](Graph* g) {
    return RemoveListArrayConverter(g) && RemoveIdentityNodes(g);
  };
  // NOTE: We are not removing Identity nodes with any control
  // dependencies yet.
  EXPECT_EQ(Optimize(remove_listarray_and_identity, func), e1);
}

}  // end namespace tensorflow
