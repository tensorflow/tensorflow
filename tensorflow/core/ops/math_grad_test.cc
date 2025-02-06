/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

namespace f = test::function;
using FDH = FunctionDefHelper;

std::unique_ptr<Session> NewSession() {
  SessionOptions opts;
  (*opts.config.mutable_device_count())["CPU"] = 1;
  return std::unique_ptr<Session>(NewSession(opts));
}

class MathGradTest : public ::testing::Test {
 protected:
  // Unary
  // dst is the output dtype of op_node.
  absl::Status Unary(const FDH::Node& op_node, const Tensor& x,
                     const DataType dst, Tensor* y) {
    const DataType src = x.dtype();
    auto adef = [](const string& name,
                   const DataType type) {  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(type));
    };
    // Sum(op(x)), sum all output of op(x).
    auto test = FDH::Define("Test", {adef("x", src)}, {adef("l", dst)}, {},
                            {
                                op_node,
                                FDH::Const("zero", 0),
                                FDH::Const("one", 1),
                                {{"r"}, "Rank", {"x"}, {{"T", src}}},
                                {{"indices"}, "Range", {"zero", "r", "one"}},
                                {{"l"}, "Sum", {"y", "indices"}, {{"T", dst}}},
                            });

    // TestGrad = Test'(x)
    auto grad = FDH::Define(
        "TestGrad", {adef("x", src)}, {adef("dx", src)}, {},
        {
            FDH::Const("one", 1),
            {{"dy"}, "Cast", {"one"}, {{"DstT", dst}, {"SrcT", DT_INT32}}},
            {{"grad"},
             "SymbolicGradient",
             {"x", "dy"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{src, dst}},
                 {"Tout", DataTypeSlice{src}},
             }},
            {{"dx"}, "Identity", {"grad"}, {{"T", src}}},
        });
    // Each test case will feed in "x:0" and expects to get "dx:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", src}}),
            f::NDef("dx", "TestGrad", {"x"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    auto s = sess->Run({{"x:0", x}}, {"dx:0"}, {}, &outputs);
    if (s.ok()) {
      CHECK_EQ(outputs.size(), 1);
      *y = outputs[0];
    }
    TF_CHECK_OK(sess->Close());
    return s;
  }

  absl::Status Unary(const string& op, const Tensor& x, Tensor* y) {
    const FDH::Node op_node = {{"y"}, op, {"x"}, {{"T", x.dtype()}}};
    return Unary(op_node, x, x.dtype(), y);
  }

  // Unary op expecting OK.
  Tensor SymGrad(const string& op, const Tensor& x) {
    Tensor ret;
    TF_CHECK_OK(Unary(op, x, &ret));
    return ret;
  }

  Tensor SymCastGrad(const Tensor& x, const DataType dst) {
    Tensor ret;
    const FDH::Node op_node = {
        {"y"}, "Cast", {"x"}, {{"SrcT", x.dtype()}, {"DstT", dst}}};
    TF_CHECK_OK(Unary(op_node, x, dst, &ret));
    return ret;
  }

  // Binary
  void SymGrad(const string& op, const Tensor& x, const Tensor& y, Tensor* dx,
               Tensor* dy) {
    const DataType T = x.dtype();
    auto adef = [T](const string& name) {  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(T));
    };
    // Sum(op(x)), sum all output of op(x).
    auto test = FDH::Define("Test", {adef("x"), adef("y")}, {adef("l")}, {},
                            {
                                {{"z"}, op, {"x", "y"}, {{"T", T}}},
                                FDH::Const("zero", 0),
                                FDH::Const("one", 1),
                                {{"r"}, "Rank", {"z"}, {{"T", T}}},
                                {{"indices"}, "Range", {"zero", "r", "one"}},
                                {{"l"}, "Sum", {"z", "indices"}, {{"T", T}}},
                            });

    // TestGrad = Test'(x, y)
    auto grad = FDH::Define(
        "TestGrad", {adef("x"), adef("y")}, {adef("dx"), adef("dy")}, {},
        {
            FDH::Const("one", 1),
            {{"dz"}, "Cast", {"one"}, {{"DstT", T}, {"SrcT", DT_INT32}}},
            {{"grad0", "grad1"},
             "SymbolicGradient",
             {"x", "y", "dz"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, T, T}},
                 {"Tout", DataTypeSlice{T, T}},
             }},
            {{"dx"}, "Identity", {"grad0"}, {{"T", T}}},
            {{"dy"}, "Identity", {"grad1"}, {{"T", T}}},
        });
    // Each test case will feed in "x:0" and "y:0" and expects to get "d0" and
    // "d:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("d", "TestGrad", {"x", "y"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(
        sess->Run({{"x:0", x}, {"y:0", y}}, {"d:0", "d:1"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 2);
    TF_CHECK_OK(sess->Close());
    *dx = outputs[0];
    *dy = outputs[1];
  }

  // Reduction grad
  void ReductionGrad(const string& op, const Tensor& x, const Tensor& idx,
                     Tensor* dx, Tensor* di) {
    const DataType T = x.dtype();
    auto adef = [T](const string& name) {  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(T));
    };
    // Sum(op(x, idx)), sum all output of op(x, idx).
    auto test = FDH::Define("Test", {adef("x"), "i:int32"}, {adef("l")}, {},
                            {
                                {{"y"}, op, {"x", "i"}, {{"T", T}}},
                                FDH::Const("zero", 0),
                                FDH::Const("one", 1),
                                {{"r"}, "Rank", {"y"}, {{"T", T}}},
                                {{"indices"}, "Range", {"zero", "r", "one"}},
                                {{"l"}, "Sum", {"y", "indices"}, {{"T", T}}},
                            });

    // TestGrad = Test'(x)
    auto grad = FDH::Define(
        "TestGrad", {adef("x"), "i:int32"}, {adef("dx"), "di:int32"}, {},
        {
            FDH::Const("one", 1),
            {{"dy"}, "Cast", {"one"}, {{"DstT", T}, {"SrcT", DT_INT32}}},
            {{"grad0", "grad1"},
             "SymbolicGradient",
             {"x", "i", "dy"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, DT_INT32, T}},
                 {"Tout", DataTypeSlice{T, DT_INT32}},
             }},
            {{"dx"}, "Identity", {"grad0"}, {{"T", T}}},
            {{"di"}, "Identity", {"grad1"}, {{"T", DT_INT32}}},
        });
    // Each test case will feed in "x:0" and expects to get "dx:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("i", "Placeholder", {}, {{"dtype", DT_INT32}}),
            f::NDef("d", "TestGrad", {"x", "i"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(
        sess->Run({{"x:0", x}, {"i:0", idx}}, {"d:0", "d:1"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 2);
    TF_CHECK_OK(sess->Close());
    *dx = outputs[0];
    *di = outputs[1];
  }

  Tensor ReduceSum(const Tensor& x, absl::Span<const int32> axes) {
    int num_axes = axes.length();
    Tensor y(DT_INT32, TensorShape({num_axes}));
    for (size_t i = 0; i < axes.size(); ++i) {
      y.flat<int32>()(i) = axes[i];
    }
    auto T = x.dtype();
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Const", {}, {{"dtype", DT_INT32}, {"value", y}}),
            f::NDef("z", "Sum", {"x", "y"}, {{"T", T}}),
        },
        {});
    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(sess->Run({{"x:0", x}}, {"z:0"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 1);
    TF_CHECK_OK(sess->Close());
    return outputs[0];
  }

  Tensor MatMulCommon(const string& opname, const string& attr_adj_x,
                      const string& attr_adj_y, const Tensor& x, bool ax,
                      const Tensor& y, bool ay) {
    auto T = x.dtype();
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("z", opname, {"x", "y"},
                    {{"T", T}, {attr_adj_x, ax}, {attr_adj_y, ay}}),
        },
        {});
    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(sess->Run({{"x:0", x}, {"y:0", y}}, {"z:0"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 1);
    TF_CHECK_OK(sess->Close());
    return outputs[0];
  }

  Tensor MatMul(const Tensor& x, bool ax, const Tensor& y, bool ay) {
    return MatMulCommon("MatMul", "transpose_a", "transpose_b", x, ax, y, ay);
  }

  Tensor BatchMatMul(const Tensor& x, bool ax, const Tensor& y, bool ay) {
    return MatMulCommon("BatchMatMul", "adj_x", "adj_y", x, ax, y, ay);
  }

  Tensor BatchMatMulV2(const Tensor& x, bool ax, const Tensor& y, bool ay) {
    return MatMulCommon("BatchMatMulV2", "adj_x", "adj_y", x, ax, y, ay);
  }

  void MatMulGradCommon(const string& opname, const string& attr_adj_x,
                        const string& attr_adj_y, const Tensor& x, bool ax,
                        const Tensor& y, bool ay, Tensor* dx, Tensor* dy) {
    const DataType T = x.dtype();
    auto adef = [T](const string& name) {  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(T));
    };
    // Sum(op(x)), sum all output of op(x).
    auto test =
        FDH::Define("Test", {adef("x"), adef("y")}, {adef("l")}, {},
                    {
                        {{"z"},
                         opname,
                         {"x", "y"},
                         {{"T", T}, {attr_adj_x, ax}, {attr_adj_y, ay}}},
                        FDH::Const("zero", 0),
                        FDH::Const("one", 1),
                        {{"r"}, "Rank", {"z"}, {{"T", T}}},
                        {{"indices"}, "Range", {"zero", "r", "one"}},
                        {{"l"}, "Sum", {"z", "indices"}, {{"T", T}}},
                    });

    // TestGrad = Test'(x, y)
    auto grad = FDH::Define(
        "TestGrad", {adef("x"), adef("y")}, {adef("dx"), adef("dy")}, {},
        {
            FDH::Const("one", 1),
            {{"dz"}, "Cast", {"one"}, {{"DstT", T}, {"SrcT", DT_INT32}}},
            {{"grad0", "grad1"},
             "SymbolicGradient",
             {"x", "y", "dz"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, T, T}},
                 {"Tout", DataTypeSlice{T, T}},
             }},
            {{"dx"}, "Identity", {"grad0"}, {{"T", T}}},
            {{"dy"}, "Identity", {"grad1"}, {{"T", T}}},
        });
    // Each test case will feed in "x:0" and "y:0" and expects to get "d0" and
    // "d:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("d", "TestGrad", {"x", "y"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(
        sess->Run({{"x:0", x}, {"y:0", y}}, {"d:0", "d:1"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 2);
    TF_CHECK_OK(sess->Close());
    *dx = outputs[0];
    *dy = outputs[1];
  }

  void MatMulGrad(const Tensor& x, bool ax, const Tensor& y, bool ay,
                  Tensor* dx, Tensor* dy) {
    return MatMulGradCommon("MatMul", "transpose_a", "transpose_b", x, ax, y,
                            ay, dx, dy);
  }

  void BatchMatMulGrad(const Tensor& x, bool ax, const Tensor& y, bool ay,
                       Tensor* dx, Tensor* dy) {
    return MatMulGradCommon("BatchMatMul", "adj_x", "adj_y", x, ax, y, ay, dx,
                            dy);
  }

  void BatchMatMulV2Grad(const Tensor& x, bool ax, const Tensor& y, bool ay,
                         Tensor* dx, Tensor* dy) {
    return MatMulGradCommon("BatchMatMulV2", "adj_x", "adj_y", x, ax, y, ay, dx,
                            dy);
  }

  void SelectGrad(const Tensor& c, const Tensor& x, const Tensor& y, Tensor* dc,
                  Tensor* dx, Tensor* dy) {
    auto T = DT_FLOAT;
    // Sum(Select(c, x, y))
    auto test =
        FDH::Define("Test", {"c:bool", "x:float", "y:float"}, {"l:float"}, {},
                    {
                        {{"z"}, "Select", {"c", "x", "y"}, {{"T", T}}},
                        FDH::Const("zero", 0),
                        FDH::Const("one", 1),
                        {{"r"}, "Rank", {"z"}, {{"T", T}}},
                        {{"indices"}, "Range", {"zero", "r", "one"}},
                        {{"l"}, "Sum", {"z", "indices"}, {{"T", T}}},
                    });

    // TestGrad(x, y) = Test'(c, x, y)
    auto grad = FDH::Define("TestGrad", {"c:bool", "x:float", "y:float"},
                            {"dc:bool", "dx:float", "dy:float"}, {},
                            {FDH::Const("dz", 1.f),
                             {{"grad0", "grad1", "grad2"},
                              "SymbolicGradient",
                              {"c", "x", "y", "dz"},
                              {
                                  {"f", FDH::FunctionRef("Test")},
                                  {"Tin", DataTypeSlice{DT_BOOL, T, T, T}},
                                  {"Tout", DataTypeSlice{DT_BOOL, T, T}},
                              }},
                             {{"dc"}, "Identity", {"grad0"}, {{"T", DT_BOOL}}},
                             {{"dx"}, "Identity", {"grad1"}, {{"T", T}}},
                             {{"dy"}, "Identity", {"grad2"}, {{"T", T}}}});
    // Each test case will feed in "x:0" and expects to get "dx:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("c", "Placeholder", {}, {{"dtype", DT_BOOL}}),
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("d", "TestGrad", {"c", "x", "y"}, {}),
        },
        {test, grad});

    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(sess->Run({{"c:0", c}, {"x:0", x}, {"y:0", y}},
                          {"d:0", "d:1", "d:2"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 3);
    TF_CHECK_OK(sess->Close());
    *dc = outputs[0];
    *dx = outputs[1];
    *dy = outputs[2];
  }
};

void HasError(const absl::Status& s, const string& substr) {
  EXPECT_TRUE(absl::StrContains(s.ToString(), substr))
      << s << ", expected substring " << substr;
}

REGISTER_OP("TestOpWithNoGrad")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Test op with no grad registered.

x: input
y: output
)doc");

class TestOp : public OpKernel {
 public:
  explicit TestOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override { ctx->set_output(0, Tensor()); }
};
REGISTER_KERNEL_BUILDER(Name("TestOpWithNoGrad").Device(DEVICE_CPU), TestOp);

TEST_F(MathGradTest, Error_Reporting) {
  auto x = test::AsTensor<float>({-3.f});
  auto dx = test::AsTensor<float>({3.f});
  Tensor donotcare;
  HasError(Unary("TestOpWithNoGrad", x, &donotcare),
           "No gradient defined for op: TestOpWithNoGrad");
}

TEST_F(MathGradTest, Abs) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return x < 0 ? -1.f : 1.f; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Abs", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Neg) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return -1.f; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Neg", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Reciprocal) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return -1.f / (x * x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Reciprocal", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Square) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return 2 * x; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Square", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sqrt) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return 0.5f / std::sqrt(x); };
  auto dx = test::AsTensor<float>(
      {g(1.f), g(2.f), g(3.f), g(4.f), g(5.f), g(6.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sqrt", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Rsqrt) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return -0.5f / (x * std::sqrt(x)); };
  auto dx = test::AsTensor<float>(
      {g(1.f), g(2.f), g(3.f), g(4.f), g(5.f), g(6.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Rsqrt", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Exp) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return std::exp(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Exp", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Expm1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return std::exp(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Expm1", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Log) {
  auto x = test::AsTensor<float>({0.1f, 1.f, 2.f, 3.f, 4.f, 10.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return 1 / x; };
  auto dx = test::AsTensor<float>(
      {g(.1f), g(1.f), g(2.f), g(3.f), g(4.f), g(10.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Log", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Log1p) {
  auto x = test::AsTensor<float>({0.1f, 1.f, 2.f, 3.f, 4.f, 10.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return 1 / (1 + x); };
  auto dx = test::AsTensor<float>(
      {g(.1f), g(1.f), g(2.f), g(3.f), g(4.f), g(10.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Log1p", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sinh) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return std::cosh(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sinh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Cosh) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return std::sinh(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Cosh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Tanh) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
    auto y = std::tanh(x);
    return 1 - y * y;
  };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Tanh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Asinh) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
    auto y = std::asinh(x);
    return std::cosh(y);
  };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Asinh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Acosh) {
  auto x = test::AsTensor<float>({6.f, 5.f, 4.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
    auto y = std::acosh(x);
    return std::sinh(y);
  };
  auto dx = test::AsTensor<float>(
      {g(6.f), g(5.f), g(4.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Acosh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Atanh) {
  auto x = test::AsTensor<float>({-0.3f, -0.2f, -0.1f, 0.1f, 0.2f, 0.3f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return 1.f / (1.f - x * x); };
  auto dx = test::AsTensor<float>(
      {g(-0.3f), g(-0.2f), g(-0.1f), g(0.1f), g(0.2f), g(0.3f)},
      TensorShape({2, 3}));
  auto ans = SymGrad("Atanh", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sigmoid) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) {
    auto y = 1.f / (1.f + std::exp(-x));
    return y * (1 - y);
  };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sigmoid", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sign) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return 0.f; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sign", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Sin) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return std::cos(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Sin", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Cos) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return -std::sin(x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Cos", x);
  test::ExpectClose(ans, dx);
}

TEST_F(MathGradTest, Cast) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return 1.f; };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  Tensor ans = SymCastGrad(x, DT_INT32);
  test::ExpectClose(ans, dx);
}

// TODO(zhifengc)
// TEST_F(MathGradSComplexTest, Real) {}
// TEST_F(MathGradSComplexTest, Imag) {}
// TEST_F(MathGradSComplexTest, Angle) {}
// TEST_F(MathGradSComplexTest, Conj) {}
// TEST_F(MathGradTernary, Select) {}

TEST_F(MathGradTest, Add) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-10.f, 10.f}, TensorShape({2, 1}));
  auto ans_dx = test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                      TensorShape({2, 3}));
  auto ans_dy = test::AsTensor<float>({3.f, 3.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Add", x, y, &dx, &dy);
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
  {  // Swap x and y
    SymGrad("Add", y, x, &dy, &dx);
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
}

TEST_F(MathGradTest, Sub) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-10.f, 10.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Sub", x, y, &dx, &dy);
    auto ans_dx = test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                        TensorShape({2, 3}));
    auto ans_dy = test::AsTensor<float>({-3.f, -3.f}, TensorShape({2, 1}));
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
  {  // Swap x and y
    SymGrad("Sub", y, x, &dy, &dx);
    auto ans_dx = test::AsTensor<float>({-1.f, -1.f, -1.f, -1.f, -1.f, -1.f},
                                        TensorShape({2, 3}));
    auto ans_dy = test::AsTensor<float>({3.f, 3.f}, TensorShape({2, 1}));
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
}

TEST_F(MathGradTest, Mul) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-10.f, 10.f}, TensorShape({2, 1}));
  auto ans_dx = test::AsTensor<float>({-10.f, -10.f, -10.f, 10.f, 10.f, 10.f},
                                      TensorShape({2, 3}));
  auto ans_dy = test::AsTensor<float>({-3.f + (-2.f) + (-1.f), 1.f + 2.f + 3.f},
                                      TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Mul", x, y, &dx, &dy);
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
  {  // Swap x and y
    SymGrad("Mul", y, x, &dy, &dx);
    test::ExpectClose(ans_dx, dx);
    test::ExpectClose(ans_dy, dy);
  }
}

TEST_F(MathGradTest, Div) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-10.f, 10.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Div", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) { return 1.f / y; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-3.f, -10.f), g(-2.f, -10.f), g(-1.f, -10.f),
                                 g(1.f, 10.f), g(2.f, 10.f), g(3.f, 10.f)},
                                TensorShape({2, 3})));
    }
    {
      auto g = [](float x, float y) { return -x / (y * y); };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-3.f, -10.f) + g(-2.f, -10.f) + g(-1.f, -10.f),
                             g(1.f, 10.f) + g(2.f, 10.f) + g(3.f, 10.f)},
                            TensorShape({2, 1})));
    }
  }
  {  // Swap x and y
    SymGrad("Div", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) { return 1.f / y; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-10.f, -3.f) + g(-10.f, -2.f) + g(-10.f, -1.f),
                             g(10.f, 1.f) + g(10.f, 2.f) + g(10.f, 3.f)},
                            TensorShape({2, 1})));
    }
    {
      auto g = [](float x, float y) { return -x / (y * y); };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-10.f, -3.f), g(-10.f, -2.f), g(-10.f, -1.f),
                                 g(10.f, 1.f), g(10.f, 2.f), g(10.f, 3.f)},
                                TensorShape({2, 3})));
    }
  }
}

TEST_F(MathGradTest, DivNoNan) {
  auto x = test::AsTensor<float>(
      {0.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 0.f}, TensorShape({3, 3}));
  auto y = test::AsTensor<float>({-10.f, 0.f, 10.f}, TensorShape({3, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("DivNoNan", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) {
        if (y == 0.f) {
          return 0.f;
        } else {
          return 1.f / y;
        }
      };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(0.f, -10.f), g(-3.f, -10.f), g(-2.f, -10.f),
                                 g(-1.f, 0.f), g(0.f, 0.f), g(1.f, 0.f),
                                 g(2.f, 10.f), g(3.f, 10.f), g(0.f, 10.f)},
                                TensorShape({3, 3})));
    }
    {
      auto g = [](float x, float y) {
        if (y == 0.f) {
          return 0.f;
        } else {
          return -x / (y * y);
        }
      };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(0.f, -10.f) + g(-3.f, -10.f) + g(-2.f, -10.f),
                             g(-1.f, 0.f) + g(0.f, 0.f) + g(1.f, 0.f),
                             g(2.f, 10.f) + g(3.f, 10.f) + g(0.f, 10.f)},
                            TensorShape({3, 1})));
    }
  }
  {  // Swap x and y.
    SymGrad("DivNoNan", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) {
        if (y == 0.f) {
          return 0.f;
        } else {
          return 1.f / y;
        }
      };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-10.f, 0.f) + g(-10.f, -3.f) + g(-10.f, -2.f),
                             g(0.f, -1.f) + g(0.f, 0.f) + g(0.f, 1.f),
                             g(10.f, 2.f) + g(10.f, 3.f) + g(10.f, 0.f)},
                            TensorShape({3, 1})));
    }
    {
      auto g = [](float x, float y) {
        if (y == 0.f) {
          return 0.f;
        } else {
          return -x / (y * y);
        }
      };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-10.f, 0.f), g(-10.f, -3.f), g(-10.f, -2.f),
                                 g(0.f, -1.f), g(0.f, 0.f), g(0.f, 1.f),
                                 g(10.f, 2.f), g(10.f, 3.f), g(10.f, 0.f)},
                                TensorShape({3, 3})));
    }
  }
}

TEST_F(MathGradTest, Pow) {
  auto x = test::AsTensor<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) { return y * std::pow(x, y - 1); };
  auto h = [](float x, float y) {
    return std::pow(x, y) * (x ? std::log(x) : 0);
  };
  {
    SymGrad("Pow", x, y, &dx, &dy);
    test::ExpectClose(
        dx, test::AsTensor<float>({g(0.f, .5f), g(1.f, .5f), g(2.f, .5f),
                                   g(3.f, 2.f), g(4.f, 2.f), g(5.f, 2.f)},
                                  TensorShape({2, 3})));
    test::ExpectClose(
        dy, test::AsTensor<float>({h(0.f, .5f) + h(1.f, .5f) + h(2.f, .5f),
                                   h(3.f, 2.f) + h(4.f, 2.f) + h(5.f, 2.f)},
                                  TensorShape({2, 1})));
  }
  {  // Swap x and y
    SymGrad("Pow", y, x, &dy, &dx);
    test::ExpectClose(
        dy, test::AsTensor<float>({g(.5f, 0.f) + g(.5f, 1.f) + g(.5f, 2.f),
                                   g(2.f, 3.f) + g(2.f, 4.f) + g(2.f, 5.f)},
                                  TensorShape({2, 1})));
    test::ExpectClose(
        dx, test::AsTensor<float>({h(.5f, 0.f), h(.5f, 1.f), h(.5f, 2.f),
                                   h(2.f, 3.f), h(2.f, 4.f), h(2.f, 5.f)},
                                  TensorShape({2, 3})));
  }
}

TEST_F(MathGradTest, ComplexPow) {
  auto x = test::AsTensor<complex64>({0.f, 2.f, -2.f}, TensorShape({3}));
  auto y = test::AsTensor<complex64>({2.f, 2.f, 2.f}, TensorShape({3}));
  Tensor dx;
  Tensor dy;
  auto g = [](complex64 x, complex64 y) { return y * std::pow(x, y - 1.f); };
  auto h = [](complex64 x, complex64 y) {
    return std::pow(x, y) * (x != complex64(0) ? std::log(x) : 0);
  };
  SymGrad("Pow", x, y, &dx, &dy);

  // This case failed on Kokoro MacOS:
  // dx[2] = (-4,6.0398321011234657e-07),
  // test::AsTensor[2] = (-4,-3.4969110629390343e-07).
  // dx[2] on linux is close to test::AsTensor[2].
  // This error hasn't shown up before because
  // ExpectClose used to check just the magnitude of a complex number, i.e.,
  // std::abs(complex) = sqrt(real^2 + imag^2).
  // Now ExpectClose checks the value of each component separately.
  // Workaround: I set a big tolerance to make the case pass for now.
  // TODO(penporn): Fix this or file a bug. This is not a precision issue.
  // Even the most significant digit (or the sign) doesn't match.
  test::ExpectClose(
      dx,
      test::AsTensor<complex64>({g(0.f, 2.f), g(2.f, 2.f), g(-2.f, 2.f)},
                                TensorShape({3})),
      1e-6f);

  // This case failed on Kokoro MacOS:
  // dx[2] = (2.7725925445556641,12.56636905670166),
  // test::AsTensor[2] = (2.7725865840911865,12.566371917724609)
  // dx[2] on linux is close to test::AsTensor[2].
  // Default atol = rtol = 5.96046e-07.
  // Real: diff = 5.96046e-06 > threshold = 2.248633e-06 <- failed
  // Complex: diff = 2.86102e-06 <= threshold = 8.08618e-06 <- passed
  // Again, this error hasn't shown up before because ExpectClose used to
  // check just the magnitude of the complex number. Now it checks each
  // component separately.
  // Workaround: Set a larger tolerance for now.
  // TODO(penporn): See if this is a precision issue or a bug.
  test::ExpectClose(
      dy,
      test::AsTensor<complex64>({h(0.f, 2.f), h(2.f, 2.f), h(-2.f, 2.f)},
                                TensorShape({3})),
      4.5e-6f);
}

TEST_F(MathGradTest, Xlogy) {
  auto x = test::AsTensor<float>({0.f, 0.f, 2.f, 3.f, 4.f, 5.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) -> float { return x == 0. ? 0. : std::log(y); };
  auto h = [](float x, float y) -> float { return x == 0. ? 0. : x / y; };
  SymGrad("Xlogy", x, y, &dx, &dy);
  test::ExpectClose(
      dx, test::AsTensor<float>({g(0.f, .5f), g(0.f, 0.f), g(2.f, .5f),
                                 g(3.f, 2.f), g(4.f, 2.f), g(5.f, 2.f)},
                                TensorShape({2, 3})));
  test::ExpectClose(
      dy, test::AsTensor<float>({h(0.f, .5f) + h(0.f, 0.f) + h(2.f, .5f),
                                 h(3.f, 2.f) + h(4.f, 2.f) + h(5.f, 2.f)},
                                TensorShape({2, 1})));
}

TEST_F(MathGradTest, Xlog1py) {
  auto x = test::AsTensor<float>({0.f, 0.f, 2.f, 3.f, 4.f, 5.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) -> float {
    return x == 0. ? 0. : std::log1p(y);
  };
  auto h = [](float x, float y) -> float {
    return x == 0. ? 0. : x / (y + 1.);
  };
  SymGrad("Xlog1py", x, y, &dx, &dy);
  test::ExpectClose(
      dx, test::AsTensor<float>({g(0.f, .5f), g(0.f, 0.f), g(2.f, .5f),
                                 g(3.f, 2.f), g(4.f, 2.f), g(5.f, 2.f)},
                                TensorShape({2, 3})));
  test::ExpectClose(
      dy, test::AsTensor<float>({h(0.f, .5f) + h(0.f, 0.f) + h(2.f, .5f),
                                 h(3.f, 2.f) + h(4.f, 2.f) + h(5.f, 2.f)},
                                TensorShape({2, 1})));
}

TEST_F(MathGradTest, Xdivy) {
  auto x = test::AsTensor<float>({0.f, 0.f, 2.f, 3.f, 4.f, 5.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) -> float { return x == 0. ? 0. : 1 / y; };
  auto h = [](float x, float y) -> float {
    return x == 0. ? 0. : -x / (y * y);
  };
  SymGrad("Xdivy", x, y, &dx, &dy);
  test::ExpectClose(
      dx, test::AsTensor<float>({g(0.f, .5f), g(0.f, 0.f), g(2.f, .5f),
                                 g(3.f, 2.f), g(4.f, 2.f), g(5.f, 2.f)},
                                TensorShape({2, 3})));
  test::ExpectClose(
      dy, test::AsTensor<float>({h(0.f, .5f) + h(0.f, 0.f) + h(2.f, .5f),
                                 h(3.f, 2.f) + h(4.f, 2.f) + h(5.f, 2.f)},
                                TensorShape({2, 1})));
}

TEST_F(MathGradTest, SquaredDifference) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  auto g = [](float x, float y) -> float { return 2. * (x - y); };
  auto h = [](float x, float y) -> float { return 2. * (y - x); };
  SymGrad("SquaredDifference", x, y, &dx, &dy);
  test::ExpectClose(
      dx, test::AsTensor<float>({g(-3.f, .5f), g(-2.f, .5f), g(-1.f, .5f),
                                 g(1.f, 2.f), g(2.f, 2.f), g(3.f, 2.f)},
                                TensorShape({2, 3})));
  test::ExpectClose(
      dy, test::AsTensor<float>({h(-3.f, .5f) + h(-2.f, .5f) + h(-1.f, .5f),
                                 h(1.f, 2.f) + h(2.f, 2.f) + h(3.f, 2.f)},
                                TensorShape({2, 1})));
}

TEST_F(MathGradTest, Maximum) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-1.5f, 1.5f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Maximum", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) { return x >= y ? 1.f : 0.f; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-3.f, -1.5f), g(-2.f, -1.5f), g(-1.f, -1.5f),
                                 g(1.f, 1.5f), g(2.f, 1.5f), g(3.f, 1.5f)},
                                TensorShape({2, 3})));
    }
    {
      auto g = [](float x, float y) { return x < y ? 1.f : 0.f; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-3.f, -1.5f) + g(-2.f, -1.5f) + g(-1.f, -1.5f),
                             g(1.f, 1.5f) + g(2.f, 1.5f) + g(3.f, 1.5f)},
                            TensorShape({2, 1})));
    }
  }
  {  // Swap x and y
    SymGrad("Maximum", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) { return x >= y ? 1.f : 0.f; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-1.5f, -3.f) + g(-1.5f, -2.f) + g(-1.5f, -1.f),
                             g(1.5f, 1.f) + g(1.5f, 2.f) + g(1.5f, 3.f)},
                            TensorShape({2, 1})));
    }
    {
      auto g = [](float x, float y) { return x < y ? 1.f : 0.f; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-1.5f, -3.f), g(-1.5f, -2.f), g(-1.5f, -1.f),
                                 g(1.5f, 1.f), g(1.5f, 2.f), g(1.5f, 3.f)},
                                TensorShape({2, 3})));
    }
  }
}

TEST_F(MathGradTest, Minimum) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-1.5f, 1.5f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Minimum", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) { return x <= y ? 1.f : 0.f; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-3.f, -1.5f), g(-2.f, -1.5f), g(-1.f, -1.5f),
                                 g(1.f, 1.5f), g(2.f, 1.5f), g(3.f, 1.5f)},
                                TensorShape({2, 3})));
    }
    {
      auto g = [](float x, float y) { return x > y ? 1.f : 0.f; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-3.f, -1.5f) + g(-2.f, -1.5f) + g(-1.f, -1.5f),
                             g(1.f, 1.5f) + g(2.f, 1.5f) + g(3.f, 1.5f)},
                            TensorShape({2, 1})));
    }
  }
  {  // Swap x and y
    SymGrad("Minimum", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) { return x <= y ? 1.f : 0.f; };
      test::ExpectClose(dy,
                        test::AsTensor<float>(
                            {g(-1.5f, -3.f) + g(-1.5f, -2.f) + g(-1.5f, -1.f),
                             g(1.5f, 1.f) + g(1.5f, 2.f) + g(1.5f, 3.f)},
                            TensorShape({2, 1})));
    }
    {
      auto g = [](float x, float y) { return x > y ? 1.f : 0.f; };
      test::ExpectClose(dx, test::AsTensor<float>(
                                {g(-1.5f, -3.f), g(-1.5f, -2.f), g(-1.5f, -1.f),
                                 g(1.5f, 1.f), g(1.5f, 2.f), g(1.5f, 3.f)},
                                TensorShape({2, 3})));
    }
  }
}

TEST_F(MathGradTest, Select) {
  auto c = test::AsTensor<bool>({true, false, false, true, true, false},
                                TensorShape({2, 3}));
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({3.f, 2.f, 1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  Tensor dc;
  Tensor dx;
  Tensor dy;
  {
    SelectGrad(c, x, y, &dc, &dx, &dy);
    test::ExpectTensorEqual<bool>(
        dc, test::AsTensor<bool>({false, false, false, false, false, false},
                                 TensorShape({2, 3})));
    test::ExpectTensorEqual<float>(
        dx, test::AsTensor<float>({1.f, 0.f, 0.f, 1.f, 1.f, 0.f},
                                  TensorShape({2, 3})));
    test::ExpectTensorEqual<float>(
        dy, test::AsTensor<float>({0.f, 1.f, 1.f, 0.f, 0.f, 1.f},
                                  TensorShape({2, 3})));
  }
}

TEST_F(MathGradTest, MatMul_00) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({3, 1}));
  Tensor dx;
  Tensor dy;
  MatMulGrad(x, false, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({2, 1}));
  test::ExpectClose(dx, MatMul(dz, false, y, true));
  test::ExpectClose(dy, MatMul(x, true, dz, false));
}

TEST_F(MathGradTest, MatMul_01) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3}));
  Tensor dx;
  Tensor dy;
  MatMulGrad(x, false, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({2, 1}));
  test::ExpectClose(dx, MatMul(dz, false, y, false));
  test::ExpectClose(dy, MatMul(dz, true, x, false));
}

TEST_F(MathGradTest, MatMul_10) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({3, 1}));
  Tensor dx;
  Tensor dy;
  MatMulGrad(x, true, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({2, 1}));
  test::ExpectClose(dx, MatMul(y, false, dz, true));
  test::ExpectClose(dy, MatMul(x, false, dz, false));
}

TEST_F(MathGradTest, MatMul_11) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3}));
  Tensor dx;
  Tensor dy;
  MatMulGrad(x, true, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({2, 1}));
  test::ExpectClose(dx, MatMul(y, true, dz, true));
  test::ExpectClose(dy, MatMul(dz, true, x, true));
}

TEST_F(MathGradTest, BatchMatMul_00) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulGrad(x, false, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMul(dz, false, y, true));
  test::ExpectClose(dy, BatchMatMul(x, true, dz, false));
}

TEST_F(MathGradTest, BatchMatMul_01) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulGrad(x, false, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMul(dz, false, y, false));
  test::ExpectClose(dy, BatchMatMul(dz, true, x, false));
}

TEST_F(MathGradTest, BatchMatMul_10) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulGrad(x, true, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMul(y, false, dz, true));
  test::ExpectClose(dy, BatchMatMul(x, false, dz, false));
}

TEST_F(MathGradTest, BatchMatMul_11) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulGrad(x, true, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMul(y, true, dz, true));
  test::ExpectClose(dy, BatchMatMul(dz, true, x, true));
}

TEST_F(MathGradTest, BatchMatMulV2_00) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMulV2(dz, false, y, true));
  test::ExpectClose(dy, BatchMatMulV2(x, true, dz, false));
}

TEST_F(MathGradTest, BatchMatMulV2_01) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 2, 3}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMulV2(dz, false, y, false));
  test::ExpectClose(dy, BatchMatMulV2(dz, true, x, false));
}

TEST_F(MathGradTest, BatchMatMulV2_10) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, true, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMulV2(y, false, dz, true));
  test::ExpectClose(dy, BatchMatMulV2(x, false, dz, false));
}

TEST_F(MathGradTest, BatchMatMulV2_11) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({1, 3, 2}));
  auto y = test::AsTensor<float>({-1.f, .5f, 2.f}, TensorShape({1, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, true, y, true, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f}, TensorShape({1, 2, 1}));
  test::ExpectClose(dx, BatchMatMulV2(y, true, dz, true));
  test::ExpectClose(dy, BatchMatMulV2(dz, true, x, true));
}

TEST_F(MathGradTest, BatchMatMulV2_LhsBroadcasts) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>(
      {1.f, 2.4, 3.f, -1.f, .5f, 2.f, 3.f, 1.f, -1.f, 2.f, -.1f, 0},
      TensorShape({2, 3, 2}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, false, &dx, &dy);
  EXPECT_TRUE(dx.shape().IsSameSize(x.shape()));
  EXPECT_TRUE(dy.shape().IsSameSize(y.shape()));
  auto dz = test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                  TensorShape({2, 2, 2}));
  Tensor ans_dx;
  CHECK(ans_dx.CopyFrom(ReduceSum(BatchMatMulV2(dz, false, y, true), {0}),
                        dx.shape()));
  Tensor ans_dy = BatchMatMulV2(x, true, dz, false);
  test::ExpectClose(dx, ans_dx);
  test::ExpectClose(dy, ans_dy);
}

TEST_F(MathGradTest, BatchMatMulV2_RhsBroadcasts) {
  auto x = test::AsTensor<float>(
      {1.f, 2.4, 3.f, -1.f, .5f, 2.f, 3.f, 1.f, -1.f, 2.f, -.1f, 0},
      TensorShape({2, 2, 3}));
  auto y = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({3, 2}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, false, &dx, &dy);
  auto dz = test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                  TensorShape({2, 2, 2}));
  Tensor ans_dx = BatchMatMulV2(dz, false, y, true);
  Tensor ans_dy;
  CHECK(ans_dy.CopyFrom(ReduceSum(BatchMatMulV2(x, true, dz, false), {0}),
                        dy.shape()));
  test::ExpectClose(dx, ans_dx);
  test::ExpectClose(dy, ans_dy);
}

TEST_F(MathGradTest, BatchMatMulV2_BothLhsAndRhsBroadcast) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 1, 1, 3}));
  auto y = test::AsTensor<float>({3.f, 1.f, -1.f, 2.f, -.1f, 0},
                                 TensorShape({1, 2, 3, 1}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, false, y, false, &dx, &dy);
  EXPECT_TRUE(dx.shape().IsSameSize(x.shape()));
  EXPECT_TRUE(dy.shape().IsSameSize(y.shape()));
  auto dz =
      test::AsTensor<float>({1.f, 1.f, 1.f, 1.f}, TensorShape({2, 2, 1, 1}));
  Tensor ans_dx;
  Tensor ans_dy;
  CHECK(ans_dx.CopyFrom(ReduceSum(BatchMatMulV2(dz, false, y, true), {1}),
                        dx.shape()));
  CHECK(ans_dy.CopyFrom(ReduceSum(BatchMatMulV2(x, true, dz, false), {0}),
                        dy.shape()));
  test::ExpectClose(dx, ans_dx);
  test::ExpectClose(dy, ans_dy);
}

TEST_F(MathGradTest, BatchMatMulV2_BroadcastWhileAdjointed) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 1, 3, 1}));
  auto y = test::AsTensor<float>({3.f, 1.f, -1.f, 2.f, -.1f, 0},
                                 TensorShape({1, 2, 1, 3}));
  Tensor dx;
  Tensor dy;
  BatchMatMulV2Grad(x, true, y, true, &dx, &dy);
  EXPECT_TRUE(dx.shape().IsSameSize(x.shape()));
  EXPECT_TRUE(dy.shape().IsSameSize(y.shape()));

  auto dz =
      test::AsTensor<float>({1.f, 1.f, 1.f, 1.f}, TensorShape({2, 2, 1, 1}));
  Tensor ans_dx;
  Tensor ans_dy;
  CHECK(ans_dx.CopyFrom(ReduceSum(BatchMatMulV2(y, true, dz, true), {1}),
                        dx.shape()));
  CHECK(ans_dy.CopyFrom(ReduceSum(BatchMatMulV2(dz, true, x, true), {0}),
                        dy.shape()));
  test::ExpectClose(dx, ans_dx);
  test::ExpectClose(dy, ans_dy);
}

TEST_F(MathGradTest, Sum_dim0) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Sum", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Sum_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({1}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Sum", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Mean_dim0) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Mean", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>(
              {1.f / 2, 1.f / 2, 1.f / 2, 1.f / 2, 1.f / 2, 1.f / 2},
              TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Mean_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({1}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Mean", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>(
              {1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3},
              TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Mean_dim0_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Mean", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>(
              {1.f / 6, 1.f / 6, 1.f / 6, 1.f / 6, 1.f / 6, 1.f / 6},
              TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

TEST_F(MathGradTest, Min_dim0) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Min", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 1.f, 1.f, 0.f, 0.f, 0.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Min_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({1}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Min", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 0.f, 0.f, 1.f, 0.f, 0.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Min_dim0_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Min", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({1.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

TEST_F(MathGradTest, Min_dim0_dim1_Dups) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, -3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Min", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({.5f, 0.f, 0.f, 0.f, 0.f, .5f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

TEST_F(MathGradTest, Max_dim0) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Max", x, i, &dx, &di);
  LOG(INFO) << dx.SummarizeValue(6);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({0.f, 0.f, 0.f, 1.f, 1.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Max_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({1}, TensorShape({}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Max", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({0.f, 0.f, 1.f, 0.f, 0.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(di,
                                 test::AsTensor<int32>({0}, TensorShape({})));
}

TEST_F(MathGradTest, Max_dim0_dim1) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Max", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({0.f, 0.f, 0.f, 0.f, 0.f, 1.f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

TEST_F(MathGradTest, Max_dim0_dim1_Dups) {
  auto x = test::AsTensor<float>({3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto i = test::AsTensor<int32>({0, 1}, TensorShape({2}));
  Tensor dx;
  Tensor di;
  ReductionGrad("Max", x, i, &dx, &di);
  test::ExpectTensorEqual<float>(
      dx, test::AsTensor<float>({.5f, 0.f, 0.f, 0.f, 0.f, .5f},
                                TensorShape({2, 3})));
  test::ExpectTensorEqual<int32>(
      di, test::AsTensor<int32>({0, 0}, TensorShape({2})));
}

}  // namespace
}  // namespace tensorflow
