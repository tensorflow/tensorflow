/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include <vector>
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

namespace f = test::function;
typedef FunctionDefHelper FDH;

namespace {
Session* NewSession() {
  SessionOptions opts;
  (*opts.config.mutable_device_count())["CPU"] = 1;
  return NewSession(opts);
}
}  // end namespace

class MathGradTest : public ::testing::Test {
 protected:
  // Unary
  Status Unary(const string& op, const Tensor& x, Tensor* y) {
    const DataType T = x.dtype();
    auto adef = [T](const string& name) {  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(T));
    };
    // Sum(op(x)), sum all output of op(x).
    auto test = FDH::Define("Test", {adef("x")}, {adef("l")}, {},
                            {
                                {{"y"}, op, {"x"}, {{"T", T}}},
                                FDH::Const("zero", 0),
                                FDH::Const("one", 1),
                                {{"r"}, "Rank", {"x"}, {{"T", T}}},
                                {{"indices"}, "Range", {"zero", "r", "one"}},
                                {{"l"}, "Sum", {"y", "indices"}, {{"T", T}}},
                            });

    // TestGrad = Test'(x)
    auto grad = FDH::Define(
        "TestGrad", {adef("x")}, {adef("dx")}, {},
        {
            FDH::Const("one", 1),
            {{"dy"}, "Cast", {"one"}, {{"DstT", T}, {"SrcT", DT_INT32}}},
            {{"grad"},
             "SymbolicGradient",
             {"x", "dy"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, T}},
                 {"Tout", DataTypeSlice{T}},
             }},
            {{"dx"}, "Identity", {"grad:0"}, {{"T", T}}},
        });
    // Each test case will feed in "x:0" and expects to get "dx:0".
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
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
    delete sess;
    return s;
  }

  // Unary op expecting OK.
  Tensor SymGrad(const string& op, const Tensor& x) {
    Tensor ret;
    TF_CHECK_OK(Unary(op, x, &ret));
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
            {{"grad"},
             "SymbolicGradient",
             {"x", "y", "dz"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, T, T}},
                 {"Tout", DataTypeSlice{T, T}},
             }},
            {{"dx"}, "Identity", {"grad:0"}, {{"T", T}}},
            {{"dy"}, "Identity", {"grad:1"}, {{"T", T}}},
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
    delete sess;
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
            {{"grad"},
             "SymbolicGradient",
             {"x", "i", "dy"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, DT_INT32, T}},
                 {"Tout", DataTypeSlice{T, DT_INT32}},
             }},
            {{"dx"}, "Identity", {"grad:0"}, {{"T", T}}},
            {{"di"}, "Identity", {"grad:1"}, {{"T", DT_INT32}}},
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
    delete sess;
    *dx = outputs[0];
    *di = outputs[1];
  }

  Tensor MatMul(const Tensor& x, bool tx, const Tensor& y, bool ty) {
    auto T = x.dtype();
    auto gdef = test::function::GDef(
        {
            f::NDef("x", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("y", "Placeholder", {}, {{"dtype", T}}),
            f::NDef("z", "MatMul", {"x", "y"},
                    {{"T", T}, {"transpose_a", tx}, {"transpose_b", ty}}),
        },
        {});
    auto sess = NewSession();
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<Tensor> outputs;
    TF_CHECK_OK(sess->Run({{"x:0", x}, {"y:0", y}}, {"z:0"}, {}, &outputs));
    CHECK_EQ(outputs.size(), 1);
    TF_CHECK_OK(sess->Close());
    delete sess;
    return outputs[0];
  }

  void MatMulGrad(const Tensor& x, bool tx, const Tensor& y, bool ty,
                  Tensor* dx, Tensor* dy) {
    const DataType T = x.dtype();
    auto adef = [T](const string& name) {  // E.g., x:float, dy:double
      return strings::StrCat(name, ":", DataTypeString(T));
    };
    // Sum(op(x)), sum all output of op(x).
    auto test =
        FDH::Define("Test", {adef("x"), adef("y")}, {adef("l")}, {},
                    {
                        {{"z"},
                         "MatMul",
                         {"x", "y"},
                         {{"T", T}, {"transpose_a", tx}, {"transpose_b", ty}}},
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
            {{"grad"},
             "SymbolicGradient",
             {"x", "y", "dz"},
             {
                 {"f", FDH::FunctionRef("Test")},
                 {"Tin", DataTypeSlice{T, T, T}},
                 {"Tout", DataTypeSlice{T, T}},
             }},
            {{"dx"}, "Identity", {"grad:0"}, {{"T", T}}},
            {{"dy"}, "Identity", {"grad:1"}, {{"T", T}}},
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
    delete sess;
    *dx = outputs[0];
    *dy = outputs[1];
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
                             {{"grad"},
                              "SymbolicGradient",
                              {"c", "x", "y", "dz"},
                              {
                                  {"f", FDH::FunctionRef("Test")},
                                  {"Tin", DataTypeSlice{DT_BOOL, T, T, T}},
                                  {"Tout", DataTypeSlice{DT_BOOL, T, T}},
                              }},
                             {{"dc"}, "Identity", {"grad:0"}, {{"T", DT_BOOL}}},
                             {{"dx"}, "Identity", {"grad:1"}, {{"T", T}}},
                             {{"dy"}, "Identity", {"grad:2"}, {{"T", T}}}});
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
    delete sess;
    *dc = outputs[0];
    *dx = outputs[1];
    *dy = outputs[2];
  }
};

static void HasError(const Status& s, const string& substr) {
  EXPECT_TRUE(StringPiece(s.ToString()).contains(substr))
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

TEST_F(MathGradTest, Inv) {
  auto x = test::AsTensor<float>({-3.f, -2.f, -1.f, 1.f, 2.f, 3.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return -1.f / (x * x); };
  auto dx = test::AsTensor<float>(
      {g(-3.f), g(-2.f), g(-1.f), g(1.f), g(2.f), g(3.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Inv", x);
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

TEST_F(MathGradTest, Log) {
  auto x = test::AsTensor<float>({0.1f, 1.f, 2.f, 3.f, 4.f, 10.f},
                                 TensorShape({2, 3}));
  auto g = [](float x) { return 1 / x; };
  auto dx = test::AsTensor<float>(
      {g(.1f), g(1.f), g(2.f), g(3.f), g(4.f), g(10.f)}, TensorShape({2, 3}));
  auto ans = SymGrad("Log", x);
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

// TODO(zhifengc)
// TEST_F(MathGradSComplexTest, Real) {}
// TEST_F(MathGradSComplexTest, Imag) {}
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

TEST_F(MathGradTest, Pow) {
  auto x = test::AsTensor<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f},
                                 TensorShape({2, 3}));
  auto y = test::AsTensor<float>({.5f, 2.f}, TensorShape({2, 1}));
  Tensor dx;
  Tensor dy;
  {
    SymGrad("Pow", x, y, &dx, &dy);
    {
      auto g = [](float x, float y) { return y * std::pow(x, y - 1); };
      test::ExpectClose(
          dx, test::AsTensor<float>({g(1.f, .5f), g(2.f, .5f), g(3.f, .5f),
                                     g(4.f, 2.f), g(5.f, 2.f), g(6.f, 2.f)},
                                    TensorShape({2, 3})));
    }
    {
      auto g = [](float x, float y) { return std::pow(x, y) * std::log(x); };
      test::ExpectClose(
          dy, test::AsTensor<float>({g(1.f, .5f) + g(2.f, .5f) + g(3.f, .5f),
                                     g(4.f, 2.f) + g(5.f, 2.f) + g(6.f, 2.f)},
                                    TensorShape({2, 1})));
    }
  }
  {  // Swap x and y
    SymGrad("Pow", y, x, &dy, &dx);
    {
      auto g = [](float x, float y) { return y * std::pow(x, y - 1); };
      test::ExpectClose(
          dy, test::AsTensor<float>({g(.5f, 1.f) + g(.5f, 2.f) + g(.5f, 3.f),
                                     g(2.f, 4.f) + g(2.f, 5.f) + g(2.f, 6.f)},
                                    TensorShape({2, 1})));
    }
    {
      auto g = [](float x, float y) { return std::pow(x, y) * std::log(x); };
      test::ExpectClose(
          dx, test::AsTensor<float>({g(.5f, 1.f), g(.5f, 2.f), g(.5f, 3.f),
                                     g(2.f, 4.f), g(2.f, 5.f), g(2.f, 6.f)},
                                    TensorShape({2, 3})));
    }
  }
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

}  // end namespace tensorflow
