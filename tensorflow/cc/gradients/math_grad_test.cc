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

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
using namespace ops;  // NOLINT(build/namespaces)

namespace {

// TODO(andydavis) Test gradient function against numeric gradients output.
// TODO(andydavis) As more gradients are added move common test functions
// to a testutil library.

class CWiseUnaryGradTest : public ::testing::Test {
 protected:
  CWiseUnaryGradTest() : scope_(Scope::NewRootScope().WithDevice("/cpu:0")) {}

  enum UnaryOpType {
    ABS,
    NEG,
    INV,
    SQUARE,
    SQRT,
    RSQRT,
    EXP,
    EXPM1,
    LOG,
    LOG1P,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    SIGMOID,
    SIGN,
    SIN,
    COS,
    ASIN,
    ACOS,
    TAN,
    ATAN
  };

  template <typename T>
  void TestCWiseGrad(UnaryOpType op_type, const std::function<T(int)>& x_fn,
                     const std::function<T(const T&)>& dy_fn,
                     const std::function<T(const T&, const T&)>& dx_fn) {
    DataType dtype = DataTypeToEnum<T>::v();
    Tensor x(dtype, {2, 3, 2});
    auto x_flat = x.flat<T>();
    for (int i = 0; i < x_flat.size(); ++i) {
      x_flat(i) = x_fn(i);
    }

    Tensor dy(dtype, {2, 3, 2});
    auto dy_flat = dy.flat<T>();
    for (int i = 0; i < dy_flat.size(); ++i) {
      dy_flat(i) = dy_fn(x_flat(i));
    }

    Tensor dx(dtype, {2, 3, 2});
    auto dx_flat = dx.flat<T>();
    for (int i = 0; i < dx_flat.size(); ++i) {
      dx_flat(i) = dx_fn(x_flat(i), dy_flat(i));
    }

    Output y;
    switch (op_type) {
      case ABS:
        y = Abs(scope_, x);
        break;
      case NEG:
        y = Neg(scope_, x);
        break;
      case INV:
        y = Reciprocal(scope_, x);
        break;
      case SQUARE:
        y = Square(scope_, x);
        break;
      case SQRT:
        y = Sqrt(scope_, x);
        break;
      case RSQRT:
        y = Rsqrt(scope_, x);
        break;
      case EXP:
        y = Exp(scope_, x);
        break;
      case EXPM1:
        y = Expm1(scope_, x);
        break;
      case LOG:
        y = Log(scope_, x);
        break;
      case LOG1P:
        y = Log1p(scope_, x);
        break;
      case SINH:
        y = Sinh(scope_, x);
        break;
      case COSH:
        y = Cosh(scope_, x);
        break;
      case TANH:
        y = Tanh(scope_, x);
        break;
      case ASINH:
        y = Asinh(scope_, x);
        break;
      case ACOSH:
        y = Acosh(scope_, x);
        break;
      case ATANH:
        y = Atanh(scope_, x);
        break;
      case SIGMOID:
        y = Sigmoid(scope_, x);
        break;
      case SIGN:
        y = Sign(scope_, x);
        break;
      case SIN:
        y = Sin(scope_, x);
        break;
      case COS:
        y = Cos(scope_, x);
        break;
      case ASIN:
        y = Asin(scope_, x);
        break;
      case ACOS:
        y = Acos(scope_, x);
        break;
      case TAN:
        y = Tan(scope_, x);
        break;
      case ATAN:
        y = Atan(scope_, x);
        break;
    }

    std::vector<Output> grad_outputs;
    TF_ASSERT_OK(test::CallGradFunction(
        scope_, Operation(y.node()), {ops::Const(scope_, dy)}, &grad_outputs));
    Tensor output;
    test::GetTensor(scope_, grad_outputs[0], &output);
    test::ExpectClose(output, dx);
  }

  float RV(const std::vector<float>& v) {
    return v[random::New64() % v.size()];
  }

  complex64 CRV(const std::vector<complex64>& v) {
    return v[random::New64() % v.size()];
  }

  complex64 conjugate(const complex64& val) {
    return complex64(val.real(), -val.imag());
  }

  const complex64 one_{1.0, 0};

  Scope scope_;
};

TEST_F(CWiseUnaryGradTest, Abs) {
  auto x_fn = [this](const int i) { return RV({-1, 0, 1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) { return x * dy; };
  TestCWiseGrad<float>(ABS, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Neg) {
  auto x_fn = [this](const int i) { return RV({-1, 0, 1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) { return -dy; };
  TestCWiseGrad<float>(NEG, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Reciprocal) {
  auto x_fn = [this](const int i) { return RV({-1, 1, -2, 2, -3, 3, -4, 4}); };
  auto dy_fn = [this](const float x) { return RV({0, -2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return -(1 / (x * x)) * dy;
  };
  TestCWiseGrad<float>(INV, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Reciprocal_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  auto dy_fn = [this](const complex64 x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64 x, const complex64 dy) {
    return -conjugate(one_ / (x * x)) * dy;
  };
  TestCWiseGrad<complex64>(INV, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Square) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return RV({0, -7, 7, -8, 8, -9, 9}); };
  auto dx_fn = [this](const float x, const float dy) { return 2 * x * dy; };
  TestCWiseGrad<float>(SQUARE, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Square_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return conjugate(complex64(2, 0) * x) * dy;
  };
  TestCWiseGrad<complex64>(SQUARE, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sqrt) {
  auto x_fn = [this](const int i) { return RV({0, 1, 2, 3, 4, 5, 6, 7}); };
  auto dy_fn = [this](const float x) { return x + RV({8, 9, 10, 11, 12, 13}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * 0.5 * (1.0 / std::sqrt(x));
  };
  TestCWiseGrad<float>(SQRT, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sqrt_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return conjugate(complex64(0.5, 0) / std::sqrt(x)) * dy;
  };
  TestCWiseGrad<complex64>(SQRT, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Rsqrt) {
  auto x_fn = [this](const int i) { return RV({1, 2, 3, 4, 5, 6, 7, 8}); };
  auto dy_fn = [this](const float x) { return x + RV({8, 9, 10, 11, 12, 13}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * -0.5 * (1 / std::sqrt(x)) * (1 / x);
  };
  TestCWiseGrad<float>(RSQRT, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Rsqrt_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return conjugate(complex64(-0.5, 0) / std::sqrt(x) / x) * dy;
  };
  TestCWiseGrad<complex64>(RSQRT, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Exp) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * std::exp(x);
  };
  TestCWiseGrad<float>(EXP, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Exp_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy * conjugate(std::exp(x));
  };
  TestCWiseGrad<complex64>(EXP, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Expm1) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1e-6, 1, -2, 3, 100}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * std::exp(x);
  };
  TestCWiseGrad<float>(EXPM1, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Expm1_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy * conjugate(std::exp(x));
  };
  TestCWiseGrad<complex64>(EXPM1, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Log) {
  auto x_fn = [this](const int i) { return RV({-1, 1, -2, 2, -3, 3, -4, 4}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) { return dy * (1.0 / x); };
  TestCWiseGrad<float>(LOG, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Log_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy * conjugate(one_ / x);
  };
  TestCWiseGrad<complex64>(LOG, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Log1p) {
  auto x_fn = [this](const int i) { return RV({0, 1e-6, 1, 2, 3, 4, 100}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (1.0 / (1.0 + x));
  };
  TestCWiseGrad<float>(LOG1P, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Log1p_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{0, 0}, {1e-6, 0}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy / (one_ + conjugate(x));
  };
  TestCWiseGrad<complex64>(LOG1P, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sinh) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * std::cosh(x);
  };
  TestCWiseGrad<float>(SINH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sinh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy * conjugate(std::cosh(x));
  };
  TestCWiseGrad<complex64>(SINH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Cosh) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * std::sinh(x);
  };
  TestCWiseGrad<float>(COSH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Cosh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy * conjugate(std::sinh(x));
  };
  TestCWiseGrad<complex64>(COSH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Tanh) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    const float y = std::tanh(x);
    return dy * (1.0 - y * y);
  };
  TestCWiseGrad<float>(TANH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Tanh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    const complex64 y = std::tanh(x);
    return dy * conjugate((one_ - y * y));
  };
  TestCWiseGrad<complex64>(TANH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Asinh) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    auto y = std::asinh(x);
    return dy / std::cosh(y);
  };
  TestCWiseGrad<float>(ASINH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Asinh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    auto y = std::asinh(x);
    return dy / conjugate(std::cosh(y));
  };
  TestCWiseGrad<complex64>(ASINH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Acosh) {
  auto x_fn = [this](const int i) { return RV({1, 2, 3, 4, 5, 6, 7}); };
  auto dy_fn = [this](const float x) {
    return x + RV({8, 9, 10, 11, 12, 13, 14});
  };
  auto dx_fn = [this](const float x, const float dy) {
    auto y = std::acosh(x);
    return dy / std::sinh(y);
  };
  TestCWiseGrad<float>(ACOSH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Acosh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 1}, {2, 1}, {1, 4}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{2, 2}, {3, 3}, {1, 4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    auto y = std::acosh(x);
    return dy / conjugate(std::sinh(y));
  };
  TestCWiseGrad<complex64>(ACOSH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Atanh) {
  auto x_fn = [this](const int i) { return RV({0, -0.5, 0.5, -0.1, 0.1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (1. / (1. - x * x));
  };
  TestCWiseGrad<float>(ATANH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Atanh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{0.1, 0}, {0, 0.1}, {0.2, -0.1}, {0.1, 0.2}, {0.3, 0.4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy / conjugate(one_ - x * x);
  };
  TestCWiseGrad<complex64>(ATANH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sigmoid) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    const float y = 1.0 / (1.0 + std::exp(-x));
    return dy * y * (1.0 - y);
  };
  TestCWiseGrad<float>(SIGMOID, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sigmoid_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 0}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    const complex64 y = one_ / (one_ + std::exp(-x));
    return dy * conjugate(y * (one_ - y));
  };
  TestCWiseGrad<complex64>(SIGMOID, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sign) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) { return 0.0; };
  TestCWiseGrad<float>(SIGN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sin) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * std::cos(x);
  };
  TestCWiseGrad<float>(SIN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sin_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy * conjugate(std::cos(x));
  };
  TestCWiseGrad<complex64>(SIN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Cos) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * -1.0 * std::sin(x);
  };
  TestCWiseGrad<float>(COS, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Cos_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy * conjugate(-std::sin(x));
  };
  TestCWiseGrad<complex64>(COS, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Asin) {
  auto x_fn = [this](const int i) { return RV({0, -0.5, 0.5, -1, 1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (1.0 / std::sqrt(1.0 - x * x));
  };
  TestCWiseGrad<float>(ASIN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Asin_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy / conjugate(std::sqrt(one_ - x * x));
  };
  // TODO(kbsriram)
  // Enable test when the asin kernel supports complex numbers
  if (false) {
    TestCWiseGrad<complex64>(ASIN, x_fn, dy_fn, dx_fn);
  }
}

TEST_F(CWiseUnaryGradTest, Acos) {
  auto x_fn = [this](const int i) { return RV({0, -0.5, 0.5, -1, 1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (-1.0 / std::sqrt(1.0 - x * x));
  };
  TestCWiseGrad<float>(ACOS, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Acos_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy / -conjugate(std::sqrt(one_ - x * x));
  };
  // TODO(kbsriram)
  // Add test when the acos kernel supports complex numbers
  if (false) {
    TestCWiseGrad<complex64>(ACOS, x_fn, dy_fn, dx_fn);
  }
}

TEST_F(CWiseUnaryGradTest, Tan) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    const float cosx = std::cos(x);
    return dy * (1 / (cosx * cosx));
  };
  TestCWiseGrad<float>(TAN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Tan_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    const complex64 cosx = std::cos(x);
    return dy / conjugate(cosx * cosx);
  };
  // TODO(kbsriram)
  // Enable when tan kernel supports complex inputs
  if (false) {
    TestCWiseGrad<complex64>(TAN, x_fn, dy_fn, dx_fn);
  }
}

TEST_F(CWiseUnaryGradTest, Atan) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (1 / (1 + x * x));
  };
  TestCWiseGrad<float>(ATAN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Atan_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  auto dy_fn = [this](const complex64& x) {
    return x + CRV({{-2, 2}, {-3, 3}, {1, -4}});
  };
  auto dx_fn = [this](const complex64& x, const complex64& dy) {
    return dy / (one_ + x * x);
  };
  // TODO(kbsriram)
  // Add test when the atan kernel supports complex numbers
  if (false) {
    TestCWiseGrad<complex64>(ATAN, x_fn, dy_fn, dx_fn);
  }
}

class CWiseUnaryComplexGradTest : public ::testing::Test {
 protected:
  CWiseUnaryComplexGradTest()
      : scope_(Scope::NewRootScope().WithDevice("/cpu:0")) {}

  enum UnaryOpType { REAL, IMAG, ANGLE, CONJ };

  void TestCWiseGradComplex(UnaryOpType op_type, const Tensor& x,
                            const Tensor& dy, const Tensor& dx_expected) {
    Output y;
    switch (op_type) {
      case REAL:
        y = Real(scope_, x);
        break;
      case IMAG:
        y = Imag(scope_, x);
        break;
      case ANGLE:
        y = Angle(scope_, x);
        break;
      case CONJ:
        y = Conj(scope_, x);
        break;
    }

    std::vector<Output> grad_outputs;
    TF_ASSERT_OK(test::CallGradFunction(
        scope_, Operation(y.node()), {ops::Const(scope_, dy)}, &grad_outputs));
    Tensor dx;
    test::GetTensor(scope_, grad_outputs[0], &dx);
    test::ExpectClose(dx, dx_expected);
  }

  Scope scope_;
};

TEST_F(CWiseUnaryComplexGradTest, Real) {
  Tensor x = test::AsTensor<complex64>(
      {{1, -1}, {-2, 2}, {3, -3}, {-4, 4}, {8, -8}, {-9, 9}}, {2, 3});
  Tensor dy = test::AsTensor<float>({11, -12, 13, -14, 15, -16}, {2, 3});
  Tensor dx_expected = test::AsTensor<complex64>(
      {{11, 0}, {-12, 0}, {13, 0}, {-14, 0}, {15, 0}, {-16, 0}}, {2, 3});
  TestCWiseGradComplex(REAL, x, dy, dx_expected);
}

TEST_F(CWiseUnaryComplexGradTest, Imag) {
  Tensor x = test::AsTensor<complex64>(
      {{1, -1}, {-2, 2}, {3, -3}, {-4, 4}, {8, -8}, {-9, 9}}, {2, 3});
  Tensor dy = test::AsTensor<float>({11, -12, 13, -14, 15, -16}, {2, 3});
  Tensor dx_expected = test::AsTensor<complex64>(
      {{0, 11}, {0, -12}, {0, 13}, {0, -14}, {0, 15}, {0, -16}}, {2, 3});
  TestCWiseGradComplex(IMAG, x, dy, dx_expected);
}

TEST_F(CWiseUnaryComplexGradTest, Angle) {
  Tensor x = test::AsTensor<complex64>(
      {{1, -1}, {-2, 2}, {3, -3}, {-4, 4}, {8, -8}, {-9, 9}}, {2, 3});
  Tensor dy = test::AsTensor<float>({11, -12, 13, -14, 15, -16}, {2, 3});
  Tensor dx_expected = test::AsTensor<complex64>(
      {{5.5, 5.5}, {3, 3},
       {2.1666666666666665, 2.1666666666666665}, {1.75, 1.75},
       {0.9375, 0.9375}, {0.8888888888888888, 0.8888888888888888}}, {2, 3});
  TestCWiseGradComplex(ANGLE, x, dy, dx_expected);
}

TEST_F(CWiseUnaryComplexGradTest, Conj) {
  Tensor x = test::AsTensor<complex64>(
      {{1, -1}, {-2, 2}, {3, -3}, {-4, 4}, {8, -8}, {-9, 9}}, {2, 3});
  Tensor dy = test::AsTensor<complex64>(
      {{1, -1}, {-2, 2}, {3, -3}, {-4, 4}, {8, -8}, {-9, 9}}, {2, 3});
  Tensor dx_expected = test::AsTensor<complex64>(
      {{1, 1}, {-2, -2}, {3, 3}, {-4, -4}, {8, 8}, {-9, -9}}, {2, 3});
  TestCWiseGradComplex(CONJ, x, dy, dx_expected);
}

class MathGradTest : public ::testing::Test {
 protected:
  MathGradTest() : root_(Scope::NewRootScope().WithDevice("/cpu:0")) {}

  void TestMatMulGrad(const bool is_batch, const bool t_x, const bool t_y) {
    // Generate random test data.
    std::vector<Tensor> data;
    RandMatMulGradData(is_batch, t_x, t_y, &data);
    auto x = Const(root_, data[0]);
    auto y = Const(root_, data[1]);
    auto dz = Const(root_, data[2]);

    std::vector<Tensor> grad_outputs;
    ComputeMatMulGrad(is_batch, x, t_x, y, t_y, dz, &grad_outputs);

    if (!t_x && !t_y) {
      test::ExpectClose(grad_outputs[0],
                        ComputeMatMul(is_batch, dz, false, y, true));
      test::ExpectClose(grad_outputs[1],
                        ComputeMatMul(is_batch, x, true, dz, false));
    } else if (t_x && !t_y) {
      test::ExpectClose(grad_outputs[0],
                        ComputeMatMul(is_batch, y, false, dz, true));
      test::ExpectClose(grad_outputs[1],
                        ComputeMatMul(is_batch, x, false, dz, false));
    } else if (!t_x && t_y) {
      test::ExpectClose(grad_outputs[0],
                        ComputeMatMul(is_batch, dz, false, y, false));
      test::ExpectClose(grad_outputs[1],
                        ComputeMatMul(is_batch, dz, true, x, false));
    } else {
      test::ExpectClose(grad_outputs[0],
                        ComputeMatMul(is_batch, y, true, dz, true));
      test::ExpectClose(grad_outputs[1],
                        ComputeMatMul(is_batch, dz, true, x, true));
    }
  }

  void ComputeMatMulGrad(const bool is_batch, const Output& x, const bool t_x,
                         const Output& y, const bool t_y, const Output& dz,
                         std::vector<Tensor>* out) {
    // Compute forward MatMul: z = MatMul(x, y).
    Output z;
    if (is_batch) {
      z = BatchMatMul(root_, x, y, BatchMatMul::AdjX(t_x).AdjY(t_y));
    } else {
      z = MatMul(root_, x, y, MatMul::TransposeA(t_x).TransposeB(t_y));
    }
    TF_ASSERT_OK(root_.status());
    CHECK_NOTNULL(z.node());
    std::vector<Output> grad_outputs;
    // Call MatMulGrad which populates 'grad_outputs'.
    TF_ASSERT_OK(test::CallGradFunction(root_, Operation(z.node()), {dz},
                                        &grad_outputs));
    ASSERT_EQ(2, grad_outputs.size());
    // Run graph and return MatMul gradient tensors for 'dx' and 'dy' in 'out'.
    test::GetTensors(root_, {grad_outputs[0], grad_outputs[1]}, out);
  }

  Tensor ComputeMatMul(const bool is_batch, const Output& x, const bool t_x,
                       const Output& y, const bool t_y) {
    Output z;
    if (is_batch) {
      z = BatchMatMul(root_, x, y, BatchMatMul::AdjX(t_x).AdjY(t_y));
    } else {
      z = MatMul(root_, x, y, MatMul::TransposeA(t_x).TransposeB(t_y));
    }
    TF_EXPECT_OK(root_.status());
    Tensor out;
    test::GetTensor(root_, z, &out);
    return out;
  }

  void RandMatMulGradData(const bool is_batch, const bool tx, const bool ty,
                          std::vector<Tensor>* data) {
    // Choose a random batch size in [1, 4]
    const int b = 1 + (random::New64() % 4);
    // z = MatMul(x, y)
    const int m = Rand();
    const int k = Rand();
    const int n = Rand();

    TensorShape x_shape;
    if (is_batch) {
      // x.shape = [b, m, k]
      x_shape = tx ? TensorShape({b, k, m}) : TensorShape({b, m, k});
    } else {
      // x.shape = [m, k]
      x_shape = tx ? TensorShape({k, m}) : TensorShape({m, k});
    }
    data->emplace_back(DT_FLOAT, x_shape);
    RandTensor(&data->back());

    TensorShape y_shape;
    if (is_batch) {
      // y.shape = [b, k, n]
      y_shape = ty ? TensorShape({b, n, k}) : TensorShape({b, k, n});
    } else {
      // y.shape = [k, n]
      y_shape = ty ? TensorShape({n, k}) : TensorShape({k, n});
    }
    data->emplace_back(DT_FLOAT, y_shape);
    RandTensor(&data->back());

    TensorShape z_shape;
    if (is_batch) {
      // z.shape = [b, m, n]
      z_shape = TensorShape({b, m, n});
    } else {
      // z.shape = [m, n]
      z_shape = TensorShape({m, n});
    }
    data->emplace_back(DT_FLOAT, z_shape);
    RandTensor(&data->back());
  }

  void RandTensor(Tensor* t) {
    test::FillFn<float>(
        t, [this](const int i) { return static_cast<float>(Rand()); });
  }

  int Rand() { return 1 + (random::New64() % 10); }

  Scope root_;
};

TEST_F(MathGradTest, MatMulGrad_NoTranspose) {
  TestMatMulGrad(false, false, false);
}

TEST_F(MathGradTest, MatMulGrad_TransposeX) {
  TestMatMulGrad(false, true, false);
}

TEST_F(MathGradTest, MatMulGrad_TransposeY) {
  TestMatMulGrad(false, false, true);
}

TEST_F(MathGradTest, MatMulGrad_TransposeX_TransposeY) {
  TestMatMulGrad(false, true, true);
}

TEST_F(MathGradTest, BatchMatMulGrad_NoTranspose) {
  TestMatMulGrad(true, false, false);
}

TEST_F(MathGradTest, BatchMatMulGrad_TransposeX) {
  TestMatMulGrad(true, true, false);
}

TEST_F(MathGradTest, BatchMatMulGrad_TransposeY) {
  TestMatMulGrad(true, false, true);
}

TEST_F(MathGradTest, BatchMatMulGrad_TransposeX_TransposeY) {
  TestMatMulGrad(true, true, true);
}

class NaryGradTest : public ::testing::Test {
 protected:
  NaryGradTest() : scope_(Scope::NewRootScope().WithDevice("/cpu:0")) {}

  void RunTest(const OutputList& xs, const std::vector<TensorShape>& x_shapes,
               const OutputList& ys, const std::vector<TensorShape>& y_shapes) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK(
        ComputeGradientError(scope_, xs, x_shapes, ys, y_shapes, &max_error));
    EXPECT_LT(max_error, 1e-3);
  }

  void RunTest(const Output& x, const Tensor& x_init_value, const Output& y,
               const TensorShape& y_shape) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK(
        ComputeGradientError(scope_, x, x_init_value, y, y_shape, &max_error));
    EXPECT_LT(max_error, 1e-3);
  }

  Scope scope_;
};

TEST_F(NaryGradTest, AddN) {
  TensorShape shape({3, 2, 5});
  std::vector<Output> xs;
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape)));
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape)));
  xs.push_back(Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape)));
  auto y = AddN(scope_, xs);
  RunTest(xs, {shape, shape, shape}, {y}, {shape});
}

TEST_F(NaryGradTest, Add) {
  TensorShape x1_shape({3, 2, 5});
  TensorShape x2_shape({2, 5});
  auto x1 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x1_shape));
  auto x2 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x2_shape));
  auto y = Add(scope_, x1, x2);
  RunTest({x1, x2}, {x1_shape, x2_shape}, {y}, {x1_shape});
}

TEST_F(NaryGradTest, Sub) {
  TensorShape x1_shape({3, 2, 5});
  TensorShape x2_shape({2, 5});
  auto x1 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x1_shape));
  auto x2 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x2_shape));
  auto y = Sub(scope_, x1, x2);
  RunTest({x1, x2}, {x1_shape, x2_shape}, {y}, {x1_shape});
}

TEST_F(NaryGradTest, Mul) {
  TensorShape x1_shape({3, 2, 5});
  TensorShape x2_shape({2, 5});
  auto x1 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x1_shape));
  auto x2 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x2_shape));
  auto y = Mul(scope_, x1, x2);
  RunTest({x1, x2}, {x1_shape, x2_shape}, {y}, {x1_shape});
}

TEST_F(NaryGradTest, Div) {
  TensorShape x_shape({3, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Test x / (1 + |x|) rather than x_1 / x_2 to avoid triggering large
  // division errors in the numeric estimator used by the gradient checker.
  auto y = Div(scope_, x, Add(scope_, Const<float>(scope_, 1), Abs(scope_, x)));
  RunTest({x}, {x_shape}, {y}, {x_shape});
}

TEST_F(NaryGradTest, RealDiv) {
  TensorShape x_shape({3, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  // Test x / (1 + |x|) rather than x_1 / x_2 to avoid triggering large
  // division errors in the numeric estimator used by the gradient checker.
  auto y =
      RealDiv(scope_, x, Add(scope_, Const<float>(scope_, 1), Abs(scope_, x)));
  RunTest({x}, {x_shape}, {y}, {x_shape});
}

TEST_F(NaryGradTest, SquaredDifference) {
  TensorShape x1_shape({3, 2, 5});
  TensorShape x2_shape({2, 5});
  auto x1 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x1_shape));
  auto x2 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x2_shape));
  auto y = SquaredDifference(scope_, x1, x2);
  RunTest({x1, x2}, {x1_shape, x2_shape}, {y}, {x1_shape});
}

TEST_F(NaryGradTest, Maximum) {
  TensorShape shape({3, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Maximum(scope_, x, Const(scope_, 1.0f));
  // Select values away from 1.0f to avoid instability when computing
  // finite differences.
  Tensor x_init_value =
      test::AsTensor<float>({0.5f, 1.5f, -1.2f, 3.0f, 0.1f, 2.8f}, {3, 2});
  RunTest(x, x_init_value, y, shape);
}

TEST_F(NaryGradTest, Minimum) {
  TensorShape shape({3, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Minimum(scope_, x, Const(scope_, 1.0f));
  // Select values away from 1.0f to avoid instability when computing
  // finite differences.
  Tensor x_init_value =
      test::AsTensor<float>({0.5f, 1.5f, -1.2f, 3.0f, 0.1f, 2.8f}, {3, 2});
  RunTest(x, x_init_value, y, shape);
}

}  // namespace
}  // namespace tensorflow
