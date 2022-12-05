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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/gradients/grad_testutil.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace {

using ops::Abs;
using ops::Add;
using ops::AddN;
using ops::AddV2;
using ops::Atan2;
using ops::BatchMatMul;
using ops::Cast;
using ops::Const;
using ops::Cumsum;
using ops::Div;
using ops::DivNoNan;
using ops::MatMul;
using ops::Max;
using ops::Maximum;
using ops::Mean;
using ops::Min;
using ops::Minimum;
using ops::Mul;
using ops::Placeholder;
using ops::Pow;
using ops::Prod;
using ops::RealDiv;
using ops::SegmentSum;
using ops::SelectV2;
using ops::SquaredDifference;
using ops::Sub;
using ops::Sum;
using ops::UnsortedSegmentMax;
using ops::UnsortedSegmentMin;
using ops::UnsortedSegmentSum;
using ops::Where3;

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
    ATAN,
    REAL,
    IMAG,
    CONJ,
    COMPLEX,
    ANGLE,
    LGAMMA,
    ERF,
    ERFINV,
    NDTRI
  };

  template <typename X_T, typename Y_T>
  void TestCWiseGrad(UnaryOpType op_type, const std::function<X_T(int)>& x_fn) {
    TF_ASSERT_OK(scope_.status());
    DataType x_type = DataTypeToEnum<X_T>::v();
    TensorShape shape({2, 3, 2});
    auto x = Placeholder(scope_, x_type, Placeholder::Shape(shape));
    Tensor x_data(x_type, shape);
    auto x_data_flat = x_data.flat<X_T>();
    for (int i = 0; i < x_data_flat.size(); ++i) {
      x_data_flat(i) = x_fn(i);
    }

    Output y;
    switch (op_type) {
      using namespace ops;  // NOLINT(build/namespaces)
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
      case REAL:
        y = Real(scope_, x);
        break;
      case IMAG:
        y = Imag(scope_, x);
        break;
      case CONJ:
        y = Conj(scope_, x);
        break;
      case COMPLEX:
        y = Complex(scope_, x, x);
        break;
      case ANGLE:
        y = Angle(scope_, x);
        break;
      case LGAMMA:
        y = Lgamma(scope_, x);
        break;
      case ERF:
        y = Erf(scope_, x);
        break;
      case ERFINV:
        y = Erfinv(scope_, x);
        break;
      case NDTRI:
        y = Ndtri(scope_, x);
        break;
    }

    float max_error;
    TF_ASSERT_OK((ComputeGradientError<X_T, Y_T, float>(scope_, x, x_data, y,
                                                        shape, &max_error)));
    EXPECT_LT(max_error, 1e-3f);
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

  Scope scope_;
};

TEST_F(CWiseUnaryGradTest, Abs) {
  auto x_fn = [this](const int i) { return RV({-1, 0, 1}); };
  TestCWiseGrad<float, float>(ABS, x_fn);
}

TEST_F(CWiseUnaryGradTest, Neg) {
  auto x_fn = [this](const int i) { return RV({-1, 0, 1}); };
  TestCWiseGrad<float, float>(NEG, x_fn);
}

TEST_F(CWiseUnaryGradTest, Reciprocal) {
  auto x_fn = [this](const int i) { return RV({-1, 1, -2, 2, -3, 3, -4, 4}); };
  TestCWiseGrad<float, float>(INV, x_fn);
}

TEST_F(CWiseUnaryGradTest, Reciprocal_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  TestCWiseGrad<complex64, complex64>(INV, x_fn);
}

TEST_F(CWiseUnaryGradTest, Square) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(SQUARE, x_fn);
}

TEST_F(CWiseUnaryGradTest, Square_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  TestCWiseGrad<complex64, complex64>(SQUARE, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sqrt) {
  auto x_fn = [this](const int i) { return RV({0.5, 1, 2, 3, 4, 5, 6, 7}); };
  TestCWiseGrad<float, float>(SQRT, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sqrt_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{-1.0f, 0.5f}, {1.0f, 0.5f}, {2, -1}});
  };
  TestCWiseGrad<complex64, complex64>(SQRT, x_fn);
}

TEST_F(CWiseUnaryGradTest, Rsqrt) {
  auto x_fn = [this](const int i) { return RV({1, 2, 3, 4, 5, 6, 7, 8}); };
  TestCWiseGrad<float, float>(RSQRT, x_fn);
}

TEST_F(CWiseUnaryGradTest, Rsqrt_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{-1.0f, 0.5f}, {1.0f, 0.5f}, {2, -1}});
  };
  TestCWiseGrad<complex64, complex64>(RSQRT, x_fn);
}

TEST_F(CWiseUnaryGradTest, Exp) {
  auto x_fn = [this](const int i) {
    return RV({0, -1, 1, -1.5f, 1.5f, -2, 2});
  };
  TestCWiseGrad<float, float>(EXP, x_fn);
}

TEST_F(CWiseUnaryGradTest, Exp_Complex) {
  auto x_fn = [this](const int i) { return CRV({{-1, 0}, {1, 0}, {2, -1}}); };
  TestCWiseGrad<complex64, complex64>(EXP, x_fn);
}

TEST_F(CWiseUnaryGradTest, Expm1) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1e-6, 1, -1.5, 1.5}); };
  TestCWiseGrad<float, float>(EXPM1, x_fn);
}

TEST_F(CWiseUnaryGradTest, Expm1_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{-1, 0}, {1, 0}, {1.5, -1.5}});
  };
  TestCWiseGrad<complex64, complex64>(EXPM1, x_fn);
}

TEST_F(CWiseUnaryGradTest, Log) {
  auto x_fn = [this](const int i) { return RV({0.5, 1, 2, 3, 4}); };
  TestCWiseGrad<float, float>(LOG, x_fn);
}

TEST_F(CWiseUnaryGradTest, Log_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{-1, 0.5f}, {1, 0.5f}, {2, -1}});
  };
  TestCWiseGrad<complex64, complex64>(LOG, x_fn);
}

TEST_F(CWiseUnaryGradTest, Log1p) {
  auto x_fn = [this](const int i) { return RV({0, 1e-6, 1, 2, 3, 4, 100}); };
  TestCWiseGrad<float, float>(LOG1P, x_fn);
}

TEST_F(CWiseUnaryGradTest, Log1p_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{0, 0}, {1e-6, 0}, {2, -1}, {1, 2}, {3, 4}});
  };
  TestCWiseGrad<complex64, complex64>(LOG1P, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sinh) {
  auto x_fn = [this](const int i) { return RV({0.5, -0.5, 1, -1, 1.5, -1.5}); };
  TestCWiseGrad<float, float>(SINH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sinh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{0.5, 0.25}, {0.25, 0.5}, {1.5, -1}, {1, 1.5}});
  };
  TestCWiseGrad<complex64, complex64>(SINH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Cosh) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(COSH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Cosh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{0.5, 0.25}, {0.25, 0.5}, {1.5, -1}, {1, 1.5}});
  };
  TestCWiseGrad<complex64, complex64>(COSH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Tanh) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(TANH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Tanh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  TestCWiseGrad<complex64, complex64>(TANH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Asinh) {
  auto x_fn = [this](const int i) { return RV({0.5, 1, -1, -1.5, 1.5}); };
  TestCWiseGrad<float, float>(ASINH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Asinh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0.5}, {0.5, 1}, {0.5, -1}, {1, 1.5}});
  };
  TestCWiseGrad<complex64, complex64>(ASINH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Acosh) {
  auto x_fn = [this](const int i) { return RV({1.5, 2, 2.5}); };
  TestCWiseGrad<float, float>(ACOSH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Acosh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0.5}, {0.5, 1}, {0.5, -1}, {1, 1.5}});
  };
  TestCWiseGrad<complex64, complex64>(ACOSH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Atanh) {
  auto x_fn = [this](const int i) { return RV({0, -0.5, 0.5, -0.1, 0.1}); };
  TestCWiseGrad<float, float>(ATANH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Atanh_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{0.1, 0}, {0, 0.1}, {0.2, -0.1}, {0.1, 0.2}, {0.3, 0.4}});
  };
  TestCWiseGrad<complex64, complex64>(ATANH, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sigmoid) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(SIGMOID, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sigmoid_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 0}, {2, -1}, {1, 2}, {3, 4}});
  };
  TestCWiseGrad<complex64, complex64>(SIGMOID, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sign) {
  auto x_fn = [this](const int i) { return RV({-1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(SIGN, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sin) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(SIN, x_fn);
}

TEST_F(CWiseUnaryGradTest, Sin_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}});
  };
  TestCWiseGrad<complex64, complex64>(SIN, x_fn);
}

TEST_F(CWiseUnaryGradTest, Cos) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(COS, x_fn);
}

TEST_F(CWiseUnaryGradTest, Cos_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}});
  };
  TestCWiseGrad<complex64, complex64>(COS, x_fn);
}

TEST_F(CWiseUnaryGradTest, Asin) {
  auto x_fn = [this](const int i) { return RV({0, 0.25, -0.25, -0.5, 0.5}); };
  TestCWiseGrad<float, float>(ASIN, x_fn);
}

TEST_F(CWiseUnaryGradTest, Asin_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{0.5, 0}, {0, 0.5}, {0.25, -0.75}, {0.5, 0.25}});
  };
  // TODO(kbsriram)
  // Enable test when the asin kernel supports complex numbers
  if (false) {
    TestCWiseGrad<complex64, complex64>(ASIN, x_fn);
  }
}

TEST_F(CWiseUnaryGradTest, Acos) {
  auto x_fn = [this](const int i) { return RV({0, -0.5, 0.5, -0.75, 0.75}); };
  TestCWiseGrad<float, float>(ACOS, x_fn);
}

TEST_F(CWiseUnaryGradTest, Acos_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{0.5, 0}, {0, 0.5}, {0.25, -0.75}, {0.5, 0.25}});
  };
  // TODO(kbsriram)
  // Add test when the acos kernel supports complex numbers
  if (false) {
    TestCWiseGrad<complex64, complex64>(ACOS, x_fn);
  }
}

TEST_F(CWiseUnaryGradTest, Tan) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(TAN, x_fn);
}

TEST_F(CWiseUnaryGradTest, Tan_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  TestCWiseGrad<complex64, complex64>(TAN, x_fn);
}

TEST_F(CWiseUnaryGradTest, Atan) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  TestCWiseGrad<float, float>(ATAN, x_fn);
}

TEST_F(CWiseUnaryGradTest, Atan_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{1, 0}, {0, 1}, {2, -1}, {1, 2}, {3, 4}});
  };
  // TODO(kbsriram)
  // Add test when the atan kernel supports complex numbers
  if (false) {
    TestCWiseGrad<complex64, complex64>(ATAN, x_fn);
  }
}

TEST_F(CWiseUnaryGradTest, Real) {
  auto x_fn = [this](const int i) {
    return CRV({{1, -1}, {-2, 2}, {2, 3}, {-2, -3}});
  };
  TestCWiseGrad<complex64, float>(REAL, x_fn);
}

TEST_F(CWiseUnaryGradTest, Imag) {
  auto x_fn = [this](const int i) {
    return CRV({{1, -1}, {-2, 2}, {2, 3}, {-2, -3}});
  };
  TestCWiseGrad<complex64, float>(IMAG, x_fn);
}

TEST_F(CWiseUnaryGradTest, Conj) {
  auto x_fn = [this](const int i) {
    return CRV({{1, -1}, {-2, 2}, {2, 3}, {-2, -3}});
  };
  TestCWiseGrad<complex64, complex64>(CONJ, x_fn);
}

TEST_F(CWiseUnaryGradTest, Complex) {
  auto x_fn = [this](const int i) { return RV({1, -1, 2, -2, 3, -3}); };
  TestCWiseGrad<float, complex64>(COMPLEX, x_fn);
}

TEST_F(CWiseUnaryGradTest, Angle) {
  auto x_fn = [this](const int i) {
    return CRV({{1.5, 1.5}, {1.5, -1.5}, {-1.5, 1.5}, {-1.5, -1.5}});
  };
  TestCWiseGrad<complex64, float>(ANGLE, x_fn);
}

TEST_F(CWiseUnaryGradTest, Lgamma) {
  auto x_fn = [this](const int i) {
    return RV({-3.5, -2.5, -1.5, 1.0, 2.0, 3.5});
  };
  TestCWiseGrad<float, float>(LGAMMA, x_fn);
}

TEST_F(CWiseUnaryGradTest, Lgamma_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{-3.5, 0.5}, {-1.5, -0.5}, {1.5, -1.0}, {3.5, 1.0}});
  };
  // TODO(kbsriram)
  // Add test when the lgamma kernel supports complex numbers
  if (false) {
    TestCWiseGrad<complex64, complex64>(LGAMMA, x_fn);
  }
}

TEST_F(CWiseUnaryGradTest, Erf) {
  auto x_fn = [this](const int i) {
    return RV({-1.2, -1.0, -0.5, 0.3, 0.5, 1.3});
  };
  TestCWiseGrad<float, float>(ERF, x_fn);
}

TEST_F(CWiseUnaryGradTest, Erf_Complex) {
  auto x_fn = [this](const int i) {
    return CRV({{-1.2, 0.5}, {-0.5, -0.5}, {0.5, 0.5}, {1.2, -0.5}});
  };
  // TODO(kbsriram)
  // Add test when the erf kernel supports complex numbers
  if (false) {
    TestCWiseGrad<complex64, complex64>(ERF, x_fn);
  }
}

TEST_F(CWiseUnaryGradTest, Ndtri) {
  auto x_fn = [this](const int i) {
    return RV({0.1, 0.2, 0.3, 0.5, 0.7, 0.9});
  };
  TestCWiseGrad<float, float>(NDTRI, x_fn);
}

TEST_F(CWiseUnaryGradTest, Erfinv) {
  auto x_fn = [this](const int i) {
    return RV({-0.9, -0.3, -0.1, 0.2, 0.6, 0.8});
  };
  TestCWiseGrad<float, float>(ERFINV, x_fn);
}

class MathGradTest : public ::testing::Test {
 protected:
  MathGradTest() : root_(Scope::NewRootScope().WithDevice("/cpu:0")) {}

  template <typename T>
  void TestMatMulGrad(const bool is_batch, const bool t_x, const bool t_y) {
    TF_ASSERT_OK(root_.status());
    // Generate random (but compatible) shapes for matrix multiplication.
    std::vector<TensorShape> shapes;
    RandMatMulShapes(is_batch, t_x, t_y, &shapes);
    TensorShape x_shape = shapes[0];
    TensorShape y_shape = shapes[1];
    TensorShape z_shape = shapes[2];
    auto x =
        Placeholder(root_, DataTypeToEnum<T>::v(), Placeholder::Shape(x_shape));
    auto y =
        Placeholder(root_, DataTypeToEnum<T>::v(), Placeholder::Shape(y_shape));
    Output z;
    if (is_batch) {
      z = BatchMatMul(root_, x, y, BatchMatMul::AdjX(t_x).AdjY(t_y));
    } else {
      z = MatMul(root_, x, y, MatMul::TransposeA(t_x).TransposeB(t_y));
    }

    float max_error;
    TF_ASSERT_OK((ComputeGradientError<T, T, float>(
        root_, {x, y}, {x_shape, y_shape}, {z}, {z_shape}, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  void RandMatMulShapes(const bool is_batch, const bool tx, const bool ty,
                        std::vector<TensorShape>* shapes) {
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
    shapes->push_back(x_shape);

    TensorShape y_shape;
    if (is_batch) {
      // y.shape = [b, k, n]
      y_shape = ty ? TensorShape({b, n, k}) : TensorShape({b, k, n});
    } else {
      // y.shape = [k, n]
      y_shape = ty ? TensorShape({n, k}) : TensorShape({k, n});
    }
    shapes->push_back(y_shape);

    TensorShape z_shape;
    if (is_batch) {
      // z.shape = [b, m, n]
      z_shape = TensorShape({b, m, n});
    } else {
      // z.shape = [m, n]
      z_shape = TensorShape({m, n});
    }
    shapes->push_back(z_shape);
  }

  int Rand() { return 1 + (random::New64() % 10); }

  Scope root_;
};

TEST_F(MathGradTest, MatMulGrad_NoTranspose) {
  TestMatMulGrad<float>(false, false, false);
}

TEST_F(MathGradTest, MatMulComplexGrad_NoTranspose) {
  TestMatMulGrad<complex64>(false, false, false);
}

TEST_F(MathGradTest, MatMulGrad_TransposeX) {
  TestMatMulGrad<float>(false, true, false);
}

TEST_F(MathGradTest, MatMulComplexGrad_TransposeX) {
  TestMatMulGrad<complex64>(false, true, false);
}

TEST_F(MathGradTest, MatMulGrad_TransposeY) {
  TestMatMulGrad<float>(false, false, true);
}

TEST_F(MathGradTest, MatMulComplexGrad_TransposeY) {
  TestMatMulGrad<complex64>(false, false, true);
}

TEST_F(MathGradTest, MatMulGrad_TransposeX_TransposeY) {
  TestMatMulGrad<float>(false, true, true);
}

TEST_F(MathGradTest, MatMulComplexGrad_TransposeX_TransposeY) {
  TestMatMulGrad<complex64>(false, true, true);
}

TEST_F(MathGradTest, BatchMatMulGrad_NoTranspose) {
  TestMatMulGrad<float>(true, false, false);
}

TEST_F(MathGradTest, BatchMatMulComplexGrad_NoTranspose) {
  TestMatMulGrad<complex64>(true, false, false);
}

TEST_F(MathGradTest, BatchMatMulGrad_TransposeX) {
  TestMatMulGrad<float>(true, true, false);
}

TEST_F(MathGradTest, BatchMatMulComplexGrad_TransposeX) {
  TestMatMulGrad<complex64>(true, true, false);
}

TEST_F(MathGradTest, BatchMatMulGrad_TransposeY) {
  TestMatMulGrad<float>(true, false, true);
}

TEST_F(MathGradTest, BatchMatMulComplexGrad_TransposeY) {
  TestMatMulGrad<complex64>(true, false, true);
}

TEST_F(MathGradTest, BatchMatMulGrad_TransposeX_TransposeY) {
  TestMatMulGrad<float>(true, true, true);
}

TEST_F(MathGradTest, BatchMatMulComplexGrad_TransposeX_TransposeY) {
  TestMatMulGrad<complex64>(true, true, true);
}

class NaryGradTest : public ::testing::Test {
 protected:
  NaryGradTest() : scope_(Scope::NewRootScope().WithDevice("/cpu:0")) {}

  void RunTest(const OutputList& xs, const std::vector<TensorShape>& x_shapes,
               const OutputList& ys, const std::vector<TensorShape>& y_shapes) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, xs, x_shapes, ys, y_shapes, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  void RunTest(const Output& x, const Tensor& x_init_value, const Output& y,
               const TensorShape& y_shape) {
    TF_ASSERT_OK(scope_.status());
    float max_error;
    TF_ASSERT_OK((ComputeGradientError<float, float, float>(
        scope_, x, x_init_value, y, y_shape, &max_error)));
    EXPECT_LT(max_error, 1e-3);
  }

  Scope scope_;
};

TEST_F(NaryGradTest, Sum) {
  TensorShape x_shape({2, 3, 5, 7});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Sum(scope_, x, {1, -1});
  // y's shape is the result of reducing x along axes 1 and -1 (= 3)
  TensorShape y_shape({2, 5});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

TEST_F(NaryGradTest, Mean) {
  TensorShape x_shape({2, 3, 5, 7});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Mean(scope_, x, {1, -1});
  // y's shape is the result of reducing x along axes 1 and -1 (= 3)
  TensorShape y_shape({2, 5});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

TEST_F(NaryGradTest, Min) {
  TensorShape x_shape({2, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Min(scope_, x, {-1});
  // y's shape is the result of reducing x along axes -1 (= 1)
  TensorShape y_shape({2});
  Tensor x_init_value =
      test::AsTensor<float>({0.5f, 0.7f, 0.2f, 1.0f, 1.5f, -2.8f}, x_shape);
  RunTest(x, x_init_value, y, y_shape);
}

TEST_F(NaryGradTest, Max) {
  TensorShape x_shape({2, 3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Max(scope_, x, {-1});
  // y's shape is the result of reducing x along axes -1 (= 1)
  TensorShape y_shape({2});
  Tensor x_init_value =
      test::AsTensor<float>({0.5f, 0.7f, 0.2f, 1.0f, 1.5f, -2.8f}, x_shape);
  RunTest(x, x_init_value, y, y_shape);
}

TEST_F(NaryGradTest, MinMulti) {
  // Test gradient when there are multiple minima.
  // Note that we cannot directly use a test Tensor with multiple
  // minima, as the numeric estimator will calculate incorrect
  // gradients when perturbing each entry in the Tensor (which then
  // changes how many minima exist.)
  // Instead, we use a single input that broadcast-multiplies a larger
  // tensor with equal values, and apply reduce_min to the multiplied
  // result.
  TensorShape x_shape({1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto all_same = Mul(scope_, Const(scope_, {1.f, 1.f, 1.f}), x);
  auto y = Min(scope_, all_same, {0});
  // y is a [3] shaped tensor reduced along dimension 0, so it is [1] shaped
  TensorShape y_shape({1});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

TEST_F(NaryGradTest, MaxMulti) {
  TensorShape x_shape({1});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto all_same = Mul(scope_, Const(scope_, {1.f, 1.f, 1.f}), x);
  auto y = Max(scope_, all_same, {0});
  TensorShape y_shape({1});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

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

TEST_F(NaryGradTest, AddV2) {
  TensorShape x1_shape({3, 2, 5});
  TensorShape x2_shape({2, 5});
  auto x1 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x1_shape));
  auto x2 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x2_shape));
  auto y = AddV2(scope_, x1, x2);
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

TEST_F(NaryGradTest, DivNoNan) {
  {
    TensorShape x_shape({3, 2, 5});
    const auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
    // Test x / (1 + |x|) rather than x_1 / x_2 to avoid triggering large
    // division errors in the numeric estimator used by the gradient checker.
    const auto y = DivNoNan(
        scope_, x, Add(scope_, Const<float>(scope_, 1), Abs(scope_, x)));
    RunTest({x}, {x_shape}, {y}, {x_shape});
  }
  {
    // Return 0 gradient (rather than NaN) for division by zero.
    const auto x = Placeholder(scope_, DT_FLOAT);
    const auto zero = Const<float>(scope_, 0.0);
    const auto y = DivNoNan(scope_, x, zero);

    std::vector<Output> grad_outputs;
    TF_EXPECT_OK(AddSymbolicGradients(scope_, {y}, {x}, &grad_outputs));
    ClientSession session(scope_);
    std::vector<Tensor> grad_result;
    TF_EXPECT_OK(
        session.Run({{x, {-3.0f, 0.0f, 3.0f}}}, grad_outputs, &grad_result));
    EXPECT_EQ(grad_result.size(), 1);
    EXPECT_EQ(grad_result[0].NumElements(), 3);
    EXPECT_EQ(grad_result[0].flat<float>()(0), 0.0f);
    EXPECT_EQ(grad_result[0].flat<float>()(1), 0.0f);
    EXPECT_EQ(grad_result[0].flat<float>()(2), 0.0f);
  }
}

TEST_F(NaryGradTest, SquaredDifference) {
  TensorShape x1_shape({3, 2, 5});
  TensorShape x2_shape({2, 5});
  auto x1 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x1_shape));
  auto x2 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x2_shape));
  auto y = SquaredDifference(scope_, x1, x2);
  RunTest({x1, x2}, {x1_shape, x2_shape}, {y}, {x1_shape});
}

TEST_F(NaryGradTest, Pow) {
  TensorShape shape({3});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  // fix exponent to avoid overflow
  auto y = Pow(scope_, x, Const(scope_, {1.f, 2.f, 3.f}));
  RunTest({x}, {shape}, {y}, {shape});
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

TEST_F(NaryGradTest, Prod) {
  TensorShape x_shape({2, 3, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Prod(scope_, x, {1});
  // y's shape is the result of reducing x along axes 1
  TensorShape y_shape({2, 1, 2});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

TEST_F(NaryGradTest, SegmentSum) {
  TensorShape x_shape({3, 4});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = SegmentSum(scope_, x, {0, 0, 1});
  // the sum is always on the first dimension
  TensorShape y_shape({2, 4});
  RunTest({x}, {x_shape}, {y}, {y_shape});
}

class CumsumGradTest
    : public NaryGradTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool, int>> {};

TEST_P(CumsumGradTest, CumsumGrad) {
  int axis = std::get<2>(GetParam());

  TensorShape shape({2, 3, 2});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  Cumsum::Attrs attrs;
  attrs.exclusive_ = std::get<0>(GetParam());
  attrs.reverse_ = std::get<1>(GetParam());
  auto y = Cumsum(scope_, x, axis, attrs);
  RunTest({x}, {shape}, {y}, {shape});
}

INSTANTIATE_TEST_SUITE_P(CumsumGrad, CumsumGradTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Range(0, 2)));

TEST_F(NaryGradTest, CastGrad) {
  TensorShape shape({2, 3, 2});
  auto x = Placeholder(scope_, DT_DOUBLE, Placeholder::Shape(shape));
  auto y = Cast(scope_, x, DT_FLOAT);
  TF_ASSERT_OK(scope_.status());
  double max_error;
  TF_ASSERT_OK((ComputeGradientError<double, float, double>(
      scope_, {x}, {shape}, {y}, {shape}, &max_error)));
  EXPECT_LT(max_error, 1e-3);
}

TEST_F(NaryGradTest, Select) {
  TensorShape shape({1, 3});
  auto cond = Const<bool>(scope_, {{false, true, true}});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto z = Where3(scope_, cond, x, y);
  RunTest({x, y}, {shape, shape}, {z}, {shape});
}

TEST_F(NaryGradTest, SelectV2_Basic) {
  TensorShape shape({1, 3});
  auto cond = Const<bool>(scope_, {{false, true, true}});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto z = SelectV2(scope_, cond, x, y);
  RunTest({x, y}, {shape, shape}, {z}, {shape});
}

TEST_F(NaryGradTest, SelectV2_Broadcast) {
  TensorShape x_shape({2, 3});
  TensorShape y_shape({});
  auto cond = Const<bool>(scope_, {{false, true, true}, {true, true, false}});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(y_shape));
  auto z = SelectV2(scope_, cond, x, y);
  RunTest({x, y}, {x_shape, y_shape}, {z}, {x_shape});
}

TEST_F(NaryGradTest, SelectV2_Broadcast2) {
  TensorShape x_shape({2, 3});
  auto cond = Const<bool>(scope_, {{false}, {true}});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto y = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(x_shape));
  auto z = SelectV2(scope_, cond, x, y);
  RunTest({x, y}, {x_shape, x_shape}, {z}, {x_shape});
}

TEST_F(NaryGradTest, Atan2Grad) {
  TensorShape shape({3, 2, 5});
  auto x1 = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  // Test with x2 = 1 + |x1| to avoid regions where the gradient
  // is unstable and might cause problems for the numeric estimator.
  auto x2 =
      Div(scope_, x1, Add(scope_, Const<float>(scope_, 1), Abs(scope_, x1)));
  auto y = Atan2(scope_, x1, x2);
  RunTest({x1}, {shape}, {y}, {shape});
}

TEST_F(NaryGradTest, UnsortedSegmentMaxGrad) {
  TensorShape shape({3, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto segment_ids = Const(scope_, {0, 0, 1});
  auto y = UnsortedSegmentMax(scope_, x, segment_ids, /*num_segments=*/2);
  TensorShape y_shape({2, 2, 5});
  RunTest({x}, {shape}, {y}, {y_shape});
}

TEST_F(NaryGradTest, UnsortedSegmentMaxGrad_Int64Ids) {
  TensorShape shape({3, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto segment_ids = Const(scope_, {0ll, 0ll, 1ll});
  auto y = UnsortedSegmentMax(scope_, x, segment_ids, /*num_segments=*/2);
  TensorShape y_shape({2, 2, 5});
  RunTest({x}, {shape}, {y}, {y_shape});
}

TEST_F(NaryGradTest, UnsortedSegmentMaxGrad_NegativeIds) {
  TensorShape shape({3, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto segment_ids = Const(scope_, {0, 0, -1});
  auto y = UnsortedSegmentMax(scope_, x, segment_ids, /*num_segments=*/1);
  TensorShape y_shape({1, 2, 5});
  RunTest({x}, {shape}, {y}, {y_shape});
}

TEST_F(NaryGradTest, UnsortedSegmentMinGrad) {
  TensorShape shape({3, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto segment_ids = Const(scope_, {0, 0, 1});
  auto y = UnsortedSegmentMin(scope_, x, segment_ids, /*num_segments=*/2);
  TensorShape y_shape({2, 2, 5});
  RunTest({x}, {shape}, {y}, {y_shape});
}

TEST_F(NaryGradTest, UnsortedSegmentSumGrad) {
  TensorShape shape({3, 2, 5});
  auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
  auto segment_ids = Const(scope_, {0, 0, 1});
  auto y = UnsortedSegmentSum(scope_, x, segment_ids, /*num_segments=*/2);
  TensorShape y_shape({2, 2, 5});
  RunTest({x}, {shape}, {y}, {y_shape});
}

}  // namespace
}  // namespace tensorflow
