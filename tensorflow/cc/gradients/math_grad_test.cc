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
    TANH,
    SIGMOID,
    SIGN,
    SIN,
    COS,
    ASIN,
    ACOS,
    TAN,
    ATAN
  };

  void TestCWiseGrad(UnaryOpType op_type, std::function<float(int)> x_fn,
                     std::function<float(float)> dy_fn,
                     std::function<float(float, float)> dx_fn) {
    Tensor x(DT_FLOAT, {2, 3, 2});
    auto x_flat = x.flat<float>();
    for (int i = 0; i < x_flat.size(); ++i) {
      x_flat(i) = x_fn(i);
    }

    Tensor dy(DT_FLOAT, {2, 3, 2});
    auto dy_flat = dy.flat<float>();
    for (int i = 0; i < dy_flat.size(); ++i) {
      dy_flat(i) = dy_fn(x_flat(i));
    }

    Tensor dx(DT_FLOAT, {2, 3, 2});
    auto dx_flat = dx.flat<float>();
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
      case TANH:
        y = Tanh(scope_, x);
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

  float RV(std::vector<float> v) { return v[random::New64() % v.size()]; }

  Scope scope_;
};

TEST_F(CWiseUnaryGradTest, Abs) {
  auto x_fn = [this](const int i) { return RV({-1, 0, 1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) { return x * dy; };
  TestCWiseGrad(ABS, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Neg) {
  auto x_fn = [this](const int i) { return RV({-1, 0, 1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) { return -dy; };
  TestCWiseGrad(NEG, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Reciprocal) {
  auto x_fn = [this](const int i) { return RV({-1, 1, -2, 2, -3, 3, -4, 4}); };
  auto dy_fn = [this](const float x) { return RV({0, -2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return -(1 / (x * x)) * dy;
  };
  TestCWiseGrad(INV, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Square) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return RV({0, -7, 7, -8, 8, -9, 9}); };
  auto dx_fn = [this](const float x, const float dy) { return 2 * x * dy; };
  TestCWiseGrad(SQUARE, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sqrt) {
  auto x_fn = [this](const int i) { return RV({0, 1, 2, 3, 4, 5, 6, 7}); };
  auto dy_fn = [this](const float x) { return x + RV({8, 9, 10, 11, 12, 13}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * 0.5 * (1.0 / std::sqrt(x));
  };
  TestCWiseGrad(SQRT, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Rsqrt) {
  auto x_fn = [this](const int i) { return RV({1, 2, 3, 4, 5, 6, 7, 8}); };
  auto dy_fn = [this](const float x) { return x + RV({8, 9, 10, 11, 12, 13}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * -0.5 * (1 / std::sqrt(x)) * (1 / x);
  };
  TestCWiseGrad(RSQRT, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Exp) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * std::exp(x);
  };
  TestCWiseGrad(EXP, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Expm1) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1e-6, 1, -2, 3, 100}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * std::exp(x);
  };
  TestCWiseGrad(EXPM1, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Log) {
  auto x_fn = [this](const int i) { return RV({-1, 1, -2, 2, -3, 3, -4, 4}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) { return dy * (1.0 / x); };
  TestCWiseGrad(LOG, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Log1p) {
  auto x_fn = [this](const int i) { return RV({0, 1e-6, 1, 2, 3, 4, 100}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (1.0 / (1.0 + x));
  };
  TestCWiseGrad(LOG1P, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Tanh) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    const float y = std::tanh(x);
    return dy * (1.0 - y * y);
  };
  TestCWiseGrad(TANH, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sigmoid) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    const float y = 1.0 / (1.0 + std::exp(-x));
    return dy * y * (1.0 - y);
  };
  TestCWiseGrad(SIGMOID, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sign) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) { return 0.0; };
  TestCWiseGrad(SIGN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Sin) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * std::cos(x);
  };
  TestCWiseGrad(SIN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Cos) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * -1.0 * std::sin(x);
  };
  TestCWiseGrad(COS, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Asin) {
  auto x_fn = [this](const int i) { return RV({0, -0.5, 0.5, -1, 1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (1.0 / std::sqrt(1.0 - x * x));
  };
  TestCWiseGrad(ASIN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Acos) {
  auto x_fn = [this](const int i) { return RV({0, -0.5, 0.5, -1, 1}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (-1.0 / std::sqrt(1.0 - x * x));
  };
  TestCWiseGrad(ACOS, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Tan) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    const float cosx = std::cos(x);
    return dy * (1 / (cosx * cosx));
  };
  TestCWiseGrad(TAN, x_fn, dy_fn, dx_fn);
}

TEST_F(CWiseUnaryGradTest, Atan) {
  auto x_fn = [this](const int i) { return RV({0, -1, 1, -2, 2, -3, 3}); };
  auto dy_fn = [this](const float x) { return x + RV({-2, 2, -3, 3, -4, 4}); };
  auto dx_fn = [this](const float x, const float dy) {
    return dy * (1 / (1 + x * x));
  };
  TestCWiseGrad(ATAN, x_fn, dy_fn, dx_fn);
}

class CWiseUnaryComplexGradTest : public ::testing::Test {
 protected:
  CWiseUnaryComplexGradTest()
      : scope_(Scope::NewRootScope().WithDevice("/cpu:0")) {}

  enum UnaryOpType { REAL, IMAG, CONJ };

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

}  // namespace
}  // namespace tensorflow
