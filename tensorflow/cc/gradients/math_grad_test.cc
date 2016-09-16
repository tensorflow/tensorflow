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
