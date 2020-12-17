/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include <complex>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GpuUnaryOpTest : public OpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }

  // 'T' is the input type, 'RT' is the input type for the callback function,
  // 'OutT' is the output type, 'ROutT' is the output type for the callback
  // function. In most cases it is enough to just provide the input type,
  // because all the types are the same.
  template <typename T, typename RT = T, typename OutT = T, typename ROutT = RT>
  void Run(std::vector<int64> input_shape, absl::InlinedVector<T, 10> input,
           const std::string op_name, ROutT (*expected_callback)(RT),
           bool expect_equal = true, bool add_tout = false,
           bool expect_buffer_reuse = true, bool add_t = true) {
    assert(std::accumulate(input_shape.begin(), input_shape.end(), 1,
                           std::multiplies<int64>()) == input.size() &&
           "Expected input length to equal to shape's number of elements.");

    TensorShape shape(input_shape);
    NodeDefBuilder builder("some_name", op_name);
    builder.Input(FakeInput(DataTypeToEnum<T>::v()));
    if (add_t) {
      builder.Attr("T", DataTypeToEnum<T>::v());
    }
    if (add_tout) {
      builder.Attr("Tout", DataTypeToEnum<OutT>::v());
    }
    TF_ASSERT_OK(builder.Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
    AddInputFromArray<T>(shape, input);
    TF_ASSERT_OK(RunOpKernel());

    // Assert buffer reuse if expected.
    if (expect_buffer_reuse) {
      void* arg_ptr_on_device = context_->input(0).data();
      void* result_ptr_on_device = context_->mutable_output(0)->data();
      ASSERT_EQ(arg_ptr_on_device, result_ptr_on_device);
    }

    // Assert expected results.
    Tensor expected_tensor(allocator(), DataTypeToEnum<OutT>::value, shape);
    absl::InlinedVector<OutT, 14> expected;
    expected.reserve(input.size());
    for (const T& inp : input) {
      expected.push_back(
          static_cast<OutT>(expected_callback(static_cast<RT>(inp))));
    }
    test::FillValues<OutT>(&expected_tensor, expected);
    if (expect_equal) {
      test::ExpectEqual(expected_tensor, *GetOutput(0));
    } else {
      test::ExpectClose(expected_tensor, *GetOutput(0));
    }
  }

  // Some helper functions to get default input values.

  std::vector<int64> DefaultInputShape() { return std::vector<int64>{2, 7}; }

  template <typename T>
  absl::InlinedVector<T, 10> DefaultInput() {
    return InputAsVector<T>({-18.0, -9.0, -1e-6, -0.0, 0.0, 1e-6, 0.1, 0.2, 0.3,
                             0.5, 0.7, 0.9, 9.0, 18.0});
  }

  template <typename T>
  absl::InlinedVector<std::complex<T>, 10> DefaultComplexInput() {
    auto input = DefaultInput<T>();
    absl::InlinedVector<std::complex<T>, 10> complex_input;
    for (T value : input) {
      complex_input.emplace_back(value, -value);
    }
    return complex_input;
  }

  template <typename T>
  absl::InlinedVector<T, 10> DefaultInputGreaterThanZero() {
    return InputAsVector<T>({18.0, 9.0, 1e-6, 1.0, 0.1, 1e-6, 0.1, 0.2, 0.3,
                             0.5, 0.7, 0.9, 9.0, 18.0});
  }

  template <typename T>
  absl::InlinedVector<T, 10> DefaultInputGreaterOrEqualToZero() {
    return InputAsVector<T>({18.0, 9.0, 1e-6, 0.0, 0.1, 1e-6, 0.1, 0.2, 0.3,
                             0.5, 0.7, 0.9, 9.0, 18.0});
  }

 private:
  template <typename T>
  absl::InlinedVector<T, 10> InputAsVector(
      std::initializer_list<double> input) {
    absl::InlinedVector<T, 10> result;
    result.reserve(input.size());
    for (const auto& value : input) {
      result.push_back(static_cast<T>(value));
    }
    return result;
  }
};

/// Test `tf.Abs`.

TEST_F(GpuUnaryOpTest, AbsFloat) {
  Run<float>(
      /*input_shape=*/{2, 3},
      /*input=*/
      {-std::numeric_limits<float>::infinity(), -0.1f, -0.0f, 0.0f, 0.1f,
       std::numeric_limits<float>::infinity()},
      /*op_name=*/"Abs",
      /*expected_callback=*/std::abs,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, AbsDouble) {
  Run<double>(
      /*input_shape=*/{2, 3},
      /*input=*/
      {-std::numeric_limits<double>::infinity(), -0.1, -0.0, 0.0, 0.1,
       std::numeric_limits<double>::infinity()},
      /*op_name=*/"Abs",
      /*expected_callback=*/std::abs,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, AbsHalf) {
  Run<Eigen::half, float>(
      /*input_shape=*/{2, 3},
      /*input=*/
      {static_cast<Eigen::half>(-std::numeric_limits<double>::infinity()),
       static_cast<Eigen::half>(-0.1), static_cast<Eigen::half>(-0.0),
       static_cast<Eigen::half>(0.0), static_cast<Eigen::half>(0.1),
       static_cast<Eigen::half>(std::numeric_limits<double>::infinity())},
      /*op_name=*/"Abs",
      /*expected_callback=*/std::abs,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, AbsInt32) {
  Run<int32>(
      /*input_shape=*/{2, 3},
      /*input=*/
      {std::numeric_limits<int32>::min(), std::numeric_limits<int32>::min() + 1,
       -1, 0, 1, std::numeric_limits<int32>::max()},
      /*op_name=*/"Abs",
      /*expected_callback=*/std::abs,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, AbsInt64) {
  Run<int64>(
      /*input_shape=*/{2, 3},
      /*input=*/
      {std::numeric_limits<int64>::min(), std::numeric_limits<int64>::min() + 1,
       -1, 0, 1, std::numeric_limits<int64>::max()},
      /*op_name=*/"Abs",
      /*expected_callback=*/std::abs,
      /*expect_equal=*/true);
}

/// Test `tf.Ceil`.

TEST_F(GpuUnaryOpTest, CeilFloat) {
  Run<float>(DefaultInputShape(), DefaultInput<float>(),
             /*op_name=*/"Ceil",
             /*expected_callback=*/std::ceil,
             /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, CeilDouble) {
  Run<double>(DefaultInputShape(), DefaultInput<double>(),
              /*op_name=*/"Ceil",
              /*expected_callback=*/std::ceil,
              /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, CeilHalf) {
  Run<Eigen::half, float>(DefaultInputShape(), DefaultInput<Eigen::half>(),
                          /*op_name=*/"Ceil",
                          /*expected_callback=*/std::ceil,
                          /*expect_equal=*/true);
}

/// Test `tf.Conj`.

TEST_F(GpuUnaryOpTest, ConjFloat) {
  Run<std::complex<float>, const std::complex<float>&, std::complex<float>,
      std::complex<float>>(DefaultInputShape(), DefaultComplexInput<float>(),
                           /*op_name=*/"Conj",
                           /*expected_callback=*/std::conj,
                           /*expect_equal=*/false,
                           /*add_tout=*/false,
                           /*expect_buffer_reuse=*/false);
}

TEST_F(GpuUnaryOpTest, ConjDouble) {
  Run<std::complex<double>, const std::complex<double>&, std::complex<double>,
      std::complex<double>>(DefaultInputShape(), DefaultComplexInput<double>(),
                            /*op_name=*/"Conj",
                            /*expected_callback=*/std::conj,
                            /*expect_equal=*/false,
                            /*add_tout=*/false,
                            /*expect_buffer_reuse=*/false);
}

/// Test `tf.Cos`.

TEST_F(GpuUnaryOpTest, CosFloat) {
  Run<float>(DefaultInputShape(), DefaultInput<float>(),
             /*op_name=*/"Cos",
             /*expected_callback=*/std::cos,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, CosDouble) {
  Run<double>(DefaultInputShape(), DefaultInput<double>(),
              /*op_name=*/"Cos",
              /*expected_callback=*/std::cos,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, CosHalf) {
  Run<Eigen::half, float>(DefaultInputShape(), DefaultInput<Eigen::half>(),
                          /*op_name=*/"Cos",
                          /*expected_callback=*/std::cos,
                          /*expect_equal=*/false);
}

/// Test `tf.Exp`.

TEST_F(GpuUnaryOpTest, ExpFloat) {
  Run<float>(DefaultInputShape(), DefaultInput<float>(),
             /*op_name=*/"Exp",
             /*expected_callback=*/std::exp,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, ExpDouble) {
  Run<double>(DefaultInputShape(), DefaultInput<double>(),
              /*op_name=*/"Exp",
              /*expected_callback=*/std::exp,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, ExpHalf) {
  Run<Eigen::half, float>(DefaultInputShape(), DefaultInput<Eigen::half>(),
                          /*op_name=*/"Exp",
                          /*expected_callback=*/std::exp,
                          /*expect_equal=*/false);
}

/// Test `tf.Floor`.

TEST_F(GpuUnaryOpTest, FloorFloat) {
  Run<float>(DefaultInputShape(), DefaultInput<float>(),
             /*op_name=*/"Floor",
             /*expected_callback=*/std::floor,
             /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, FloorDouble) {
  Run<double>(DefaultInputShape(), DefaultInput<double>(),
              /*op_name=*/"Floor",
              /*expected_callback=*/std::floor,
              /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, FloorHalf) {
  Run<Eigen::half, float>(DefaultInputShape(), DefaultInput<Eigen::half>(),
                          /*op_name=*/"Floor",
                          /*expected_callback=*/std::floor,
                          /*expect_equal=*/true);
}

/// Test `tf.Imag`.

TEST_F(GpuUnaryOpTest, ImagFloat) {
  Run<std::complex<float>, const std::complex<float>&, float, float>(
      DefaultInputShape(), DefaultComplexInput<float>(),
      /*op_name=*/"Imag",
      /*expected_callback=*/std::imag,
      /*expect_equal=*/false,
      /*add_tout=*/true,
      /*expect_buffer_reuse=*/false);
}

TEST_F(GpuUnaryOpTest, ImagDouble) {
  Run<std::complex<double>, const std::complex<double>&, double, double>(
      DefaultInputShape(), DefaultComplexInput<double>(),
      /*op_name=*/"Imag",
      /*expected_callback=*/std::imag,
      /*expect_equal=*/false,
      /*add_tout=*/true,
      /*expect_buffer_reuse=*/false);
}

/// Test `tf.IsInf`.

// TODO(b/162575339): The tests currently still fails with CUDA_ILLEGAL_ADDRESS
// when run with unranked kernels.
TEST_F(GpuUnaryOpTest, DISABLED_IsInfFloat) {
  Run<float, float, bool, bool>(DefaultInputShape(), DefaultInput<float>(),
                                /*op_name=*/"IsInf",
                                /*expected_callback=*/std::isinf,
                                /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, DISABLED_IsInfDouble) {
  // Workaround for gcc bug, it would fail with "unresolved overloaded function
  // type" if passing std::isinf with type double. So we use type float for
  // comparing expected values.
  Run<double, float, bool, bool>(DefaultInputShape(), DefaultInput<double>(),
                                 /*op_name=*/"IsInf",
                                 /*expected_callback=*/std::isinf,
                                 /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, DISABLED_IsInfHalf) {
  Run<Eigen::half, float, bool, bool>(DefaultInputShape(),
                                      DefaultInput<Eigen::half>(),
                                      /*op_name=*/"IsInf",
                                      /*expected_callback=*/std::isinf,
                                      /*expect_equal=*/true);
}

/// Test `tf.Log`.

TEST_F(GpuUnaryOpTest, LogFloat) {
  Run<float>(DefaultInputShape(), DefaultInputGreaterThanZero<float>(),
             /*op_name=*/"Log",
             /*expected_callback=*/std::log,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, LogDouble) {
  Run<double>(DefaultInputShape(), DefaultInputGreaterThanZero<double>(),
              /*op_name=*/"Log",
              /*expected_callback=*/std::log,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, LogHalf) {
  Run<Eigen::half, float>(DefaultInputShape(),
                          /*input=*/
                          DefaultInputGreaterThanZero<Eigen::half>(),
                          /*op_name=*/"Log",
                          /*expected_callback=*/std::log,
                          /*expect_equal=*/false);
}

/// Test `tf.LogicalNot`

TEST_F(GpuUnaryOpTest, LogicalNot) {
  Run<bool, bool, bool, bool>(
      DefaultInputShape(), DefaultInput<bool>(),
      /*op_name=*/"LogicalNot",
      /*expected_callback=*/[](bool v) { return !v; },
      /*expect_equal=*/true,
      /*add_tout=*/false,
      /*expect_buffer_reuse=*/true,
      /*add_t=*/false);
}

/// Test `tf.Neg`.

/// Reference implementation.
template <typename T>
T expected_neg(T x) {
  return -x;
}

TEST_F(GpuUnaryOpTest, NegFloat) {
  Run<float>(DefaultInputShape(), DefaultInput<float>(),
             /*op_name=*/"Neg",
             /*expected_callback=*/expected_neg,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, NegDouble) {
  Run<double>(DefaultInputShape(), DefaultInput<double>(),
              /*op_name=*/"Neg",
              /*expected_callback=*/expected_neg,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, NegHalf) {
  Run<Eigen::half, float>(DefaultInputShape(), DefaultInput<Eigen::half>(),
                          /*op_name=*/"Neg",
                          /*expected_callback=*/expected_neg,
                          /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, NegInt8) {
  Run<int8>(DefaultInputShape(), DefaultInput<int8>(),
            /*op_name=*/"Neg",
            /*expected_callback=*/expected_neg,
            /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, NegInt16) {
  Run<int16>(DefaultInputShape(), DefaultInput<int16>(),
             /*op_name=*/"Neg",
             /*expected_callback=*/expected_neg,
             /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, NegInt64) {
  Run<int64>(DefaultInputShape(), DefaultInput<int64>(),
             /*op_name=*/"Neg",
             /*expected_callback=*/expected_neg,
             /*expect_equal=*/true);
}

/// Test `tf.Real`.

TEST_F(GpuUnaryOpTest, RealFloat) {
  Run<std::complex<float>, const std::complex<float>&, float, float>(
      DefaultInputShape(), DefaultComplexInput<float>(),
      /*op_name=*/"Real",
      /*expected_callback=*/std::real,
      /*expect_equal=*/false,
      /*add_tout=*/true,
      /*expect_buffer_reuse=*/false);
}

TEST_F(GpuUnaryOpTest, RealDouble) {
  Run<std::complex<double>, const std::complex<double>&, double, double>(
      DefaultInputShape(), DefaultComplexInput<double>(),
      /*op_name=*/"Real",
      /*expected_callback=*/std::real,
      /*expect_equal=*/false,
      /*add_tout=*/true,
      /*expect_buffer_reuse=*/false);
}

/// Test `tf.Rsqrt`.

/// Reference implementation.
template <typename T>
T expected_rsqrt(T x) {
  return 1.0 / std::sqrt(x);
}

TEST_F(GpuUnaryOpTest, RsqrtFloat) {
  Run<float>(DefaultInputShape(), DefaultInputGreaterThanZero<float>(),
             /*op_name=*/"Rsqrt",
             /*expected_callback=*/expected_rsqrt,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, RsqrtDouble) {
  Run<double>(DefaultInputShape(), DefaultInputGreaterThanZero<double>(),
              /*op_name=*/"Rsqrt",
              /*expected_callback=*/expected_rsqrt,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, RsqrtHalf) {
  Run<Eigen::half, float>(DefaultInputShape(),
                          /*input=*/
                          DefaultInputGreaterThanZero<Eigen::half>(),
                          /*op_name=*/"Rsqrt",
                          /*expected_callback=*/expected_rsqrt,
                          /*expect_equal=*/false);
}

/// Test `tf.Sign`.

// Reference implementation
template <typename T>
T expected_sign(T x) {
  if (x == 0) return 0;
  if (x < 0) return -1;
  return 1;
}

TEST_F(GpuUnaryOpTest, SignFloat) {
  Run<float>(DefaultInputShape(), DefaultInput<float>(),
             /*op_name=*/"Sign",
             /*expected_callback=*/expected_sign,
             /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, SignDouble) {
  Run<double>(DefaultInputShape(), DefaultInput<double>(),
              /*op_name=*/"Sign",
              /*expected_callback=*/expected_sign,
              /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, SignHalf) {
  Run<Eigen::half, float>(DefaultInputShape(), DefaultInput<Eigen::half>(),
                          /*op_name=*/"Sign",
                          /*expected_callback=*/expected_sign,
                          // TODO(b/162577610): We should actually use true
                          // here. This requires returning 0.0 for input -0.0.
                          /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SignInt64) {
  Run<int64>(DefaultInputShape(), DefaultInput<int64>(),
             /*op_name=*/"Sign",
             /*expected_callback=*/expected_sign,
             /*expect_equal=*/true);
}

/// Test `tf.Sin`.

TEST_F(GpuUnaryOpTest, SinFloat) {
  Run<float>(DefaultInputShape(), DefaultInput<float>(),
             /*op_name=*/"Sin",
             /*expected_callback=*/std::sin,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SinDouble) {
  Run<double>(DefaultInputShape(), DefaultInput<double>(),
              /*op_name=*/"Sin",
              /*expected_callback=*/std::sin,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SinHalf) {
  Run<Eigen::half, float>(DefaultInputShape(), DefaultInput<Eigen::half>(),
                          /*op_name=*/"Sin",
                          /*expected_callback=*/std::sin,
                          /*expect_equal=*/false);
}

/// Test `tf.Sqrt`.

TEST_F(GpuUnaryOpTest, SqrtFloat) {
  Run<float>(DefaultInputShape(), DefaultInputGreaterOrEqualToZero<float>(),
             /*op_name=*/"Sqrt",
             /*expected_callback=*/std::sqrt,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SqrtDouble) {
  Run<double>(DefaultInputShape(), DefaultInputGreaterOrEqualToZero<double>(),
              /*op_name=*/"Sqrt",
              /*expected_callback=*/std::sqrt,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SqrtHalf) {
  Run<Eigen::half, float>(DefaultInputShape(),
                          DefaultInputGreaterOrEqualToZero<Eigen::half>(),
                          /*op_name=*/"Sqrt",
                          /*expected_callback=*/std::sqrt,
                          /*expect_equal=*/false);
}

/// Test `tf.Tanh`.

TEST_F(GpuUnaryOpTest, TanhFloat) {
  Run<float>(DefaultInputShape(), DefaultInput<float>(),
             /*op_name=*/"Tanh",
             /*expected_callback=*/std::tanh,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, TanhDouble) {
  Run<double>(DefaultInputShape(), DefaultInput<double>(),
              /*op_name=*/"Tanh",
              /*expected_callback=*/std::tanh,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, TanhHalf) {
  Run<Eigen::half, float>(DefaultInputShape(), DefaultInput<Eigen::half>(),
                          /*op_name=*/"Tanh",
                          /*expected_callback=*/std::tanh,
                          /*expect_equal=*/false);
}

}  // namespace
}  // end namespace tensorflow
