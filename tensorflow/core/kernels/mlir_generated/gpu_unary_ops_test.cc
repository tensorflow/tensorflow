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
#include "tensorflow/core/kernels/mlir_generated/gpu_ops_test_util.h"
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

  template <typename T, typename OutT>
  void SetOpKernel(const std::string& op_name, const TensorShape& shape,
                   const absl::InlinedVector<T, 10>& input, bool add_t,
                   bool add_tout) {
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
  }

  template <typename T, typename OutT>
  void RunAndExpectResult(const std::string& op_name, const TensorShape& shape,
                          const absl::InlinedVector<T, 10>& input,
                          const absl::InlinedVector<OutT, 10>& expected_output,
                          bool add_t, bool add_tout, bool expect_buffer_reuse,
                          bool expect_equal) {
    SetOpKernel<T, OutT>(op_name, shape, input, add_t, add_tout);
    TF_ASSERT_OK(RunOpKernel());

    // Assert buffer reuse if expected.
    if (expect_buffer_reuse) {
      void* arg_ptr_on_device = context_->input(0).data();
      void* result_ptr_on_device = context_->mutable_output(0)->data();
      ASSERT_EQ(arg_ptr_on_device, result_ptr_on_device);
    }

    // Assert expected results.
    Tensor expected_tensor(allocator(), DataTypeToEnum<OutT>::value, shape);
    test::FillValues<OutT>(&expected_tensor, expected_output);
    if (expect_equal) {
      test::ExpectEqual(expected_tensor, *GetOutput(0));
    } else {
      test::ExpectClose(expected_tensor, *GetOutput(0));
    }
  }

  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  void Test(const std::string op_name, const TensorShape& shape,
            absl::InlinedVector<T, 10> input,
            BaselineOutT (*baseline_callback)(BaselineT),
            bool expect_equal = true, bool add_tout = false,
            bool expect_buffer_reuse = true, bool add_t = true) {
    // Prepare inputs and compute expected results.
    auto repeated_input =
        test::RepeatInputToMatchShape(input, shape.num_elements());
    absl::InlinedVector<OutT, 10> expected_output =
        ComputeExpectedOutput<T, BaselineT, OutT, BaselineOutT>(
            repeated_input, baseline_callback);

    RunAndExpectResult<T, OutT>(op_name, shape, repeated_input, expected_output,
                                add_t, add_tout, expect_buffer_reuse,
                                expect_equal);
  }

 private:
  template <typename T, typename BaselineT, typename OutT,
            typename BaselineOutT>
  absl::InlinedVector<OutT, 10> ComputeExpectedOutput(
      absl::InlinedVector<T, 10> input,
      BaselineOutT (*baseline_callback)(BaselineT)) {
    absl::InlinedVector<OutT, 10> expected_output;
    for (int i = 0; i < input.size(); i++) {
      auto arg = static_cast<BaselineT>(input[i]);
      auto result = static_cast<OutT>(baseline_callback(arg));
      expected_output.push_back(result);
    }
    return expected_output;
  }
};

/// Test `tf.Abs`.

TEST_F(GpuUnaryOpTest, AbsFloat) {
  Test<float, float, float, float>(
      /*op_name=*/"Abs", test::DefaultInputShape(),
      test::NearZeroAndExtremeInput<float>(),
      /*baseline_callback=*/std::abs,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, AbsDouble) {
  Test<double, double, double, double>(
      /*op_name=*/"Abs", test::DefaultInputShape(),
      test::NearZeroAndExtremeInput<double>(),
      /*baseline_callback=*/std::abs,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, AbsHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Abs", test::DefaultInputShape(),
      test::NearZeroAndExtremeInput<Eigen::half>(),
      /*baseline_callback=*/std::abs,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, AbsInt32) {
  Test<int32, int32, int32, int32>(
      /*op_name=*/"Abs", test::DefaultInputShape(),
      test::NearZeroAndExtremeInput<int32>(),
      /*baseline_callback=*/std::abs,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, AbsInt64) {
  Test<int64, int64, int64, int64>(
      /*op_name=*/"Abs", test::DefaultInputShape(),
      test::NearZeroAndExtremeInput<int64>(),
      /*baseline_callback=*/std::abs,
      /*expect_equal=*/true);
}

/// Test `tf.Ceil`.

TEST_F(GpuUnaryOpTest, CeilFloat) {
  Test<float, float, float, float>(
      /*op_name=*/"Ceil", test::DefaultInputShape(),
      test::DefaultInput<float>("Ceil"),
      /*baseline_callback=*/std::ceil,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, CeilDouble) {
  Test<double, double, double, double>(
      /*op_name=*/"Ceil", test::DefaultInputShape(),
      test::DefaultInput<double>(),
      /*baseline_callback=*/std::ceil,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, CeilHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Ceil", test::DefaultInputShape(),
      test::DefaultInput<Eigen::half>(),
      /*baseline_callback=*/std::ceil,
      /*expect_equal=*/true);
}

/// Test `tf.Conj`.

TEST_F(GpuUnaryOpTest, ConjFloat) {
  Test<std::complex<float>, const std::complex<float>&, std::complex<float>,
       std::complex<float>>(/*op_name=*/"Conj", test::DefaultInputShape(),
                            test::DefaultComplexInput<float>(),
                            /*baseline_callback=*/std::conj,
                            /*expect_equal=*/false,
                            /*add_tout=*/false,
                            /*expect_buffer_reuse=*/false);
}

TEST_F(GpuUnaryOpTest, ConjDouble) {
  Test<std::complex<double>, const std::complex<double>&, std::complex<double>,
       std::complex<double>>(
      /*op_name=*/"Conj", test::DefaultInputShape(),
      test::DefaultComplexInput<double>(),
      /*baseline_callback=*/std::conj,
      /*expect_equal=*/false,
      /*add_tout=*/false,
      /*expect_buffer_reuse=*/false);
}

/// Test `tf.Cos`.

TEST_F(GpuUnaryOpTest, CosFloat) {
  Test<float, float, float, float>(
      /*op_name=*/"Cos", test::DefaultInputShape(), test::DefaultInput<float>(),
      /*baseline_callback=*/std::cos,
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, CosDouble) {
  Test<double, double, double, double>(/*op_name=*/"Cos",
                                       test::DefaultInputShape(),
                                       test::DefaultInput<double>(),
                                       /*baseline_callback=*/std::cos,
                                       /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, CosHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Cos", test::DefaultInputShape(),
      test::DefaultInput<Eigen::half>(),
      /*baseline_callback=*/std::cos,
      /*expect_equal=*/false);
}

/// Test `tf.Exp`.

TEST_F(GpuUnaryOpTest, ExpFloat) {
  Test<float, float, float, float>(/*op_name=*/"Exp", test::DefaultInputShape(),
                                   test::DefaultInput<float>(),
                                   /*baseline_callback=*/std::exp,
                                   /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, ExpDouble) {
  Test<double, double, double, double>(/*op_name=*/"Exp",
                                       test::DefaultInputShape(),
                                       test::DefaultInput<double>(),
                                       /*baseline_callback=*/std::exp,
                                       /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, ExpHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Exp", test::DefaultInputShape(),
      test::DefaultInput<Eigen::half>(),
      /*baseline_callback=*/std::exp,
      /*expect_equal=*/false);
}

/// Test `tf.Floor`.

TEST_F(GpuUnaryOpTest, FloorFloat) {
  Test<float, float, float, float>(/*op_name=*/"Floor",
                                   test::DefaultInputShape(),
                                   test::DefaultInput<float>(),
                                   /*baseline_callback=*/std::floor,
                                   /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, FloorDouble) {
  Test<double, double, double, double>(/*op_name=*/"Floor",
                                       test::DefaultInputShape(),
                                       test::DefaultInput<double>(),
                                       /*baseline_callback=*/std::floor,
                                       /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, FloorHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Floor", test::DefaultInputShape(),
      test::DefaultInput<Eigen::half>(),
      /*baseline_callback=*/std::floor,
      /*expect_equal=*/true);
}

/// Test `tf.Imag`.

TEST_F(GpuUnaryOpTest, ImagFloat) {
  Test<std::complex<float>, const std::complex<float>&, float, float>(
      /*op_name=*/"Imag", test::DefaultInputShape(),
      test::DefaultComplexInput<float>(),
      /*baseline_callback=*/std::imag,
      /*expect_equal=*/false,
      /*add_tout=*/true,
      /*expect_buffer_reuse=*/false);
}

TEST_F(GpuUnaryOpTest, ImagDouble) {
  Test<std::complex<double>, const std::complex<double>&, double, double>(
      /*op_name=*/"Imag", test::DefaultInputShape(),
      test::DefaultComplexInput<double>(),
      /*baseline_callback=*/std::imag,
      /*expect_equal=*/false,
      /*add_tout=*/true,
      /*expect_buffer_reuse=*/false);
}

/// Test `tf.IsInf`.

// TODO(b/162575339): The tests currently still fails with CUDA_ILLEGAL_ADDRESS
// when Test with unranked kernels.
TEST_F(GpuUnaryOpTest, DISABLED_IsInfFloat) {
  Test<float, float, bool, bool>(/*op_name=*/"IsInf", test::DefaultInputShape(),
                                 test::DefaultInput<float>(),
                                 /*baseline_callback=*/std::isinf,
                                 /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, DISABLED_IsInfDouble) {
  // Workaround for gcc bug, it would fail with "unresolved overloaded function
  // type" if passing std::isinf with type double. So we use type float for
  // comparing expected values.
  Test<double, float, bool, bool>(/*op_name=*/"IsInf",
                                  test::DefaultInputShape(),
                                  test::DefaultInput<double>(),
                                  /*baseline_callback=*/std::isinf,
                                  /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, DISABLED_IsInfHalf) {
  Test<Eigen::half, float, bool, bool>(/*op_name=*/"IsInf",
                                       test::DefaultInputShape(),
                                       test::DefaultInput<Eigen::half>(),
                                       /*baseline_callback=*/std::isinf,
                                       /*expect_equal=*/true);
}

/// Test `tf.Log`.

TEST_F(GpuUnaryOpTest, LogFloat) {
  Test<float, float, float, float>(/*op_name=*/"Log", test::DefaultInputShape(),
                                   test::DefaultInputGreaterThanZero<float>(),
                                   /*baseline_callback=*/std::log,
                                   /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, LogDouble) {
  Test<double, double, double, double>(
      /*op_name=*/"Log", test::DefaultInputShape(),
      test::DefaultInputGreaterThanZero<double>(),
      /*baseline_callback=*/std::log,
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, LogHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Log", test::DefaultInputShape(),
      test::DefaultInputGreaterThanZero<Eigen::half>(),
      /*baseline_callback=*/std::log,
      /*expect_equal=*/false);
}

/// Test `tf.LogicalNot`

TEST_F(GpuUnaryOpTest, LogicalNot) {
  Test<bool, bool, bool, bool>(
      /*op_name=*/"LogicalNot", test::DefaultInputShape(),
      test::DefaultInput<bool>(),
      /*baseline_callback=*/[](bool v) { return !v; },
      /*expect_equal=*/true,
      /*add_tout=*/false,
      /*expect_buffer_reuse=*/true,
      /*add_t=*/false);
}

/// Test `tf.Neg`.

/// Reference implementation.
template <typename T>
T baseline_neg(T x) {
  return -x;
}

TEST_F(GpuUnaryOpTest, NegFloat) {
  Test<float, float, float, float>(
      /*op_name=*/"Neg", test::DefaultInputShape(), test::DefaultInput<float>(),
      /*baseline_callback=*/baseline_neg,
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, NegDouble) {
  Test<double, double, double, double>(
      /*op_name=*/"Neg", test::DefaultInputShape(),
      test::DefaultInput<double>(),
      /*baseline_callback=*/baseline_neg,
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, NegHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Neg", test::DefaultInputShape(),
      test::DefaultInput<Eigen::half>(),
      /*baseline_callback=*/baseline_neg,
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, NegInt8) {
  Test<int8, int8, int8, int8>(
      /*op_name=*/"Neg", test::DefaultInputShape(), test::DefaultInput<int8>(),
      /*baseline_callback=*/baseline_neg,
      /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, NegInt16) {
  Test<int16, int16, int16, int16>(/*op_name=*/"Neg", test::DefaultInputShape(),
                                   test::DefaultInput<int16>(),
                                   /*baseline_callback=*/baseline_neg,
                                   /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, NegInt64) {
  Test<int64, int64, int64, int64>(/*op_name=*/"Neg", test::DefaultInputShape(),
                                   test::DefaultInput<int64>(),
                                   /*baseline_callback=*/baseline_neg,
                                   /*expect_equal=*/true);
}

/// Test `tf.Real`.

TEST_F(GpuUnaryOpTest, RealFloat) {
  Test<std::complex<float>, const std::complex<float>&, float, float>(
      /*op_name=*/"Real", test::DefaultInputShape(),
      test::DefaultComplexInput<float>(),
      /*baseline_callback=*/std::real,
      /*expect_equal=*/false,
      /*add_tout=*/true,
      /*expect_buffer_reuse=*/false);
}

TEST_F(GpuUnaryOpTest, RealDouble) {
  Test<std::complex<double>, const std::complex<double>&, double, double>(
      /*op_name=*/"Real", test::DefaultInputShape(),
      test::DefaultComplexInput<double>(),
      /*baseline_callback=*/std::real,
      /*expect_equal=*/false,
      /*add_tout=*/true,
      /*expect_buffer_reuse=*/false);
}

/// Test `tf.Rsqrt`.

/// Reference implementation.
template <typename T>
T baseline_rsqrt(T x) {
  return 1.0 / std::sqrt(x);
}

TEST_F(GpuUnaryOpTest, RsqrtFloat) {
  Test<float, float, float, float>(/*op_name=*/"Rsqrt",
                                   test::DefaultInputShape(),
                                   test::DefaultInputGreaterThanZero<float>(),
                                   /*baseline_callback=*/baseline_rsqrt,
                                   /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, RsqrtDouble) {
  Test<double, double, double, double>(
      /*op_name=*/"Rsqrt", test::DefaultInputShape(),
      test::DefaultInputGreaterThanZero<double>(),
      /*baseline_callback=*/baseline_rsqrt,
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, RsqrtHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Rsqrt", test::DefaultInputShape(),
      test::DefaultInputGreaterThanZero<Eigen::half>(),
      /*baseline_callback=*/baseline_rsqrt,
      /*expect_equal=*/false);
}

/// Test `tf.Sign`.

// Reference implementation
template <typename T>
T baseline_sign(T x) {
  if (x == 0) return 0;
  if (x < 0) return -1;
  return 1;
}

TEST_F(GpuUnaryOpTest, SignFloat) {
  Test<float, float, float, float>(/*op_name=*/"Sign",
                                   test::DefaultInputShape(),
                                   test::DefaultInput<float>(),
                                   /*baseline_callback=*/baseline_sign,
                                   /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, SignDouble) {
  Test<double, double, double, double>(/*op_name=*/"Sign",
                                       test::DefaultInputShape(),
                                       test::DefaultInput<double>(),
                                       /*baseline_callback=*/baseline_sign,
                                       /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, SignHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Sign", test::DefaultInputShape(),
      test::DefaultInput<Eigen::half>(),
      /*expected_callback=*/baseline_sign,
      // TODO(b/162577610): We should actually use true
      // here. This requires returning 0.0 for input -0.0.
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SignInt64) {
  Test<int64, int64, int64, int64>(
      /*op_name=*/"Sign", test::DefaultInputShape(),
      test::DefaultInput<int64>(),
      /*expected_callback=*/baseline_sign,
      /*expect_equal=*/true);
}

/// Test `tf.Sin`.

TEST_F(GpuUnaryOpTest, SinFloat) {
  Test<float, float, float, float>(/*op_name=*/"Sin", test::DefaultInputShape(),
                                   test::DefaultInput<float>(),
                                   /*baseline_callback=*/std::sin,
                                   /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SinDouble) {
  Test<double, double, double, double>(/*op_name=*/"Sin",
                                       test::DefaultInputShape(),
                                       test::DefaultInput<double>(),
                                       /*baseline_callback=*/std::sin,
                                       /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SinHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Sin", test::DefaultInputShape(),
      test::DefaultInput<Eigen::half>(),
      /*baseline_callback=*/std::sin,
      /*expect_equal=*/false);
}

/// Test `tf.Sqrt`.

TEST_F(GpuUnaryOpTest, SqrtFloat) {
  Test<float, float, float, float>(
      /*op_name=*/"Sqrt", test::DefaultInputShape(),
      test::DefaultInputGreaterOrEqualToZero<float>(),
      /*baseline_callback=*/std::sqrt,
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SqrtDouble) {
  Test<double, double, double, double>(
      /*op_name=*/"Sqrt", test::DefaultInputShape(),
      test::DefaultInputGreaterOrEqualToZero<double>(),
      /*baseline_callback=*/std::sqrt,
      /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, SqrtHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Sqrt", test::DefaultInputShape(),
      test::DefaultInputGreaterOrEqualToZero<Eigen::half>(),
      /*baseline_callback=*/std::sqrt,
      /*expect_equal=*/false);
}

/// Test `tf.Tanh`.

TEST_F(GpuUnaryOpTest, TanhFloat) {
  Test<float, float, float, float>(/*op_name=*/"Tanh",
                                   test::DefaultInputShape(),
                                   test::DefaultInput<float>(),
                                   /*baseline_callback=*/std::tanh,
                                   /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, TanhDouble) {
  Test<double, double, double, double>(/*op_name=*/"Tanh",
                                       test::DefaultInputShape(),
                                       test::DefaultInput<double>(),
                                       /*baseline_callback=*/std::tanh,
                                       /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, TanhHalf) {
  Test<Eigen::half, float, Eigen::half, float>(
      /*op_name=*/"Tanh", test::DefaultInputShape(),
      test::DefaultInput<Eigen::half>(),
      /*baseline_callback=*/std::tanh,
      /*expect_equal=*/false);
}

}  // namespace
}  // end namespace tensorflow
