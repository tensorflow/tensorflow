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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class DequantizeOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void ComputeDequantizeMinCombinedUsingEigen(const Tensor& input,
                                              float min_range, float max_range,
                                              Tensor* output) {
    float half_range =
        !std::is_signed<T>::value
            ? 0.0f
            : (static_cast<float>(std::numeric_limits<T>::max()) -
               std::numeric_limits<T>::min() + 1) /
                  2.0f;
    const float scale_factor =
        (max_range - min_range) /
        (static_cast<float>(std::numeric_limits<T>::max()) -
         std::numeric_limits<T>::min());
    output->flat<float>() =
        ((input.flat<T>().template cast<int>().template cast<float>() +
          half_range) *
         scale_factor) +
        min_range;
  }

  // Compares dequantize min vs the same using eigen. This tests that a change
  // to not use eigen gives equivalent results to using eigen.
  template <typename T>
  void RunDequantizeMinCombinedTest(float min_range, float max_range) {
    TF_ASSERT_OK(NodeDefBuilder("dequantize_op", "Dequantize")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("mode", "MIN_COMBINED")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    std::vector<T> input;
    for (int64 i = std::numeric_limits<T>::min();
         i < std::numeric_limits<T>::max(); ++i) {
      input.push_back(static_cast<T>(i));
    }
    TensorShape shape({static_cast<int64>(input.size())});
    AddInputFromArray<T>(shape, input);
    AddInputFromArray<float>(TensorShape({}), {min_range});
    AddInputFromArray<float>(TensorShape({}), {max_range});
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected(allocator(), DT_FLOAT, shape);
    ComputeDequantizeMinCombinedUsingEigen<T>(GetInput(0), min_range, max_range,
                                              &expected);
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  }

  template <typename T>
  void RunDequantizeScaledTest(float min_range, float max_range, int input_int,
                               float expected_output) {
    TF_ASSERT_OK(NodeDefBuilder("dequantize_op", "Dequantize")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("mode", "SCALED")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    std::vector<T> input;
    input.push_back(static_cast<T>(input_int));
    TensorShape shape({static_cast<int64>(input.size())});
    AddInputFromArray<T>(shape, input);
    AddInputFromArray<float>(TensorShape({}), {min_range});
    AddInputFromArray<float>(TensorShape({}), {max_range});
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected(allocator(), DT_FLOAT, shape);
    test::FillValues<float>(&expected, {expected_output});
    test::ExpectClose(expected, *GetOutput(0));
  }
};

TEST_F(DequantizeOpTest, DequantizeMinCombinedQuint8) {
  RunDequantizeMinCombinedTest<quint8>(0, 255.0f);
}
TEST_F(DequantizeOpTest, DequantizeMinCombinedQint8) {
  RunDequantizeMinCombinedTest<qint8>(0, 255.0f);
}
TEST_F(DequantizeOpTest, DequantizeMinCombinedQint16) {
  RunDequantizeMinCombinedTest<qint16>(0, 255.0f);
}
TEST_F(DequantizeOpTest, DequantizeMinCombinedQuint16) {
  RunDequantizeMinCombinedTest<quint16>(0, 255.0f);
}

TEST_F(DequantizeOpTest, DequantizeScaledQuint8Zero) {
  RunDequantizeScaledTest<quint8>(-255.0f, 127.0f, 0, 0.0);
}
TEST_F(DequantizeOpTest, DequantizeScaledQuint8CheckIgnoresNegative) {
  RunDequantizeScaledTest<quint8>(-512.0f, 255.0f, 255, 255.0);
}
TEST_F(DequantizeOpTest, DequantizeScaledQuint8ScaleDown) {
  RunDequantizeScaledTest<quint8>(-1.0f, 2.0f, 255, 2.0);
}
TEST_F(DequantizeOpTest, DequantizeScaledQuint8ScaleUp) {
  RunDequantizeScaledTest<quint8>(200.0f, 400.0f, 255, 400.0);
}

TEST_F(DequantizeOpTest, DequantizeScaledQint8Zero) {
  RunDequantizeScaledTest<qint8>(-255.0f, 127.0f, 0, 0.0);
}
TEST_F(DequantizeOpTest, DequantizeScaledQint8ScaleIdentity) {
  RunDequantizeScaledTest<qint8>(-10.0f, 127.0f, -127, -127.0);
}
TEST_F(DequantizeOpTest, DequantizeScaledQint8ScaleDown) {
  RunDequantizeScaledTest<qint8>(-2.0f, 1.0f, -128, -2.0);
}
TEST_F(DequantizeOpTest, DequantizeScaledQint8ScaleUp) {
  RunDequantizeScaledTest<qint8>(-1.0f, 300.0f, 42, 99.212601);
}

template <typename T>
static void BM_DequantizeMinCombinedCpu(int iters) {
  auto root = Scope::NewRootScope().ExitOnError();
  const int64 num_values = 1500 * 250;
  std::vector<T> inputs;
  inputs.reserve(num_values);
  for (int i = 0; i < num_values; ++i) inputs.push_back(i);
  ops::Dequantize(root, test::AsTensor<T>(inputs), test::AsScalar<float>(-1.5f),
                  test::AsScalar<float>(20.5f),
                  ops::Dequantize::Attrs().Mode("MIN_COMBINED"));
  TF_CHECK_OK(root.status());
  Graph* g = new Graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(g));

  test::Benchmark("cpu", g).Run(iters);
  testing::BytesProcessed(iters * num_values * (sizeof(float) + sizeof(T)));
  testing::ItemsProcessed(iters);
}

static void BM_DequantizeMinCombinedCpuQuint16(int iters) {
  BM_DequantizeMinCombinedCpu<quint16>(iters);
}

static void BM_DequantizeMinCombinedCpuQint16(int iters) {
  BM_DequantizeMinCombinedCpu<qint16>(iters);
}

static void BM_DequantizeMinCombinedCpuQuint8(int iters) {
  BM_DequantizeMinCombinedCpu<quint8>(iters);
}

static void BM_DequantizeMinCombinedCpuQint8(int iters) {
  BM_DequantizeMinCombinedCpu<qint8>(iters);
}

BENCHMARK(BM_DequantizeMinCombinedCpuQuint16);
BENCHMARK(BM_DequantizeMinCombinedCpuQint16);
BENCHMARK(BM_DequantizeMinCombinedCpuQuint8);
BENCHMARK(BM_DequantizeMinCombinedCpuQint8);

}  // namespace
}  // namespace tensorflow
