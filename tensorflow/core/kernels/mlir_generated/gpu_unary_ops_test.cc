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
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

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

  template <typename T, typename RT = T>
  void Run(std::initializer_list<int64> input_shape,
           std::initializer_list<T> input, const std::string op_name,
           RT (*expected_callback)(RT), bool expect_equal = true) {
    assert(std::accumulate(input_shape.begin(), input_shape.end(), 1,
                           std::multiplies<int64>()) == input.size() &&
           "Expected input length to equal to shape's number of elements.");

    TensorShape shape(input_shape);
    TF_ASSERT_OK(NodeDefBuilder("some_name", op_name)
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
    AddInputFromArray<T>(shape, input);
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected_tensor(allocator(), DataTypeToEnum<T>::value, shape);
    std::vector<T> expected;
    expected.reserve(input.size());
    for (const T& inp : input) {
      expected.push_back(
          static_cast<T>(expected_callback(static_cast<RT>(inp))));
    }
    test::FillValues<T>(&expected_tensor, expected);
    if (expect_equal) {
      test::ExpectEqual(expected_tensor, *GetOutput(0));
    } else {
      test::ExpectClose(expected_tensor, *GetOutput(0));
    }
  }
};

/// Test `tf.Tanh`.

TEST_F(GpuUnaryOpTest, TanhFloat) {
  Run<float>(/*input_shape=*/{2, 7},
             /*input=*/
             {-18.0f, -9.0f, -1e-6f, -0.0f, 0.0f, 1e-6, 0.1f, 0.2f, 0.3f, 0.5f,
              0.7f, 0.9f, 9.0f, 18.0f},
             /*op_name=*/"Tanh",
             /*expected_callback=*/std::tanh,
             /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, TanhDouble) {
  Run<double>(/*input_shape=*/{2, 7},
              /*input=*/
              {-18.0, -9.0, -1e-6, -0.0, 0.0, 1e-6, 0.1, 0.2, 0.3, 0.5, 0.7,
               0.9, 9.0, 18.0},
              /*op_name=*/"Tanh",
              /*expected_callback=*/std::tanh,
              /*expect_equal=*/false);
}

TEST_F(GpuUnaryOpTest, TanhHalf) {
  Run<Eigen::half, float>(
      /*input_shape=*/{2, 7},
      /*input=*/
      {static_cast<Eigen::half>(-18.0), static_cast<Eigen::half>(-9.0),
       static_cast<Eigen::half>(-1e-6), static_cast<Eigen::half>(-0.0),
       static_cast<Eigen::half>(0.0), static_cast<Eigen::half>(1e-6),
       static_cast<Eigen::half>(0.1), static_cast<Eigen::half>(0.2),
       static_cast<Eigen::half>(0.3), static_cast<Eigen::half>(0.5),
       static_cast<Eigen::half>(0.7), static_cast<Eigen::half>(0.9),
       static_cast<Eigen::half>(9.0), static_cast<Eigen::half>(18.0)},
      /*op_name=*/"Tanh",
      /*expected_callback=*/std::tanh,
      /*expect_equal=*/false);
}

/// Test `tf.Ceil`.

TEST_F(GpuUnaryOpTest, CeilFloat) {
  Run<float>(/*input_shape=*/{2, 7},
             /*input=*/
             {-18.0f, -9.0f, -1e-6f, -0.0f, 0.0f, 1e-6, 0.1f, 0.2f, 0.3f, 0.5f,
              0.7f, 0.9f, 9.0f, 18.0f},
             /*op_name=*/"Ceil",
             /*expected_callback=*/std::ceil,
             /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, CeilDouble) {
  Run<double>(/*input_shape=*/{2, 7},
              /*input=*/
              {-18.0, -9.0, -1e-6, -0.0, 0.0, 1e-6, 0.1, 0.2, 0.3, 0.5, 0.7,
               0.9, 9.0, 18.0},
              /*op_name=*/"Ceil",
              /*expected_callback=*/std::ceil,
              /*expect_equal=*/true);
}

TEST_F(GpuUnaryOpTest, CeilHalf) {
  Run<Eigen::half, float>(
      /*input_shape=*/{2, 7},
      /*input=*/
      {static_cast<Eigen::half>(-18.0), static_cast<Eigen::half>(-9.0),
       static_cast<Eigen::half>(-1e-6), static_cast<Eigen::half>(-0.0),
       static_cast<Eigen::half>(0.0), static_cast<Eigen::half>(1e-6),
       static_cast<Eigen::half>(0.1), static_cast<Eigen::half>(0.2),
       static_cast<Eigen::half>(0.3), static_cast<Eigen::half>(0.5),
       static_cast<Eigen::half>(0.7), static_cast<Eigen::half>(0.9),
       static_cast<Eigen::half>(9.0), static_cast<Eigen::half>(18.0)},
      /*op_name=*/"Ceil",
      /*expected_callback=*/std::ceil,
      /*expect_equal=*/true);
}

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

}  // namespace
}  // end namespace tensorflow
