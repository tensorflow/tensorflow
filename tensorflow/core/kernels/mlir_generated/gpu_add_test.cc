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
#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GpuAddTest : public OpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }

  template <typename T, typename BaselineType = T>
  void SetAddOp(std::vector<T> input_1, TensorShape shape_1,
                std::vector<T> input_2, TensorShape shape_2) {
    TF_ASSERT_OK(NodeDefBuilder("add_op", "AddV2")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
    inputs_.clear();
    AddInputFromArray<T>(shape_1, input_1);
    AddInputFromArray<T>(shape_2, input_2);
  }

  template <typename T, typename BaselineType = T>
  void RunAndCompareAddOp(std::vector<T> input_1, TensorShape shape_1,
                          std::vector<T> input_2, TensorShape shape_2,
                          std::vector<T> output, TensorShape output_shape) {
    SetAddOp<T>(input_1, shape_1, input_2, shape_2);
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected_tensor(allocator(), DataTypeToEnum<T>::value, output_shape);
    test::FillValues<T>(&expected_tensor, output);
    test::ExpectEqual(expected_tensor, *GetOutput(0));
  }

  template <typename T, typename BaselineType = T>
  void TestBroadcastingExpandAddOp() {
    auto input_1 = {static_cast<T>(10)};
    auto input_2 = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
                    static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)};
    std::vector<T> expected{
        static_cast<T>(11), static_cast<T>(12), static_cast<T>(13),
        static_cast<T>(14), static_cast<T>(15), static_cast<T>(16),
    };
    auto expected_shape = TensorShape({6});
    RunAndCompareAddOp<T, BaselineType>(input_1, TensorShape({1}), input_2,
                                        TensorShape({6}), expected,
                                        expected_shape);
  }

  template <typename T, typename BaselineType = T>
  void TestBroadcastingInDimAddOp() {
    auto input_1 = {static_cast<T>(10), static_cast<T>(20), static_cast<T>(30)};
    auto input_2 = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
                    static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)};
    std::vector<T> expected{
        static_cast<T>(11), static_cast<T>(22), static_cast<T>(33),
        static_cast<T>(14), static_cast<T>(25), static_cast<T>(36),
    };
    auto expected_shape = TensorShape({2, 3});
    RunAndCompareAddOp<T, BaselineType>(input_1, TensorShape({3}), input_2,
                                        TensorShape({2, 3}), expected,
                                        expected_shape);
  }

  template <typename T, typename BaselineType = T>
  void TestBroadcastingAddOp() {
    auto input_1 = {static_cast<T>(10), static_cast<T>(20)};
    auto input_2 = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};
    std::vector<T> expected{
        static_cast<T>(11), static_cast<T>(12), static_cast<T>(13),
        static_cast<T>(21), static_cast<T>(22), static_cast<T>(23),
    };
    auto expected_shape = TensorShape({2, 3});
    RunAndCompareAddOp<T, BaselineType>(input_1, TensorShape({2, 1}), input_2,
                                        TensorShape({3}), expected,
                                        expected_shape);
  }

  template <typename T, typename BaselineType = T>
  void RunAddOp() {
    auto input_1 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    auto input_2 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    std::vector<T> expected;
    for (const T& inp : input_2) {
      expected.push_back(static_cast<T>(static_cast<BaselineType>(inp) +
                                        static_cast<BaselineType>(inp)));
    }
    RunAndCompareAddOp<T, BaselineType>(input_1, TensorShape{2, 3}, input_2,
                                        TensorShape{2, 3}, expected,
                                        TensorShape{2, 3});
  }

  template <typename T, typename BaselineType = T>
  void TestEqualShapesAddOp() {
    auto input_1 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    auto input_2 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    std::vector<T> expected;
    for (const T& inp : input_2) {
      expected.push_back(static_cast<T>(static_cast<BaselineType>(inp) +
                                        static_cast<BaselineType>(inp)));
    }
    RunAndCompareAddOp<T, BaselineType>(input_1, TensorShape{2, 3}, input_2,
                                        TensorShape{2, 3}, expected,
                                        TensorShape{2, 3});
  }

  template <typename T, typename BaselineType = T>
  void TestOneIsScalarAddOp() {
    auto input_1 = static_cast<T>(42);
    auto input_2 = {
        static_cast<T>(-std::numeric_limits<BaselineType>::infinity()),
        static_cast<T>(-0.1),
        static_cast<T>(-0.0),
        static_cast<T>(0.0),
        static_cast<T>(0.1),
        static_cast<T>(std::numeric_limits<BaselineType>::infinity())};
    std::vector<T> expected;
    for (const T& inp : input_2) {
      expected.push_back(static_cast<T>(static_cast<BaselineType>(input_1) +
                                        static_cast<BaselineType>(inp)));
    }
    RunAndCompareAddOp<T, BaselineType>({input_1}, TensorShape{}, input_2,
                                        TensorShape{2, 3}, expected,
                                        TensorShape{2, 3});
  }

  template <typename T, typename RT = T>
  void TestIncompatibleShapes() {
    auto input_1 = {static_cast<T>(-0.1), static_cast<T>(-0.0),
                    static_cast<T>(0.0)};
    auto input_2 = {static_cast<T>(-0.1), static_cast<T>(0.0)};

    SetAddOp<T>(input_1, TensorShape{3}, input_2, TensorShape{2});
    auto status = RunOpKernel();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  }

  template <typename T, typename BaselineType = T>
  void TestEmptyShapeWithBroadcastingAddOp() {
    TensorShape input_shape_a{2, 0, 1};
    TensorShape input_shape_b{2, 0, 5};
    TensorShape expected_shape{2, 0, 5};
    std::vector<T> empty_input = {};
    RunAndCompareAddOp<T, BaselineType>(empty_input, input_shape_a, empty_input,
                                        input_shape_b, empty_input,
                                        expected_shape);
    RunAndCompareAddOp<T, BaselineType>(empty_input, input_shape_b, empty_input,
                                        input_shape_a, empty_input,
                                        expected_shape);
  }
};

TEST_F(GpuAddTest, AddFloat) { RunAddOp<float>(); }
TEST_F(GpuAddTest, AddDouble) { RunAddOp<double>(); }
TEST_F(GpuAddTest, AddHalf) { RunAddOp<Eigen::half, float>(); }
TEST_F(GpuAddTest, AddInt64) { RunAddOp<int64, int64>(); }

TEST_F(GpuAddTest, AddEqShapesFloat) { TestEqualShapesAddOp<float>(); }
TEST_F(GpuAddTest, AddEqShapesDouble) { TestEqualShapesAddOp<double>(); }
TEST_F(GpuAddTest, AddEqShapesHalf) {
  TestEqualShapesAddOp<Eigen::half, float>();
}
TEST_F(GpuAddTest, AddEqShapesInt64) { TestEqualShapesAddOp<int64>(); }

TEST_F(GpuAddTest, AddScalarFloat) { TestOneIsScalarAddOp<float>(); }
TEST_F(GpuAddTest, AddScalarDouble) { TestOneIsScalarAddOp<double>(); }
TEST_F(GpuAddTest, AddScalarHalf) {
  TestOneIsScalarAddOp<Eigen::half, float>();
}
TEST_F(GpuAddTest, AddScalarInt64) { TestOneIsScalarAddOp<int64>(); }

TEST_F(GpuAddTest, BCastExpandAddFloat) {
  TestBroadcastingExpandAddOp<float>();
}
TEST_F(GpuAddTest, BCastExpandAddDouble) {
  TestBroadcastingExpandAddOp<double>();
}
TEST_F(GpuAddTest, BCastExpandAddHalf) {
  TestBroadcastingExpandAddOp<Eigen::half, float>();
}
TEST_F(GpuAddTest, BCastExpandAddInt64) {
  TestBroadcastingExpandAddOp<int64>();
}

TEST_F(GpuAddTest, BCastInDimAddFloat) { TestBroadcastingInDimAddOp<float>(); }
TEST_F(GpuAddTest, BCastInDimAddDouble) {
  TestBroadcastingInDimAddOp<double>();
}
TEST_F(GpuAddTest, BCastInDimAddHalf) {
  TestBroadcastingInDimAddOp<Eigen::half, float>();
}
TEST_F(GpuAddTest, BCastInDimAddInt64) { TestBroadcastingInDimAddOp<int64>(); }

TEST_F(GpuAddTest, BCastAddFloat) { TestBroadcastingAddOp<float>(); }
TEST_F(GpuAddTest, BCastAddDouble) { TestBroadcastingAddOp<double>(); }
TEST_F(GpuAddTest, BCastAddHalf) {
  TestBroadcastingAddOp<Eigen::half, float>();
}
TEST_F(GpuAddTest, BCastAddInt64) { TestBroadcastingAddOp<int64>(); }

TEST_F(GpuAddTest, IncompatibleShapes) { TestIncompatibleShapes<float>(); }

TEST_F(GpuAddTest, EmptyShapeBCastAddFloat) {
  TestEmptyShapeWithBroadcastingAddOp<float>();
}
TEST_F(GpuAddTest, EmptyShapeBCastAddDouble) {
  TestEmptyShapeWithBroadcastingAddOp<double>();
}

// TEST_F(GpuAddTest, AddV2Half) { RunAddOp<Eigen::half, float>(); }
}  // namespace
}  // end namespace tensorflow
