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

  template <typename T, typename RT = T>
  void RunAddOp(std::vector<T> input1, TensorShape shape1,
                std::vector<T> input2, TensorShape shape2,
                std::vector<T> output, TensorShape output_shape) {
    TF_ASSERT_OK(NodeDefBuilder("add_op", "AddV2")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
    AddInputFromArray<T>(shape1, input1);
    AddInputFromArray<T>(shape2, input2);
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected_tensor(allocator(), DataTypeToEnum<T>::value, output_shape);
    test::FillValues<T>(&expected_tensor, output);
    test::ExpectEqual(expected_tensor, *GetOutput(0));
  }

  template <typename T, typename RT = T>
  void RunBroadcastingAddOp() {
    auto input1 = {
        static_cast<T>(10),
        static_cast<T>(20),
    };
    auto shape1 = TensorShape({2, 1});
    auto input2 = {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)};
    auto shape2 = TensorShape({3});
    std::vector<T> expected{
        static_cast<T>(11), static_cast<T>(12), static_cast<T>(13),
        static_cast<T>(21), static_cast<T>(22), static_cast<T>(23),
    };
    auto expected_shape = TensorShape({2, 3});
    RunAddOp<T, RT>(input1, shape1, input2, shape2, expected, expected_shape);
  }

  template <typename T, typename RT = T>
  void RunAddOp() {
    auto input1 = {static_cast<T>(-std::numeric_limits<RT>::infinity()),
                   static_cast<T>(-0.1),
                   static_cast<T>(-0.0),
                   static_cast<T>(0.0),
                   static_cast<T>(0.1),
                   static_cast<T>(std::numeric_limits<RT>::infinity())};
    auto input2 = {static_cast<T>(-std::numeric_limits<RT>::infinity()),
                   static_cast<T>(-0.1),
                   static_cast<T>(-0.0),
                   static_cast<T>(0.0),
                   static_cast<T>(0.1),
                   static_cast<T>(std::numeric_limits<RT>::infinity())};
    std::vector<T> expected;
    for (const T& inp : input2) {
      expected.push_back(
          static_cast<T>(static_cast<RT>(inp) + static_cast<RT>(inp)));
    }
    RunAddOp<T, RT>(input1, {2, 3}, input2, {2, 3}, expected, {2, 3});
  }
};

TEST_F(GpuAddTest, AddFloat) { RunAddOp<float>(); }
TEST_F(GpuAddTest, AddDouble) { RunAddOp<double>(); }
TEST_F(GpuAddTest, AddHalf) { RunAddOp<Eigen::half, float>(); }
TEST_F(GpuAddTest, BCastAddFloat) { RunBroadcastingAddOp<float>(); }
TEST_F(GpuAddTest, BCastAddDouble) { RunBroadcastingAddOp<double>(); }
TEST_F(GpuAddTest, BCastAddHalf) { RunBroadcastingAddOp<Eigen::half, float>(); }

// TEST_F(GpuAddTest, AddV2Half) { RunAddOp<Eigen::half, float>(); }
}  // namespace
}  // end namespace tensorflow
