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

class GpuAbsTest : public OpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }
  template <typename T, typename RT = T>
  void RunAbsOp(std::initializer_list<T> input) {
    TensorShape shape({2, 3});
    TF_ASSERT_OK(NodeDefBuilder("abs_op", "Abs")
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
      expected.push_back(static_cast<T>(std::abs(static_cast<RT>(inp))));
    }
    test::FillValues<T>(&expected_tensor, expected);
    test::ExpectEqual(expected_tensor, *GetOutput(0));
  }
};

TEST_F(GpuAbsTest, AbsFloat) {
  RunAbsOp<float>({-std::numeric_limits<float>::infinity(), -0.1f, -0.0f, 0.0f,
                   0.1f, std::numeric_limits<float>::infinity()});
}

TEST_F(GpuAbsTest, AbsDouble) {
  RunAbsOp<double>({-std::numeric_limits<double>::infinity(), -0.1, -0.0, 0.0,
                    0.1, std::numeric_limits<double>::infinity()});
}

TEST_F(GpuAbsTest, AbsHalf) {
  RunAbsOp<Eigen::half, float>(
      {static_cast<Eigen::half>(-std::numeric_limits<double>::infinity()),
       static_cast<Eigen::half>(-0.1), static_cast<Eigen::half>(-0.0),
       static_cast<Eigen::half>(0.0), static_cast<Eigen::half>(0.1),
       static_cast<Eigen::half>(std::numeric_limits<double>::infinity())});
}

TEST_F(GpuAbsTest, AbsInt32) {
  RunAbsOp<int32>({std::numeric_limits<int32>::min(),
                   std::numeric_limits<int32>::min() + 1, -1, 0, 1,
                   std::numeric_limits<int32>::max()});
}

TEST_F(GpuAbsTest, AbsInt64) {
  RunAbsOp<int64>({std::numeric_limits<int64>::min(),
                   std::numeric_limits<int64>::min() + 1, -1, 0, 1,
                   std::numeric_limits<int64>::max()});
}

}  // namespace
}  // end namespace tensorflow
