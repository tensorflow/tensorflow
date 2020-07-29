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

class MlirGeneratedOpGpuTanhTest : public OpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }
  template <typename T, typename RT = T>
  void RunTanhOp(std::initializer_list<T> input) {
    TensorShape shape({2, 7});
    TF_ASSERT_OK(NodeDefBuilder("tanh_op", "Tanh")
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
      expected.push_back(static_cast<T>(std::tanh(static_cast<RT>(inp))));
    }
    test::FillValues<T>(&expected_tensor, expected);
    test::ExpectClose(expected_tensor, *GetOutput(0));
  }
};

TEST_F(MlirGeneratedOpGpuTanhTest, TanhFloat) {
  RunTanhOp<float>({-18.0f, -9.0f, -1e-6f, -0.0f, 0.0f, 1e-6, 0.1f, 0.2f, 0.3f,
                    0.5f, 0.7f, 0.9f, 9.0f, 18.0f});
}

TEST_F(MlirGeneratedOpGpuTanhTest, TanhDouble) {
  RunTanhOp<double>({-18.0, -9.0, -1e-6, -0.0, 0.0, 1e-6, 0.1, 0.2, 0.3, 0.5,
                     0.7, 0.9, 9.0, 18.0});
}

TEST_F(MlirGeneratedOpGpuTanhTest, TanhHalf) {
  RunTanhOp<Eigen::half, float>(
      {static_cast<Eigen::half>(-18.0), static_cast<Eigen::half>(-9.0),
       static_cast<Eigen::half>(-1e-6), static_cast<Eigen::half>(-0.0),
       static_cast<Eigen::half>(0.0), static_cast<Eigen::half>(1e-6),
       static_cast<Eigen::half>(0.1), static_cast<Eigen::half>(0.2),
       static_cast<Eigen::half>(0.3), static_cast<Eigen::half>(0.5),
       static_cast<Eigen::half>(0.7), static_cast<Eigen::half>(0.9),
       static_cast<Eigen::half>(9.0), static_cast<Eigen::half>(18.0)});
}

}  // namespace
}  // end namespace tensorflow
