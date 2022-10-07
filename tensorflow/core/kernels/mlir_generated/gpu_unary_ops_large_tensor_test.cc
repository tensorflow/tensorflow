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

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"
#include "tensorflow/core/kernels/mlir_generated/base_unary_ops_test.h"

namespace tensorflow {
namespace {
// Test fixture `UnaryOpsTest` that sets the TF device is expected by the TEST
// macros below.
class UnaryOpsTest : public UnaryOpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }
};
/// Test `tf.Atanh`.

// GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
//     Atanh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
//     test::VeryLargeVector<Eigen::half>(), std::atanh,
//     test::OpsTestConfig())
    
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Atanh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
    test::DefaultInputBetweenZeroAndOne<Eigen::half>(), std::atanh,
    test::OpsTestConfig())

}  // namespace
}  // namespace tensorflow
