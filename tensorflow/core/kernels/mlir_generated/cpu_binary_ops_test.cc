/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/kernels/mlir_generated/base_binary_ops_test.h"
#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"

namespace tensorflow {
namespace {

// Test fixture `BinaryOpsTest` that sets the TF device is expected by the TEST
// macros below.
class BinaryOpsTest : public BinaryOpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_cpu(
        tensorflow::DeviceFactory::NewDevice("CPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_CPU, std::move(device_cpu));
  }
};

/// Test `tf.AddV2`.

template <typename T>
T baseline_add(T lhs, T rhs) {
  return lhs + rhs;
}

GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Half, Eigen::half, Eigen::half,
                       baseline_add)
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Float, float, float, baseline_add)
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Double, double, double,
                       baseline_add)
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int32, int32, int32, baseline_add)
GENERATE_DEFAULT_TESTS(AddV2, /*test_name=*/Int64, int64, int64, baseline_add)

}  // namespace
}  // namespace tensorflow
