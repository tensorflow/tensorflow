/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "tensorflow/core/kernels/mlir_generated/base_binary_ops_test.h"
#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"

namespace tensorflow {
namespace {
// Test fixture `BianryOpsLargeTensorTest` that sets the TF device is expected
// by the TEST macros below.
class BinaryOpsLargeTensorTest : public BinaryOpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }
};

template <typename T>
T baseline_add(T lhs, T rhs) {
  return lhs + rhs;
}

template <typename T>
T baseline_sub(T lhs, T rhs) {
  return lhs - rhs;
}

/// Test `tf.Addv2`.

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED) && \
    defined(MLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED)

TEST_F(BinaryOpsLargeTensorTest, AddV2LargeTensors) {
  TestEqualShapes<float, float, float, float>(
      "AddV2", /*shape=*/test::DefaultInputShapeExceedingInt32(),
      test::DefaultInput<float>(), test::DefaultInput<float>(), baseline_add,
      test::OpsTestConfig().ExpectStrictlyEqual());
}

#endif

/// Test `tf.Sub`.

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED) && \
    defined(MLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED)

TEST_F(BinaryOpsLargeTensorTest, SubLargeTensors) {
  TestEqualShapes<float, float, float, float>(
      "Sub", /*shape=*/test::DefaultInputShapeExceedingInt32(),
      test::DefaultInput<float>(), test::DefaultInput<float>(), baseline_sub,
      test::OpsTestConfig().ExpectStrictlyEqual());
}

#endif

}  // namespace
}  // namespace tensorflow
