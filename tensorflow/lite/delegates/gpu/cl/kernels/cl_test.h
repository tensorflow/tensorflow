/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CL_TEST_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CL_TEST_H_

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"

namespace tflite {
namespace gpu {
namespace cl {

#ifndef ASSERT_OK
#define ASSERT_OK(x) ASSERT_TRUE(x.ok());
#endif

class ClExecutionEnvironment : public TestExecutionEnvironment {
 public:
  ClExecutionEnvironment() = default;
  ~ClExecutionEnvironment() override = default;

  absl::Status Init();

  std::vector<CalculationsPrecision> GetSupportedPrecisions() const override;
  std::vector<TensorStorageType> GetSupportedStorages(
      DataType data_type) const override;

  const GpuInfo& GetGpuInfo() const override;

  absl::Status ExecuteGpuOperationInternal(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation) override;

 private:
  Environment env_;
};

class OpenCLOperationTest : public ::testing::Test {
 public:
  void SetUp() override {
    ASSERT_OK(LoadOpenCL());
    ASSERT_OK(exec_env_.Init());
  }

 protected:
  ClExecutionEnvironment exec_env_;
};
}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CL_TEST_H_
