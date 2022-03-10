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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

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
  std::vector<TensorStorageType> GetSupportedStoragesWithHWZeroClampSupport(
      DataType data_type) const override;

  const GpuInfo& GetGpuInfo() const override;

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWC>& dst_sizes,
      const std::vector<TensorFloat32*>& dst_cpu) override;

  absl::Status ExecuteGPUOperation(
      const std::vector<Tensor5DFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWDC>& dst_sizes,
      const std::vector<Tensor5DFloat32*>& dst_cpu) override;

  absl::Status ExecuteGPUOperation(
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
    ASSERT_OK(CreateEnvironment(&env_));
    creation_context_.device = env_.GetDevicePtr();
    creation_context_.context = &env_.context();
    creation_context_.queue = env_.queue();
    creation_context_.cache = env_.program_cache();

    ASSERT_OK(exec_env_.Init());
  }

 protected:
  Environment env_;
  CreationContext creation_context_;

  ClExecutionEnvironment exec_env_;
};

absl::Status ExecuteGPUOperation(const TensorFloat32& src_cpu,
                                 const CreationContext& creation_context,
                                 std::unique_ptr<GPUOperation>&& operation,
                                 const BHWC& dst_size, TensorFloat32* result);

absl::Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                                 const CreationContext& creation_context,
                                 std::unique_ptr<GPUOperation>&& operation,
                                 const BHWC& dst_size, TensorFloat32* result);

absl::Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                                 const CreationContext& creation_context,
                                 std::unique_ptr<GPUOperation>&& operation,
                                 const std::vector<BHWC>& dst_sizes,
                                 const std::vector<TensorFloat32*>& dst_cpu);
}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CL_TEST_H_
