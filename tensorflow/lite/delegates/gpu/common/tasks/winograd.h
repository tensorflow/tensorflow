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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_WINOGRAD_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_WINOGRAD_H_

#include <string>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_linear_desc.h"

namespace tflite {
namespace gpu {

// You can read https://arxiv.org/pdf/1509.09308.pdf for understanding of basic
// principles. In this kernels used different matrices for transformations than
// in original work.
class Winograd4x4To36 : public GPUOperation {
 public:
  Winograd4x4To36() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;
  absl::Status BindArguments(ArgumentsBinder* args) override;

  // Move only
  Winograd4x4To36(Winograd4x4To36&& kernel) = default;
  Winograd4x4To36& operator=(Winograd4x4To36&& kernel) = default;
  Winograd4x4To36(const Winograd4x4To36&) = delete;
  Winograd4x4To36& operator=(const Winograd4x4To36&) = delete;

 private:
  Winograd4x4To36(const OperationDef& definition, const Padding2D& padding)
      : GPUOperation(definition), padding_(padding) {}
  friend Winograd4x4To36 CreateWinograd4x4To36(const OperationDef& definition,
                                               const Padding2D& padding,
                                               const GpuInfo& gpu_info);

  Padding2D padding_;
};

Winograd4x4To36 CreateWinograd4x4To36(const OperationDef& definition,
                                      const Padding2D& padding,
                                      const GpuInfo& gpu_info);

class Winograd4x4To36TileX6 : public GPUOperation {
 public:
  Winograd4x4To36TileX6() = default;
  Winograd4x4To36TileX6(const OperationDef& definition,
                        const Padding2D& padding, const GpuInfo& gpu_info);
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;

  // Move only
  Winograd4x4To36TileX6(Winograd4x4To36TileX6&& operation) = default;
  Winograd4x4To36TileX6& operator=(Winograd4x4To36TileX6&& operation) = default;
  Winograd4x4To36TileX6(const Winograd4x4To36TileX6&) = delete;
  Winograd4x4To36TileX6& operator=(const Winograd4x4To36TileX6&) = delete;

 private:
  friend Winograd4x4To36TileX6 CreateWinograd4x4To36TileX6(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const Padding2D& padding);

  void UploadBt();

  std::string GetWinograd4x4To36TileX6Code(const OperationDef& op_def,
                                           const GpuInfo& gpu_info);

  // Must be called after kernel compilation
  int3 SelectBestWorkGroup(const KernelInfo& kernel_info) const;

  Padding2D padding_;
};

Winograd4x4To36TileX6 CreateWinograd4x4To36TileX6(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const Padding2D& padding);

class Winograd36To4x4 : public GPUOperation {
 public:
  Winograd36To4x4() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  int3 GetGridSize() const override;

  // Move only
  Winograd36To4x4(Winograd36To4x4&& kernel) = default;
  Winograd36To4x4& operator=(Winograd36To4x4&& kernel) = default;
  Winograd36To4x4(const Winograd36To4x4&) = delete;
  Winograd36To4x4& operator=(const Winograd36To4x4&) = delete;

 private:
  explicit Winograd36To4x4(const OperationDef& definition)
      : GPUOperation(definition) {}
  friend Winograd36To4x4 CreateWinograd36To4x4(
      const OperationDef& definition,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);
};

Winograd36To4x4 CreateWinograd36To4x4(
    const OperationDef& definition,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);

class Winograd36To4x4Tile4x1 : public GPUOperation {
 public:
  Winograd36To4x4Tile4x1() = default;
  Winograd36To4x4Tile4x1(const OperationDef& definition,
                         const GpuInfo& gpu_info);
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override;

  // Move only
  Winograd36To4x4Tile4x1(Winograd36To4x4Tile4x1&& operation) = default;
  Winograd36To4x4Tile4x1& operator=(Winograd36To4x4Tile4x1&& operation) =
      default;
  Winograd36To4x4Tile4x1(const Winograd36To4x4Tile4x1&) = delete;
  Winograd36To4x4Tile4x1& operator=(const Winograd36To4x4Tile4x1&) = delete;

 private:
  friend Winograd36To4x4Tile4x1 CreateWinograd36To4x4Tile4x1(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);

  void UploadAt();

  std::string GetWinograd36To4x4Tile4x1Code(const OperationDef& op_def,
                                            const GpuInfo& gpu_info);

  // Must be called after kernel compilation
  int3 SelectBestWorkGroup(const KernelInfo& kernel_info) const;
};

Winograd36To4x4Tile4x1 CreateWinograd36To4x4Tile4x1(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_WINOGRAD_H_
