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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_WINOGRAD_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_WINOGRAD_H_

#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/linear_storage.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {
namespace cl {

// You can read https://arxiv.org/pdf/1509.09308.pdf for understanding of basic
// principles. In this kernels used different matrices for transformations than
// in original work.
class Winograd4x4To36 : public GPUOperation {
 public:
  Winograd4x4To36() = default;
  Winograd4x4To36(const OperationDef& definition, const Padding2D& padding)
      : GPUOperation(definition), padding_(padding) {}
  absl::Status AddToQueue(CLCommandQueue* queue) override;
  absl::Status Tune(const TuningParameters& params) override;
  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  Winograd4x4To36(Winograd4x4To36&& operation);
  Winograd4x4To36& operator=(Winograd4x4To36&& operation);
  Winograd4x4To36(const Winograd4x4To36&) = delete;
  Winograd4x4To36& operator=(const Winograd4x4To36&) = delete;

 private:
  friend absl::Status CreateWinograd4x4To36(
      const CreationContext& creation_context, const OperationDef& definition,
      const Padding2D& padding, Winograd4x4To36* result);

  absl::Status UploadBt(CLContext* context);

  // Must be called after kernel compilation
  int3 SelectBestWorkGroup();

  absl::Status BindArguments();
  int3 GetGridSize() const;

  Padding2D padding_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(128, 1, 1);
};

absl::Status CreateWinograd4x4To36(const CreationContext& creation_context,
                                   const OperationDef& definition,
                                   const Padding2D& padding,
                                   Winograd4x4To36* result);

class Winograd36To4x4 : public GPUOperation {
 public:
  Winograd36To4x4() = default;
  explicit Winograd36To4x4(const OperationDef& definition)
      : GPUOperation(definition) {}
  absl::Status AddToQueue(CLCommandQueue* queue) override;
  absl::Status Tune(const TuningParameters& params) override;
  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  Winograd36To4x4(Winograd36To4x4&& operation);
  Winograd36To4x4& operator=(Winograd36To4x4&& operation);
  Winograd36To4x4(const Winograd36To4x4&) = delete;
  Winograd36To4x4& operator=(const Winograd36To4x4&) = delete;

 private:
  friend absl::Status CreateWinograd36To4x4(
      const CreationContext& creation_context, const OperationDef& definition,
      const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases,
      Winograd36To4x4* result);

  absl::Status UploadAt(CLContext* context);

  // Must be called after kernel compilation
  int3 SelectBestWorkGroup();

  absl::Status BindArguments();
  int3 GetGridSize() const;

  LinearStorage at_;
  LinearStorage biases_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(128, 1, 1);
};

absl::Status CreateWinograd36To4x4(
    const CreationContext& creation_context, const OperationDef& definition,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases,
    Winograd36To4x4* result);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_WINOGRAD_H_
