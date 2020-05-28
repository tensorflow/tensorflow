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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_TRANSPOSE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_TRANSPOSE_H_

#include "tensorflow/lite/delegates/gpu/cl/arguments.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class Transpose : public GPUOperation {
 public:
  Transpose(const OperationDef& definition, const TransposeAttributes& attr)
      : GPUOperation(definition), attr_(attr), work_group_size_(8, 4, 1) {}
  absl::Status AddToQueue(CLCommandQueue* queue) override;
  absl::Status Tune(const TuningParameters& params) override;
  absl::Status Compile(const CreationContext& creation_context) override;

  // Move only
  Transpose(Transpose&& operation);
  Transpose& operator=(Transpose&& operation);
  Transpose(const Transpose&) = delete;
  Transpose& operator=(const Transpose&) = delete;

 private:
  absl::Status BindArguments();
  int3 GetGridSize() const;

  TransposeAttributes attr_;
  Arguments args_;
  CLKernel kernel_;
  int3 work_group_size_;
};

Transpose CreateTranspose(const OperationDef& definition,
                          const TransposeAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_TRANSPOSE_H_
