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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UPSAMPLE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UPSAMPLE_H_

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class Upsample : public GPUOperation {
 public:
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  Upsample(Upsample&& operation);
  Upsample& operator=(Upsample&& operation);
  Upsample(const Upsample&) = delete;
  Upsample& operator=(const Upsample&) = delete;

  friend Upsample CreateUpsample(const OperationDef& definition,
                                 const Upsample2DAttributes& attr);

 private:
  Upsample(const OperationDef& definition, const Upsample2DAttributes& attr)
      : GPUOperation(definition), attr_(attr) {}

  Status BindArguments();
  int3 GetGridSize() const;

  Upsample2DAttributes attr_;
  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

Upsample CreateUpsample(const OperationDef& definition,
                        const Upsample2DAttributes& attr);

class Upsample3D : public GPUOperation {
 public:
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  Upsample3D(Upsample3D&& operation);
  Upsample3D& operator=(Upsample3D&& operation);
  Upsample3D(const Upsample3D&) = delete;
  Upsample3D& operator=(const Upsample3D&) = delete;

  friend Upsample3D CreateUpsample3D(const OperationDef& definition,
                                     const Upsample3DAttributes& attr);

 private:
  Upsample3D(const OperationDef& definition, const Upsample3DAttributes& attr)
      : GPUOperation(definition), attr_(attr) {}

  Status BindArguments();
  int3 GetGridSize() const;

  Upsample3DAttributes attr_;
  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

Upsample3D CreateUpsample3D(const OperationDef& definition,
                            const Upsample3DAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_UPSAMPLE_H_
