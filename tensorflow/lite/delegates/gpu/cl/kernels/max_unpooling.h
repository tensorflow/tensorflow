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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MAX_UNPOOLING_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MAX_UNPOOLING_H_

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class MaxUnpooling : public GPUOperation {
 public:
  MaxUnpooling(const OperationDef& definition,
               const MaxUnpooling2DAttributes& attr);
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  MaxUnpooling(MaxUnpooling&& kernel);
  MaxUnpooling& operator=(MaxUnpooling&& kernel);
  MaxUnpooling(const MaxUnpooling&) = delete;
  MaxUnpooling& operator=(const MaxUnpooling&) = delete;

 private:
  Status BindArguments();
  int3 GetGridSize() const;

  int2 stride_;
  int2 padding_;
  int2 kernel_size_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

MaxUnpooling CreateMaxUnpooling(const OperationDef& definition,
                                const MaxUnpooling2DAttributes& attr);

class MaxUnpooling3D : public GPUOperation {
 public:
  MaxUnpooling3D(const OperationDef& definition,
                 const MaxUnpooling3DAttributes& attr);
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  MaxUnpooling3D(MaxUnpooling3D&& kernel);
  MaxUnpooling3D& operator=(MaxUnpooling3D&& kernel);
  MaxUnpooling3D(const MaxUnpooling3D&) = delete;
  MaxUnpooling3D& operator=(const MaxUnpooling3D&) = delete;

 private:
  Status BindArguments();
  int3 GetGridSize() const;

  int3 stride_;
  int3 padding_;
  int3 kernel_size_;

  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

MaxUnpooling3D CreateMaxUnpooling3D(const OperationDef& definition,
                                    const MaxUnpooling3DAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MAX_UNPOOLING_H_
