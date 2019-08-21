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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONCAT_XY_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONCAT_XY_H_

#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class ConcatXY : public GPUOperation {
 public:
  ConcatXY(const OperationDef& definition, const ConcatAttributes& attr,
           int tensors_count)
      : GPUOperation(definition), attr_(attr), tensors_count_(tensors_count) {}
  Status AddToQueue(CLCommandQueue* queue) override;
  Status Tune(const TuningParameters& params) override;

  Status Compile(const CreationContext& creation_context) override;

  // Move only
  ConcatXY(ConcatXY&& operation);
  ConcatXY& operator=(ConcatXY&& operation);
  ConcatXY(const ConcatXY&) = delete;
  ConcatXY& operator=(const ConcatXY&) = delete;

 private:
  Status BindArguments();
  int3 GetGridSize() const;

  ConcatAttributes attr_;
  int tensors_count_;
  CLKernel kernel_;
  int3 work_group_size_ = int3(8, 4, 1);
};

ConcatXY CreateConcatXY(const OperationDef& definition,
                        const ConcatAttributes& attr, int tensors_count);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_CONCAT_XY_H_
