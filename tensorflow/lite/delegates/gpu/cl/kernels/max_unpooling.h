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
               const MaxUnpooling2DAttributes& attr,
               const DeviceInfo& device_info);
  MaxUnpooling(const OperationDef& definition,
               const MaxUnpooling3DAttributes& attr,
               const DeviceInfo& device_info);

  absl::Status BindArguments() override;
  int3 GetGridSize() const override;

  // Move only
  MaxUnpooling(MaxUnpooling&& kernel);
  MaxUnpooling& operator=(MaxUnpooling&& kernel);
  MaxUnpooling(const MaxUnpooling&) = delete;
  MaxUnpooling& operator=(const MaxUnpooling&) = delete;

 private:
  std::string GetMaxUnpoolingKernelCode(const OperationDef& op_def,
                                        const DeviceInfo& device_info);

  int4 stride_;
  int4 padding_;
  int4 kernel_size_;
};

MaxUnpooling CreateMaxUnpooling(const OperationDef& definition,
                                const MaxUnpooling2DAttributes& attr,
                                const DeviceInfo& device_info);

MaxUnpooling CreateMaxUnpooling(const OperationDef& definition,
                                const MaxUnpooling3DAttributes& attr,
                                const DeviceInfo& device_info);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_MAX_UNPOOLING_H_
