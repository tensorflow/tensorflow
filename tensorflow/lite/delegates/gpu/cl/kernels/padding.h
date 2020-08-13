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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_PADDING_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_PADDING_H_

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace cl {

class Padding : public GPUOperation {
 public:
  Padding(const OperationDef& definition, const PadAttributes& attr);
  int3 GetGridSize() const override;

  // Move only
  Padding(Padding&& kernel);
  Padding& operator=(Padding&& kernel);
  Padding(const Padding&) = delete;
  Padding& operator=(const Padding&) = delete;

 private:
  std::string GetPaddingCode(const OperationDef& op_def,
                             const PadAttributes& attr);
};

Padding CreatePadding(const OperationDef& definition,
                      const PadAttributes& attr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_PADDING_H_
