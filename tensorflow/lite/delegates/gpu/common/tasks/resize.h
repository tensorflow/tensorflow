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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_RESIZE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_RESIZE_H_

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {

class Resize : public GPUOperation {
 public:
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  Resize(Resize&& operation);
  Resize& operator=(Resize&& operation);
  Resize(const Resize&) = delete;
  Resize& operator=(const Resize&) = delete;

  friend Resize CreateResize(const OperationDef& definition,
                             const Resize2DAttributes& attr);

 private:
  Resize(const OperationDef& definition, const Resize2DAttributes& attr);

  std::string GetResizeCode(const OperationDef& op_def,
                            const Resize2DAttributes& attr);

  Resize2DAttributes attr_;
};

Resize CreateResize(const OperationDef& definition,
                    const Resize2DAttributes& attr);

class Resize3D : public GPUOperation {
 public:
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  Resize3D(Resize3D&& operation);
  Resize3D& operator=(Resize3D&& operation);
  Resize3D(const Resize3D&) = delete;
  Resize3D& operator=(const Resize3D&) = delete;

  friend Resize3D CreateResize3D(const OperationDef& definition,
                                 const Resize3DAttributes& attr);

 private:
  Resize3D(const OperationDef& definition, const Resize3DAttributes& attr);

  std::string GetResize3DCode(const OperationDef& op_def,
                              const Resize3DAttributes& attr);

  Resize3DAttributes attr_;
};

Resize3D CreateResize3D(const OperationDef& definition,
                        const Resize3DAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_RESIZE_H_
