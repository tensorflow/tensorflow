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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_TENSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_TENSOR_H_

#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"

namespace tflite {
namespace gpu {

// Interface for GpuSpatialTensor.
// Spatial means that it has Width/Height/Depth dimensions(or their combination)
// and Channels dimension
// Batch dimension optional
class GpuSpatialTensor {
 public:
  GpuSpatialTensor() = default;
  virtual ~GpuSpatialTensor() = default;

  virtual int Width() const = 0;
  virtual int Height() const = 0;
  virtual int Depth() const = 0;
  virtual int Channels() const = 0;
  virtual int Slices() const = 0;
  virtual int Batch() const = 0;

  virtual TensorDescriptor GetDescriptor() const = 0;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_GPU_TENSOR_H_
