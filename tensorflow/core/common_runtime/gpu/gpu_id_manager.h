/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"

namespace tensorflow {

// Class that manages the translation between Tensorflow GPU ids and CUDA GPU
// ids.
class GpuIdManager {
 public:
  static void InsertTfCudaGpuIdPair(TfGpuId tf_gpu_id, CudaGpuId cuda_gpu_id);
  static CudaGpuId TfToCudaGpuId(TfGpuId tf_gpu_id);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
