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
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Class that maintains a map from TfGpuId to CudaGpuId, and manages the
// translation between them.
class GpuIdManager {
 public:
  // Adds a mapping from tf_gpu_id to cuda_gpu_id.
  static void InsertTfCudaGpuIdPair(TfGpuId tf_gpu_id, CudaGpuId cuda_gpu_id);

  // Gets the cuda_gpu_id associated with tf_gpu_id. Returns OK if found.
  static Status TfToCudaGpuId(TfGpuId tf_gpu_id, CudaGpuId* cuda_gpu_id);
  // Similar to the above version, but returns the result, and checks fail if
  // no result is found.
  static CudaGpuId TfToCudaGpuId(TfGpuId tf_gpu_id);

  // Clears the map. Used in unit tests only.
  static void TestOnlyReset();
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
