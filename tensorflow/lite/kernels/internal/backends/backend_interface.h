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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_INTERFACE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_INTERFACE_H_

#include <memory>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/kernels/internal/backends/backend_kernel.h"

namespace tflite {
namespace internal {
namespace backends {

// Backend interface
class IBackend {
 public:
  // Default virtual destructor
  virtual ~IBackend() = default;
  // Check if backend has a kernel implementation for a given builtin
  virtual bool is_kernel_supported(TfLiteBuiltinOperator op) = 0;
  // Create a kernel wrapper for a given builtin
  //
  // Returns nullptr if the builtin is not supported
  virtual std::unique_ptr<IBackendKernel> create_backend_kernel(
      TfLiteBuiltinOperator op) = 0;
  // Set maximum number of threads
  virtual void set_max_num_threads(int max_num_threads) = 0;
};

}  // namespace backends
}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_INTERFACE_H_