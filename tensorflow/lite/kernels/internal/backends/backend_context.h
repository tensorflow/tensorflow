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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_CONTEXT_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_CONTEXT_H_

#include <map>
#include <memory>
#include <string>

#include "tensorflow/lite/kernels/internal/backends/backend_interface.h"
#include "tensorflow/lite/kernels/internal/backends/backend_kernel.h"

namespace tflite {
namespace internal {
namespace backends {

// Context tracking all the external backends
class KernelBackendContext final {
 public:
  // Constructor
  KernelBackendContext();
  // Destructor
  ~KernelBackendContext();

  // Backend accessor given the backend name
  //
  // Return pointer to the backend on success else nullptr
  IBackend* backend(std::string backend);
  // Extract a backend kernel object of a given backend
  //
  // Return a backend kernel on success else nullptr
  std::unique_ptr<IBackendKernel> backend_kernel(std::string backend,
                                                 TfLiteBuiltinOperator op);
  // Extract a backend kernel object of the first backend that supports it
  //
  // Return a backend kernel on success else nullptr
  std::unique_ptr<IBackendKernel> backend_kernel(TfLiteBuiltinOperator op);
  // Set number of threads on a given backend
  void set_max_num_threads(std::string backend, int max_num_threads);
  // Set number of threads on all backends
  void set_max_num_threads_all(int max_num_threads);

 private:
  std::map<std::string, std::unique_ptr<IBackend>> _backends;

  KernelBackendContext(const KernelBackendContext&) = delete;
};
}  // namespace backends
}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_CONTEXT_H_
