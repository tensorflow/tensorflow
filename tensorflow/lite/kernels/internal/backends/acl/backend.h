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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_BACKEND_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_BACKEND_H_

#include "tensorflow/lite/kernels/internal/backends/backend_interface.h"
#include "tensorflow/lite/kernels/internal/backends/backend_kernel.h"

namespace tflite {
namespace internal {
namespace backends {
namespace acl {

// ACL Backend
class ACLBackend final : public IBackend {
 public:
  // Inherited methods overidden:
  bool is_kernel_supported(TfLiteBuiltinOperator op) override;
  std::unique_ptr<IBackendKernel> create_backend_kernel(
      TfLiteBuiltinOperator op) override;
  void set_max_num_threads(int max_num_threads) override;
};

}  // namespace acl
}  // namespace backends
}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_BACKEND_H_