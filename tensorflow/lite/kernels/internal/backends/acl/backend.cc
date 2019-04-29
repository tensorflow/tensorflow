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
#include "tensorflow/lite/kernels/internal/backends/acl/backend.h"

// Supported builtins
#include "tensorflow/lite/kernels/internal/backends/acl/conv2d.h"
#include "tensorflow/lite/kernels/internal/backends/acl/depthwise_conv2d.h"

#include "arm_compute/runtime/Scheduler.h"
#include "support/ToolchainSupport.h"

namespace tflite {
namespace internal {
namespace backends {
namespace acl {

bool ACLBackend::is_kernel_supported(TfLiteBuiltinOperator op) {
  switch (op) {
    case TfLiteBuiltinOperator::kTfLiteBuiltinConv2d:
    case TfLiteBuiltinOperator::kTfLiteBuiltinDepthwiseConv2d:
      return true;
    default:
      return false;
  }
}

std::unique_ptr<IBackendKernel> ACLBackend::create_backend_kernel(
    TfLiteBuiltinOperator op) {
  switch (op) {
    case TfLiteBuiltinOperator::kTfLiteBuiltinConv2d:
      return arm_compute::support::cpp14::make_unique<ACLConv2dBackendKernel>();
    case TfLiteBuiltinOperator::kTfLiteBuiltinDepthwiseConv2d:
      return arm_compute::support::cpp14::make_unique<
          ACLDepthwiseConv2dBackendKernel>();
    default:
      return nullptr;
  }
}

void ACLBackend::set_max_num_threads(int max_num_threads) {
  arm_compute::Scheduler::get().set_num_threads(max_num_threads);
}

}  // namespace acl
}  // namespace backends
}  // namespace internal
}  // namespace tflite
