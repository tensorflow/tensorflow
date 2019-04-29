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
// This file defines a kernel abstraction API for external kernel
// implementations.
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_DEPTHWISE_CONV2D_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_DEPTHWISE_CONV2D_H_

#include "tensorflow/lite/kernels/internal/backends/backend_kernel.h"

#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"

namespace tflite {
namespace internal {
namespace backends {
namespace acl {

// Backend ACL kernel for depthwise convolution
class ACLDepthwiseConv2dBackendKernel final : public IBackendKernel {
 public:
  // Inherited methods overridden:
  void init(TfLiteContext* context, const char* buffer, size_t length) override;
  void free(TfLiteContext* context) override;
  TfLiteStatus prepare(TfLiteContext* context, TfLiteNode* node) override;
  TfLiteStatus invoke(TfLiteContext* context, TfLiteNode* node) override;

 private:
  arm_compute::NEDepthwiseConvolutionLayer3x3 _conv_func{};
  arm_compute::Tensor _input{};
  arm_compute::Tensor _filter{};
  arm_compute::Tensor _bias{};
  arm_compute::Tensor _output{};
  bool _is_configured{false};
};

}  // namespace acl
}  // namespace backends
}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_ACL_DEPTHWISE_CONV2D_H_