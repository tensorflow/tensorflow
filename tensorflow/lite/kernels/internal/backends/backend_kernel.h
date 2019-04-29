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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_KERNEL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_KERNEL_H_

#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {
namespace internal {
namespace backends {

// Backend kernel interface
class IBackendKernel {
 public:
  // Default virtual destructor
  virtual ~IBackendKernel() = default;
  // Initializes the backend operation from serialized data.
  // If a built-in op:
  //   `buffer` is the op's params data (TfLiteLSTMParams*).
  //   `length` is zero.
  // If custom op:
  //   `buffer` is the op's `custom_options`.
  //   `length` is the size of the buffer.
  //
  // All required data are stored internally to the object to encapsulate
  // backend resource handling.
  virtual void init(TfLiteContext* context, const char* buffer,
                    size_t length) = 0;
  // Free custom resources
  // Note: This could be handled by the object destructor, adding this for
  // alignment.
  virtual void free(TfLiteContext* context) = 0;
  // Prepare is called when the inputs this node depends on have been resized.
  // context->ResizeTensor() can be called to request output tensors to be
  // resized. Moreover, prepare is responsible for validating if the backend
  // kernel can execute the given workload.
  //
  // Returns kTfLiteOk on success.
  virtual TfLiteStatus prepare(TfLiteContext* context, TfLiteNode* node) = 0;
  // Execute the node (should read node->inputs and output to node->outputs).
  // Returns kTfLiteOk on success.
  virtual TfLiteStatus invoke(TfLiteContext* context, TfLiteNode* node) = 0;
};
}  // namespace backends
}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_BACKENDS_BACKEND_KERNEL_H_