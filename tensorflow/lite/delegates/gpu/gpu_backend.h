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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GPU_BACKEND_INTERNAL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GPU_BACKEND_INTERNAL_H_

#include <functional>

#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

namespace tflite {
namespace gpu {

class GpuBackend {
 public:
  virtual absl::Status Prepare(
      const TfLiteGpuDelegateOptionsV2& delegate_options, GraphFloat32* graph,
      std::function<absl::Status(GraphFloat32* graph)> initialize_graph,
      std::unique_ptr<InferenceBuilder>* builder) = 0;

  bool enforce_same_thread() const { return enforce_same_thread_; }

  virtual ~GpuBackend() = default;

  static InferencePriority ToPriority(int32_t priority);
  static InferenceUsage ToUsage(int32_t usage);

 protected:
  bool enforce_same_thread_ = false;
};

TfLiteDelegate* TfLiteGpuDelegateCreateInternal(
    GpuBackend* backend, const TfLiteGpuDelegateOptionsV2* options);

// Destroys a delegate created with `TfLiteGpuDelegateCreateInternal` call.
void TfLiteGpuDelegateDeleteInternal(TfLiteDelegate* delegate);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GPU_BACKEND_INTERNAL_H_
