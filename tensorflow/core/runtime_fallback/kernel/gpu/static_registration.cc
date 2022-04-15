/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file uses a static constructor to automatically register conversion
// functions for TFKernelFallback tensor.

#include "tensorflow/core/runtime_fallback/kernel/gpu/conversion_function.h"
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

static bool kernel_fallback_to_gpu_conversion_fn_registration = []() {
  tfrt::AddStaticTensorConversionFn(
      RegisterTFKernelFallbackTensorToGpuConversionFn);
  return true;
}();

}  // namespace tfd
}  // namespace tensorflow
