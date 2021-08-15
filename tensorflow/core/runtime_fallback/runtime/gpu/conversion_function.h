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

// This file declares TFRuntimeFallback tensor conversion functions for copying
// between gpu and host.

#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_GPU_CONVERSION_FUNCTION_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_GPU_CONVERSION_FUNCTION_H_

#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// Register conversion functions for TFRuntimeFallbackTensors.
void RegisterTFRuntimeFallbackTensorToGpuConversionFn(
    tfrt::TensorConversionFnRegistry* registry);

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_RUNTIME_GPU_CONVERSION_FUNCTION_H_
