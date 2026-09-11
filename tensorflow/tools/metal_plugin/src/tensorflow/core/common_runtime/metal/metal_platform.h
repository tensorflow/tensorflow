/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_PLATFORM_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_PLATFORM_H_

// Names no Objective-C types; includable from plain C++.

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace metal {

// Device type reported to TensorFlow.
//
// "GPU" rather than a new type such as "METAL", so that Metal devices show up
// as /device:GPU:0 and existing user code, Keras, tf.distribute and the
// placement rules all work unmodified. There is no clash with the CUDA GPU
// device: CUDA is never built on macOS.
inline constexpr char kMetalDeviceType[] = "GPU";

// Platform (subtype) name, which is what distinguishes this backend from any
// other registered under the GPU device type.
inline constexpr char kMetalPlatformName[] = "METAL";

// StreamExecutor C API plugin entry point.
//
// Has the shape of SE_InitPlugin but is not exported under that name: this
// backend is linked into TensorFlow rather than dlopen'd, so it is registered
// by passing this function pointer to RegisterPluggableDevicePlugin. Keeping
// the symbol namespaced also avoids colliding with an out-of-tree plugin's
// SE_InitPlugin in the same process.
void MetalInitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status);

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_PLATFORM_H_
