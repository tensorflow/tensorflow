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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMMON_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMMON_H_

#import <Metal/Metal.h>

#include <utility>

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace metal {

/// Returns system default device on iOS or Intel GPU on macOS.
id<MTLDevice> GetBestSupportedMetalDevice();

/// Metal compute shader compilation
/// @param device The device on which that shader program will be stored.
/// @param code Shader source.
/// @param functionName The name of the main shader function.
/// @param macros Compile-time definitions.
/// @param program A non-nil pointer to the program object that will be filled.
/// @return Returns a valid program pointer or error string. At least one pointer is valid but not
///     both.
/// @discussion The function autoselects the maximum shader language version supported by the target
///     OS. FastMath is enabled.
absl::Status CreateComputeProgram(id<MTLDevice> device, NSString* code, NSString* functionName,
                                  NSDictionary<NSString*, NSString*>* macros,
                                  id<MTLComputePipelineState>* program);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMMON_H_
