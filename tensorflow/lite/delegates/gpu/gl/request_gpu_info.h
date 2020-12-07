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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_REQUEST_GPU_INFO_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_REQUEST_GPU_INFO_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {

// This method performs multiple GL calls, therefore, egl context needs to be
// created upfront.
absl::Status RequestOpenGlInfo(OpenGlInfo* gl_info);

// This method performs multiple GL calls, therefore, egl context needs to be
// created upfront.
absl::Status RequestGpuInfo(GpuInfo* gpu_info);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_REQUEST_GPU_INFO_H_
