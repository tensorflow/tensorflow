// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GPU_GL_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GPU_GL_UTILS_H_

#include <GLES3/gl32.h>

#include <cstddef>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"

namespace litert {

// Allocates a GL buffer that is backed by a GL buffer object.
LiteRtStatus AllocateSsbo(GLuint* gl_buffer, size_t bytes);

// Allocates a GL buffer that is backed by an AHardwareBuffer.
LiteRtStatus AllocateSsboBackedByAhwb(GLuint* gl_buffer, size_t bytes,
                                      AHardwareBuffer* ahwb_buffer);

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_GPU_GL_UTILS_H_
