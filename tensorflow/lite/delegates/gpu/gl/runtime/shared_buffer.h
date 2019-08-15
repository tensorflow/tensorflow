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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_SHARED_BUFFER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_SHARED_BUFFER_H_

#include <algorithm>
#include <iterator>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

// Class accumulates readonly data and creates a single buffer out of it.
// User should call Add one or more times and complete shared buffer creation
// with CreateSharedBuffer() call.
class SharedBufferData {
 public:
  SharedBufferData() {
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment_);
  }

  // @return true if data was added to the shared buffer.
  bool Add(const ObjectData& data, GlBuffer* buffer) {
    // TODO(akulik): Does it make sense to bundle even big buffers > 1MB?

    // align buffer's data.
    shared_data_.resize(AlignByN(shared_data_.size(), alignment_), 0);
    // Accumulate readonly data in a single shared buffer buffer.
    *buffer = GlBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id_.id(), data.size(),
                       shared_data_.size(), /*has_ownership=*/false);
    std::copy(data.begin(), data.end(), std::back_inserter(shared_data_));
    return true;
  }

  bool empty() const { return shared_data_.empty(); }

  // Returns a single GlBuffer that owns entire shared data.
  Status CreateSharedGlBuffer(GlBuffer* gl_buffer) {
    // Upload data to a buffer
    gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER,
                                            buffer_id_.id());
    RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glBufferData, GL_SHADER_STORAGE_BUFFER,
                                       shared_data_.size(), shared_data_.data(),
                                       GL_STATIC_READ));
    *gl_buffer = GlBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id_.Release(),
                          shared_data_.size(), 0, /*has_ownership=*/true);
    return OkStatus();
  }

 private:
  GLint alignment_ = 256;
  gl_buffer_internal::BufferId buffer_id_;
  ObjectData shared_data_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_SHARED_BUFFER_H_
