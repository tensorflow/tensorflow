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

#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {

Status CopyBuffer(const GlBuffer& read_buffer, const GlBuffer& write_buffer) {
  if (read_buffer.bytes_size() != write_buffer.bytes_size()) {
    return InvalidArgumentError(
        "Read buffer does not match write buffer size.");
  }
  gl_buffer_internal::BufferBinder read_buffer_binder(GL_COPY_READ_BUFFER,
                                                      read_buffer.id());
  gl_buffer_internal::BufferBinder write_buffer_binder(GL_COPY_WRITE_BUFFER,
                                                       write_buffer.id());
  return TFLITE_GPU_CALL_GL(glCopyBufferSubData, GL_COPY_READ_BUFFER,
                            GL_COPY_WRITE_BUFFER, read_buffer.offset(),
                            write_buffer.offset(), read_buffer.bytes_size());
}

GlBuffer::GlBuffer(GlBuffer&& buffer)
    : GlBuffer(buffer.target_, buffer.id_, buffer.bytes_size_, buffer.offset_,
               buffer.has_ownership_) {
  buffer.has_ownership_ = false;
}

GlBuffer& GlBuffer::operator=(GlBuffer&& buffer) {
  if (this != &buffer) {
    Invalidate();

    target_ = buffer.target_;
    bytes_size_ = buffer.bytes_size_;
    offset_ = buffer.offset_;
    has_ownership_ = buffer.has_ownership_;
    id_ = buffer.id_;
    buffer.has_ownership_ = false;
  }
  return *this;
}

GlBuffer::~GlBuffer() { Invalidate(); }

void GlBuffer::Invalidate() {
  if (has_ownership_ && id_ != GL_INVALID_INDEX) {
    TFLITE_GPU_CALL_GL(glDeleteBuffers, 1, &id_).IgnoreError();
    id_ = GL_INVALID_INDEX;
  }
}

Status GlBuffer::BindToIndex(uint32_t index) const {
  return TFLITE_GPU_CALL_GL(glBindBufferRange, target_, index, id_, offset_,
                            bytes_size_);
}

Status GlBuffer::MakeView(size_t offset, size_t bytes_size,
                          GlBuffer* gl_buffer) {
  if (offset + bytes_size > bytes_size_) {
    return OutOfRangeError("GlBuffer view is out of range.");
  }
  *gl_buffer = GlBuffer(target_, id_, bytes_size, offset_ + offset,
                        /*has_ownership=*/false);
  return OkStatus();
}

GlBuffer GlBuffer::MakeRef() {
  return GlBuffer(target_, id_, bytes_size_, offset_,
                  /* has_ownership = */ false);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
