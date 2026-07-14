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

#include <cstddef>
#include <cstdint>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {

absl::Status CopyBuffer(const GlBuffer& read_buffer,
                        const GlBuffer& write_buffer) {
  if (read_buffer.bytes_size() != write_buffer.bytes_size()) {
    return absl::InvalidArgumentError(
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

absl::Status GetSSBOSize(GLuint id, int64_t* size_bytes) {
  GLuint prev_id;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glGetIntegerv,
                                     GL_SHADER_STORAGE_BUFFER_BINDING,
                                     reinterpret_cast<GLint*>(&prev_id)));
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id,
                                          prev_id);
  return TFLITE_GPU_CALL_GL(glGetBufferParameteri64v, GL_SHADER_STORAGE_BUFFER,
                            GL_BUFFER_SIZE, size_bytes);
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

absl::Status GlBuffer::BindToIndex(uint32_t index) const {
  return TFLITE_GPU_CALL_GL(glBindBufferRange, target_, index, id_, offset_,
                            bytes_size_);
}

absl::Status GlBuffer::MakeView(size_t offset, size_t bytes_size,
                                GlBuffer* gl_buffer) {
  if (offset + bytes_size > bytes_size_) {
    return absl::OutOfRangeError("GlBuffer view is out of range.");
  }
  *gl_buffer = GlBuffer(target_, id_, bytes_size, offset_ + offset,
                        /*has_ownership=*/false);
  return absl::OkStatus();
}

GlBuffer GlBuffer::MakeRef() {
  return GlBuffer(target_, id_, bytes_size_, offset_,
                  /* has_ownership = */ false);
}

GlPersistentBuffer::GlPersistentBuffer(GLenum target, GLuint id,
                                       size_t bytes_size, size_t offset,
                                       bool has_ownership, void* data)
    : GlBuffer(target, id, bytes_size, offset, has_ownership), data_(data) {}

GlPersistentBuffer::GlPersistentBuffer()
    : GlPersistentBuffer(GL_INVALID_ENUM, GL_INVALID_INDEX, 0, 0, false,
                         nullptr) {}

GlPersistentBuffer::GlPersistentBuffer(GlPersistentBuffer&& buffer)
    : GlBuffer(std::move(buffer)), data_(buffer.data_) {}

GlPersistentBuffer& GlPersistentBuffer::operator=(GlPersistentBuffer&& buffer) {
  if (this != &buffer) {
    data_ = buffer.data_;
    GlBuffer::operator=(std::move(buffer));
  }
  return *this;
}

GlPersistentBuffer::~GlPersistentBuffer() {
  if (!data_) return;
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id());
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

absl::Status CreatePersistentBuffer(size_t size,
                                    GlPersistentBuffer* gl_buffer) {
  PFNGLBUFFERSTORAGEEXTPROC glBufferStorageEXT = nullptr;
  glBufferStorageEXT = reinterpret_cast<PFNGLBUFFERSTORAGEEXTPROC>(
      eglGetProcAddress("glBufferStorageEXT"));
  if (!glBufferStorageEXT) {
    return absl::UnavailableError("glBufferStorageEXT is not supported");
  }
  gl_buffer_internal::BufferId id;
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id.id());
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(
      glBufferStorageEXT, GL_SHADER_STORAGE_BUFFER, size, nullptr,
      GL_MAP_COHERENT_BIT_EXT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
          GL_MAP_PERSISTENT_BIT_EXT));
  void* data = nullptr;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(
      glMapBufferRange, &data, GL_SHADER_STORAGE_BUFFER, 0, size,
      GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT));
  *gl_buffer = GlPersistentBuffer{
      GL_SHADER_STORAGE_BUFFER, id.Release(), size, 0, true, data};
  return absl::OkStatus();
}

namespace gl_buffer_internal {

BufferMapper::BufferMapper(GLenum target, size_t offset, size_t bytes,
                           GLbitfield access)
    : target_(target),
      data_(glMapBufferRange(target_, offset, bytes, access)) {}

BufferMapper::~BufferMapper() {
  TFLITE_GPU_CALL_GL(glUnmapBuffer, target_).IgnoreError();
}

};  // namespace gl_buffer_internal

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
