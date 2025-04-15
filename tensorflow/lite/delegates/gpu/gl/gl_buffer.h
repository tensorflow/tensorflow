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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_BUFFER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_BUFFER_H_

#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {

// Buffer is an RAII wrapper for OpenGL buffer object.
// See https://www.khronos.org/opengl/wiki/Buffer_Object for more information.
//
// Buffer is moveable but not copyable.
class GlBuffer {
 public:
  // @param has_ownership indicates that GlBuffer is responsible for
  // corresponding GL buffer deletion.
  GlBuffer(GLenum target, GLuint id, size_t bytes_size, size_t offset,
           bool has_ownership)
      : target_(target),
        id_(id),
        bytes_size_(bytes_size),
        offset_(offset),
        has_ownership_(has_ownership) {}

  // Creates invalid buffer.
  GlBuffer() : GlBuffer(GL_INVALID_ENUM, GL_INVALID_INDEX, 0, 0, false) {}

  // Move-only
  GlBuffer(GlBuffer&& buffer);
  GlBuffer& operator=(GlBuffer&& buffer);
  GlBuffer(const GlBuffer&) = delete;
  GlBuffer& operator=(const GlBuffer&) = delete;

  ~GlBuffer();

  // Reads data from buffer into CPU memory. Data should point to a region that
  // has at least bytes_size available.
  template <typename T>
  absl::Status Read(absl::Span<T> data) const;

  // Writes data to a buffer.
  template <typename T>
  absl::Status Write(absl::Span<const T> data);

  // Maps GPU memory to CPU address space and calls reader that may read from
  // that memory.
  template <typename T>
  absl::Status MappedRead(
      const std::function<absl::Status(absl::Span<const T>)>& reader) const;

  // Maps GPU memory to CPU address space and calls writer that may write into
  // that memory.
  template <typename T>
  absl::Status MappedWrite(
      const std::function<absl::Status(absl::Span<T>)>& writer);

  absl::Status MakeView(size_t offset, size_t bytes_size, GlBuffer* gl_buffer);

  // Makes a copy without ownership of the buffer.
  GlBuffer MakeRef();

  // Binds a buffer to an index.
  absl::Status BindToIndex(uint32_t index) const;

  // Releases the ownership of the buffer object.
  void Release() { has_ownership_ = false; }

  size_t bytes_size() const { return bytes_size_; }

  const GLenum target() const { return target_; }

  const GLuint id() const { return id_; }

  bool is_valid() const { return id_ != GL_INVALID_INDEX; }

  size_t offset() const { return offset_; }

  // @return true if this object actually owns corresponding GL buffer
  //         and manages it's lifetime.
  bool has_ownership() const { return has_ownership_; }

 private:
  void Invalidate();

  GLenum target_;
  GLuint id_;
  size_t bytes_size_;
  size_t offset_;
  bool has_ownership_;
};

absl::Status CopyBuffer(const GlBuffer& read_buffer,
                        const GlBuffer& write_buffer);

absl::Status GetSSBOSize(GLuint id, int64_t* size_bytes);

// Creates new shader storage buffer that will be modified and used many
// times.
// Buffer will be initialized with 0's.
//
// See https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object for
// details.
template <typename T>
absl::Status CreateReadWriteShaderStorageBuffer(uint32_t num_elements,
                                                GlBuffer* gl_buffer);

// Creates new shader storage buffer that will be filled with data once which
// will be used many times.
template <typename T>
absl::Status CreateReadOnlyShaderStorageBuffer(absl::Span<const T> data,
                                               GlBuffer* gl_buffer);

// Adapts raw Buffer::Read method to read data into a vector.
template <typename T>
absl::Status AppendFromBuffer(const GlBuffer& buffer, std::vector<T>* data) {
  if (buffer.bytes_size() % sizeof(T) != 0) {
    return absl::InvalidArgumentError("Buffer is not aligned");
  }
  size_t num_elements = buffer.bytes_size() / sizeof(T);
  data->resize(data->size() + num_elements);
  return buffer.Read<T>(
      absl::MakeSpan(data->data() + data->size() - num_elements, num_elements));
}

// Persistent buffer provides CPU pointer to the buffer that is valid all the
// time. A user should properly synchronize the access to the buffer on CPU and
// GPU sides.
class GlPersistentBuffer : public GlBuffer {
 public:
  GlPersistentBuffer(GLenum target, GLuint id, size_t bytes_size, size_t offset,
                     bool has_ownership, void* data);
  GlPersistentBuffer();

  // Move-only
  GlPersistentBuffer(GlPersistentBuffer&& buffer);
  GlPersistentBuffer& operator=(GlPersistentBuffer&& buffer);
  GlPersistentBuffer(const GlPersistentBuffer&) = delete;
  GlPersistentBuffer& operator=(const GlPersistentBuffer&) = delete;

  ~GlPersistentBuffer();

  void* data() { return data_; }

 private:
  void* data_;
};

// Creates read-write persistent buffer with valid CPU pointer
absl::Status CreatePersistentBuffer(size_t size, GlPersistentBuffer* gl_buffer);

////////////////////////////////////////////////////////////////////////////////
// Implementation details are below.

namespace gl_buffer_internal {

// RAII for creating and/or owning buffer id.
class BufferId {
 public:
  BufferId() : id_(GL_INVALID_INDEX) {
    TFLITE_GPU_CALL_GL(glGenBuffers, 1 /* number of buffers */, &id_)
        .IgnoreError();
    // only possible error here is when a number of buffers is negative.
  }

  explicit BufferId(GLuint id) : id_(id) {}

  ~BufferId() {
    if (id_ != GL_INVALID_INDEX) {
      TFLITE_GPU_CALL_GL(glDeleteBuffers, 1, &id_).IgnoreError();
    }
  }

  GLuint id() const { return id_; }

  GLuint Release() {
    GLuint id = GL_INVALID_INDEX;
    std::swap(id, id_);
    return id;
  }

 private:
  GLuint id_;
};

// RAII for binding and unbinding a buffer.
class BufferBinder {
 public:
  BufferBinder(GLenum target, GLuint id) : target_(target), prev_id_(0) {
    TFLITE_GPU_CALL_GL(glBindBuffer, target_, id).IgnoreError();
  }

  BufferBinder(GLenum target, GLuint id, GLuint prev_id)
      : target_(target), prev_id_(prev_id) {
    TFLITE_GPU_CALL_GL(glBindBuffer, target_, id).IgnoreError();
  }

  ~BufferBinder() {
    TFLITE_GPU_CALL_GL(glBindBuffer, target_, prev_id_).IgnoreError();
  }

 private:
  const GLenum target_;
  GLuint prev_id_;
};

// RAII for mapping and unmapping a buffer.
class BufferMapper {
 public:
  BufferMapper(GLenum target, size_t offset, size_t bytes, GLbitfield access);

  ~BufferMapper();

  void* data() { return data_; }

 private:
  const GLenum target_;
  void* data_;
};

}  // namespace gl_buffer_internal

template <typename T>
absl::Status CreateReadWriteShaderStorageBuffer(uint32_t num_elements,
                                                GlBuffer* gl_buffer) {
  gl_buffer_internal::BufferId id;
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id.id());
  // TODO(akulik): benchmark DYNAMIC vs STREAM buffer
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(
      glBufferData, GL_SHADER_STORAGE_BUFFER, num_elements * sizeof(T),
      std::vector<T>(num_elements).data(), GL_STREAM_COPY));
  *gl_buffer = GlBuffer{GL_SHADER_STORAGE_BUFFER, id.Release(),
                        num_elements * sizeof(T), 0, true};
  return absl::OkStatus();
}

template <typename T>
absl::Status CreateReadOnlyShaderStorageBuffer(absl::Span<const T> data,
                                               GlBuffer* gl_buffer) {
  gl_buffer_internal::BufferId id;
  gl_buffer_internal::BufferBinder binder(GL_SHADER_STORAGE_BUFFER, id.id());
  RETURN_IF_ERROR(TFLITE_GPU_CALL_GL(glBufferData, GL_SHADER_STORAGE_BUFFER,
                                     data.size() * sizeof(T), data.data(),
                                     GL_STATIC_READ));
  *gl_buffer = GlBuffer{GL_SHADER_STORAGE_BUFFER, id.Release(),
                        data.size() * sizeof(T), 0, true};
  return absl::OkStatus();
}

template <typename T>
absl::Status GlBuffer::Read(absl::Span<T> data) const {
  if (data.size() * sizeof(T) < bytes_size()) {
    return absl::InvalidArgumentError(
        "Read from buffer failed. Destination data is shorter than buffer.");
  }
  // TODO(akulik): glCopyBufferSubData is actually available in ES 3.1, try it.
  return MappedRead<T>([this, data](absl::Span<const T> src) {
    std::memcpy(data.data(), src.data(), bytes_size());
    return absl::OkStatus();
  });
}

template <typename T>
absl::Status GlBuffer::Write(absl::Span<const T> data) {
  if (data.size() * sizeof(T) > bytes_size_) {
    return absl::InvalidArgumentError(
        "Write to buffer failed. Source data is larger than buffer.");
  }
  gl_buffer_internal::BufferBinder binder(target_, id_);
  return TFLITE_GPU_CALL_GL(glBufferSubData, target_, offset_, bytes_size_,
                            data.data());
}

template <typename T>
absl::Status GlBuffer::MappedRead(
    const std::function<absl::Status(absl::Span<const T> d)>& reader) const {
  if (bytes_size_ % sizeof(T) != 0) {
    return absl::InvalidArgumentError("Buffer is not aligned");
  }
  gl_buffer_internal::BufferBinder binder(target_, id_);
  gl_buffer_internal::BufferMapper mapper(target_, offset_, bytes_size_,
                                          GL_MAP_READ_BIT);
  if (!mapper.data()) {
    return GetOpenGlErrors();
  }
  return reader(absl::MakeSpan(reinterpret_cast<const T*>(mapper.data()),
                               bytes_size_ / sizeof(T)));
}

template <typename T>
absl::Status GlBuffer::MappedWrite(
    const std::function<absl::Status(absl::Span<T> d)>& writer) {
  if (bytes_size_ % sizeof(T) != 0) {
    return absl::InvalidArgumentError("Buffer is not aligned");
  }
  gl_buffer_internal::BufferBinder binder(target_, id_);
  gl_buffer_internal::BufferMapper mapper(target_, offset_, bytes_size_,
                                          GL_MAP_WRITE_BIT);
  if (!mapper.data()) {
    return GetOpenGlErrors();
  }
  return writer(absl::MakeSpan(reinterpret_cast<T*>(mapper.data()),
                               bytes_size_ / sizeof(T)));
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_GL_BUFFER_H_
