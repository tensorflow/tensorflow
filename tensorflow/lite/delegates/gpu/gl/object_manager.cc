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

#include "tensorflow/lite/delegates/gpu/gl/object_manager.h"

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {

Status CreatePHWC4BufferFromTensor(const TensorFloat32& tensor,
                                   GlBuffer* gl_buffer) {
  std::vector<float> transposed(GetElementsSizeForPHWC4(tensor.shape));
  RETURN_IF_ERROR(
      ConvertToPHWC4(tensor.data, tensor.shape, absl::MakeSpan(transposed)));
  return CreateReadOnlyShaderStorageBuffer<float>(transposed, gl_buffer);
}

Status CreatePHWC4BufferFromTensorRef(const TensorRef<BHWC>& tensor_ref,
                                      GlBuffer* gl_buffer) {
  return CreateReadWriteShaderStorageBuffer<float>(
      GetElementsSizeForPHWC4(tensor_ref.shape), gl_buffer);
}

Status CopyFromPHWC4Buffer(const GlBuffer& buffer, TensorFloat32* tensor) {
  return buffer.MappedRead<float>(
      [tensor, &buffer](absl::Span<const float> data) {
        tensor->data.resize(tensor->shape.DimensionsProduct());
        return ConvertFromPHWC4(absl::MakeConstSpan(data), tensor->shape,
                                absl::MakeSpan(tensor->data));
      });
}

Status ObjectManager::RegisterBuffer(uint32_t id, GlBuffer buffer) {
  if (id >= buffers_.size()) {
    buffers_.resize(id + 1);
  }
  buffers_[id] = absl::make_unique<GlBuffer>(std::move(buffer));
  return OkStatus();
}

void ObjectManager::RemoveBuffer(uint32_t id) {
  if (id < buffers_.size()) {
    buffers_[id].reset(nullptr);
  }
}

GlBuffer* ObjectManager::FindBuffer(uint32_t id) const {
  return id >= buffers_.size() ? nullptr : buffers_[id].get();
}

Status ObjectManager::RegisterTexture(uint32_t id, GlTexture texture) {
  if (id >= textures_.size()) {
    textures_.resize(id + 1);
  }
  textures_[id] = absl::make_unique<GlTexture>(std::move(texture));
  return OkStatus();
}

void ObjectManager::RemoveTexture(uint32_t id) {
  if (id < textures_.size()) {
    textures_[id].reset(nullptr);
  }
}

GlTexture* ObjectManager::FindTexture(uint32_t id) const {
  return id >= textures_.size() ? nullptr : textures_[id].get();
}

ObjectsStats ObjectManager::stats() const {
  ObjectsStats stats;
  for (auto& texture : textures_) {
    if (!texture || !texture->has_ownership()) continue;
    stats.textures.count++;
    stats.textures.total_bytes += texture->bytes_size();
  }
  for (auto& buffer : buffers_) {
    if (!buffer || !buffer->has_ownership()) continue;
    stats.buffers.count++;
    stats.buffers.total_bytes += buffer->bytes_size();
  }
  return stats;
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
