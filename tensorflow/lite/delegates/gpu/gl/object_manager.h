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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_OBJECT_MANAGER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_OBJECT_MANAGER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/delegates/gpu/gl/stats.h"

namespace tflite {
namespace gpu {
namespace gl {

// ObjectManager is a registry that owns corresponding objects and provides
// discovery functionality. All objects are kept until manager is destroyed.
//
// All buffers and textures share the same id space, therefore, it is an error
// to register two objects with the same id.
// TODO(akulik): make ObjectManager templated by object type.
class ObjectManager {
 public:
  // Moves ownership over the given buffer to the manager.
  Status RegisterBuffer(uint32_t id, GlBuffer buffer);

  void RemoveBuffer(uint32_t id);

  // Return a permanent pointer to a buffer for the given id or nullptr.
  GlBuffer* FindBuffer(uint32_t id) const;

  // Moves ownership over the given texture to the manager.
  Status RegisterTexture(uint32_t id, GlTexture texture);

  void RemoveTexture(uint32_t id);

  // Return a permanent pointer to a texture for the given id or nullptr.
  GlTexture* FindTexture(uint32_t id) const;

  ObjectsStats stats() const;

 private:
  std::vector<std::unique_ptr<GlBuffer>> buffers_;
  std::vector<std::unique_ptr<GlTexture>> textures_;
};

// TODO(akulik): find better place for functions below.

// Creates read-only buffer from the given tensor. Tensor data is converted to
// PHWC4 layout.
Status CreatePHWC4BufferFromTensor(const TensorFloat32& tensor,
                                   GlBuffer* gl_buffer);

// Creates read-write buffer for the given tensor shape, where data layout is
// supposed to be PHWC4.
Status CreatePHWC4BufferFromTensorRef(const TensorRefFloat32& tensor_ref,
                                      GlBuffer* gl_buffer);

// Copies data from a buffer that holds data in PHWC4 layout to the given
// tensor.
Status CopyFromPHWC4Buffer(const GlBuffer& buffer, TensorFloat32* tensor);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_OBJECT_MANAGER_H_
