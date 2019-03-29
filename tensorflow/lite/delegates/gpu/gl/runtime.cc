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

#include "tensorflow/lite/delegates/gpu/gl/runtime.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

struct TextureF16Maker {
  Status operator()(const uint3& size) const {
    return CreateReadOnlyImageTextureF16(size, data, gl_texture);
  }
  Status operator()(const uint2& size) const {
    return CreateReadOnlyImageTextureF16(size, data, gl_texture);
  }
  Status operator()(const uint32_t& size) const {
    return CreateReadOnlyImageTextureF16(uint2(size, 1U), data, gl_texture);
  }
  absl::Span<const uint16_t> data;
  GlTexture* gl_texture;
};

struct TextureF32Maker {
  Status operator()(const uint3& size) const {
    return CreateReadOnlyImageTexture(size, data, gl_texture);
  }
  Status operator()(const uint2& size) const {
    return CreateReadOnlyImageTexture(size, data, gl_texture);
  }
  Status operator()(const uint32_t& size) const {
    return CreateReadOnlyImageTexture(uint2(size, 1U), data, gl_texture);
  }
  absl::Span<const float> data;
  GlTexture* gl_texture;
};

Status MakeGlTexture(const Object& object, const ObjectData& data,
                     GlTexture* gl_texture) {
  if (object.access == AccessType::READ_WRITE ||
      object.access == AccessType::WRITE) {
    return InvalidArgumentError("Read-write textures are not supported");
  }
  if (object.data_type != DataType::FLOAT16 &&
      object.data_type != DataType::FLOAT32) {
    return InvalidArgumentError("Textures support float16 or float32 only.");
  }
  switch (object.data_type) {
    case DataType::FLOAT16: {
      if (data.size() % 2 != 0) {
        return InvalidArgumentError("Texture size is not aligned");
      }
      return absl::visit(
          TextureF16Maker{
              .data = absl::MakeConstSpan(
                  reinterpret_cast<const uint16_t*>(data.data()),
                  data.size() / 2),
              .gl_texture = gl_texture,
          },
          object.size);
    }
    case DataType::FLOAT32: {
      if (data.size() % sizeof(float) != 0) {
        return InvalidArgumentError("Texture size is not aligned");
      }
      return absl::visit(
          TextureF32Maker{
              .data = absl::MakeConstSpan(
                  reinterpret_cast<const float*>(data.data()),
                  data.size() / sizeof(float)),
              .gl_texture = gl_texture,
          },
          object.size);
    }
    default:
      return InvalidArgumentError("Unsupported textures data type.");
  }
}

struct TextureRefMaker {
  Status operator()(const uint3& size) const {
    return CreateReadWriteRgbaImageTexture(type, size, gl_texture);
  }
  Status operator()(const uint2& size) const {
    return CreateReadWriteRgbaImageTexture(type, size, gl_texture);
  }
  Status operator()(const uint32_t& size) const {
    return CreateReadWriteRgbaImageTexture(type, uint2(size, 1U), gl_texture);
  }
  DataType type;
  GlTexture* gl_texture;
};

// Makes read-write gl texture
Status MakeGlTextureRef(const Object& object, GlTexture* gl_texture) {
  return absl::visit(TextureRefMaker{object.data_type, gl_texture},
                     object.size);
}

Status MakeGlBuffer(const Object& object, const ObjectData& data,
                    GlBuffer* gl_buffer) {
  if (data.size() % SizeOf(object.data_type) != 0) {
    return InvalidArgumentError("Buffer size is not aligned");
  }
  return CreateReadOnlyShaderStorageBuffer(absl::MakeConstSpan(data),
                                           gl_buffer);
}

// Looks up an object with the given id. If found, makes a binding function.
Status MakeBindingFunc(const Object& object, uint32_t id,
                       const ObjectManager& objects,
                       std::function<Status()>* binding_func) {
  const uint32_t binding = object.binding;
  switch (object.object_type) {
    case ObjectType::BUFFER: {
      auto ptr = objects.FindBuffer(id);
      if (!ptr) {
        return NotFoundError(absl::StrCat("Buffer ", id, " is not found"));
      }

      // Validate buffer.
      size_t size_in_bytes = ByteSizeOf(object);
      // TODO(akulik): make comparison != instead of <
      if (ptr->bytes_size() < size_in_bytes) {
        return FailedPreconditionError(
            absl::StrCat("Buffer ", id, " size in bytes ", ptr->bytes_size(),
                         " < requested size_in_bytes ", size_in_bytes));
      }
      *binding_func = [=]() { return ptr->BindToIndex(binding); };
      break;
    }
    case ObjectType::TEXTURE: {
      auto ptr = objects.FindTexture(id);
      if (!ptr) {
        return NotFoundError(absl::StrCat("Texture ", id, " is not found"));
      }
      *binding_func = [=]() { return ptr->BindAsReadWriteImage(binding); };
      break;
    }
    case ObjectType::UNKNOWN:
      return InvalidArgumentError("Unknown object type");
  }
  return OkStatus();
}

}  // namespace

Runtime::Runtime(const RuntimeOptions& options, const GpuInfo& gpu_info,
                 CommandQueue* command_queue,
                 const ObjectManager* external_objects)
    : options_(options),
      gpu_info_(gpu_info),
      external_objects_(external_objects),
      command_queue_(command_queue) {
  programs_.reserve(256);
  if (options_.bundle_readonly_objects) {
    shared_readonly_buffer_ = absl::make_unique<SharedBufferData>();
  }
}

Status Runtime::AddProgram(const GlShader& shader,
                           const std::vector<UniformParameter>& parameters,
                           const std::vector<Object>& objects,
                           const uint3& num_workgroups) {
  GlProgram program;
  RETURN_IF_ERROR(GlProgram::CreateWithShader(shader, &program));

  for (auto& parameter : parameters) {
    RETURN_IF_ERROR(program.SetParameter(parameter));
  }

  programs_.emplace_back(
      CompiledProgramDescriptor{std::move(program), num_workgroups, {}});

  // Create const buffers, resolve external references and collect internal
  // buffer references.
  for (auto& object : objects) {
    auto& program = programs_.back();
    BindFunc binding_func;
    if (IsRef(object)) {
      // Reference object could be provided externally as a model input/output
      // but also for debugging purposes. Otherwise all references are collected
      // and allocated later.
      Status status = MakeBindingFunc(object, GetRef(object),
                                      *external_objects_, &binding_func);
      if (!status.ok()) {
        if (status.code() == StatusCode::kNotFound) {
          program.refs.push_back(object);
          continue;  // don't add to binding.
        }
        return status;
      }
    } else {
      // Allocate const object.
      uint32_t id;
      RETURN_IF_ERROR(AllocateConstObject(object, &id));
      RETURN_IF_ERROR(
          MakeBindingFunc(object, id, const_objects_, &binding_func));
    }
    program.bindings.push_back(std::move(binding_func));
  }

  // All parameters once set stay with program, therefore, we only need to keep
  // program and bindings for execution.
  return OkStatus();
}

Status Runtime::AllocateInternalObject(const Object& object) {
  const ObjectRef ref = GetRef(object);
  switch (object.object_type) {
    case ObjectType::BUFFER: {
      GlBuffer gl_buffer;
      RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<uint8_t>(
          ByteSizeOf(object), &gl_buffer));
      RETURN_IF_ERROR(
          internal_objects_.RegisterBuffer(ref, std::move(gl_buffer)));
      break;
    }
    case ObjectType::TEXTURE: {
      GlTexture gl_texture;
      RETURN_IF_ERROR(MakeGlTextureRef(object, &gl_texture));
      RETURN_IF_ERROR(
          internal_objects_.RegisterTexture(ref, std::move(gl_texture)));
      break;
    }
    default:
      return InternalError("Unexpected internal object type");
  }
  return OkStatus();
}

Status Runtime::AllocateConstObject(const Object& object, uint32_t* id) {
  const ObjectData* data = GetData(object);
  if (data == nullptr) {
    return InternalError("Unable to allocate reference as a const object");
  }
  *id = next_const_id_++;
  switch (object.object_type) {
    case ObjectType::BUFFER: {
      GlBuffer gl_buffer;
      if (!shared_readonly_buffer_ ||
          !shared_readonly_buffer_->Add(*data, &gl_buffer)) {
        RETURN_IF_ERROR(MakeGlBuffer(object, *data, &gl_buffer));
      }
      RETURN_IF_ERROR(const_objects_.RegisterBuffer(*id, std::move(gl_buffer)));
      break;
    }
    case ObjectType::TEXTURE: {
      GlTexture gl_texture;
      RETURN_IF_ERROR(MakeGlTexture(object, *data, &gl_texture));
      RETURN_IF_ERROR(
          const_objects_.RegisterTexture(*id, std::move(gl_texture)));
      break;
    }
    case ObjectType::UNKNOWN:
      return InternalError("Unknown object type");
  }
  return OkStatus();
}

Status Runtime::PrepareForExecution() {
  if (shared_readonly_buffer_ && !shared_readonly_buffer_->empty()) {
    GlBuffer shared_buffer;
    RETURN_IF_ERROR(
        shared_readonly_buffer_->CreateSharedGlBuffer(&shared_buffer));
    shared_readonly_buffer_.reset(nullptr);
    RETURN_IF_ERROR(const_objects_.RegisterBuffer(next_const_id_++,
                                                  std::move(shared_buffer)));
  }

  if (options_.reuse_internal_objects) {
    // Analyze internal objects and make a pool of shared objects to be re-used
    // by them. These shared objects need to be allocated upfront.
    std::vector<Object> shared_objects;
    RETURN_IF_ERROR(AssignInternalObjects(&shared_objects));
    for (const Object& object : shared_objects) {
      RETURN_IF_ERROR(AllocateInternalObject(object));
    }
  }

  // Allocate all internal objects and create bindings for them.
  for (auto& program : programs_) {
    for (auto& object : program.refs) {
      // Check whether it is created already.
      BindFunc binding;
      ObjectRef ref = GetRef(object);
      Status status = MakeBindingFunc(object, ref, internal_objects_, &binding);
      if (!status.ok()) {
        if (status.code() != StatusCode::kNotFound) {
          return status;
        }
        RETURN_IF_ERROR(AllocateInternalObject(object));
        RETURN_IF_ERROR(
            MakeBindingFunc(object, ref, internal_objects_, &binding));
      }
      program.bindings.push_back(std::move(binding));
    }
    program.refs.clear();
  }
  return OkStatus();
}

namespace {

struct FitSizeFunc {
  bool operator()(const uint3& size) const {
    auto s = absl::get_if<uint3>(&b);
    if (!s) return false;
    *result = uint3(std::max(s->x, size.x), std::max(s->y, size.y),
                    std::max(s->z, size.z));
    return true;
  }

  bool operator()(const uint2& size) const {
    auto s = absl::get_if<uint2>(&b);
    if (!s) return false;
    *result = uint2(std::max(s->x, size.x), std::max(s->y, size.y));
    return true;
  }

  bool operator()(uint32_t size) const {
    auto s = absl::get_if<uint32_t>(&b);
    if (!s) return false;
    *result = std::max(*s, size);
    return true;
  }

  const ObjectSize& b;
  ObjectSize* result;
};

// Makes new size which combines largest dimensions of both given sizes.
//
// @return false if sizes have different number of dimensions
bool FitSize(const ObjectSize& a, const ObjectSize& b, ObjectSize* result) {
  return absl::visit(FitSizeFunc{b, result}, a);
}

// Texture fitting policy is:
//  - 1D: source texture will always fit into target because it is linear
//  - 2D: source texture should fit without growing target texture
//  - 3D: source texture should fit without growing target texture
//
struct TextureFitPolicy {
  bool operator()(const uint3& size) const {
    auto s = absl::get_if<uint3>(&target);
    return s && size.x <= s->x && size.y <= s->y && size.z <= s->z;
  }

  bool operator()(const uint2& size) const {
    auto s = absl::get_if<uint2>(&target);
    return s && size.x <= s->x && size.y <= s->y;
  }

  bool operator()(uint32_t size) const {
    return absl::get_if<uint32_t>(&target);
  }

  const ObjectSize& target;
};

// Makes new size which combines largest dimensions of both given sizes.
//
// @return false if sizes have different number of dimensions
bool WillTextureFit(const ObjectSize& source, const ObjectSize& target) {
  return absl::visit(TextureFitPolicy{target}, source);
}

struct TextureNumElementsFunc {
  size_t operator()(const uint3& size) const {
    auto s = absl::get_if<uint3>(&target);
    return s ? size.z * s->x * s->y + size.y * s->x + size.x : 0;
  }

  size_t operator()(const uint2& size) const {
    auto s = absl::get_if<uint2>(&target);
    return s ? size.y * s->x + size.x : 0;
  }

  size_t operator()(uint32_t size) const {
    auto s = absl::get_if<uint32_t>(&target);
    return s ? size : 0;
  }

  const ObjectSize& target;
};

// @return estimated number of elements if target texture is used to keep source
// texture data assuming XYZ layout.
size_t TextureNumElements(const ObjectSize& source, const ObjectSize& target) {
  return absl::visit(TextureNumElementsFunc{target}, source);
}

// Checks whether the given object fits into 'to' object. Returns number of
// bytes used if an object fits, or 0 otherwise.
//
// Fitting policy:
//   - buffer will always fit into another buffer because they all are linear.
//   - textures are handles by the policy above
//
size_t WillItFit(const Object& object, const Object& to) {
  if (object.object_type != to.object_type ||
      object.data_type != to.data_type) {
    return 0;
  }
  switch (object.object_type) {
    case ObjectType::BUFFER:
      return ByteSizeOf(object);
    case ObjectType::TEXTURE: {
      if (!WillTextureFit(object.size, to.size)) return 0;
      // Expand 'to' dimensions to ensure an object fits.
      ObjectSize new_texture_size;
      if (!FitSize(object.size, to.size, &new_texture_size)) return 0;
      return /* RGBA = */ 4 * SizeOf(object.data_type) *
             TextureNumElements(object.size, new_texture_size);
    }
    default:
      return 0;
  }
}

}  // namespace

// Algorithm works as follows:
//
//   1. First it collects usage intervals for each object reference.
//      For example: buffer #3 is introduced in program #2 and used for the
//      last time in program #7.
//
//   2. Iterates through all programs where for every object reference
//      assigns shared object from the pool. When object reference is used
//      for the last time, corresponding shared object is returned back to
//      the pool.
//
//   3. Shared object pool grows when there are no free shared object
//      available.
//
//   4. Shared object size may increase when object reference requests bigger
//      size.
//
// Therefore, in the end all references are remapped to ids in the range
// [0..num_shared_objects]. To avoid ref space collision with global reference
// all shared objects are allocated in internal_objects_.
Status Runtime::AssignInternalObjects(std::vector<Object>* shared_objects) {
  // Build interval set for objects to know where each object is introduced
  // and used for the last time.
  std::vector<std::pair<int32_t, int32_t>> usage_intervals;
  for (int32_t i = 0; i < programs_.size(); ++i) {
    for (auto& object : programs_[i].refs) {
      auto ref = GetRef(object);
      if (ref >= usage_intervals.size()) {
        usage_intervals.resize(ref + 1, std::make_pair(programs_.size(), -1));
      }
      auto& it = usage_intervals[ref];
      it.first = std::min(it.first, i);
      it.second = std::max(it.second, i);
    }
  }

  std::vector<bool> is_used_shared_object;
  std::vector<ObjectRef> global_ref_to_shared_ref(usage_intervals.size(),
                                                  kInvalidObjectRef);

  for (size_t i = 0; i < programs_.size(); ++i) {
    auto& program = programs_[i];
    // list of object indices to return to the pool.
    std::vector<ObjectRef> object_refs_to_return;

    // Assign to every internal buffer, that is not yet allocated, appropriate
    // shared buffer from a heap of unused.
    for (auto& object : program.refs) {
      const ObjectRef ref = GetRef(object);
      ObjectRef shared_ref = global_ref_to_shared_ref[ref];
      const auto& usage = usage_intervals[ref];

      if (usage.first == i) {
        // First time a reference is introduced. Assign shared object.
        if (shared_ref != kInvalidObjectRef) {
          return InternalError(
              "Internal object is introduced for the first time but is already "
              "assigned");
        }

        // Try to find a free shared object that is as close as possible by
        // size. Here we assume that number of shared objects is relatively
        // small (< 100), therefore, search linearly over all of them.
        size_t selected_waste_bytes = 0;
        for (int32_t b = 0; b < shared_objects->size(); ++b) {
          // Check whether shared object is available.
          if (is_used_shared_object[b]) continue;
          auto& shared_object = (*shared_objects)[b];

          // Bytes needed to fit object in the shared object.
          size_t alloc_bytes = WillItFit(object, shared_object);
          if (alloc_bytes == 0) continue;

          // Prefer shared object that will waste less memory.
          size_t shared_byte_size = ByteSizeOf(shared_object);
          // sizes are unsigned, therefore '-' may undeflow. Take smallest.
          size_t waste_bytes = std::min(shared_byte_size - alloc_bytes,
                                        alloc_bytes - shared_byte_size);
          if (shared_ref == kInvalidObjectRef ||
              waste_bytes < selected_waste_bytes) {
            selected_waste_bytes = waste_bytes;
            shared_ref = b;
          }
        }

        if (shared_ref == kInvalidObjectRef) {
          // Didn't find an object to share. Create new one.
          shared_ref = shared_objects->size();
          Object shared_object = object;
          shared_object.access = AccessType::READ_WRITE;
          shared_object.object = shared_ref;
          if (shared_object.object_type == ObjectType::BUFFER) {
            // Make a buffer linear.
            shared_object.size = NumElements(object.size);
          }
          shared_objects->push_back(std::move(shared_object));
          is_used_shared_object.push_back(false);
        } else {
          // Check chosen shared object and update it's size.
          Object& shared_object = (*shared_objects)[shared_ref];
          switch (object.object_type) {
            case ObjectType::BUFFER:
              shared_object.size = std::max(NumElements(object.size),
                                            NumElements(shared_object.size));
              break;
            case ObjectType::TEXTURE: {
              if (!FitSize(object.size, shared_object.size,
                           &shared_object.size)) {
                return InternalError(
                    "Already assigned shared texture does not fit an object");
              }
              break;
            }
            default:
              return InternalError("Unexpected shared object type");
          }
        }
      }

      // Mark shared object as used and map internal object to it.
      is_used_shared_object[shared_ref] = true;
      global_ref_to_shared_ref[ref] = shared_ref;
      object.object = shared_ref;

      // At this point we want to return unused object, but it should be
      // returned later to avoid re-using the same object in this operation
      // for a different purpose.
      if (usage.second == i) {
        object_refs_to_return.push_back(shared_ref);
      }
    }

    // Mark all returned objects from this program as unused.
    for (size_t ref : object_refs_to_return) {
      is_used_shared_object[ref] = false;
    }
  }
  return OkStatus();
}

Status Runtime::Execute() {
  for (const auto& descriptor : programs_) {
    for (auto& b : descriptor.bindings) {
      RETURN_IF_ERROR(b());
    }
    RETURN_IF_ERROR(command_queue_->Dispatch(descriptor.program,
                                             descriptor.num_workgroups));
  }
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
