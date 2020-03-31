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
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management/types.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_errors.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

struct TextureF16Maker {
  absl::Status operator()(const uint3& size) const {
    return CreateReadOnlyImageTextureF16(size, data, gl_texture);
  }
  absl::Status operator()(const uint2& size) const {
    return CreateReadOnlyImageTextureF16(size, data, gl_texture);
  }
  absl::Status operator()(const size_t& size) const {
    return CreateReadOnlyImageTextureF16(uint2(static_cast<uint32_t>(size), 1U),
                                         data, gl_texture);
  }
  absl::Span<const uint16_t> data;
  GlTexture* gl_texture;
};

struct TextureF32Maker {
  absl::Status operator()(const uint3& size) const {
    return CreateReadOnlyImageTexture(size, data, gl_texture);
  }
  absl::Status operator()(const uint2& size) const {
    return CreateReadOnlyImageTexture(size, data, gl_texture);
  }
  absl::Status operator()(const size_t& size) const {
    return CreateReadOnlyImageTexture(uint2(static_cast<uint32_t>(size), 1U),
                                      data, gl_texture);
  }
  absl::Span<const float> data;
  GlTexture* gl_texture;
};

absl::Status MakeGlTexture(const Object& object, const ObjectData& data,
                           GlTexture* gl_texture) {
  if (object.access == AccessType::READ_WRITE ||
      object.access == AccessType::WRITE) {
    return absl::InvalidArgumentError("Read-write textures are not supported");
  }
  if (object.data_type != DataType::FLOAT16 &&
      object.data_type != DataType::FLOAT32) {
    return absl::InvalidArgumentError(
        "Textures support float16 or float32 only.");
  }
  switch (object.data_type) {
    case DataType::FLOAT16: {
      if (data.size() % 2 != 0) {
        return absl::InvalidArgumentError("Texture size is not aligned");
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
        return absl::InvalidArgumentError("Texture size is not aligned");
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
      return absl::InvalidArgumentError("Unsupported textures data type.");
  }
}

struct TextureRefMaker {
  absl::Status operator()(const uint3& size) const {
    return CreateReadWriteRgbaImageTexture(type, size, gl_texture);
  }
  absl::Status operator()(const uint2& size) const {
    return CreateReadWriteRgbaImageTexture(type, size, gl_texture);
  }
  absl::Status operator()(const size_t& size) const {
    return CreateReadWriteRgbaImageTexture(
        type, uint2(static_cast<uint32_t>(size), 1U), gl_texture);
  }
  DataType type;
  GlTexture* gl_texture;
};

// Makes read-write gl texture
absl::Status MakeGlTextureRef(const Object& object, GlTexture* gl_texture) {
  return absl::visit(TextureRefMaker{object.data_type, gl_texture},
                     object.size);
}

absl::Status MakeGlBuffer(const Object& object, const ObjectData& data,
                          GlBuffer* gl_buffer) {
  if (data.size() % SizeOf(object.data_type) != 0) {
    return absl::InvalidArgumentError("Buffer size is not aligned");
  }
  return CreateReadOnlyShaderStorageBuffer(absl::MakeConstSpan(data),
                                           gl_buffer);
}

// Looks up an object with the given id. If found, makes a binding function.
absl::Status MakeBindingFunc(const Object& object, uint32_t id,
                             const ObjectManager& objects,
                             std::function<absl::Status()>* binding_func) {
  const uint32_t binding = object.binding;
  switch (object.object_type) {
    case ObjectType::BUFFER: {
      auto ptr = objects.FindBuffer(id);
      if (!ptr) {
        return absl::NotFoundError(
            absl::StrCat("Buffer ", id, " is not found"));
      }

      // Validate buffer.
      size_t size_in_bytes = ByteSizeOf(object);
      // TODO(akulik): make comparison != instead of <
      if (ptr->bytes_size() < size_in_bytes) {
        return absl::FailedPreconditionError(
            absl::StrCat("Buffer ", id, " size in bytes ", ptr->bytes_size(),
                         " < requested size_in_bytes ", size_in_bytes));
      }
      *binding_func = [=]() { return ptr->BindToIndex(binding); };
      break;
    }
    case ObjectType::TEXTURE: {
      auto ptr = objects.FindTexture(id);
      if (!ptr) {
        return absl::NotFoundError(
            absl::StrCat("Texture ", id, " is not found"));
      }
      *binding_func = [=]() { return ptr->BindAsReadWriteImage(binding); };
      break;
    }
    case ObjectType::UNKNOWN:
      return absl::InvalidArgumentError("Unknown object type");
  }
  return absl::OkStatus();
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

absl::Status Runtime::AddProgram(const GlShader& shader,
                                 const std::vector<Variable>& parameters,
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
      absl::Status status = MakeBindingFunc(object, GetRef(object),
                                            *external_objects_, &binding_func);
      if (!status.ok()) {
        if (absl::IsNotFound(status)) {
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
  return absl::OkStatus();
}

absl::Status Runtime::AllocateInternalObject(const Object& object) {
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
      return absl::InternalError("Unexpected internal object type");
  }
  return absl::OkStatus();
}

absl::Status Runtime::AllocateConstObject(const Object& object, uint32_t* id) {
  const ObjectData* data = GetData(object);
  if (data == nullptr) {
    return absl::InternalError(
        "Unable to allocate reference as a const object");
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
      return absl::InternalError("Unknown object type");
  }
  return absl::OkStatus();
}

absl::Status Runtime::PrepareForExecution() {
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
      absl::Status status =
          MakeBindingFunc(object, ref, internal_objects_, &binding);
      if (!status.ok()) {
        if (absl::IsNotFound(status)) return status;
        RETURN_IF_ERROR(AllocateInternalObject(object));
        RETURN_IF_ERROR(
            MakeBindingFunc(object, ref, internal_objects_, &binding));
      }
      program.bindings.push_back(std::move(binding));
    }
    program.refs.clear();
  }
  return absl::OkStatus();
}

namespace {

const size_t kNotAssigned = std::numeric_limits<size_t>::max();

struct CombinedUsageRecords {
  std::vector<TensorUsageRecord<size_t>> buffers;
  std::vector<TensorUsageRecord<size_t>> textures_1d;
  std::vector<TensorUsageRecord<uint2>> textures_2d;
  std::vector<TensorUsageRecord<uint3>> textures_3d;
  std::vector<size_t> usage_refs;
};

template <typename TensorSizeT>
void UpdateUsageRecord(TensorUsageRecord<TensorSizeT>* usage_rec,
                       size_t task_id) {
  usage_rec->first_task = std::min(usage_rec->first_task, task_id);
  usage_rec->last_task = std::max(usage_rec->last_task, task_id);
}

struct AddUsageRecordForTextureFunc {
  void operator()(const uint3& size) const {
    auto& usage_ref = usage_records->usage_refs[object_ref];
    if (usage_ref == kNotAssigned) {
      usage_ref = usage_records->textures_3d.size();
      usage_records->textures_3d.emplace_back(/*tensor_size=*/size,
                                              /*first_task=*/program_id,
                                              /*last_task=*/program_id);
    } else {
      UpdateUsageRecord(&usage_records->textures_3d[usage_ref], program_id);
    }
  }

  void operator()(const uint2& size) const {
    auto& usage_ref = usage_records->usage_refs[object_ref];
    if (usage_ref == kNotAssigned) {
      usage_ref = usage_records->textures_2d.size();
      usage_records->textures_2d.emplace_back(/*tensor_size=*/size,
                                              /*first_task=*/program_id,
                                              /*last_task=*/program_id);
    } else {
      UpdateUsageRecord(&usage_records->textures_2d[usage_ref], program_id);
    }
  }

  void operator()(size_t size) const {
    auto& usage_ref = usage_records->usage_refs[object_ref];
    if (usage_ref == kNotAssigned) {
      usage_ref = usage_records->textures_1d.size();
      usage_records->textures_1d.emplace_back(/*tensor_size=*/size,
                                              /*first_task=*/program_id,
                                              /*last_task=*/program_id);
    } else {
      UpdateUsageRecord(&usage_records->textures_1d[usage_ref], program_id);
    }
  }

  CombinedUsageRecords* usage_records;
  const ObjectRef& object_ref;
  const size_t program_id;
};

// We assume that AddUsageRecord for different objects is called in order of
// program_id.
absl::Status AddUsageRecord(CombinedUsageRecords* usage_records,
                            const Object& object, const size_t program_id) {
  auto ref = GetRef(object);
  if (ref >= usage_records->usage_refs.size()) {
    usage_records->usage_refs.resize(ref + 1, kNotAssigned);
  }
  auto& usage_ref = usage_records->usage_refs[ref];
  if (object.object_type == ObjectType::BUFFER) {
    if (usage_ref == kNotAssigned) {
      usage_ref = usage_records->buffers.size();
      usage_records->buffers.emplace_back(
          /*tensor_size=*/NumElements(object.size),
          /*first_task=*/program_id,
          /*last_task=*/program_id);
    } else {
      UpdateUsageRecord(&usage_records->buffers[usage_ref], program_id);
    }
    return absl::OkStatus();
  }
  if (object.object_type == ObjectType::TEXTURE) {
    absl::visit(AddUsageRecordForTextureFunc{usage_records, ref, program_id},
                object.size);
    return absl::OkStatus();
  }
  return absl::InternalError("Unexpected object type");
}

absl::Status ApplyBuffersAssignment(
    const ObjectsAssignment<size_t>& assignment,
    const std::vector<size_t>& global_ref_to_usage_rec,
    const std::vector<Object*>& global_ref_to_object_ptr,
    std::vector<ObjectRef>* global_ref_to_shared_ref,
    std::vector<Object>* shared_objects) {
  std::vector<ObjectRef> assigned_id_to_shared_ref(
      assignment.object_sizes.size(), kInvalidObjectRef);
  for (size_t global_ref = 0; global_ref < global_ref_to_usage_rec.size();
       ++global_ref) {
    const auto& usage_rec_id = global_ref_to_usage_rec[global_ref];
    Object* object = global_ref_to_object_ptr[global_ref];
    if (usage_rec_id == kNotAssigned || object == nullptr ||
        object->object_type != ObjectType::BUFFER) {
      // Skip objects with other data type and non-buffers.
      continue;
    }

    // id of shared object, returned by memory allocation algorithm.
    size_t assigned_id = assignment.object_ids[usage_rec_id];

    // id of corresponding shared object in vector share_objects.
    ObjectRef shared_ref = assigned_id_to_shared_ref[assigned_id];

    if (shared_ref == kInvalidObjectRef) {
      // We need to create new shared object for current buffer.
      shared_ref = shared_objects->size();
      Object shared_object = *object;
      shared_object.access = AccessType::READ_WRITE;
      shared_object.object = shared_ref;
      shared_object.size = assignment.object_sizes[assigned_id];
      shared_objects->push_back(std::move(shared_object));
      assigned_id_to_shared_ref[assigned_id] = shared_ref;
    }
    (*global_ref_to_shared_ref)[global_ref] = shared_ref;
  }
  return absl::OkStatus();
}

template <typename ObjectSizeT>
absl::Status ApplyTexturesAssignment(
    const ObjectsAssignment<ObjectSizeT>& assignment,
    const std::vector<size_t>& global_ref_to_usage_rec,
    const std::vector<Object*>& global_ref_to_object_ptr,
    std::vector<ObjectRef>* global_ref_to_shared_ref,
    std::vector<Object>* shared_objects) {
  std::vector<ObjectRef> assigned_id_to_shared_ref(
      assignment.object_sizes.size(), kInvalidObjectRef);
  for (size_t global_ref = 0; global_ref < global_ref_to_usage_rec.size();
       ++global_ref) {
    const auto& usage_rec_id = global_ref_to_usage_rec[global_ref];
    Object* object = global_ref_to_object_ptr[global_ref];
    if (usage_rec_id == kNotAssigned || object == nullptr ||
        object->object_type != ObjectType::TEXTURE ||
        !absl::get_if<ObjectSizeT>(&object->size)) {
      // Skip objects with other data type, non-textures and textures with wrong
      // number of dimensions.
      continue;
    }

    // id of shared object, returned by memory allocation algorithm.
    size_t assigned_id = assignment.object_ids[usage_rec_id];

    // id of corresponding shared object in vector share_objects.
    ObjectRef shared_ref = assigned_id_to_shared_ref[assigned_id];

    if (shared_ref == kInvalidObjectRef) {
      // We need to create new shared object for current texture.
      shared_ref = shared_objects->size();
      Object shared_object = *object;
      shared_object.access = AccessType::READ_WRITE;
      shared_object.object = shared_ref;
      shared_object.size = assignment.object_sizes[assigned_id];
      shared_objects->push_back(std::move(shared_object));
      assigned_id_to_shared_ref[assigned_id] = shared_ref;
    }
    (*global_ref_to_shared_ref)[global_ref] = shared_ref;
  }
  return absl::OkStatus();
}

}  // namespace

// Assign shared objects to internal objects, using memory allocation
// algorithms. Usage records for the algorithms are calculated separately for
// each data type and object type.
absl::Status Runtime::AssignInternalObjects(
    std::vector<Object>* shared_objects) {
  // Build tensor usage records, clusterized by object type and data type.
  std::map<DataType, CombinedUsageRecords> usage_records_by_data_type;
  std::vector<Object*> global_ref_to_object_ptr;
  for (size_t i = 0; i < programs_.size(); ++i) {
    for (auto& object : programs_[i].refs) {
      auto ref = GetRef(object);
      if (ref >= global_ref_to_object_ptr.size()) {
        global_ref_to_object_ptr.resize(ref + 1, nullptr);
      }
      if (global_ref_to_object_ptr[ref] == nullptr) {
        global_ref_to_object_ptr[ref] = &object;
      }
      RETURN_IF_ERROR(AddUsageRecord(
          &usage_records_by_data_type[object.data_type], object, i));
    }
  }

  std::vector<ObjectRef> global_ref_to_shared_ref(
      global_ref_to_object_ptr.size(), kInvalidObjectRef);

  // Calculate and apply shared objects assignment for each data type.
  for (const auto& it : usage_records_by_data_type) {
    const CombinedUsageRecords& usage_records = it.second;
    if (!usage_records.buffers.empty()) {
      ObjectsAssignment<size_t> buffer_assignment;
      RETURN_IF_ERROR(AssignObjectsToTensors(usage_records.buffers,
                                             MemoryStrategy::GREEDY_BEST,
                                             &buffer_assignment));
      RETURN_IF_ERROR(ApplyBuffersAssignment(
          buffer_assignment, usage_records.usage_refs, global_ref_to_object_ptr,
          &global_ref_to_shared_ref, shared_objects));
    }
    if (!usage_records.textures_1d.empty()) {
      ObjectsAssignment<size_t> texture_1d_assignment;
      RETURN_IF_ERROR(AssignObjectsToTensors(usage_records.textures_1d,
                                             MemoryStrategy::GREEDY_BEST,
                                             &texture_1d_assignment));
      RETURN_IF_ERROR(ApplyTexturesAssignment(
          texture_1d_assignment, usage_records.usage_refs,
          global_ref_to_object_ptr, &global_ref_to_shared_ref, shared_objects));
    }
    if (!usage_records.textures_2d.empty()) {
      ObjectsAssignment<uint2> texture_2d_assignment;
      RETURN_IF_ERROR(AssignObjectsToTensors(usage_records.textures_2d,
                                             MemoryStrategy::GREEDY_IN_ORDER,
                                             &texture_2d_assignment));
      RETURN_IF_ERROR(ApplyTexturesAssignment(
          texture_2d_assignment, usage_records.usage_refs,
          global_ref_to_object_ptr, &global_ref_to_shared_ref, shared_objects));
    }
    if (!usage_records.textures_3d.empty()) {
      ObjectsAssignment<uint3> texture_3d_assignment;
      RETURN_IF_ERROR(AssignObjectsToTensors(usage_records.textures_3d,
                                             MemoryStrategy::GREEDY_IN_ORDER,
                                             &texture_3d_assignment));
      RETURN_IF_ERROR(ApplyTexturesAssignment(
          texture_3d_assignment, usage_records.usage_refs,
          global_ref_to_object_ptr, &global_ref_to_shared_ref, shared_objects));
    }
  }

  for (size_t i = 0; i < programs_.size(); ++i) {
    for (auto& object : programs_[i].refs) {
      object.object = global_ref_to_shared_ref[GetRef(object)];
    }
  }
  return absl::OkStatus();
}

absl::Status Runtime::Execute() {
  for (const auto& descriptor : programs_) {
    for (auto& b : descriptor.bindings) {
      RETURN_IF_ERROR(b());
    }
    RETURN_IF_ERROR(command_queue_->Dispatch(descriptor.program,
                                             descriptor.num_workgroups));
  }
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
