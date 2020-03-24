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

#include "tensorflow/lite/delegates/gpu/gl/serialization.h"

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

using flatbuffers::Offset;
using flatbuffers::Vector;

namespace {

struct ParameterValueGetter {
  Offset<void> operator()(int32_t value) {
    auto offset = builder->CreateVector(std::vector<int32_t>{value});
    data::DataInt32Builder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(const int2& value) {
    auto offset = builder->CreateVector(std::vector<int32_t>{value.x, value.y});
    data::DataInt32Builder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(const int4& value) {
    auto offset = builder->CreateVector(
        std::vector<int32_t>{value.x, value.y, value.z, value.w});
    data::DataInt32Builder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(const std::vector<int2>& value) {
    std::vector<int32_t> d(value.size() * 2);
    for (size_t i = 0; i < value.size(); ++i) {
      d[i * 2] = value[i].x;
      d[i * 2 + 1] = value[i].y;
    }
    auto offset = builder->CreateVector(d);
    data::DataInt32Builder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(uint32_t value) {
    auto offset = builder->CreateVector(std::vector<uint32_t>{value});
    data::DataUint32Builder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(const uint4& value) {
    auto offset = builder->CreateVector(
        std::vector<uint32_t>{value.x, value.y, value.z, value.w});
    data::DataUint32Builder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(float value) {
    auto offset = builder->CreateVector(std::vector<float>{value});
    data::DataFloatBuilder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(const float2& value) {
    auto offset = builder->CreateVector(std::vector<float>{value.x, value.y});
    data::DataFloatBuilder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(const float4& value) {
    auto offset = builder->CreateVector(
        std::vector<float>{value.x, value.y, value.z, value.w});
    data::DataFloatBuilder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  Offset<void> operator()(const std::vector<float4>& value) {
    std::vector<float> d(value.size() * 4);
    for (size_t i = 0; i < value.size(); ++i) {
      d[i * 4] = value[i].x;
      d[i * 4 + 1] = value[i].y;
      d[i * 4 + 2] = value[i].z;
      d[i * 4 + 3] = value[i].w;
    }
    auto offset = builder->CreateVector(d);
    data::DataFloatBuilder data(*builder);
    data.add_data(offset);
    return data.Finish().Union();
  }

  ::flatbuffers::FlatBufferBuilder* builder;
};

struct DataVariantTypeGetter {
  data::DataVariant operator()(int32_t) const {
    return data::DataVariant::DataInt32;
  }

  data::DataVariant operator()(const int2&) const {
    return data::DataVariant::DataInt32;
  }

  data::DataVariant operator()(const int4&) const {
    return data::DataVariant::DataInt32;
  }

  data::DataVariant operator()(const std::vector<int2>&) const {
    return data::DataVariant::DataInt32;
  }

  data::DataVariant operator()(uint32_t) const {
    return data::DataVariant::DataUint32;
  }

  data::DataVariant operator()(const uint4&) const {
    return data::DataVariant::DataUint32;
  }

  data::DataVariant operator()(float) const {
    return data::DataVariant::DataFloat;
  }

  data::DataVariant operator()(const float2&) const {
    return data::DataVariant::DataFloat;
  }

  data::DataVariant operator()(const float4&) const {
    return data::DataVariant::DataFloat;
  }

  data::DataVariant operator()(const std::vector<float4>&) const {
    return data::DataVariant::DataFloat;
  }
};

struct ParameterTypeGetter {
  data::ParameterType operator()(int32_t) const {
    return data::ParameterType::INT32;
  }

  data::ParameterType operator()(const int2&) const {
    return data::ParameterType::INT32;
  }

  data::ParameterType operator()(const int4&) const {
    return data::ParameterType::INT32;
  }

  data::ParameterType operator()(const std::vector<int2>&) const {
    return data::ParameterType::INT32_2;
  }

  data::ParameterType operator()(uint32_t) const {
    return data::ParameterType::UINT32;
  }

  data::ParameterType operator()(const uint4&) const {
    return data::ParameterType::UINT32;
  }

  data::ParameterType operator()(float) const {
    return data::ParameterType::FLOAT32;
  }

  data::ParameterType operator()(const float2&) const {
    return data::ParameterType::FLOAT32;
  }

  data::ParameterType operator()(const float4&) const {
    return data::ParameterType::FLOAT32;
  }

  data::ParameterType operator()(const std::vector<float4>&) const {
    return data::ParameterType::FLOAT32;
  }
};

data::DataType ToFB(DataType type) {
  switch (type) {
    case DataType::INT16:
      return data::DataType::INT16;
    case DataType::INT32:
      return data::DataType::INT32;
    case DataType::FLOAT16:
      return data::DataType::FLOAT16;
    case DataType::FLOAT32:
      return data::DataType::FLOAT32;
    default:
      return data::DataType::UNKNOWN;
  }
}

data::ObjectType ToFB(ObjectType type) {
  switch (type) {
    case ObjectType::TEXTURE:
      return data::ObjectType::TEXTURE;
    case ObjectType::BUFFER:
      return data::ObjectType::BUFFER;
    default:
      return data::ObjectType::UNKNOWN;
  }
}

struct ObjectSizeGetter {
  Offset<void> operator()(const uint3& shape) {
    data::Uint3Builder shape_builder(*builder);
    shape_builder.add_x(shape.x);
    shape_builder.add_y(shape.y);
    shape_builder.add_z(shape.z);
    return shape_builder.Finish().Union();
  }
  Offset<void> operator()(const uint2& shape) {
    data::Uint2Builder shape_builder(*builder);
    shape_builder.add_x(shape.x);
    shape_builder.add_y(shape.y);
    return shape_builder.Finish().Union();
  }
  Offset<void> operator()(uint32_t shape) {
    data::Uint1Builder shape_builder(*builder);
    shape_builder.add_x(shape);
    return shape_builder.Finish().Union();
  }

  ::flatbuffers::FlatBufferBuilder* builder;
};

struct ObjectSizeTypeGetter {
  data::ObjectSize operator()(const uint3&) const {
    return data::ObjectSize::Uint3;
  }
  data::ObjectSize operator()(const uint2&) const {
    return data::ObjectSize::Uint2;
  }
  data::ObjectSize operator()(const uint32_t&) const {
    return data::ObjectSize::Uint1;
  }
};

struct ObjectGetter {
  Offset<void> operator()(const ObjectData& data) {
    auto fb_data = builder->CreateVector(data);
    data::ObjectDataBuilder data_builder(*builder);
    data_builder.add_data(fb_data);
    return data_builder.Finish().Union();
  }
  Offset<void> operator()(ObjectRef ref) {
    data::ObjectRefBuilder ref_builder(*builder);
    ref_builder.add_global_id(ref);
    return ref_builder.Finish().Union();
  }

  ::flatbuffers::FlatBufferBuilder* builder;
};

struct ObjectTypeGetter {
  data::ObjectVariant operator()(const ObjectData&) const {
    return data::ObjectVariant::ObjectData;
  }
  data::ObjectVariant operator()(const ObjectRef&) const {
    return data::ObjectVariant::ObjectRef;
  }
};

data::AccessType ToFB(AccessType type) {
  switch (type) {
    case AccessType::READ:
      return data::AccessType::READ;
    case AccessType::WRITE:
      return data::AccessType::WRITE;
    case AccessType::READ_WRITE:
      return data::AccessType::READ_WRITE;
  }
}

Offset<data::Uint3> Encode(const uint3& v,
                           ::flatbuffers::FlatBufferBuilder* builder) {
  data::Uint3Builder uint3_builder(*builder);
  uint3_builder.add_x(v.x);
  uint3_builder.add_y(v.y);
  uint3_builder.add_z(v.z);
  return uint3_builder.Finish();
}

Offset<data::Parameters> Encode(const CompiledModelOptions& options,
                                ::flatbuffers::FlatBufferBuilder* builder) {
  data::ParametersBuilder params_builder(*builder);
  params_builder.add_dynamic_batch(options.dynamic_batch);
  return params_builder.Finish();
}

}  // namespace

void SerializedCompiledModelBuilder::AddShader(const std::string& shader_src) {
  shaders_.push_back(builder_.CreateString(shader_src));
}

void SerializedCompiledModelBuilder::AddProgram(
    const std::vector<Variable>& parameters, const std::vector<Object>& objects,
    const uint3& workgroup_size, const uint3& num_workgroups,
    size_t shader_index) {
  Offset<data::Uint3> fb_workgroups = Encode(num_workgroups, &builder_);
  Offset<data::Uint3> fb_workgroup_size = Encode(workgroup_size, &builder_);

  Offset<Vector<Offset<data::UniformParameter>>> fb_params;
  {
    std::vector<Offset<data::UniformParameter>> offsets;
    for (const Variable& param : parameters) {
      auto name = builder_.CreateString(param.name);
      auto data = absl::visit(ParameterValueGetter{&builder_}, param.value);
      data::UniformParameterBuilder builder(builder_);
      builder.add_name(name);
      builder.add_data_type(absl::visit(DataVariantTypeGetter{}, param.value));
      builder.add_data(data);
      builder.add_type(absl::visit(ParameterTypeGetter{}, param.value));
      offsets.push_back(builder.Finish());
    }
    fb_params = builder_.CreateVector(offsets);
  }

  Offset<Vector<Offset<data::Object>>> fb_objects;
  {
    std::vector<Offset<data::Object>> offsets;
    for (const Object& object : objects) {
      auto object_variant = absl::visit(ObjectGetter{&builder_}, object.object);
      auto size = absl::visit(ObjectSizeGetter{&builder_}, object.size);

      data::ObjectBuilder builder(builder_);
      builder.add_access(ToFB(object.access));
      builder.add_binding(object.binding);
      builder.add_type(ToFB(object.object_type));
      builder.add_data_type(ToFB(object.data_type));
      builder.add_size_type(absl::visit(ObjectSizeTypeGetter{}, object.size));
      builder.add_size(size);
      builder.add_object_type(absl::visit(ObjectTypeGetter{}, object.object));
      builder.add_object(object_variant);
      offsets.push_back(builder.Finish());
    }
    fb_objects = builder_.CreateVector(offsets);
  }

  data::ProgramBuilder program_builder(builder_);
  program_builder.add_number_workgroups(fb_workgroups);
  program_builder.add_workgroup_size(fb_workgroup_size);
  program_builder.add_parameters(fb_params);
  program_builder.add_objects(fb_objects);
  program_builder.add_shader_index(shader_index);
  programs_.push_back(program_builder.Finish());
}

absl::Span<const uint8_t> SerializedCompiledModelBuilder::Finalize(
    const CompiledModelOptions& options) {
  auto shaders = builder_.CreateVector(shaders_);
  auto programs = builder_.CreateVector(programs_);
  auto parameters = Encode(options, &builder_);
  data::CompiledModelBuilder model_builder(builder_);
  model_builder.add_shaders(shaders);
  model_builder.add_programs(programs);
  model_builder.add_parameters(parameters);
  data::FinishCompiledModelBuffer(builder_, model_builder.Finish());
  return absl::MakeConstSpan(builder_.GetBufferPointer(), builder_.GetSize());
}

namespace {

Status ParseParameter(const data::UniformParameter& fb_parameter,
                      Variable* parameter) {
  parameter->name = fb_parameter.name()->str();
  switch (fb_parameter.type()) {
    case data::ParameterType::INT32: {
      auto* ptr = fb_parameter.data_as_DataInt32();
      if (ptr == nullptr) {
        return InvalidArgumentError("Unexpected data type '" + parameter->name +
                                    "'");
      }
      switch (ptr->data()->size()) {
        case 1:
          parameter->value = (*ptr->data())[0];
          break;
        case 2:
          parameter->value = int2((*ptr->data())[0], (*ptr->data())[1]);
          break;
        case 4:
          parameter->value = int4((*ptr->data())[0], (*ptr->data())[1],
                                  (*ptr->data())[2], (*ptr->data())[3]);
          break;
        default:
          return InvalidArgumentError("Unexpected size for parameter '" +
                                      parameter->name + "'");
      }
      break;
    }
    case data::ParameterType::UINT32: {
      auto* ptr = fb_parameter.data_as_DataUint32();
      if (ptr == nullptr) {
        return InvalidArgumentError("Unexpected data type '" + parameter->name +
                                    "'");
      }
      switch (ptr->data()->size()) {
        case 1:
          parameter->value = (*ptr->data())[0];
          break;
        case 4:
          parameter->value = uint4((*ptr->data())[0], (*ptr->data())[1],
                                   (*ptr->data())[2], (*ptr->data())[3]);
          break;
        default:
          return InvalidArgumentError("Unexpected size for parameter '" +
                                      parameter->name + "'");
      }
      break;
    }
    case data::ParameterType::FLOAT32: {
      auto* ptr = fb_parameter.data_as_DataFloat();
      if (ptr == nullptr) {
        return InvalidArgumentError("Unexpected data type '" + parameter->name +
                                    "'");
      }
      switch (ptr->data()->size()) {
        case 1:
          parameter->value = (*ptr->data())[0];
          break;
        case 2:
          parameter->value = float2((*ptr->data())[0], (*ptr->data())[1]);
          break;
        case 4:
          parameter->value = float4((*ptr->data())[0], (*ptr->data())[1],
                                    (*ptr->data())[2], (*ptr->data())[3]);
          break;
        default:
          return InvalidArgumentError("Unexpected size for parameter '" +
                                      parameter->name + "'");
      }
      break;
    }
    case data::ParameterType::INT32_2: {
      auto* ptr = fb_parameter.data_as_DataInt32();
      if (ptr == nullptr) {
        return InvalidArgumentError("Unexpected data type '" + parameter->name +
                                    "'");
      }

      if (ptr->data()->size() % 2 != 0) {
        return InvalidArgumentError("Unexpected size for parameter '" +
                                    parameter->name + "'");
      }

      std::vector<int2> values(ptr->data()->size() / 2);
      for (int i = 0; i < values.size(); ++i) {
        values[i] = int2((*ptr->data())[i * 2], (*ptr->data())[i * 2 + 1]);
      }
      parameter->value = values;
      break;
    }
  }
  return OkStatus();
}

DataType ToEnum(data::DataType type) {
  switch (type) {
    case data::DataType::INT16:
      return DataType::INT16;
    case data::DataType::INT32:
      return DataType::INT32;
    case data::DataType::FLOAT16:
      return DataType::FLOAT16;
    case data::DataType::FLOAT32:
      return DataType::FLOAT32;
    default:
      return DataType::UNKNOWN;
  }
}

ObjectType ToEnum(data::ObjectType type) {
  switch (type) {
    case data::ObjectType::TEXTURE:
      return ObjectType::TEXTURE;
    case data::ObjectType::BUFFER:
      return ObjectType::BUFFER;
    default:
      return ObjectType::UNKNOWN;
  }
}

AccessType ToEnum(data::AccessType type) {
  switch (type) {
    case data::AccessType::READ:
      return AccessType::READ;
    case data::AccessType::WRITE:
      return AccessType::WRITE;
    case data::AccessType::READ_WRITE:
      return AccessType::READ_WRITE;
  }
}

Status ParseObject(const data::Object& fb_object, Object* object) {
  object->access = ToEnum(fb_object.access());
  object->binding = fb_object.binding();
  object->object_type = ToEnum(fb_object.type());
  object->data_type = ToEnum(fb_object.data_type());

  switch (fb_object.size_type()) {
    case data::ObjectSize::Uint3: {
      auto* size = fb_object.size_as_Uint3();
      object->size = uint3(size->x(), size->y(), size->z());
      break;
    }
    case data::ObjectSize::Uint2: {
      auto* size = fb_object.size_as_Uint2();
      object->size = uint2(size->x(), size->y());
      break;
    }
    case data::ObjectSize::Uint1: {
      auto* size = fb_object.size_as_Uint1();
      object->size = size->x();
      break;
    }
    case data::ObjectSize::NONE:
      return InvalidArgumentError("Texture size is not set");
  }

  switch (fb_object.object_type()) {
    case data::ObjectVariant::ObjectData: {
      auto* fb_data = fb_object.object_as_ObjectData();
      object->object = std::vector<uint8_t>(
          fb_data->data()->data(),
          fb_data->data()->data() + fb_data->data()->size());
      break;
    }
    case data::ObjectVariant::ObjectRef: {
      auto* fb_ref = fb_object.object_as_ObjectRef();
      object->object = fb_ref->global_id();
      break;
    }
    case data::ObjectVariant::NONE: {
      return InvalidArgumentError("Object is not set");
    }
  }
  return OkStatus();
}

CompiledModelOptions ParseParameters(const data::Parameters& fb_parameters) {
  CompiledModelOptions options;
  options.dynamic_batch = fb_parameters.dynamic_batch();
  return options;
}

}  // namespace

Status DeserializeCompiledModel(absl::Span<const uint8_t> serialized,
                                DeserializationHandler* handler) {
  flatbuffers::Verifier verifier(serialized.data(), serialized.size());
  if (!data::VerifyCompiledModelBuffer(verifier)) {
    return InvalidArgumentError("Serialized model is corrupted.");
  }

  auto model = data::GetCompiledModel(serialized.data());
  for (auto shader : *model->shaders()) {
    RETURN_IF_ERROR(
        handler->OnShader(absl::MakeSpan(shader->c_str(), shader->size())));
  }
  std::vector<Variable> parameters;
  std::vector<Object> objects;
  for (auto program : *model->programs()) {
    parameters.clear();
    objects.clear();
    for (auto fb_parameter : *program->parameters()) {
      Variable parameter;
      RETURN_IF_ERROR(ParseParameter(*fb_parameter, &parameter));
      parameters.push_back(std::move(parameter));
    }
    for (auto fb_object : *program->objects()) {
      Object object;
      RETURN_IF_ERROR(ParseObject(*fb_object, &object));
      objects.push_back(std::move(object));
    }
    uint3 workgroup_size(program->workgroup_size()->x(),
                         program->workgroup_size()->y(),
                         program->workgroup_size()->z());
    uint3 num_workgroups(program->number_workgroups()->x(),
                         program->number_workgroups()->y(),
                         program->number_workgroups()->z());
    RETURN_IF_ERROR(handler->OnProgram(parameters, objects, workgroup_size,
                                       num_workgroups,
                                       program->shader_index()));
  }
  handler->OnOptions(ParseParameters(*model->parameters()));
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
