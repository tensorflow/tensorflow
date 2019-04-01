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

#include <stddef.h>
#include <sys/types.h>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/uniform_parameter.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

struct ProgramDesc {
  std::vector<UniformParameter> parameters;
  std::vector<Object> objects;
  uint3 workgroup_size;
  uint3 num_workgroups;
  size_t shader_index;
};

struct Handler : public DeserializationHandler {
  Status OnShader(absl::Span<const char> shader_src) final {
    shaders.push_back(std::string(shader_src.data(), shader_src.size()));
    return OkStatus();
  }

  Status OnProgram(const std::vector<UniformParameter>& parameters,
                   const std::vector<Object>& objects,
                   const uint3& workgroup_size, const uint3& num_workgroups,
                   size_t shader_index) final {
    programs.push_back(
        {parameters, objects, workgroup_size, num_workgroups, shader_index});
    return OkStatus();
  }

  void OnOptions(const CompiledModelOptions& o) final { options = o; }

  std::vector<std::string> shaders;
  std::vector<ProgramDesc> programs;
  CompiledModelOptions options;
};

struct ParameterComparator {
  bool operator()(int32_t value) const {
    return value == absl::get<int32_t>(a.value);
  }
  bool operator()(const int2& value) const {
    auto v = absl::get<int2>(a.value);
    return value.x == v.x && value.y == v.y;
  }
  bool operator()(const int4& value) const {
    auto v = absl::get<int4>(a.value);
    return value.x == v.x && value.y == v.y && value.z == v.z && value.w == v.w;
  }
  bool operator()(const std::vector<int2>& value) const {
    auto v = absl::get<std::vector<int2>>(a.value);
    if (v.size() != value.size()) {
      return false;
    }
    for (int i = 0; i < v.size(); ++i) {
      if (v[i].x != value[i].x || v[i].y != value[i].y) {
        return false;
      }
    }
    return true;
  }
  bool operator()(uint32_t value) const {
    return value == absl::get<uint32_t>(a.value);
  }
  bool operator()(const uint4& value) const {
    auto v = absl::get<uint4>(a.value);
    return value.x == v.x && value.y == v.y && value.z == v.z && value.w == v.w;
  }
  bool operator()(float value) const {
    return value == absl::get<float>(a.value);
  }
  bool operator()(float2 value) const {
    auto v = absl::get<float2>(a.value);
    return value.x == v.x && value.y == v.y;
  }
  bool operator()(const float4& value) const {
    auto v = absl::get<float4>(a.value);
    return value.x == v.x && value.y == v.y && value.z == v.z && value.w == v.w;
  }
  UniformParameter a;
};

bool Eq(const UniformParameter& a, const UniformParameter& b) {
  return a.name == b.name && absl::visit(ParameterComparator{a}, b.value);
}

struct ObjectComparator {
  bool operator()(const ObjectData& data) const {
    return absl::get<ObjectData>(a.object) == data;
  }
  bool operator()(const ObjectRef& ref) const {
    return absl::get<ObjectRef>(a.object) == ref;
  }

  Object a;
};

bool Eq(const Object& a, const Object& b) {
  return a.access == b.access && a.binding == b.binding &&
         absl::visit(ObjectComparator{a}, b.object);
}

TEST(Smoke, Read) {
  std::string shader1 = "A";
  std::string shader2 = "B";

  SerializedCompiledModelBuilder builder;
  builder.AddShader(shader1);
  builder.AddShader(shader2);

  std::vector<UniformParameter> parameters;
  parameters.push_back(UniformParameter{"1", int32_t(1)});
  parameters.push_back(UniformParameter{"2", int2(1, 2)});
  parameters.push_back(UniformParameter{"3", int4(1, 2, 3, 4)});
  parameters.push_back(UniformParameter{"4", uint32_t(10)});
  parameters.push_back(UniformParameter{"5", uint4(10, 20, 30, 40)});
  parameters.push_back(UniformParameter{"6", -2.0f});
  parameters.push_back(UniformParameter{"7", float2(1, -1)});
  parameters.push_back(UniformParameter{"8", float4(1, -1, 2, -2)});
  parameters.push_back(UniformParameter{
      "9", std::vector<int2>{int2(1, 2), int2(3, 4), int2(5, 6)}});

  std::vector<Object> objects;
  objects.push_back(MakeReadonlyBuffer(std::vector<float>{1, 2, 3, 4}));
  objects.push_back(Object{AccessType::WRITE, DataType::FLOAT32,
                           ObjectType::TEXTURE, 5, uint3(1, 2, 3), 100});
  objects.push_back(Object{AccessType::READ_WRITE, DataType::INT8,
                           ObjectType::BUFFER, 6, uint2(2, 1),
                           std::vector<uint8_t>{7, 9}});
  uint3 num_workgroups(10, 20, 30);
  uint3 workgroup_size(1, 2, 3);
  builder.AddProgram(parameters, objects, workgroup_size, num_workgroups, 1);

  Handler handler;
  CompiledModelOptions options;
  options.dynamic_batch = true;
  ASSERT_TRUE(
      DeserializeCompiledModel(builder.Finalize(options), &handler).ok());
  EXPECT_EQ(num_workgroups.data_, handler.programs[0].num_workgroups.data_);
  EXPECT_EQ(workgroup_size.data_, handler.programs[0].workgroup_size.data_);
  EXPECT_THAT(handler.shaders, ::testing::ElementsAre(shader1, shader2));
  EXPECT_EQ(handler.programs[0].parameters.size(), parameters.size());
  for (int i = 0; i < parameters.size(); ++i) {
    EXPECT_TRUE(Eq(parameters[i], handler.programs[0].parameters[i])) << i;
  }
  EXPECT_EQ(handler.programs[0].objects.size(), objects.size());
  for (int i = 0; i < objects.size(); ++i) {
    EXPECT_TRUE(Eq(objects[i], handler.programs[0].objects[i])) << i;
  }
  EXPECT_TRUE(handler.options.dynamic_batch);
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
