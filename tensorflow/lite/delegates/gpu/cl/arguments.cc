/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/arguments.h"

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

std::string GetNextWord(const std::string& code, size_t first_position) {
  size_t pos = first_position;
  char t = code[pos];
  while (IsWordSymbol(t)) {
    pos++;
    t = code[pos];
  }
  return code.substr(first_position, pos - first_position);
}
}  // namespace

Arguments::Arguments(Arguments&& args)
    : int_values_(std::move(args.int_values_)),
      shared_int4s_data_(std::move(args.shared_int4s_data_)),
      float_values_(std::move(args.float_values_)),
      shared_float4s_data_(std::move(args.shared_float4s_data_)),
      buffers_(std::move(args.buffers_)),
      images2d_(std::move(args.images2d_)),
      objects_(std::move(args.objects_)) {}
Arguments& Arguments::operator=(Arguments&& args) {
  if (this != &args) {
    int_values_ = std::move(args.int_values_);
    shared_int4s_data_ = std::move(args.shared_int4s_data_);
    float_values_ = std::move(args.float_values_);
    shared_float4s_data_ = std::move(args.shared_float4s_data_);
    buffers_ = std::move(args.buffers_);
    images2d_ = std::move(args.images2d_);
    objects_ = std::move(args.objects_);
  }
  return *this;
}

void Arguments::AddFloat(const std::string& name, float value) {
  float_values_[name].value = value;
}
void Arguments::AddInt(const std::string& name, int value) {
  int_values_[name].value = value;
}
void Arguments::AddBuffer(const std::string& name,
                          const GPUBufferDescriptor& desc) {
  buffers_[name] = desc;
}
void Arguments::AddImage2D(const std::string& name,
                           const GPUImage2DDescriptor& desc) {
  images2d_[name] = desc;
}

void Arguments::AddObject(const std::string& name, GPUObjectPtr&& object) {
  objects_[name] = {AccessType::READ, std::move(object)};
}

void Arguments::AddGPUResources(const std::string& name,
                                const GPUResources& resources) {
  for (const auto& r : resources.ints) {
    AddInt(absl::StrCat(name, "_", r));
  }
  for (const auto& r : resources.floats) {
    AddFloat(absl::StrCat(name, "_", r));
  }
  for (const auto& r : resources.buffers) {
    AddBuffer(absl::StrCat(name, "_", r.first), r.second);
  }
  for (const auto& r : resources.images2d) {
    AddImage2D(absl::StrCat(name, "_", r.first), r.second);
  }
}

absl::Status Arguments::SetInt(const std::string& name, int value) {
  auto ii = int_values_.find(name);
  if (ii == int_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No int argument with name - ", name));
  }
  ii->second.value = value;
  if (ii->second.active) {
    shared_int4s_data_[ii->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status Arguments::SetFloat(const std::string& name, float value) {
  auto fi = float_values_.find(name);
  if (fi == float_values_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No float argument with name - ", name));
  }
  fi->second.value = value;
  if (fi->second.active) {
    shared_float4s_data_[fi->second.offset] = value;
  }
  return absl::OkStatus();
}

absl::Status Arguments::SetImage2D(const std::string& name, cl_mem memory) {
  auto ti = images2d_.find(name);
  if (ti == images2d_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No image2D argument with name - ", name));
  }
  ti->second.memory = memory;
  return absl::OkStatus();
}

absl::Status Arguments::SetBuffer(const std::string& name, cl_mem memory) {
  auto it = buffers_.find(name);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("No buffer argument with name - ", name));
  }
  it->second.memory = memory;
  return absl::OkStatus();
}

absl::Status Arguments::SetGPUResources(
    const std::string& name, const GPUResourcesWithValue& resources) {
  for (const auto& r : resources.ints) {
    RETURN_IF_ERROR(SetInt(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.floats) {
    RETURN_IF_ERROR(SetFloat(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.buffers) {
    RETURN_IF_ERROR(SetBuffer(absl::StrCat(name, "_", r.first), r.second));
  }
  for (const auto& r : resources.images2d) {
    RETURN_IF_ERROR(SetImage2D(absl::StrCat(name, "_", r.first), r.second));
  }
  return absl::OkStatus();
}

absl::Status Arguments::TransformToCLCode(std::string* code) {
  RETURN_IF_ERROR(AddObjectArgs());
  ResolveArgsPass(code);
  return absl::OkStatus();
}

std::string Arguments::GetListOfArgs() {
  std::string result;
  for (auto& t : buffers_) {
    const std::string type_name =
        t.second.data_type == DataType::FLOAT32 ? "float" : "half";
    absl::StrAppend(&result, ",\n  __global ", type_name, t.second.element_size,
                    "* ", t.first);
  }
  for (auto& t : images2d_) {
    absl::StrAppend(&result, ",\n  __read_only image2d_t ", t.first);
  }
  for (int i = 0; i < shared_int4s_data_.size() / 4; ++i) {
    absl::StrAppend(&result, ",\n  int4 shared_int4_", i);
  }
  for (int i = 0; i < shared_float4s_data_.size() / 4; ++i) {
    absl::StrAppend(&result, ",\n  float4 shared_float4_", i);
  }
  return result;
}

absl::Status Arguments::Bind(cl_kernel kernel, int offset) {
  for (auto& t : buffers_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (auto& t : images2d_) {
    const int error_code =
        clSetKernelArg(kernel, offset, sizeof(cl_mem), &t.second.memory);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (int i = 0; i < shared_int4s_data_.size() / 4; ++i) {
    const int error_code = clSetKernelArg(kernel, offset, sizeof(int32_t) * 4,
                                          &shared_int4s_data_[i * 4]);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  for (int i = 0; i < shared_float4s_data_.size() / 4; ++i) {
    const int error_code = clSetKernelArg(kernel, offset, sizeof(int32_t) * 4,
                                          &shared_float4s_data_[i * 4]);
    if (error_code != CL_SUCCESS) {
      return absl::UnknownError(absl::StrCat(
          "Failed to set kernel arguments - ", CLErrorCodeToString(error_code),
          "(at index - ", offset, ")"));
    }
    offset++;
  }
  return absl::OkStatus();
}

std::string Arguments::AddActiveArgument(const std::string& arg_name) {
  if (auto it = int_values_.find(arg_name); it != int_values_.end()) {
    int int_index;
    if (it->second.active) {
      int_index = it->second.offset;
    } else {
      it->second.active = true;
      it->second.offset = shared_int4s_data_.size();
      int_index = it->second.offset;
      shared_int4s_data_.push_back(it->second.value);
    }
    std::string index = std::to_string(int_index / 4);
    std::string postfixes[4] = {"x", "y", "z", "w"};
    return "shared_int4_" + index + "." + postfixes[int_index % 4];
  }
  if (auto it = float_values_.find(arg_name); it != float_values_.end()) {
    int float_index;
    if (it->second.active) {
      float_index = it->second.offset;
    } else {
      it->second.active = true;
      it->second.offset = shared_float4s_data_.size();
      float_index = it->second.offset;
      shared_float4s_data_.push_back(it->second.value);
    }
    std::string index = std::to_string(float_index / 4);
    std::string postfixes[4] = {"x", "y", "z", "w"};
    return "shared_float4_" + index + "." + postfixes[float_index % 4];
  }
  return arg_name;
}

void Arguments::ResolveArgsPass(std::string* code) {
  constexpr char kPrefix[] = "args.";
  std::string result;
  size_t position = 0;
  size_t next_position = code->find(kPrefix);
  while (next_position != std::string::npos) {
    size_t arg_pos = next_position;
    next_position += strlen(kPrefix);
    std::string object_name = GetNextWord(*code, next_position);
    std::string new_name = AddActiveArgument(object_name);
    code->replace(arg_pos, object_name.size() + strlen(kPrefix), new_name);
    position = arg_pos + new_name.size();
    next_position = code->find(kPrefix, position);
  }

  int shared_int4s_aligned_size = AlignByN(shared_int4s_data_.size(), 4);
  shared_int4s_data_.resize(shared_int4s_aligned_size);
  int shared_float4s_aligned_size = AlignByN(shared_float4s_data_.size(), 4);
  shared_float4s_data_.resize(shared_float4s_aligned_size);
}

absl::Status Arguments::AddObjectArgs() {
  for (auto& t : objects_) {
    AddGPUResources(t.first,
                    t.second.obj_ptr->GetGPUDescriptor()->GetGPUResources());
    RETURN_IF_ERROR(
        SetGPUResources(t.first, t.second.obj_ptr->GetGPUResources()));
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
