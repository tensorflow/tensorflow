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
std::string GetNextWord(const std::string& code, size_t first_position) {
  size_t pos = first_position;
  char t = code[pos];
  while (absl::ascii_isalnum(t) || t == '_') {
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
      shared_float4s_data_(std::move(args.shared_float4s_data_)) {}
Arguments& Arguments::operator=(Arguments&& args) {
  if (this != &args) {
    int_values_ = std::move(args.int_values_);
    shared_int4s_data_ = std::move(args.shared_int4s_data_);
    float_values_ = std::move(args.float_values_);
    shared_float4s_data_ = std::move(args.shared_float4s_data_);
  }
  return *this;
}

void Arguments::AddFloat(const std::string& name, float value) {
  float_values_[name].value = value;
}
void Arguments::AddInt(const std::string& name, int value) {
  int_values_[name].value = value;
}

absl::Status Arguments::SetInt(const std::string& name, int value) {
  auto ii = int_values_.find(name);
  if (ii == int_values_.end()) {
    return absl::NotFoundError(absl::StrCat("No argument with name - ", name));
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
    return absl::NotFoundError(absl::StrCat("No argument with name - ", name));
  }
  fi->second.value = value;
  if (fi->second.active) {
    shared_float4s_data_[fi->second.offset] = value;
  }
  return absl::OkStatus();
}

std::string Arguments::GetListOfArgs() {
  std::string result;
  for (int i = 0; i < shared_int4s_data_.size() / 4; ++i) {
    absl::StrAppend(&result, ",\n  int4 shared_int4_", i);
  }
  for (int i = 0; i < shared_float4s_data_.size() / 4; ++i) {
    absl::StrAppend(&result, ",\n  float4 shared_float4_", i);
  }
  return result;
}

absl::Status Arguments::Bind(cl_kernel kernel, int offset) {
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
  std::string result;
  constexpr char kPrefix[] = "args.";
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

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
