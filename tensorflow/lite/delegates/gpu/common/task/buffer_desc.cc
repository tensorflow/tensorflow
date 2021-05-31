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

#include "tensorflow/lite/delegates/gpu/common/task/buffer_desc.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {

void BufferDescriptor::Release() { data.clear(); }

GPUResources BufferDescriptor::GetGPUResources(const GpuInfo& gpu_info) const {
  GPUResources resources;
  GPUBufferDescriptor desc;
  desc.data_type = element_type;
  desc.access_type = access_type_;
  desc.element_size = element_size;
  desc.memory_type = memory_type;
  desc.attributes = attributes;
  resources.buffers.push_back({"buffer", desc});
  return resources;
}

absl::Status BufferDescriptor::PerformSelector(
    const GpuInfo& gpu_info, const std::string& selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Read") {
    return PerformReadSelector(args, result);
  } else if (selector == "GetPtr") {
    return PerformGetPtrSelector(args, template_args, result);
  } else {
    return absl::NotFoundError(absl::StrCat(
        "BufferDescriptor don't have selector with name - ", selector));
  }
}

absl::Status BufferDescriptor::PerformReadSelector(
    const std::vector<std::string>& args, std::string* result) const {
  if (args.size() != 1) {
    return absl::NotFoundError(
        absl::StrCat("BufferDescriptor Read require one argument, but ",
                     args.size(), " was passed"));
  }
  *result = absl::StrCat("buffer[", args[0], "]");
  return absl::OkStatus();
}

absl::Status BufferDescriptor::PerformGetPtrSelector(
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (args.size() > 1) {
    return absl::NotFoundError(absl::StrCat(
        "BufferDescriptor GetPtr require one or zero arguments, but ",
        args.size(), " was passed"));
  }
  if (template_args.size() > 1) {
    return absl::NotFoundError(
        absl::StrCat("BufferDescriptor GetPtr require one or zero teemplate "
                     "arguments, but ",
                     template_args.size(), " was passed"));
  }
  std::string conversion;
  if (template_args.size() == 1) {
    const std::string type_name = ToCLDataType(element_type, element_size);
    if (type_name != template_args[0]) {
      conversion = absl::StrCat("(", MemoryTypeToCLType(memory_type), " ",
                                template_args[0], "*)&");
    }
  }
  if (args.empty()) {
    *result = absl::StrCat(conversion, "buffer");
  } else if (conversion.empty()) {
    *result = absl::StrCat("(buffer + ", args[0], ")");
  } else {
    *result = absl::StrCat(conversion, "buffer[", args[0], "]");
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
