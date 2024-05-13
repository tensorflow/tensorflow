/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/task/qcom_thin_filter_desc.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tflite {
namespace gpu {

GPUResources QcomThinFilterDescriptor::GetGPUResources(
    const GpuInfo& gpu_info) const {
  GPUResources resources;
  GPUCustomMemoryDescriptor desc;
  desc.type_name = "__read_only qcom_weight_image_t";
  resources.custom_memories.push_back({"filter", desc});
  return resources;
}

absl::Status QcomThinFilterDescriptor::PerformSelector(
    const GpuInfo& gpu_info, absl::string_view selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "GetHandle" && args.empty()) {
    *result = "filter";
    return absl::OkStatus();
  } else {
    return absl::NotFoundError(absl::StrCat(
        "QcomThinFilterDescriptor don't have selector with name - ", selector));
  }
}

}  // namespace gpu
}  // namespace tflite
