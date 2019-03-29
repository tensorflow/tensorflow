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

#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator_from_metadata.h"

#ifndef TFLITE_GPU_BINARY_RELEASE

#include <memory>
#include <unordered_map>

#include "tensorflow/lite/delegates/gpu/gl/metadata_generated.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/calculator.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/default_calculator.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups_generated.h"

#include "absl/memory/memory.h"
#include "flatbuffers/flatbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/delegates/gpu/common/types.h"

#endif  // TFLITE_GPU_BINARY_RELEASE

namespace tflite {
namespace gpu {
namespace gl {

#ifndef TFLITE_GPU_BINARY_RELEASE
namespace {
class WorkgroupsCalculatorFromMetadata : public WorkgroupsCalculator {
 public:
  WorkgroupsCalculatorFromMetadata(const data::HardcodedWorkgroups& workgroups,
                                   const GpuInfo& gpu_info)
      : WorkgroupsCalculator(gpu_info),
        default_calculator_(NewDefaultWorkgroupsCalculator(gpu_info)) {
    for (const auto* workgroup : *workgroups.workgroups()) {
      uint3 size(workgroup->size()->x(), workgroup->size()->y(),
                 workgroup->size()->z());
      // Class implementation relies on the fact that it uses unique graph
      // representation where each node id appears in a single workgroup.
      for (auto node_id : *workgroup->node_indices()) {
        workgroups_.insert({node_id, size});
      }
    }
  }

  uint3 CalculateInternal(const ShaderCode& shader_code) const final {
    auto it = workgroups_.find(shader_code.node_indices[0]);
    return it != workgroups_.end()
               ? it->second
               : default_calculator_->Calculate(shader_code);
  }

 private:
  std::unordered_map<NodeId, uint3> workgroups_;
  std::unique_ptr<WorkgroupsCalculator> default_calculator_;
};

const data::HardcodedWorkgroups* FindWorkgroups(
    const data::CustomWorkgroups& workgroups, const GpuInfo& gpu_info) {
  for (auto workgroup : *workgroups.hardcoded_workgroups()) {
    if (workgroup->gpu_info()->c_str() == gpu_info.renderer_name) {
      return workgroup;
    }
  }
  return nullptr;
}

}  // namespace

std::unique_ptr<WorkgroupsCalculator> NewWorkgroupsCalculatorFromMetadata(
    const uint8_t* metadata, const GpuInfo& gpu_info) {
  if (!metadata) return nullptr;
  const auto* flow_metadata =
      flatbuffers::GetRoot<data::FlowMetadata>(metadata);
  if (!flow_metadata || !flow_metadata->workgroups()) return nullptr;
  const data::HardcodedWorkgroups* workgroups =
      FindWorkgroups(*flow_metadata->workgroups(), gpu_info);
  if (!workgroups) return nullptr;
  return absl::make_unique<WorkgroupsCalculatorFromMetadata>(*workgroups,
                                                             gpu_info);
}

#else  // TFLITE_GPU_BINARY_RELEASE

std::unique_ptr<WorkgroupsCalculator> NewWorkgroupsCalculatorFromMetadata(
    const uint8_t* metadata, const GpuInfo& gpu_info) {
  return nullptr;
}

#endif  // TFLITE_GPU_BINARY_RELEASE

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
