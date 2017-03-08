/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/utils.h"

#include <stddef.h>
#include <utility>

#include "third_party/eigen3/Eigen/Core"

#if GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime_api.h"
#include "cuda/include/cudnn.h"
#endif

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

std::vector<OpInfo::TensorProperties> FindInputFeatures(
    const NodeDef& node,
    const std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost) {
  std::vector<OpInfo::TensorProperties> inputs;
  for (const auto& input_name : node.input()) {
    // Skip control inputs. These are prefixed with the ^ character.
    CHECK(!input_name.empty());
    if (input_name[0] == '^') {
      continue;
    }

    // Each input is "node_name:output_imdex" with "node_name" being a string
    // name and "output_index" indicating which output tensor to use from
    // "node_name". If "output_index" is 0 the ":0" suffix can be omitted.
    string input_node_name;
    int output_index = -1;
    const size_t pos = input_name.rfind(':');
    if (pos == string::npos) {
      input_node_name = input_name;
      output_index = 0;
    } else {
      string index = input_name.substr(pos);
      if (strings::safe_strto32(index, &output_index)) {
        input_node_name = input_name.substr(0, pos);
      }
    }

    auto it = name_to_cost.find(input_name);
    if (it == name_to_cost.end() || output_index < 0) {
      OpInfo::TensorProperties input;
      input.set_dtype(DataType::DT_INVALID);
      input.mutable_shape()->set_unknown_rank(true);
      inputs.push_back(input);
    } else {
      const CostGraphDef::Node* input_cost = it->second;
      const CostGraphDef::Node::OutputInfo& output =
          input_cost->output_info(output_index);
      OpInfo::TensorProperties input;
      input.set_dtype(output.dtype());
      *input.mutable_shape() = output.shape();
      inputs.push_back(input);
    }
  }

  return inputs;
}

OpInfo::DeviceProperties GetDeviceInfo(const CostGraphDef::Node& node) {
  DeviceNameUtils::ParsedName parsed;
  if (DeviceNameUtils::ParseFullName(node.device(), &parsed)) {
    if (parsed.type == "GPU") {
      return GetLocalGPUInfo(parsed.id);
    } else if (parsed.type == "CPU") {
      return GetLocalCPUInfo();
    }
  }
  OpInfo::DeviceProperties device;
  device.set_type("UNKNOWN");
  return device;
}

OpInfo::DeviceProperties GetLocalCPUInfo() {
  OpInfo::DeviceProperties device;
  device.set_type("CPU");

  device.set_num_cores(port::NumSchedulableCPUs());
  device.set_l1_cache_size(Eigen::l1CacheSize());
  device.set_l2_cache_size(Eigen::l2CacheSize());
  device.set_l3_cache_size(Eigen::l3CacheSize());

  (*device.mutable_environment())["cpu_instruction_set"] =
      Eigen::SimdInstructionSetsInUse();

  (*device.mutable_environment())["eigen"] = strings::StrCat(
      EIGEN_WORLD_VERSION, ".", EIGEN_MAJOR_VERSION, ".", EIGEN_MINOR_VERSION);
#ifdef EIGEN_USE_LIBXSMM
  (*device.mutable_environment())["libxsmm"] = LIBXSMM_VERSION;
#endif

  return device;
}

OpInfo::DeviceProperties GetLocalGPUInfo(int gpu_id) {
  OpInfo::DeviceProperties device;
  device.set_type("GPU");

#if GOOGLE_CUDA
  cudaDeviceProp properties;
  cudaError_t error = cudaGetDeviceProperties(&properties, gpu_id);
  if (error == cudaSuccess) {
    device.set_vendor("NVidia");
    device.set_model(properties.name);
    device.set_frequency(properties.clockRate / 1000);
    device.set_num_cores(properties.multiProcessorCount);
    device.set_num_registers(properties.regsPerMultiprocessor);
    // For compute capability less than 5, l1 cache size is configurable to
    // either 16 KB or 48 KB. We use the initial configuration 16 KB here. For
    // compute capability larger or equal to 5, l1 cache (unified with texture
    // cache) size is 24 KB. This number may need to be updated for future
    // compute capabilities.
    device.set_l1_cache_size((properties.major < 5) ? 16 * 1024 : 24 * 1024);
    device.set_l2_cache_size(properties.l2CacheSize);
    device.set_l3_cache_size(0);
    device.set_shared_memory_size_per_multiprocessor(
        properties.sharedMemPerMultiprocessor);
    device.set_memory_size(properties.totalGlobalMem);
    // 8 is the number of bits per byte. 2 is accounted for
    // double data rate (DDR).
    device.set_bandwidth(properties.memoryBusWidth / 8 *
                         properties.memoryClockRate * 2);
  }

  (*device.mutable_environment())["cuda"] = strings::StrCat(CUDA_VERSION);
  (*device.mutable_environment())["cudnn"] = strings::StrCat(CUDNN_VERSION);
#endif

  return device;
}

}  // end namespace grappler
}  // end namespace tensorflow
