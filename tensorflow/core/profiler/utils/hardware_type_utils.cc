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

#include "tensorflow/core/profiler/utils/hardware_type_utils.h"

#include <algorithm>

#include "absl/container/btree_map.h"
#include "absl/strings/match.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"

namespace tensorflow {
namespace profiler {
namespace {

// The calculation methods is referred from Nvidia developer forum:
// https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727
// Below data are calculated from the various NVidia whitepapers/specs.

// https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_9_0 = {
    .cuda_core =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 256,
            .bf16_tflops = 512,
            .fp16_tflops = 512,
            .int8_tops = 1024,
        },
    .tensor_core =
        {
            .fp64_tflops = 256,
            .fp32_tflops = 2048,
            .bf16_tflops = 4096,
            .fp16_tflops = 4096,
            .fp8_tflops = 8192,
            .int8_tops = 8192,
        },
    .has_tensor_core_sparsity_support = true,
};

// https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_8_9 = {
    .cuda_core =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 256,
            .bf16_tflops = 256,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .tensor_core =
        {
            .fp32_tflops = 512,
            .bf16_tflops = 1024,
            .fp16_tflops = 1024,
            .fp8_tflops = 2048,
            .int8_tops = 2048,
            .int4_tops = 4096,
        },
    .has_tensor_core_sparsity_support = true,
};

// https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_8_6 = {
    .cuda_core =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 256,
            .bf16_tflops = 256,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .tensor_core =
        {
            .fp32_tflops = 256,
            .bf16_tflops = 512,
            .fp16_tflops = 1024,
            .int8_tops = 2048,
            .int4_tops = 4096,
        },
    .has_tensor_core_sparsity_support = true,
};

// https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_8_0 = {
    .cuda_core =
        {
            .fp64_tflops = 64,
            .fp32_tflops = 128,
            .bf16_tflops = 256,
            .fp16_tflops = 512,
            .int8_tops = 512,
        },
    .tensor_core =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 1024,
            .bf16_tflops = 2048,
            .fp16_tflops = 2048,
            .int8_tops = 4096,
        },
    .has_tensor_core_sparsity_support = true,
};

// https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_7_5 = {
    .cuda_core =
        {
            .fp64_tflops = 64,
            .fp32_tflops = 128,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .tensor_core =
        {
            .fp16_tflops = 1024,
            .int8_tops = 2048,
            .int4_tops = 4096,
        },
    .has_tensor_core_sparsity_support = false,
};

// https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_7_0 = {
    .cuda_core =
        {
            .fp64_tflops = 64,
            .fp32_tflops = 128,
            .bf16_tflops = 0.0,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .tensor_core =
        {
            .fp16_tflops = 1024,
        },
    .has_tensor_core_sparsity_support = false,
};

// https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_6_1 = {
    .cuda_core =
        {
            .fp64_tflops = 8,
            .fp32_tflops = 256,
            .fp16_tflops = 4,
            .int8_tops = 1024,
        },
    .tensor_core = {},
    .has_tensor_core_sparsity_support = false,
};

// https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_6_0 = {
    .cuda_core =
        {
            .fp64_tflops = 64,
            .fp32_tflops = 128,
            .fp16_tflops = 256,
            .int8_tops = 512,
        },
    .tensor_core = {},
    .has_tensor_core_sparsity_support = false,
};

// https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_5_0 = {
    .cuda_core =
        {
            .fp64_tflops = 4,
            .fp32_tflops = 256,
        },
    .tensor_core = {},
    .has_tensor_core_sparsity_support = false,
};

// https://www.nvidia.com/content/PDF/product-specifications/GeForce_GTX_680_Whitepaper_FINAL.pdf
const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_3_0 = {
    .cuda_core =
        {
            .fp64_tflops = 128,
            .fp32_tflops = 384,
        },
    .tensor_core = {},
    .has_tensor_core_sparsity_support = false,
};

const GpuFlopCapabilities kComputeCap_PerSM_PerCycle_2_0 = {
    .cuda_core =
        {
            .fp64_tflops = 8,
            .fp32_tflops = 64,
        },
    .tensor_core = {},
    .has_tensor_core_sparsity_support = false,
};

GpuFlopCapabilities GetNvidiaFlopCapsPerSMPerCycle(int major_comp_cap,
                                                   int minor_comp_cap) {
  static const auto& kPerSMFlopCapsTable =
      *new absl::btree_map<int, GpuFlopCapabilities const*>{
          // TODO: Add incoming blackwell, and other old GPUS
          {9000, &kComputeCap_PerSM_PerCycle_9_0},
          {8090, &kComputeCap_PerSM_PerCycle_8_9},
          {8060, &kComputeCap_PerSM_PerCycle_8_6},
          {8000, &kComputeCap_PerSM_PerCycle_8_0},
          {7050, &kComputeCap_PerSM_PerCycle_7_5},
          {7000, &kComputeCap_PerSM_PerCycle_7_0},
          {6010, &kComputeCap_PerSM_PerCycle_6_1},
          {6000, &kComputeCap_PerSM_PerCycle_6_0},
          {5000, &kComputeCap_PerSM_PerCycle_5_0},
          {3000, &kComputeCap_PerSM_PerCycle_3_0},
          {2000, &kComputeCap_PerSM_PerCycle_2_0},
      };

  const int normalized_compute_cap =
      major_comp_cap * 1000 + minor_comp_cap * 10;
  GpuFlopCapabilities flops_cap{};
  auto it = kPerSMFlopCapsTable.lower_bound(normalized_compute_cap);
  if (it == kPerSMFlopCapsTable.end()) {
    LOG(WARNING) << "GPU compute capability " << major_comp_cap << "."
                 << minor_comp_cap << " is too old to support.";
  } else {
    flops_cap = *it->second;
    if (it->first != normalized_compute_cap) {
      LOG(WARNING) << "GPU compute capability " << major_comp_cap << "."
                   << minor_comp_cap
                   << " is not found. Use the highest compute cap known "
                   << (it->first / 1000) << "." << ((it->first % 1000) / 10)
                   << " instead.";
    }
  }
  return flops_cap;
}

GpuFlopCapabilities GetGpuFlopCapabilitiesPerSM(
    const DeviceCapabilities& device_cap) {
  GpuFlopCapabilities flops_cap{};
  if (device_cap.device_vendor() == kDeviceVendorNvidia) {
    flops_cap =
        GetNvidiaFlopCapsPerSMPerCycle(device_cap.compute_capability().major(),
                                       device_cap.compute_capability().minor());
  } else {
    LOG(WARNING) << "Unsupported device vendor " << device_cap.device_vendor();
  }

  flops_cap.ScaleWith(device_cap.clock_rate_in_ghz());
  return flops_cap;
}

}  // namespace

double GetFlopMaxThroughputPerSM(const DeviceCapabilities& device_cap) {
  GpuFlopCapabilities sm_flops = GetGpuFlopCapabilitiesPerSM(device_cap);
  double result = std::max(
      {sm_flops.cuda_core.fp32_tflops, sm_flops.cuda_core.fp16_tflops,
       sm_flops.tensor_core.fp32_tflops, sm_flops.tensor_core.fp16_tflops});
  VLOG(3) << "GetFlopMaxThroughputPerSM get result: " << result << " GFLOPs";
  return result;
}

double GetSharedMemoryBandwidthPerSM(const DeviceCapabilities& device_cap) {
  // https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/memorystatisticsshared.htm
  // Compute capability 2.0, each bank has bandwidth of 4 bytes per 2 cycles.
  // For compute capability 3.0 and above, each bank has bandwidth 8 bytes per
  // cycle. Each SM has 32 banks.
  double transaction_byts_per_cycle =
      device_cap.compute_capability().major() <= 2 ? (32 * 4 / 2) : (32 * 8);
  double GiBPS = transaction_byts_per_cycle * device_cap.clock_rate_in_ghz();
  return tsl::profiler::GigaToUni(GiBPS);
}

absl::string_view GpuModelName(const DeviceCapabilities& device_cap) {
  if (device_cap.device_vendor() == kDeviceVendorNvidia) {
    switch (device_cap.compute_capability().major()) {
      case 2:
        return "Nvidia GPU (Fermi)";
      case 3:
        return "Nvidia GPU (Kepler)";
      case 5:
        return "Nvidia GPU (Maxwell)";
      case 6:
        return "Nvidia GPU (Pascal)";
      case 7:
        if (device_cap.compute_capability().minor() < 5) {
          return "Nvidia GPU (Volta)";
        } else {
          return "Nvidia GPU (Turing)";
        }
      case 8:
        if (device_cap.compute_capability().minor() < 9) {
          return "Nvidia GPU (Ampere)";
        } else {
          return "Nvidia GPU (Ada Lovelace)";
        }
      case 9:
        return "Nvidia GPU (Hopper)";
      case 10:
        return "Nvidia GPU (Blackwell)";
      default:
        return "Nvidia GPU";
    }
  } else if (device_cap.device_vendor() == kDeviceVendorAMD) {
    switch (device_cap.compute_capability().major()) {
      case 9:
        return "AMD GPU - gfx-9XX series";
      case 10:
        return "AMD GPU - gfx-10XX series";
      case 11:
        return "AMD GPU - gfx-11XX series";
      default:
        return "AMD GPU";
    }
  } else {
    LOG(ERROR) << "Unknown device vendor " << device_cap.device_vendor();
    return "";
  }
}

HardwareType ParseHardwareType(absl::string_view device_type) {
  if (absl::StrContains(device_type, "GPU")) return HardwareType::GPU;
  if (device_type == "CPU") return HardwareType::CPU_ONLY;
  if (absl::StrContains(device_type, "TPU")) return HardwareType::TPU;
  return HardwareType::UNKNOWN_HARDWARE;
}

bool HasDevice(HardwareType x) { return x > tensorflow::profiler::CPU_ONLY; }

}  // namespace profiler
}  // namespace tensorflow
