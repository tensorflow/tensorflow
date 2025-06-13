/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_

#include <array>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#if defined(PLATFORM_POSIX) || defined(PLATFORM_GOOGLE)
#include <dlfcn.h>
#endif

#include "third_party/gpus/cuda/nvml/include/nvml.h"
// Below is a list of function pointers to be used
// for querying device properties through nvml library.
#define NVML_FUNCTOR(name, rettype, args) \
  inline rettype(*xla_##name) args = nullptr;

NVML_FUNCTOR(nvmlInit, nvmlReturn_t, ())
NVML_FUNCTOR(nvmlShutdown, nvmlReturn_t, ())
NVML_FUNCTOR(nvmlDeviceGetHandleByIndex, nvmlReturn_t,
             (unsigned int index, nvmlDevice_t* device))
NVML_FUNCTOR(nvmlDeviceGetNvLinkCapability, nvmlReturn_t,
             (nvmlDevice_t device, unsigned int link,
              nvmlNvLinkCapability_t capability, unsigned int* capResult))
NVML_FUNCTOR(nvmlSystemGetNVMLVersion, nvmlReturn_t,
             (char* version, size_t versionSize))

#if CUDA_VERSION >= 12040
NVML_FUNCTOR(nvmlDeviceGetHandleByPciBusId_v2, nvmlReturn_t,
             (const char* pciBusId, nvmlDevice_t* device))
NVML_FUNCTOR(nvmlDeviceGetGpuFabricInfoV, nvmlReturn_t,
             (nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo))
#endif  // CUDA_VERSION >= 12040

#endif

namespace xla {
namespace gpu {

class GpuPerformanceWithCollectiveModel : public GpuPerformanceModelBase {
 public:
  static absl::Duration ComputeCollectiveTime(
      const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
      const se::DeviceDescription& gpu_device_info);

  // Initialize nvml library.
  static bool InitNvml();

  // Shut down nvml library.
  static bool ShutdownNvml();

  // This checks if the nvlink supports direct P2P communication,
  // If not, we will use PCIE bandwidth to estimate latency.
  static uint32_t CheckIfNvlinkSupportsP2P();

 private:
  static absl::Duration ComputeAllreduceTime(
      const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
      const se::DeviceDescription& gpu_device_info);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_
