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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_HARDWARE_TYPE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_HARDWARE_TYPE_UTILS_H_

#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"

namespace tensorflow {
namespace profiler {

struct GpuFlopCapabilities {
  struct FlopCapabilityOnPrecisions {
    double fp64_tflops = 0;
    double fp32_tflops = 0;  // also for tf32 for nvidia tensor core
    double bf16_tflops = 0;
    double fp16_tflops = 0;
    double fp8_tflops = 0;
    double int8_tops = 0;
    double fp4_tflops = 0;
    double int4_tops = 0;

    void ScaleWith(double scale) {
      fp64_tflops *= scale;
      fp32_tflops *= scale;
      bf16_tflops *= scale;
      fp16_tflops *= scale;
      fp8_tflops *= scale;
      int8_tops *= scale;
      fp4_tflops *= scale;
      int4_tops *= scale;
    }
  };

  FlopCapabilityOnPrecisions cuda_core;
  FlopCapabilityOnPrecisions tensor_core;
  bool has_tensor_core_sparsity_support = false;

  void ScaleWith(double scale) {
    cuda_core.ScaleWith(scale);
    tensor_core.ScaleWith(scale);
  }
};

// Get peak single precision throughput of the GPU in GFLOPS per
// streaming multiprocessor.
// TODO: Need design on how to use the sparsity capability of FLOPs.
double GetFlopMaxThroughputPerSM(const DeviceCapabilities& device_cap);

// for Nvidia GPU, return shared memory bandwidth in Bytes Per Second on
// one single SM given the GPU core freq in device_cap.
double GetSharedMemoryBandwidthPerSM(const DeviceCapabilities& device_cap);

// Returns the GPU model name from the given DeviceCapabilities.
// For nvidia GPUs, the name is like "Nvidia GPU (Kepler)" or "Nvidia GPU
// (Turing)". For AMD GPUs, the name is like "AMD GPU - gfx-10XX series".
// The model name here for Nvidia GPU in fact refers to its microarchitecture
// name.
absl::string_view GpuModelName(const DeviceCapabilities& device_cap);

HardwareType ParseHardwareType(absl::string_view device_type);

// Returns true if the given hardware type has a device.
bool HasDevice(HardwareType x);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HARDWARE_TYPE_UTILS_H_
