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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {
namespace {

// Get theoretical upperbound of single precision FMA throughput of the GPU per
// cycle per streaming multiprocessor.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions__throughput-native-arithmetic-instructions
uint32 GetFmaMaxThroughputPerSMPerCycle(const DeviceCapabilities& device_cap) {
  uint32 n_fp32_cores = 0;
  uint32 n_tc_cores = 0;
  switch (device_cap.compute_capability().major()) {
    case 2:
      // Fermi
      n_fp32_cores = 32;
      break;
    case 3:
      // Kepler
      n_fp32_cores = 192;
      break;
    case 5:
      // Maxwell
      n_fp32_cores = 128;
      break;
    case 6:
      // Pascal
      if (device_cap.compute_capability().minor() > 0) {
        // Pascal SM61/62
        n_fp32_cores = 128;
      } else {
        // Pascal SM60
        n_fp32_cores = 64;
      }
      break;
    case 7:
      // Volta and Turing
      n_fp32_cores = 64;
      n_tc_cores = 8;
      break;
    default:
      LOG(ERROR) << "Invalid GPU compute capability.";
      break;
  }
  // GPU TensorCore can execute 64 FMAs per cycle.
  // https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/
  return n_fp32_cores + n_tc_cores * 64;
}

}  // namespace

double GetFlopMaxThroughputPerSM(const DeviceCapabilities& device_cap) {
  // One FMA = 2 floating point operations, one multiply and one add.
  return GetFmaMaxThroughputPerSMPerCycle(device_cap) * 2 *
         device_cap.clock_rate_in_ghz();
}

}  // namespace profiler
}  // namespace tensorflow
