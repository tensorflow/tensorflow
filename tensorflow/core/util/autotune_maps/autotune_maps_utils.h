/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// For Google-internal use only.

#ifndef TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_AUTOTUNE_MAPS_UTILS_H_
#define TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_AUTOTUNE_MAPS_UTILS_H_

#include <string>
#include <vector>

#include "tensorflow/core/platform/protobuf.h"
namespace tensorflow {
namespace autotune_maps_utils {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Given a device_id, this function computes an identifier string that
// represents the corresponding GPU device type. Currently the identifier is
// computed as
// "<device_name> <compute_compatibility> <GPU_memory> <multiprocessor_count>".
// We cannot simply use <device_name> output by GetDeviceName here because for
// some GPUs the it will output uninformative names like "Graphics Device",
// which cannot identify device types of GPUs.
// TODO(ruochengw): Replace the identifier with something that uniquely
// determines a GPU device type, e.g. PCI device ID.
std::string DeviceIdToIdentifier(int device_id);

// Precomputes a map storing the results of DeviceIdToIdentifierHelper for all
// device_ids available and outputs "Unknown Graphics Device" when
// DeviceIdToIdentifierHelper returns an error.
std::vector<std::string> GetDeviceIdToIdentifierMap();

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
std::string SerializeProtoDeterministic(const protobuf::Message& proto);

uint64_t HashProto(const protobuf::Message& proto);

}  // namespace autotune_maps_utils
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_AUTOTUNE_MAPS_UTILS_H_
