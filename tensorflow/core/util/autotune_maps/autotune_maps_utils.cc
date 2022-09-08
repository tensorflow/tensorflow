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

#include "tensorflow/core/util/autotune_maps/autotune_maps_utils.h"

#include <string>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/hash.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_driver.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

namespace autotune_maps_utils {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace {

using ::stream_executor::gpu::GpuDeviceHandle;
using ::stream_executor::gpu::GpuDriver;

StatusOr<string> DeviceIdToIdentifierHelper(int device_id) {
  GpuDeviceHandle device;
  TF_RETURN_IF_ERROR(GpuDriver::GetDevice(device_id, &device));
  std::string device_name;
  TF_RETURN_IF_ERROR(GpuDriver::GetDeviceName(device, &device_name));
  int cc_major;
  int cc_minor;
  TF_RETURN_IF_ERROR(
      GpuDriver::GetComputeCapability(&cc_major, &cc_minor, device));

  uint64 device_memory_size = -1;
  if (!GpuDriver::GetDeviceTotalMemory(device, &device_memory_size)) {
    return errors::Internal("Failed to get device's total memory");
  }

  TF_ASSIGN_OR_RETURN(int core_count,
                      GpuDriver::GetMultiprocessorCount(device));
  return absl::StrFormat("%s sm_%d.%d with %dB RAM and %d cores", device_name,
                         cc_major, cc_minor, device_memory_size, core_count);
}

}  // namespace

std::vector<std::string> GetDeviceIdToIdentifierMap() {
  int device_count = GpuDriver::GetDeviceCount();
  std::vector<string> map(device_count);
  for (int device_id = 0; device_id < device_count; device_id++) {
    StatusOr<string> device_identifier_or_status =
        DeviceIdToIdentifierHelper(device_id);
    if (device_identifier_or_status.ok()) {
      map[device_id] = device_identifier_or_status.value();
    } else {
      map[device_id] = "Unknown Graphics Device";
    }
  }
  return map;
}

std::string DeviceIdToIdentifier(int device_id) {
  // Ensure the static variable is trivially destructible and thus safe to be
  // destruct in multi-thread setting.
  static const auto& map =
      *new std::vector<string>(GetDeviceIdToIdentifierMap());
  return device_id < map.size() ? map[device_id] : "Unknown Graphics Device";
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

std::string SerializeProtoDeterministic(const protobuf::Message& proto) {
  std::string proto_serialized_string;
  protobuf::io::StringOutputStream string_stream(&proto_serialized_string);
  protobuf::io::CodedOutputStream stream(&string_stream);
  // Ensure the serialization is deterministic so that equal ConvParameters
  // have equal serialized strings and therefore equal hash codes.
  stream.SetSerializationDeterministic(true);
  proto.SerializeToCodedStream(&stream);
  return proto_serialized_string;
}

uint64 HashProto(const protobuf::Message& proto) {
  return Hash64(SerializeProtoDeterministic(proto));
}

}  // namespace autotune_maps_utils
}  // namespace tensorflow
