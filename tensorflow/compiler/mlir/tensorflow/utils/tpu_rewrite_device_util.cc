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

#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <type_traits>

#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

constexpr char kDeviceTPUSystem[] = "TPU_SYSTEM";
constexpr char kDeviceTPU[] = "TPU";

using Device = DeviceNameUtils::ParsedName;
using Devices = llvm::ArrayRef<DeviceNameUtils::ParsedName>;

namespace {
// Finds matching devices in `devices` based on pattern `spec`.
void FindMatchingDevices(Devices devices, const Device& spec,
                         llvm::SmallVectorImpl<Device>* matched_devices) {
  for (const auto& device : devices)
    if (DeviceNameUtils::IsCompleteSpecification(spec, device))
      matched_devices->push_back(device);
}

// Creates error message for a conflicting attribute of a device.
template <typename T>
Status MismatchedTPUSystemAttributeErr(absl::string_view attribute, T a, T b) {
  return errors::InvalidArgument("found ", kDeviceTPUSystem,
                                 " devices with conflicting ", attribute, "s '",
                                 a, "' and '", b, "'");
}

// Finds TPU_SYSTEM:0 devices in `devices`. If multiple TPU_SYSTEM devices are
// found, the first one lexicographically is returned. If no TPU_SYSTEM device
// is found or if there are multiple TPU_SYSTEM devices with different jobs or
// replicas, a failure will be returned.
Status GetTPUSystemDevices(Devices devices,
                           llvm::SmallVectorImpl<Device>* matched_devices) {
  Device spec;
  spec.type = kDeviceTPUSystem;
  spec.has_type = true;
  spec.id = 0;
  spec.has_id = true;

  llvm::SmallVector<Device, 8> system_devices;
  FindMatchingDevices(devices, spec, &system_devices);
  if (system_devices.empty())
    return errors::InvalidArgument("no ", kDeviceTPUSystem, " devices found");

  // Check that all system devices are part of the same job.
  const auto& job = system_devices[0].job;
  auto replica = system_devices[0].replica;
  for (const auto& device : llvm::make_range(std::next(system_devices.begin()),
                                             system_devices.end())) {
    if (device.job != job)
      return MismatchedTPUSystemAttributeErr("job", job, device.job);

    if (device.replica != replica)
      return MismatchedTPUSystemAttributeErr("replica", replica,
                                             device.replica);
  }

  // Sort by task to be deterministic.
  std::sort(system_devices.begin(), system_devices.end(),
            [](const Device& a, const Device& b) { return a.task < b.task; });

  matched_devices->swap(system_devices);

  return Status::OK();
}

// Finds TPU devices associated to system device based on spec (e.g. from
// GetTPUSystemDevices). If the number of TPU devices per host do not match for
// every host, a failure will be returned.
Status GetTPUDevices(
    Devices devices, llvm::ArrayRef<Device> system_devices,
    llvm::SmallVectorImpl<llvm::SmallVector<Device, 8>>* tpu_devices) {
  tpu_devices->reserve(system_devices.size());

  auto lookup = [&devices](Device device_spec) {
    device_spec.has_type = true;
    device_spec.type = kDeviceTPU;
    // Enumerate all the available TPUs.
    device_spec.has_id = false;

    llvm::SmallVector<Device, 8> host_tpu_devices;
    FindMatchingDevices(devices, device_spec, &host_tpu_devices);

    // Sort devices by id.
    std::sort(host_tpu_devices.begin(), host_tpu_devices.end(),
              [](const Device& i, const Device& j) { return i.id < j.id; });
    return host_tpu_devices;
  };

  int num_tpus_per_host = 0;
  {
    const auto& device = system_devices[0];
    auto host_tpu_devices = lookup(device);
    num_tpus_per_host = host_tpu_devices.size();
    tpu_devices->push_back(std::move(host_tpu_devices));
  }

  for (const auto& device_spec : llvm::make_range(
           std::next(system_devices.begin()), system_devices.end())) {
    auto host_tpu_devices = lookup(device_spec);
    // Check number of TPU devices per host all match.
    if (num_tpus_per_host != host_tpu_devices.size())
      return errors::InvalidArgument(
          "expected the number of TPU devices per host to be ",
          num_tpus_per_host, ", got ", host_tpu_devices.size());

    tpu_devices->push_back(std::move(host_tpu_devices));
  }

  return Status::OK();
}

// Finds the compilation device from system device.
std::string GetTPUCompilationDevice(Device system_device) {
  // TODO(b/110910013) GetTPUSystemDevices parses the spec and returns the
  // TPU_SYSTEM device, which we replace with the CPU device. We do this
  // replacement because we want to place the `tf._TPUCompileMlir` explicitly on
  // CPU devices of the same job as the TPU_SYSTEM device.
  system_device.type = tensorflow::DEVICE_CPU;
  return DeviceNameUtils::ParsedNameToString(system_device);
}

}  // anonymous namespace

Status GetTPUCompilationAndExecutionDevices(
    Devices devices, int num_replicas, int num_cores_per_replica,
    std::string* compilation_device,
    llvm::SmallVectorImpl<std::string>* execution_devices) {
  if (num_cores_per_replica != 1)
    return errors::Unimplemented(
        "num_cores_per_replica must be equal to 1, got ",
        num_cores_per_replica);

  // Collect TPU_SYSTEM devices.
  llvm::SmallVector<Device, 8> system_devices;
  TF_RETURN_IF_ERROR(GetTPUSystemDevices(devices, &system_devices));

  // Collect TPU devices based on TPU_SYSTEM devices collected earlier.
  llvm::SmallVector<llvm::SmallVector<Device, 8>, 8> tpu_devices;
  TF_RETURN_IF_ERROR(GetTPUDevices(devices, system_devices, &tpu_devices));

  const int num_tasks = tpu_devices.size();
  const int num_tpus_per_task = tpu_devices[0].size();
  const int num_tpu_devices = num_tasks * num_tpus_per_task;

  if (num_replicas != 1 && num_replicas != num_tpu_devices)
    return errors::Unimplemented("num_replicas must be equal to 1 or ",
                                 num_tpu_devices, ", got ", num_replicas);

  *compilation_device = GetTPUCompilationDevice(system_devices[0]);

  // TODO(lyandy): Update `execution_devices` to be 2D when support for explicit
  // topologies is added.
  execution_devices->reserve(num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    const int task = i / num_tpus_per_task;
    const int device = i % num_tpus_per_task;
    execution_devices->push_back(
        tensorflow::DeviceNameUtils::ParsedNameToString(
            tpu_devices[task][device]));
  }

  return Status::OK();
}

}  // namespace tensorflow
