/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_DEVICE_DESCRIPTION_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_DEVICE_DESCRIPTION_H_

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_common.h"

namespace xla {

using PjRtDeviceAttribute = PjRtValueType;

class PjRtDeviceDescription {
 public:
  virtual ~PjRtDeviceDescription() = default;

  // The ID of this device. IDs are unique among devices of this type
  // (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
  // hosts' devices.  This is the ID that should be used in a DeviceAssignment.
  virtual int id() const = 0;

  // The index of the process that this device belongs to, i.e. is addressable
  // from. This is not always identical to PjRtClient::process_index() in a
  // multi-process setting, where each client can see devices from all
  // processes, but only a subset of them are addressable and have the same
  // process_index as the client.
  virtual int process_index() const = 0;

  // A vendor-dependent string that uniquely identifies the kind of device,
  // e.g., "Tesla V100-SXM2-16GB". May be used to determine whether two GPUs are
  // compatible compilation.
  virtual std::string_view device_kind() const = 0;

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current device unambiguously.
  virtual std::string_view DebugString() const = 0;

  // Debug string suitable for reading by end users, should be reasonably terse,
  // for example: "CpuDevice(id=0)".
  virtual std::string_view ToString() const = 0;

  // Returns vendor specific attributes about the device. For example the model
  // number of a GPU, or the mesh coordinates of a TPU device. The returned
  // reference will remain valid for the lifetime of the PjRtDevice.
  virtual const absl::flat_hash_map<std::string, PjRtDeviceAttribute>&
  Attributes() const = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_DEVICE_DESCRIPTION_H_
