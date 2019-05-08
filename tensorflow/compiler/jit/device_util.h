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

#ifndef TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
#define TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_

#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace jit {
// Caches some miscellaneous information about TF devices.  Thread compatible.
class DeviceInfoCache {
 public:
  xla::StatusOr<const XlaOpRegistry::DeviceRegistration*> GetCompilationDevice(
      absl::string_view device_name);
  xla::StatusOr<std::reference_wrapper<const DeviceType>> GetDeviceTypeFor(
      absl::string_view device_name);

 private:
  absl::flat_hash_map<string, const XlaOpRegistry::DeviceRegistration*>
      device_to_device_registration_;
  absl::flat_hash_map<string, std::unique_ptr<DeviceType>>
      device_to_device_type_;
};

}  // namespace jit

// Returns the DeviceType corresponding to 'device'.
Status DeviceNameToDeviceType(const string& device, DeviceType* device_type);

// Picks the device for which XLA should compile a cluster that contains
// operations placed in devices in `device_names`.  For instance a cluster that
// contains operations solely placed on the CPU will be compiled into a CPU
// executable by XLA, whereas a cluster that contains operations placed on the
// CPU and also operations placed on the GPU will be compiled into a GPU
// executable.
//
// Returns a non-OK Status if no unambiguous choice of device exists.
//
// We choose the device using the following rules:
//
//  - It is an error for `device_names` to contain more than one device of the
//    same type.
//  - GPU is preferred over CPU.
//  - If `allow_mixing_unknown_and_cpu` is true then unknown devices are
//    preferred over CPU.
//  - XLA devices count as "unrecognized devices".
//
// This set of rules above implicitly assume that XLA:GPU can compile all
// operations in the cluster that XLA:CPU can compile, and if
// `allow_mixing_unknown_and_cpu` then the unrecognized device can also compile
// all operations in the cluster that XLA:CPU can compile.
//
// We provide the `allow_mixing_unknown_and_cpu` knob so that we can do both of
// the following things:
//
// - Let MarkForCompilationPass not inject CPU-placed operations into clusters
//   that will run on unknown devices (because the unknown XLA backend may not
//   support every operation supported by CPU).
// - Let BuildXlaOpsPass successfully infer a compilation device for a cluster
//   that contains nodes placed on both the CPU and on unknown devices.  In this
//   case it is the responsibility of the optimization pass that injected the
//   CPU nodes into the cluster to ensure that these nodes can be compiled by
//   the unknown XLA backend.
Status PickDeviceForXla(absl::Span<const string> device_names,
                        bool allow_mixing_unknown_and_cpu,
                        string* out_device_picked);

// This is like `PickDeviceForXla` except that it returns false (instead of a
// non-OK Status) in `out_can_pick_device` if no unambiguous choice of device
// exists.
Status CanPickDeviceForXla(absl::Span<const string> device_names,
                           bool allow_mixing_unknown_and_cpu,
                           bool* out_can_pick_device);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_DEVICE_INFO_CACHE_H_
