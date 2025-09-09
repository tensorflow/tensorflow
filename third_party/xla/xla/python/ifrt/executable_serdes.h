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

#ifndef XLA_PYTHON_IFRT_EXECUTABLE_SERDES_H_
#define XLA_PYTHON_IFRT_EXECUTABLE_SERDES_H_

#include <optional>
#include <utility>

#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/serdes.h"

namespace xla {
namespace ifrt {

// Abstract options for deserializing an `Executable` and load it as
// `LoadedExecutable`. This option structure is to express legacy compilation
// options that are not included in the program.
//
// TODO(hyeontaek): Make an new `LoadOptions` that is specific for loading.
struct DeserializeExecutableOptions
    : llvm::RTTIExtends<DeserializeExecutableOptions, DeserializeOptions> {
  DeserializeExecutableOptions() = default;
  explicit DeserializeExecutableOptions(std::optional<DeviceListRef> devices)
      : devices(std::move(devices)) {}

  // The devices to load the executable on to.
  // If unspecified, let the runtime decide the devices to use for loading.
  // TODO(hyeontaek): Require `devices` to be always provided.
  std::optional<DeviceListRef> devices;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_EXECUTABLE_SERDES_H_
