/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_IR_EXECUTABLE_VERSION_H_
#define XLA_PYTHON_IFRT_IR_IFRT_IR_EXECUTABLE_VERSION_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_executable_version.pb.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

struct IfrtIrExecutableVersionDeserializeOptions
    : llvm::RTTIExtends<IfrtIrExecutableVersionDeserializeOptions,
                        xla::ifrt::DeserializeOptions> {
  explicit IfrtIrExecutableVersionDeserializeOptions(xla::ifrt::Client* client)
      : client(client) {}

  static char ID;  // NOLINT

  xla::ifrt::Client* client = nullptr;
};

struct IfrtIrExecutableVersion
    : llvm::RTTIExtends<IfrtIrExecutableVersion, ExecutableVersion> {
  // Tracking the runtime ABI version of an atom executable and the devices that
  // it is to be used on.
  struct AtomExecutableVersion {
    std::unique_ptr<xla::ifrt::ExecutableVersion> runtime_abi_version;
    DeviceListRef devices;
  };

  IfrtIrExecutableVersion() = default;
  explicit IfrtIrExecutableVersion(
      std::string ifrt_version,
      std::vector<AtomExecutableVersion> runtime_abi_versions = {});

  // String of the form "major.minor.patch", representing the IFRT IR version.
  std::string ifrt_version;
  // Atom executable runtime ABI versions and their device assignments.
  std::vector<AtomExecutableVersion> runtime_abi_versions;

  // Returns true if the IFRT IR version is compatible with the other version.
  bool IsCompatibleWith(const ExecutableVersion& other) const override;

  // Returns true if the IFRT IR version is compatible with the other version
  // and the runtime ABI version is compatible with the given client on the
  // given devices.
  bool IsCompatibleWith(xla::ifrt::Client& client,
                        const xla::ifrt::DeviceListRef& devices,
                        const ExecutableVersion& other) const;

  absl::StatusOr<IfrtIrExecutableVersionProto> ToProto(
      SerDesVersion version = SerDesVersion::current()) const;
  static absl::StatusOr<std::unique_ptr<IfrtIrExecutableVersion>> FromProto(
      xla::ifrt::Client* client, const IfrtIrExecutableVersionProto& proto);

  static char ID;  // NOLINT
};

absl::StatusOr<std::unique_ptr<IfrtIrExecutableVersion>>
ToIfrtIrExecutableVersion(
    std::unique_ptr<ExecutableVersion> executable_version);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IFRT_IR_EXECUTABLE_VERSION_H_
