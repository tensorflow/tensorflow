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

#include "xla/python/ifrt/ir/ifrt_ir_executable_version.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_executable_version.pb.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/pjrt_ifrt/xla_executable_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

[[maybe_unused]] char IfrtIrExecutableVersion::ID = 0;
[[maybe_unused]] char IfrtIrExecutableVersionDeserializeOptions::ID = 0;

IfrtIrExecutableVersion::IfrtIrExecutableVersion(
    std::string ifrt_version,
    std::vector<AtomExecutableVersion> runtime_abi_versions)
    : ifrt_version(std::move(ifrt_version)),
      runtime_abi_versions(std::move(runtime_abi_versions)) {}

bool IfrtIrExecutableVersion::IsCompatibleWith(
    const ExecutableVersion& other) const {
  if (this == &other) {
    return true;
  }
  if (auto other_ifrt_ir_executable_version =
          llvm::dyn_cast<IfrtIrExecutableVersion>(&other)) {
    return (ifrt_version == other_ifrt_ir_executable_version->ifrt_version);
  }
  return false;
}

bool IfrtIrExecutableVersion::IsCompatibleWith(
    xla::ifrt::Client& client, const xla::ifrt::DeviceListRef& devices,
    const ExecutableVersion& other) const {
  if (!IsCompatibleWith(other)) {
    return false;
  }
  if (auto other_ifrt_ir_executable_version =
          llvm::dyn_cast<IfrtIrExecutableVersion>(&other)) {
    if (other_ifrt_ir_executable_version->runtime_abi_versions.empty()) {
      return true;
    }
    // This version is compatible with the other IFRT IR version if the other's
    // atom executables are compatible with the client on the given devices.
    for (const auto& [runtime_abi_version, atom_devices] :
         other_ifrt_ir_executable_version->runtime_abi_versions) {
      if (!client.GetDefaultCompiler()
               ->IsExecutableVersionCompatible(*runtime_abi_version,
                                               atom_devices)
               .ok()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

absl::StatusOr<IfrtIrExecutableVersionProto> IfrtIrExecutableVersion::ToProto(
    SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for IfrtIrExecutableVersion serialization"));
  }

  IfrtIrExecutableVersionProto proto;
  proto.set_version_number(SerDesVersionNumber(0).value());
  proto.set_ifrt_ir_version(ifrt_version);
  for (const auto& [runtime_abi_version, atom_devices] : runtime_abi_versions) {
    if (auto xla_executable_version =
            llvm::dyn_cast<XlaExecutableVersion>(runtime_abi_version.get())) {
      AtomExecutableVersionProto atom_executable_version_proto;
      TF_ASSIGN_OR_RETURN(
          *atom_executable_version_proto.mutable_executable_version(),
          xla_executable_version->ToProto(version));
      *atom_executable_version_proto.mutable_devices() =
          atom_devices->ToProto(version);
      *proto.add_executable_versions() = atom_executable_version_proto;
    } else {
      return absl::UnimplementedError(
          "IfrtIrExecutableVersion::ToProto only supports atom executable "
          "versions of type XlaExecutableVersion");
    }
  }

  return proto;
}

absl::StatusOr<std::unique_ptr<IfrtIrExecutableVersion>>
IfrtIrExecutableVersion::FromProto(xla::ifrt::Client* client,
                                   const IfrtIrExecutableVersionProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version_number,
                     " for IfrtIrExecutableVersion deserialization"));
  }
  std::vector<AtomExecutableVersion> atom_executable_versions;
  for (const auto& atom_executable_version_proto :
       proto.executable_versions()) {
    AtomExecutableVersion atom_executable_version;
    TF_ASSIGN_OR_RETURN(
        atom_executable_version.runtime_abi_version,
        XlaExecutableVersion::FromProto(
            atom_executable_version_proto.executable_version()));
    TF_ASSIGN_OR_RETURN(
        atom_executable_version.devices,
        DeviceList::FromProto(client, atom_executable_version_proto.devices()));
    atom_executable_versions.emplace_back(std::move(atom_executable_version));
  }
  return std::make_unique<IfrtIrExecutableVersion>(
      proto.ifrt_ir_version(), std::move(atom_executable_versions));
}

absl::StatusOr<std::unique_ptr<IfrtIrExecutableVersion>>
ToIfrtIrExecutableVersion(
    std::unique_ptr<ExecutableVersion> executable_version) {
  if (!executable_version) {
    return absl::InvalidArgumentError("executable_version is null");
  }
  if (auto* ifrt_ir_executable_version =
          llvm::dyn_cast<IfrtIrExecutableVersion>(executable_version.get())) {
    executable_version.release();
    return absl::WrapUnique(ifrt_ir_executable_version);
  }
  return absl::InvalidArgumentError(
      "executable_version is not IfrtIrExecutableVersion");
}

namespace {

class IfrtIrExecutableVersionSerDes
    : public llvm::RTTIExtends<IfrtIrExecutableVersionSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::IfrtIrExecutableVersion";
  }

  absl::StatusOr<std::string> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions> options) override {
    const SerDesVersion version = GetRequestedSerDesVersion(options.get());

    const auto& ifrt_ir_executable_version =
        llvm::cast<IfrtIrExecutableVersion>(serializable);

    TF_ASSIGN_OR_RETURN(IfrtIrExecutableVersionProto proto,
                        ifrt_ir_executable_version.ToProto(version));
    return proto.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    if (auto* ifrt_ir_executable_version_options =
            llvm::dyn_cast<IfrtIrExecutableVersionDeserializeOptions>(
                options.get())) {
      IfrtIrExecutableVersionProto proto;
      if (!proto.ParseFromString(serialized)) {
        return absl::InvalidArgumentError(
            "Failed to parse IfrtIrExecutableVersionProto");
      }
      return IfrtIrExecutableVersion::FromProto(
          ifrt_ir_executable_version_options->client, proto);
    }
    return absl::InvalidArgumentError(
        "IfrtIrExecutableVersionDeserializeOptions not found");
  }

  IfrtIrExecutableVersionSerDes() = default;
  ~IfrtIrExecutableVersionSerDes() override = default;

  static char ID;  // NOLINT
};

}  // namespace

[[maybe_unused]] char IfrtIrExecutableVersionSerDes::ID = 0;

bool register_ifrt_ir_executable_version_serdes = ([]{
    RegisterSerDes<IfrtIrExecutableVersion>(
        std::make_unique<IfrtIrExecutableVersionSerDes>());
    }(), true);

}  // namespace ifrt
}  // namespace xla
