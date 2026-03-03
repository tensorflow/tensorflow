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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_executable_version.pb.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/serdes_week_4_old_version_accessor.h"
#include "xla/python/pjrt_ifrt/xla_executable_version.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

[[maybe_unused]] char IfrtIrExecutableVersionDeserializeOptions::ID = 0;
[[maybe_unused]] char IfrtIrExecutableVersion::ID = 0;

namespace {

absl::StatusOr<xla::ifrt::DeviceListRef> MakeDeviceListFromAtomDeviceIds(
    xla::ifrt::Client& client,
    const std::vector<xla::ifrt::DeviceId>& device_assignments,
    const std::vector<IfrtIrLogicalDeviceId>& atom_logical_device_ids) {
  std::vector<xla::ifrt::Device*> device_ptrs;
  device_ptrs.reserve(atom_logical_device_ids.size());
  for (const auto& atom_logical_device_id : atom_logical_device_ids) {
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::Device * device,
        client.LookupDevice(
            device_assignments[atom_logical_device_id.value()]));
    device_ptrs.push_back(device);
  }
  return client.MakeDeviceList(device_ptrs);
}

}  // namespace

IfrtIrExecutableVersion::IfrtIrExecutableVersion(
    Version ifrt_version,
    absl::Span<const xla::ifrt::DeviceId> device_assignments,
    std::vector<AtomExecutableVersion> runtime_abi_versions)
    : ifrt_version(std::move(ifrt_version)),
      device_assignments(device_assignments.begin(), device_assignments.end()),
      runtime_abi_versions(std::move(runtime_abi_versions)) {}

absl::Status IfrtIrExecutableVersion::IsCompatibleWith(
    const ExecutableVersion& other) const {
  if (this == &other) {
    return absl::OkStatus();
  }
  if (auto other_ifrt_ir_executable_version =
          llvm::dyn_cast<IfrtIrExecutableVersion>(&other)) {
    if (ifrt_version == other_ifrt_ir_executable_version->ifrt_version) {
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        "Executable version is not compatible with current version");
  }
  return absl::InvalidArgumentError(
      "Other ExecutableVersion is not IfrtIrExecutableVersion");
}

absl::Status IfrtIrExecutableVersion::IsCompatibleWith(
    xla::ifrt::Client& client, const ExecutableVersion& other) const {
  TF_RETURN_IF_ERROR(IsCompatibleWith(other));
  const auto* other_ifrt_ir_executable_version =
      llvm::cast<IfrtIrExecutableVersion>(&other);
  // This version is compatible with the other IFRT IR version if the other's
  // atom executables are compatible with this client on the assigned devices.
  for (const auto& [other_atom_abi_version, other_atom_logical_device_ids] :
       other_ifrt_ir_executable_version->runtime_abi_versions) {
    TF_ASSIGN_OR_RETURN(
        DeviceListRef other_atom_device_list,
        MakeDeviceListFromAtomDeviceIds(
            client, other_ifrt_ir_executable_version->device_assignments,
            other_atom_logical_device_ids));
    TF_RETURN_IF_ERROR(
        client.GetDefaultCompiler()->IsExecutableVersionCompatible(
            *other_atom_abi_version, other_atom_device_list));
  }
  return absl::OkStatus();
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
  proto.set_ifrt_ir_version(ifrt_version.toString());
  for (const auto& [runtime_abi_version, atom_logical_device_ids] :
       runtime_abi_versions) {
    xla::ifrt::Serialized serialized_runtime_abi_version;
    TF_ASSIGN_OR_RETURN(
        serialized_runtime_abi_version,
        xla::ifrt::Serialize(
            *runtime_abi_version,
            std::make_unique<xla::ifrt::SerializeOptions>(
                xla::ifrt::SerDesWeek4OldVersionAccessor::Get())));

    AtomExecutableVersionProto atom_executable_version_proto;
    if (!serialized_runtime_abi_version.SerializeToString(
            atom_executable_version_proto.mutable_executable_version())) {
      return absl::InternalError("Failed to serialize runtime ABI version");
    }

    for (const auto& atom_logical_device_id : atom_logical_device_ids) {
      atom_executable_version_proto.add_logical_device_indexes(
          atom_logical_device_id.value());
    }
    *proto.add_executable_versions() = atom_executable_version_proto;
  }

  return proto;
}

absl::StatusOr<std::unique_ptr<IfrtIrExecutableVersion>>
IfrtIrExecutableVersion::FromProto(
    std::vector<xla::ifrt::DeviceId> device_assignments,
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
    xla::ifrt::Serialized serialized_runtime_abi_version;
    if (!serialized_runtime_abi_version.ParseFromString(
            atom_executable_version_proto.executable_version())) {
      return absl::InvalidArgumentError(
          "Failed to parse serialized runtime ABI version");
    }

    TF_ASSIGN_OR_RETURN(atom_executable_version.runtime_abi_version,
                        xla::ifrt::Deserialize<xla::ifrt::ExecutableVersion>(
                            serialized_runtime_abi_version,
                            std::make_unique<xla::ifrt::DeserializeOptions>()));

    for (auto logical_device_index :
         atom_executable_version_proto.logical_device_indexes()) {
      if (logical_device_index >= device_assignments.size()) {
        return absl::InvalidArgumentError("Logical device id is out of range");
      }
      atom_executable_version.logical_device_ids.push_back(
          IfrtIrLogicalDeviceId(logical_device_index));
    }

    atom_executable_versions.emplace_back(std::move(atom_executable_version));
  }

  auto ifrt_version = Version::fromString(proto.ifrt_ir_version());
  if (mlir::failed(ifrt_version)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse IFRT IR version: ", proto.ifrt_ir_version()));
  }
  return std::make_unique<IfrtIrExecutableVersion>(
      *ifrt_version, device_assignments, std::move(atom_executable_versions));
}

std::string IfrtIrExecutableVersion::ToString() const {
  std::vector<std::string> runtime_abi_version_strs;
  runtime_abi_version_strs.reserve(runtime_abi_versions.size());
  for (const auto& [runtime_abi_version, atom_devices] : runtime_abi_versions) {
    if (auto xla_executable_version =
            llvm::dyn_cast<XlaExecutableVersion>(runtime_abi_version.get())) {
      runtime_abi_version_strs.push_back(absl::StrCat(
          "{platform_id=", xla_executable_version->platform_id, " devices=[",
          absl::StrJoin(atom_devices, ",",
                        [](std::string* out, IfrtIrLogicalDeviceId device_id) {
                          absl::StrAppend(out, device_id.value());
                        }),
          "]}"));
    } else {
      runtime_abi_version_strs.push_back("(unknown)");
    }
  }
  return absl::StrCat("IfrtIrExecutableVersion(", ifrt_version.toString(),
                      ", device_assignments=[",
                      absl::StrJoin(device_assignments, ", "),
                      ", runtime_abi_versions=[",
                      absl::StrJoin(runtime_abi_version_strs, ", "), "])");
}

absl::StatusOr<std::unique_ptr<IfrtIrExecutableVersion>>
ToIfrtIrExecutableVersion(
    std::unique_ptr<ExecutableVersion> executable_version) {
  if (!executable_version) {
    return absl::InvalidArgumentError("executable_version is null");
  }
  if (llvm::isa_and_nonnull<IfrtIrExecutableVersion>(
          executable_version.get())) {
    return xla::unique_ptr_down_cast<IfrtIrExecutableVersion>(
        std::move(executable_version));
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
    std::string serialized;
    if (!proto.SerializeToString(&serialized)) {
      return absl::InternalError(
          "Failed to serialize IfrtIrExecutableVersionProto");
    }
    return serialized;
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    auto* deserialize_options =
        llvm::dyn_cast<IfrtIrExecutableVersionDeserializeOptions>(
            options.get());
    if (deserialize_options == nullptr) {
      return absl::InvalidArgumentError(
          "IfrtIrExecutableVersionDeserializeOptions not found");
    }
    IfrtIrExecutableVersionProto proto;
    if (!proto.ParseFromString(serialized)) {
      return absl::InvalidArgumentError(
          "Failed to parse IfrtIrExecutableVersionProto");
    }
    return IfrtIrExecutableVersion::FromProto(
        deserialize_options->device_assignments, proto);
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
