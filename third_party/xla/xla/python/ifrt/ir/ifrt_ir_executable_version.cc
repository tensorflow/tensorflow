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

#include "absl/container/flat_hash_map.h"
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
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

[[maybe_unused]] char IfrtIrExecutableVersionDeserializeOptions::ID = 0;
[[maybe_unused]] char IfrtIrExecutableVersion::ID = 0;

namespace {
absl::StatusOr<int> FindLogicalDeviceId(
    const absl::flat_hash_map<xla::ifrt::DeviceId, int>&
        device_id_to_logical_device_id,
    xla::ifrt::DeviceId device_id) {
  auto it = device_id_to_logical_device_id.find(device_id);
  if (it == device_id_to_logical_device_id.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Device id ", device_id.value(), " not found in device assignments"));
  }
  return it->second;
}

absl::StatusOr<xla::ifrt::Device*> FindDevice(
    const xla::ifrt::DeviceListRef& devices, xla::ifrt::DeviceId device_id) {
  for (xla::ifrt::Device* device : devices->devices()) {
    if (device->Id() == device_id) {
      return device;
    }
  }
  return absl::NotFoundError(absl::StrCat("Device id ", device_id.value(),
                                          " not found in device list"));
}

absl::StatusOr<xla::ifrt::DeviceListRef> MakeDeviceListFromAtomDeviceIds(
    xla::ifrt::Client& client, const xla::ifrt::DeviceListRef& devices,
    const std::vector<xla::ifrt::DeviceId>& atom_device_ids) {
  std::vector<xla::ifrt::Device*> device_ptrs;
  device_ptrs.reserve(atom_device_ids.size());
  for (const auto& atom_device_id : atom_device_ids) {
    TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                        FindDevice(devices, atom_device_id));
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
  const auto* other_ifrt_ir_executable_version =
      llvm::cast<IfrtIrExecutableVersion>(&other);
  // This version is compatible with the other IFRT IR version if the other's
  // atom executables are compatible with the client on the given devices.
  for (const auto& [runtime_abi_version, atom_devices] :
       other_ifrt_ir_executable_version->runtime_abi_versions) {
    absl::StatusOr<xla::ifrt::DeviceListRef> atom_device_list =
        MakeDeviceListFromAtomDeviceIds(client, devices, atom_devices);
    if (!atom_device_list.ok()) {
      LOG(ERROR) << "Failed to make device list from atom device ids: "
                 << atom_device_list.status();
      return false;
    }
    absl::Status status =
        client.GetDefaultCompiler()->IsExecutableVersionCompatible(
            *runtime_abi_version, *atom_device_list);
    if (!status.ok()) {
      LOG(ERROR) << "Executable version not compatible: " << status;
      return false;
    }
  }
  return true;
}

absl::StatusOr<IfrtIrExecutableVersionProto> IfrtIrExecutableVersion::ToProto(
    SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for IfrtIrExecutableVersion serialization"));
  }

  // Create a map from runtime device id to logical device id.
  absl::flat_hash_map<xla::ifrt::DeviceId, int> device_id_to_logical_device_id;
  for (int i = 0; i < device_assignments.size(); ++i) {
    auto [_, inserted] =
        device_id_to_logical_device_id.insert({device_assignments[i], i});
    if (!inserted) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate device id ", device_assignments[i].value(),
                       " found in device assignments"));
    }
  }

  IfrtIrExecutableVersionProto proto;
  proto.set_version_number(SerDesVersionNumber(0).value());
  proto.set_ifrt_ir_version(ifrt_version.toString());
  for (const auto& [runtime_abi_version, atom_devices] : runtime_abi_versions) {
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

    for (auto& device : atom_devices) {
      TF_ASSIGN_OR_RETURN(
          int logical_device_id,
          FindLogicalDeviceId(device_id_to_logical_device_id, device));
      atom_executable_version_proto.add_logical_device_indexes(
          logical_device_id);
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

    for (auto logical_device_id :
         atom_executable_version_proto.logical_device_indexes()) {
      if (logical_device_id >= device_assignments.size()) {
        return absl::InvalidArgumentError("Logical device id is out of range");
      }
      atom_executable_version.devices.push_back(
          device_assignments[logical_device_id]);
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
          "{platform_id=", xla_executable_version->platform_id,
          ", runtime_abi_version=", xla_executable_version->runtime_abi_version,
          ", devices=[",
          absl::StrJoin(atom_devices, ",",
                        [](std::string* out, xla::ifrt::DeviceId device_id) {
                          absl::StrAppend(out, device_id.value());
                        }),
          "]}"));
    } else {
      runtime_abi_version_strs.push_back("(unknown)");
    }
  }
  return absl::StrCat("IfrtIrExecutableVersion(", ifrt_version.toString(),
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
