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

#include "xla/python/ifrt/ir/serialization_utils.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/message_lite.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/messages/serialize_message.h"
#include "stablehlo/dialect/Version.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/serialized_executable_metadata.pb.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

void SerializeAndUpdateLocation(riegeli::CordWriter<absl::Cord*>& writer,
                                SerializedObjectLocation* location,
                                absl::Cord serialized_data) {
  location->set_offset(writer.pos());
  writer.Write(serialized_data);
  location->set_size(writer.pos() - location->offset());
}

absl::Status SerializeAndUpdateLocation(
    riegeli::CordWriter<absl::Cord*>& writer,
    SerializedObjectLocation* location, const google::protobuf::MessageLite& message) {
  location->set_offset(writer.pos());
  TF_RETURN_IF_ERROR(riegeli::SerializeMessage(message, writer));
  location->set_size(writer.pos() - location->offset());
  return absl::OkStatus();
}

absl::Status SerializeIfrtIrProgram(riegeli::CordWriter<absl::Cord*>& writer,
                                    SerializedObjectLocation* location,
                                    xla::ifrt::IfrtIRProgram* ifrt_ir_program) {
  auto options = std::make_unique<SerializeIfrtIRProgramOptions>(
      Version::getCurrentVersion().toString(),
      ::mlir::vhlo::Version::getCurrentVersion().toString(),
      /*version_in_place=*/false);

  Serialized serialized;
  TF_ASSIGN_OR_RETURN(
      serialized, xla::ifrt::Serialize(*ifrt_ir_program, std::move(options)));

  return SerializeAndUpdateLocation(writer, location, serialized);
}

absl::StatusOr<std::unique_ptr<xla::ifrt::IfrtIRProgram>>
DeserializeIfrtIrProgram(
    const SerializedIfrtIrExecutableMetadata& metadata,
    absl::string_view serialized_executable_payload,
    std::unique_ptr<xla::ifrt::DeserializeIfrtIRProgramOptions> options) {
  absl::string_view program_data = serialized_executable_payload.substr(
      metadata.ifrt_ir_program_location().offset(),
      metadata.ifrt_ir_program_location().size());
  Serialized serialized_program;
  if (!serialized_program.ParseFromString(program_data)) {
    return absl::InvalidArgumentError(
        "Failed to parse program from serialized loaded executable");
  }
  return xla::ifrt::Deserialize<xla::ifrt::IfrtIRProgram>(serialized_program,
                                                          std::move(options));
}

absl::StatusOr<int> FindLogicalDeviceId(
    const absl::flat_hash_map<xla::ifrt::DeviceId, int>&
        device_id_to_logical_device_id,
    xla::ifrt::Device* device) {
  if (!device) {
    return absl::InvalidArgumentError("Device is null");
  }
  auto it = device_id_to_logical_device_id.find(device->Id());
  if (it == device_id_to_logical_device_id.end()) {
    return absl::NotFoundError(
        absl::StrCat("Device id ", device->Id().value(),
                     " not found in device assignments"));
  }
  return it->second;
}

absl::Status SerializeIfrtIrAtomExecutable(
    riegeli::CordWriter<absl::Cord*>& writer,
    absl::flat_hash_map<xla::ifrt::DeviceId, int>&
        device_id_to_logical_device_id,
    SerializedIfrtIrAtomExecutableMetadata* metadata, absl::string_view name,
    std::shared_ptr<LoadedExecutable> executable) {
  metadata->set_name(name);
  std::optional<xla::ifrt::DeviceListRef> device_list = executable->devices();
  if (!device_list.has_value()) {
    return absl::InvalidArgumentError(
        "Portable executables are not supported.");
  }
  absl::Span<xla::ifrt::Device* const> devices = (*device_list)->devices();

  // Map used devices back to logical device ids for serialization.
  for (const auto& device : devices) {
    TF_ASSIGN_OR_RETURN(
        int logical_device_id,
        FindLogicalDeviceId(device_id_to_logical_device_id, device));
    metadata->add_logical_device_ids(logical_device_id);
  }

  TF_ASSIGN_OR_RETURN(std::string serialized_executable,
                      executable->Serialize());
  SerializeAndUpdateLocation(writer, metadata->mutable_executable_location(),
                             absl::Cord(std::move(serialized_executable)));
  return absl::OkStatus();
}

absl::Status SerializeIfrtIrAtomExecutables(
    riegeli::CordWriter<absl::Cord*>& writer,
    SerializedIfrtIrExecutableMetadata* metadata,
    std::shared_ptr<CompiledIfrtIrProgram> ifrt_ir_program) {
  // Create a map from runtime device id to logical device id.
  absl::flat_hash_map<xla::ifrt::DeviceId, int> device_id_to_logical_device_id;
  for (int i = 0; i < ifrt_ir_program->device_assignments.size(); ++i) {
    const xla::ifrt::DeviceId device_id =
        ifrt_ir_program->device_assignments[i];
    auto [_, inserted] = device_id_to_logical_device_id.insert({device_id, i});
    if (!inserted) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate device id ", device_id.value(),
                       " found in device assignments"));
    }
  }

  for (const auto& [name, executable] :
       *ifrt_ir_program->atom_program_executables) {
    TF_RETURN_IF_ERROR(SerializeIfrtIrAtomExecutable(
        writer, device_id_to_logical_device_id,
        metadata->add_atom_program_executables(), name, executable));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<xla::ifrt::XlaDeserializeExecutableOptions>>
CreateXlaDeserializeExecutableOptions(
    xla::ifrt::Client* client,
    absl::Span<const xla::ifrt::DeviceId> device_assignments,
    const SerializedIfrtIrAtomExecutableMetadata& metadata) {
  std::vector<Device*> atom_devices;
  for (auto logical_device_id : metadata.logical_device_ids()) {
    if (logical_device_id >= device_assignments.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Logical device id ", logical_device_id,
                       " is out of range. Device assignments size: ",
                       device_assignments.size()));
    }
    TF_ASSIGN_OR_RETURN(
        Device * device,
        client->LookupDevice(device_assignments[logical_device_id]));
    atom_devices.push_back(device);
  }
  TF_ASSIGN_OR_RETURN(DeviceListRef atom_device_list,
                      client->MakeDeviceList(std::move(atom_devices)));

  return std::make_unique<xla::ifrt::XlaDeserializeExecutableOptions>(
      std::nullopt, atom_device_list);
}

absl::Status DeserializeAndRegisterAtomPrograms(
    xla::ifrt::Client* client,
    const SerializedIfrtIrExecutableMetadata& metadata,
    absl::string_view serialized_executable_payload,
    xla::ifrt::IfrtIRCompileOptions* compile_options) {
  for (const auto& atom_meta : metadata.atom_program_executables()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::ifrt::XlaDeserializeExecutableOptions> options,
        CreateXlaDeserializeExecutableOptions(
            client, compile_options->device_assignments, atom_meta));

    absl::string_view serialized_atom_program =
        serialized_executable_payload.substr(
            atom_meta.executable_location().offset(),
            atom_meta.executable_location().size());

    TF_ASSIGN_OR_RETURN(LoadedExecutableRef loaded_executable,
                        client->GetDefaultCompiler()
                            ->DeserializeLoadedExecutable(
                                serialized_atom_program, std::move(options))
                            .Await());

    compile_options->loaded_exec_binding.emplace(atom_meta.name(),
                                                 std::move(loaded_executable));
  }
  return absl::OkStatus();
}

}  // namespace

// Serializes an IFRT executable into a string with the following format:
// where the SerializedIfrtIrExecutableMetadata contains offsets and sizes of
// the other components after the metadata.
//
// <serialized_executable> ::= <serialized_metadata>
//                             <serialized_ifrt_ir_program>
//                             <atom_executable_list>
//
// <serialized_metadata> ::= <SerializedIfrtIrExecutableMetadata>
//
// <serialized_ifrt_ir_program> ::= <serialized IfrtIRProgram>
//
// <atom_executable_list> ::= <serialized executable>*
//
absl::StatusOr<std::string> SerializeIfrtIrExecutable(
    std::shared_ptr<CompiledIfrtIrProgram> ifrt_ir_program) {
  SerializedIfrtIrExecutableMetadata metadata;
  absl::Cord serialized_executable_payload;
  riegeli::CordWriter writer(&serialized_executable_payload);

  TF_RETURN_IF_ERROR(SerializeIfrtIrProgram(
      writer, metadata.mutable_ifrt_ir_program_location(),
      ifrt_ir_program->program.get()));

  TF_ASSIGN_OR_RETURN(*metadata.mutable_compile_options(),
                      ifrt_ir_program->compile_options->ToProto());

  TF_RETURN_IF_ERROR(
      SerializeIfrtIrAtomExecutables(writer, &metadata, ifrt_ir_program));

  writer.Close();

  std::string serialized_loaded_executable;
  {
    tsl::protobuf::io::StringOutputStream output_stream(
        &serialized_loaded_executable);
    if (!tsl::protobuf::util::SerializeDelimitedToZeroCopyStream(
            metadata, &output_stream)) {
      return absl::InternalError(
          "Failed to serialize IFRT IR executable metadata");
    }
  }

  return absl::StrCat(serialized_loaded_executable,
                      serialized_executable_payload);
}

absl::StatusOr<DeserializedIfrtIRProgram> DeserializeIfrtIrExecutable(
    xla::ifrt::Client* client, absl::string_view serialized,
    std::unique_ptr<xla::ifrt::DeserializeIfrtIRProgramOptions> options) {
  SerializedIfrtIrExecutableMetadata metadata;
  tsl::protobuf::io::ArrayInputStream input_stream(serialized.data(),
                                                   serialized.size());
  if (!tsl::protobuf::util::ParseDelimitedFromZeroCopyStream(
          &metadata, &input_stream, nullptr)) {
    return absl::InvalidArgumentError(
        "Failed to parse SerializedIfrtIrExecutableMetadata");
  }

  absl::string_view serialized_executable_payload =
      serialized.substr(input_stream.ByteCount());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::IfrtIRProgram> program,
      DeserializeIfrtIrProgram(metadata, serialized_executable_payload,
                               std::move(options)));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::IfrtIRCompileOptions> compile_options,
      xla::ifrt::IfrtIRCompileOptions::FromProto(metadata.compile_options()));

  TF_RETURN_IF_ERROR(DeserializeAndRegisterAtomPrograms(
      client, metadata, serialized_executable_payload, compile_options.get()));

  return DeserializedIfrtIRProgram{std::move(program),
                                   std::move(compile_options)};
}

}  // namespace ifrt
}  // namespace xla
