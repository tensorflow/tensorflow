/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/pjrt_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/util/delimited_message_util.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/pjrt_ifrt/executable_metadata.pb.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/pjrt_ifrt/pjrt_layout.h"
#include "xla/python/pjrt_ifrt/pjrt_topology.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/python/pjrt_ifrt/xla_executable_version.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char PjRtCompiler::ID = 0;

// Translates IFRT device IDs to PjRt global device IDs in place. When an error
// occurs, `options` may have invalid device IDs.
absl::Status TranslateDeviceIds(PjRtClient* client,
                                xla::CompileOptions& options) {
  if (options.executable_build_options.device_ordinal() != -1) {
    TF_ASSIGN_OR_RETURN(
        auto pjrt_global_device_id,
        client->GetPjRtGlobalDeviceId(
            DeviceId(options.executable_build_options.device_ordinal())));
    options.executable_build_options.set_device_ordinal(
        pjrt_global_device_id.value());
  }
  if (options.executable_build_options.has_device_assignment()) {
    absl::Status result;
    xla::DeviceAssignment device_assignment =
        options.executable_build_options.device_assignment();
    device_assignment.Each(
        [&](int64_t replica, int64_t computation, int64_t* device_id) {
          if (!result.ok()) {
            return;
          }
          auto pjrt_global_device_id =
              client->GetPjRtGlobalDeviceId(DeviceId(*device_id));
          if (pjrt_global_device_id.ok()) {
            *device_id = pjrt_global_device_id->value();
          } else {
            result.Update(pjrt_global_device_id.status());
          }
        });
    TF_RETURN_IF_ERROR(result);
    options.executable_build_options.set_device_assignment(
        std::move(device_assignment));
  }
  return absl::OkStatus();
}

absl::StatusOr<LoadedExecutableRef> PjRtCompiler::CompileAndLoad(
    std::unique_ptr<Program> program, std::unique_ptr<CompileOptions> options) {
  DCHECK(this);
  const auto* xla_program = llvm::dyn_cast<HloProgram>(program.get());
  if (xla_program == nullptr) {
    return absl::InvalidArgumentError("PjRtCompiler requires an HloProgram");
  }
  TF_ASSIGN_OR_RETURN(auto xla_compile_options,
                      GetXlaCompileOptions(std::move(options)));
  TF_RETURN_IF_ERROR(
      TranslateDeviceIds(client_, xla_compile_options->compile_options));
  return PjRtLoadedExecutable::Create(
      client_, xla_program->mlir_module(),
      std::move(xla_compile_options->compile_options),
      std::move(xla_compile_options->loaded_host_callbacks),
      std::move(xla_compile_options->devices));
}

absl::StatusOr<ExecutableRef> PjRtCompiler::Compile(
    std::unique_ptr<Program> program, const Topology& topology,
    std::unique_ptr<CompileOptions> options) {
  DCHECK(this);
  const auto* xla_program = llvm::dyn_cast<HloProgram>(program.get());
  if (xla_program == nullptr) {
    return absl::InvalidArgumentError("PjRtCompiler requires an HloProgram");
  }
  TF_ASSIGN_OR_RETURN(auto xla_compile_options,
                      GetXlaCompileOptions(std::move(options)));
  TF_RETURN_IF_ERROR(
      TranslateDeviceIds(client_, xla_compile_options->compile_options));
  const auto* pjrt_topology = llvm::dyn_cast<PjRtTopology>(&topology);
  if (pjrt_topology == nullptr) {
    return absl::InvalidArgumentError("PjRtCompiler requires a PjRtTopology");
  }
  return PjRtExecutable::Create(xla_program->mlir_module(),
                                std::move(xla_compile_options->compile_options),
                                *pjrt_topology->description());
}

absl::StatusOr<LoadedExecutableRef> PjRtCompiler::DeserializeLoadedExecutable(
    absl::string_view serialized,
    std::unique_ptr<DeserializeExecutableOptions> options) {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(auto xla_deserialize_options,
                      GetXlaDeserializeExecutableOptions(std::move(options)));
  if (xla_deserialize_options->compile_options.has_value()) {
    TF_RETURN_IF_ERROR(
        TranslateDeviceIds(client_, *xla_deserialize_options->compile_options));
  }

  xla::ifrt::SerializedXlaExecutableMetadata metadata;
  google::protobuf::io::ArrayInputStream input_stream(serialized.data(),
                                            serialized.size());
  if (!google::protobuf::util::ParseDelimitedFromZeroCopyStream(&metadata, &input_stream,
                                                      nullptr)) {
    return absl::InvalidArgumentError(
        "Failed to parse SerializedXlaExecutableMetadata");
  }

  absl::string_view serialized_pjrt_executable =
      serialized.substr(input_stream.ByteCount());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::XlaExecutableVersion> executable_version,
      xla::ifrt::XlaExecutableVersion::FromProto(
          metadata.executable_version()));
  // PjRt-IFRT currently does not track XLA executable versions.
  // TF_RETURN_IF_ERROR(IsExecutableVersionCompatible(
  //     *executable_version, xla_deserialize_options->devices.value()));

  std::vector<int> donated_input_indices;
  std::vector<DType> output_dtypes;
  std::vector<Shape> output_shapes;
  std::optional<std::vector<xla::HloSharding>> output_hlo_shardings;
  std::vector<absl::string_view> output_memory_kinds;
  std::optional<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
      output_layouts;
  output_layouts.emplace();

  output_dtypes.reserve(metadata.output_specs_size());
  output_shapes.reserve(metadata.output_specs_size());
  output_memory_kinds.reserve(metadata.output_specs_size());
  output_layouts->reserve(metadata.output_specs_size());

  for (int i = 0; i < metadata.parameter_specs_size(); ++i) {
    if (metadata.parameter_specs(i).donated_input()) {
      donated_input_indices.push_back(i);
    }
  }

  for (const auto& output_spec : metadata.output_specs()) {
    TF_ASSIGN_OR_RETURN(auto dtype, DType::FromProto(output_spec.dtype()));
    output_dtypes.push_back(dtype);
    TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(output_spec.shape()));
    output_shapes.push_back(std::move(shape));
    if (output_spec.has_op_sharding()) {
      if (!output_hlo_shardings.has_value()) {
        output_hlo_shardings.emplace();
        output_hlo_shardings->reserve(metadata.output_specs_size());
      }
      TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                          HloSharding::FromProto(output_spec.op_sharding()));
      output_hlo_shardings->push_back(std::move(hlo_sharding));
    } else {
      if (output_hlo_shardings.has_value()) {
        return absl::InvalidArgumentError(
            "All outputs must use either HloSharding or ConcreteEvenSharding, "
            "not a mix of the two.");
      }
    }
    output_memory_kinds.push_back(output_spec.memory_kind());
    if (output_spec.has_layout()) {
      TF_ASSIGN_OR_RETURN(auto layout, Layout::FromProto(output_spec.layout()));
      output_layouts->push_back(
          llvm::cast<PjRtLayout>(layout.get())->pjrt_layout());
    } else {
      output_layouts->push_back(nullptr);
    }
  }

  TF_ASSIGN_OR_RETURN(auto pjrt_loaded_executable,
                      client_->pjrt_client()->LoadSerializedExecutable(
                          serialized_pjrt_executable,
                          std::move(xla_deserialize_options->compile_options),
                          xla::LoadOptions()));
  // TODO(emilyaf): Remove the else branch once devices are plumbed through from
  // Australis and are always present in the DeserializeExecutableOptions.
  DeviceListRef device_list;
  if (xla_deserialize_options->devices.has_value()) {
    device_list = std::move(xla_deserialize_options->devices.value());
  } else {
    TF_ASSIGN_OR_RETURN(
        device_list, GetDeviceListFromDeviceAssignment(
                         client_, pjrt_loaded_executable->device_assignment()));
  }
  return PjRtLoadedExecutable::Create(
      client_,
      std::shared_ptr<xla::PjRtLoadedExecutable>(
          std::move(pjrt_loaded_executable)),
      std::move(xla_deserialize_options->loaded_host_callbacks),
      std::move(device_list), std::move(donated_input_indices),
      std::move(output_dtypes), std::move(output_shapes),
      std::move(output_hlo_shardings), std::move(output_memory_kinds),
      std::move(output_layouts));
}

}  // namespace ifrt
}  // namespace xla
