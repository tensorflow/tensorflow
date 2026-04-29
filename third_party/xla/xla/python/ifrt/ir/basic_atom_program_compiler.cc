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

#include "xla/python/ifrt/ir/basic_atom_program_compiler.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/compilation_utils.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

absl::StatusOr<std::unique_ptr<AtomProgramCompiler>>
BasicAtomProgramCompiler::Create(Client* absl_nonnull client,
                                 absl::Span<const DeviceId> device_assignments,
                                 bool strict_memory_reservation) {
  for (const DeviceId device_id : device_assignments) {
    TF_RETURN_IF_ERROR(client->LookupDevice(device_id).status());
  }
  return absl::WrapUnique(new BasicAtomProgramCompiler(
      client, device_assignments, strict_memory_reservation));
}

BasicAtomProgramCompiler::BasicAtomProgramCompiler(
    Client* absl_nonnull client, absl::Span<const DeviceId> device_assignments,
    bool strict_memory_reservation)
    : client_(client),
      device_assignments_(device_assignments.begin(), device_assignments.end()),
      strict_memory_reservation_(strict_memory_reservation) {}

tsl::Future<LoadedExecutableRef> BasicAtomProgramCompiler::CompileXla(
    std::unique_ptr<HloProgram> hlo_program, xla::CompileOptions options) {
  // Rewrite device assignment from logical ids to IFRT device ids.
  xla::DeviceAssignment device_assignment =
      options.executable_build_options.device_assignment();
  TF_RETURN_IF_ERROR(device_assignment.EachStatus(
      [&](absl::Span<const int64_t>, int64_t* id) -> absl::Status {
        if (*id < 0 || *id >= device_assignments_.size()) {
          return absl::NotFoundError(
              absl::StrFormat("Unknown logical device id %d (expected the id "
                              "to be between 0 and %d ) ",
                              *id, (device_assignments_.size() - 1)));
        }
        *id = device_assignments_[*id].value();
        return absl::OkStatus();
      }));
  options.executable_build_options.set_device_assignment(device_assignment);

  TF_ASSIGN_OR_RETURN(DeviceListRef devices,
                      GetDeviceListFromXlaCompileOptions(client_, options));

  TF_RET_CHECK(!devices->devices().empty())
      << "CompileXla was called with empty device assignment.";
  xla::ifrt::Device* first_device = devices->devices().front();
  absl::string_view platform_name = first_device->PlatformName();
  TF_RET_CHECK(platform_name == "tpu" || platform_name == "cpu" ||
               platform_name == "cuda" || platform_name == "rocm")
      << "Unsupported platform: " << platform_name;

  if (strict_memory_reservation_) {
    TF_RET_CHECK(first_device->PlatformName() == "tpu")
        << "IFRT IR `strict_memory_reservation` option is only supported for "
           "TPU devices. Got device platform: "
        << first_device->PlatformName();
    TF_ASSIGN_OR_RETURN(
        int64_t device_memory,
        first_device->Attributes().Get<int64_t>("device_memory_bytes_limit"));

    TF_RETURN_IF_ERROR(SetStrictMemoryReservation(hlo_program->name(),
                                                  device_memory, options));
  }

  return client_->GetDefaultCompiler()->CompileAndLoad(
      std::move(hlo_program), std::make_unique<XlaCompileOptions>(
                                  std::move(options), std::move(devices)));
}

tsl::Future<LoadedExecutableRef> BasicAtomProgramCompiler::CompileMpmdReshard(
    std::vector<DType> dtypes, std::vector<Shape> shapes,
    std::vector<IfrtArrayType> in_array_types,
    std::vector<IfrtArrayType> out_array_types) {
  return absl::UnimplementedError(
      "BasicAtomProgramCompiler does not support MPMD resharding");
}

}  // namespace ifrt
}  // namespace xla
