/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/local_service_utils.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/xla_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/backend.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/service.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {
// Retrieves the parameter metadata for the given computation and parameter
// number.
//
// If the parameter number is invalid for this computation, nullopt is
// returned. When the return value has_value(), nullptr will never be
// the held value.
std::optional<const OpMetadata*> ParameterMetadata(
    const XlaComputation& computation, int parameter_number) {
  for (const HloComputationProto& comp : computation.proto().computations()) {
    if (comp.id() == computation.proto().entry_computation_id()) {
      for (const HloInstructionProto& instr : comp.instructions()) {
        if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter) &&
            instr.parameter_number() == parameter_number) {
          if (!instr.has_metadata()) {
            return std::nullopt;
          }
          return &instr.metadata();
        }
      }
    }
  }
  return std::nullopt;
}
}  // namespace

absl::StatusOr<std::unique_ptr<HloModuleConfig>> GetHloModuleConfig(
    const XlaComputation& computation,
    absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options, ServiceOptions* options,
    Backend* backend) {
  const HloModuleProto& proto = computation.proto();
  TF_RET_CHECK(proto.has_host_program_shape());
  ProgramShape program_shape(proto.host_program_shape());

  // Validate incoming layouts.
  if (argument_layouts.size() != program_shape.parameters_size()) {
    return InvalidArgument(
        "Invalid number of arguments for computation: expected %d, got %u.",
        program_shape.parameters_size(), argument_layouts.size());
  }

  for (int i = 0; i < argument_layouts.size(); ++i) {
    const Shape& argument_shape = *argument_layouts[i];
    TF_RETURN_IF_ERROR(
        ShapeUtil::ValidateShapeWithOptionalLayout(argument_shape));
    if (!ShapeUtil::Compatible(argument_shape, program_shape.parameters(i))) {
      std::optional<const OpMetadata*> metadata =
          ParameterMetadata(computation, /*parameter_number=*/i);
      auto metadata_string = [&metadata]() -> std::string {
        if (!metadata.has_value()) {
          return "";
        }
        CHECK(metadata.value() != nullptr);
        const OpMetadata& m = *metadata.value();
        if (!m.source_file().empty()) {
          return absl::StrFormat(" (%s:%d)", m.source_file(), m.source_line());
        }
        return "";
      };
      return InvalidArgument(
          "Invalid argument shape for argument %d%s, expected %s, got %s.", i,
          metadata_string(),
          ShapeUtil::HumanString(program_shape.parameters(i)),
          ShapeUtil::HumanString(argument_shape));
    }
  }
  if (build_options.result_layout() != nullptr) {
    TF_RETURN_IF_ERROR(Service::ValidateResultShape(
        *build_options.result_layout(), program_shape.result()));
  }

  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  int default_num_replicas =
      options == nullptr ? 1 : options->number_of_replicas();
  std::optional<int> num_threads;
  if (backend != nullptr && backend->eigen_intra_op_thread_pool() != nullptr) {
    num_threads = backend->eigen_intra_op_thread_pool()->NumThreads();
  }
  return xla::CreateModuleConfig(program_shape, argument_layouts,
                                 &execution_options, default_num_replicas,
                                 num_threads);
}
}  // namespace xla
