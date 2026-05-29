/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_xla_transform_internal.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_xla_transform_extension.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/xla_transform.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {

class CApiXlaTransformAdapter : public HloXlaTransform {
 public:
  explicit CApiXlaTransformAdapter(std::string name,
                                   PJRT_XlaTransform_Callbacks callbacks)
      : HloXlaTransform(std::move(name)), callbacks_(callbacks) {}

  absl::StatusOr<bool> Transform(xla::HloModule* module) override {
    xla::HloModuleProto proto = module->ToProto();
    std::string serialized_proto;
    if (!tsl::SerializeToStringDeterministic(proto, &serialized_proto)) {
      return absl::InternalError(
          "CApiXlaTransformAdapter: failed to serialize HLO module");
    }

    PJRT_XlaTransform_Args args = {};
    args.struct_size = PJRT_XlaTransform_Args_STRUCT_SIZE;
    args.header.api_version = PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION;
    args.header.has_error = false;
    args.hlo_module.data = serialized_proto.data();
    args.hlo_module.size = serialized_proto.size();

    if (callbacks_.transform_hlo_module == nullptr) {
      return absl::InternalError("transform_hlo_module callback is null");
    }
    callbacks_.transform_hlo_module(&callbacks_, &args);

    if (args.header.has_error) {
      return absl::InternalError("Error in C callback");
    }

    if (args.changed) {
      xla::HloModuleProto transformed_proto;
      if (!transformed_proto.ParseFromString(
              absl::string_view(args.transformed_hlo_module.data,
                                args.transformed_hlo_module.size))) {
        return absl::InternalError("Failed to parse transformed HLO module");
      }

      // We need a HloModuleConfig. We can use the existing module's config.
      ASSIGN_OR_RETURN(
          auto temp_module,
          HloModule::CreateFromProto(transformed_proto, module->config()));

      // Capture schedule from temp_module if it has one.
      absl::flat_hash_map<HloComputation*, HloInstructionSequence>
          comp_to_sequence;
      if (temp_module->has_schedule()) {
        absl::flat_hash_map<int64_t, HloComputation*> temp_comp_map;
        for (HloComputation* comp : temp_module->computations()) {
          temp_comp_map[comp->unique_id()] = comp;
        }
        for (const auto& [comp_id, sequence] :
             temp_module->schedule().sequences()) {
          comp_to_sequence[temp_comp_map[comp_id]] = sequence;
        }
      }

      // Capture the entry computation pointer before calling
      // MoveComputationsFrom, as MoveComputationsFrom can rename computations
      // on collision.
      HloComputation* new_entry = temp_module->entry_computation();
      module->MoveComputationsFrom(temp_module.get());

      module->ReplaceEntryComputation(new_entry);

      RETURN_IF_ERROR(module->RemoveUnusedComputations());

      // Restore schedule if we captured one.
      if (!comp_to_sequence.empty()) {
        HloSchedule new_schedule(module);
        absl::flat_hash_set<HloComputation*> remaining_computations(
            module->computations().begin(), module->computations().end());
        for (auto& [comp, sequence] : comp_to_sequence) {
          if (remaining_computations.contains(comp)) {
            sequence.update_id_sequence();
            new_schedule.set_sequence(comp, std::move(sequence));
          }
        }
        RETURN_IF_ERROR(module->set_schedule(std::move(new_schedule)));
      }

      return true;
    }
    return false;
  }

 private:
  PJRT_XlaTransform_Callbacks callbacks_;
};

}  // namespace

}  // namespace xla

namespace pjrt {
namespace {

PJRT_Error* RegisterXlaTransform(PJRT_Register_Xla_Transform_Args* args) {
  if (args->struct_size < PJRT_Register_Xla_Transform_Args_STRUCT_SIZE) {
    return pjrt::StatusToPjRtError(
        absl::InvalidArgumentError("Invalid struct_size"));
  }
  if (args->callbacks == nullptr) {
    return pjrt::StatusToPjRtError(
        absl::InvalidArgumentError("Callbacks cannot be null"));
  }
  if (args->callbacks->version != PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION) {
    return pjrt::StatusToPjRtError(
        absl::InvalidArgumentError("Invalid callback version"));
  }

  std::string name;
  if (args->name != nullptr) {
    name = std::string(args->name, args->name_size);
  } else {
    name = "pjrt_c_api_transform_" +
           std::to_string(reinterpret_cast<uintptr_t>(args->callbacks));
  }

  auto transform =
      std::make_shared<xla::CApiXlaTransformAdapter>(name, *args->callbacks);
  xla::HloXlaTransform::PipelineStage stage;
  switch (args->stage) {
    case PJRT_XlaTransform_PipelineStage_kPreScheduler:
      stage = xla::HloXlaTransform::PipelineStage::kPreScheduler;
      break;
    case PJRT_XlaTransform_PipelineStage_kPostScheduler:
      stage = xla::HloXlaTransform::PipelineStage::kPostScheduler;
      break;
    default:
      return pjrt::StatusToPjRtError(
          absl::InvalidArgumentError("Invalid pipeline stage"));
  }

  xla::RegisterHloXlaTransform(stage, transform);

  return nullptr;
}

}  // namespace

PJRT_Xla_Transform_Extension CreateXlaTransformExtension(
    PJRT_Extension_Base* next) {
  return PJRT_Xla_Transform_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_Xla_Transform_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_XlaTransform,
          /*next=*/next,
      },
      /*register_xla_transform=*/RegisterXlaTransform,
  };
}

}  // namespace pjrt
