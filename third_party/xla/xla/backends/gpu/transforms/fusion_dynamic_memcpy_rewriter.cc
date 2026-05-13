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

#include "xla/backends/gpu/transforms/fusion_dynamic_memcpy_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/codegen/copy.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

const HloInstruction* SkipOptionalBitcast(const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kBitcast) {
    instr = instr->operand(0);
  }
  return instr;
}

std::optional<DynamicSliceConfig> GetDynamicSliceConfig(
    const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  if (!config.ok() || !config->has_dynamic_slice_config()) {
    return std::nullopt;
  }
  return config->dynamic_slice_config();
}

const HloInstruction* FindEnclosingWhileLoop(const HloInstruction* instr) {
  const HloComputation* computation = instr->parent();
  while (computation != nullptr) {
    auto callers = computation->caller_instructions(HloOpcode::kWhile);
    if (!callers.empty()) {
      return callers.front();
    }
    auto all_callers = computation->caller_instructions();
    if (all_callers.empty()) {
      break;
    }
    computation = all_callers.front()->parent();
  }
  return nullptr;
}

int64_t GetSliceByteSize(const HloInstruction* ds_or_dus) {
  if (ds_or_dus->opcode() == HloOpcode::kDynamicSlice) {
    return ShapeUtil::ByteSizeOf(ds_or_dus->shape());
  }
  return ShapeUtil::ByteSizeOf(ds_or_dus->operand(1)->shape());
}

}  // namespace

absl::StatusOr<bool> FusionDynamicMemcpyRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool has_changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }

    HloFusionInstruction* fusion =
        Cast<HloFusionInstruction>(computation->FusionInstruction());
    if (!DynamicMemcpyFusion::IsCandidateFusion(*fusion)) {
      continue;
    }

    const HloInstruction* root =
        SkipOptionalBitcast(fusion->fused_expression_root());

    std::optional<DynamicSliceConfig> ds_config = GetDynamicSliceConfig(root);
    if (!ds_config.has_value()) {
      continue;
    }

    ASSIGN_OR_RETURN(auto backend_config,
                     fusion->backend_config<GpuBackendConfig>());
    auto* fusion_config = backend_config.mutable_fusion_backend_config();
    fusion_config->set_kind(kDynamicMemcpyFusionKind);
    auto* memcpy_config = fusion_config->mutable_dynamic_memcpy_config();

    bool is_src = root->opcode() == HloOpcode::kDynamicSlice;
    int64_t buffer_size = ShapeUtil::ByteSizeOf(root->operand(0)->shape());
    int64_t max_offset = buffer_size - GetSliceByteSize(root);

    if (ds_config->byte_stride() == 0) {
      int64_t offset =
          std::clamp(ds_config->byte_offset(), int64_t{0}, max_offset);
      if (is_src) {
        memcpy_config->add_src_offset_bytes(offset);
        memcpy_config->add_dst_offset_bytes(0);
      } else {
        memcpy_config->add_src_offset_bytes(0);
        memcpy_config->add_dst_offset_bytes(offset);
      }
    } else {
      const HloInstruction* while_loop = FindEnclosingWhileLoop(fusion);
      if (while_loop == nullptr) {
        LOG(INFO) << "Cannot find enclosing while loop for " << fusion->name();
        continue;
      }

      auto loop_config = while_loop->backend_config<WhileLoopBackendConfig>();
      if (!loop_config.ok() || !loop_config->has_known_trip_count()) {
        LOG(INFO) << "While loop has unknown trip count for " << fusion->name();
        continue;
      }

      int64_t trip_count = loop_config->known_trip_count().n();
      memcpy_config->set_depends_on_loop(true);

      for (int64_t i = 0; i < trip_count; ++i) {
        int64_t offset =
            std::clamp(ds_config->byte_offset() + i * ds_config->byte_stride(),
                       int64_t{0}, max_offset);
        if (is_src) {
          memcpy_config->add_src_offset_bytes(offset);
          memcpy_config->add_dst_offset_bytes(0);
        } else {
          memcpy_config->add_src_offset_bytes(0);
          memcpy_config->add_dst_offset_bytes(offset);
        }
      }
    }

    RETURN_IF_ERROR(fusion->set_backend_config(backend_config));
    has_changed = true;
  }

  return has_changed;
}

}  // namespace xla::gpu
