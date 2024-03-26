/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/stream_attribute_annotator.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

bool IsOnlyRootNonDefaultStream(HloComputation* computation) {
  HloInstruction* root = computation->root_instruction();
  auto root_gpu_config = root->backend_config<GpuBackendConfig>();
  if (!root_gpu_config.ok() || root->opcode() == HloOpcode::kTuple) {
    return false;
  }
  int64_t root_stream_id = root_gpu_config->operation_queue_id();
  VLOG(2) << "Found fusion computation's root stream id to be "
          << root_stream_id;
  if (root_stream_id == Thunk::kDefaultExecutionStreamId.value()) {
    return false;
  }
  for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
    if (instr == root) {
      continue;
    }
    int64_t instr_stream_id =
        instr->backend_config<GpuBackendConfig>()->operation_queue_id();
    if (instr_stream_id != Thunk::kDefaultExecutionStreamId.value() &&
        instr_stream_id != root_stream_id) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<bool> AnnotateStreamAttributesForInstruction(
    HloInstruction* instr, GpuBackendConfig& instr_gpu_config) {
  if (instr->called_computations().size() != 1) {
    return false;
  }
  HloComputation* called_comp = instr->called_computations()[0];
  int64_t stream_id = instr_gpu_config.operation_queue_id();

  if (!IsOnlyRootNonDefaultStream(called_comp) ||
      stream_id != Thunk::kDefaultExecutionStreamId.value()) {
    return false;
  }

  auto comp_root_gpu_config =
      called_comp->root_instruction()->backend_config<GpuBackendConfig>();

  instr_gpu_config.set_operation_queue_id(
      comp_root_gpu_config->operation_queue_id());
  *instr_gpu_config.mutable_wait_on_operation_queues() =
      comp_root_gpu_config->wait_on_operation_queues();
  TF_RETURN_IF_ERROR(instr->set_backend_config(instr_gpu_config));
  return true;
}

absl::StatusOr<bool> AnnotateStreamAttributesForCopyStart(
    HloInstruction* instr, int64_t channel_id,
    GpuBackendConfig& instr_gpu_config) {
  // Do nothing if copy-start has already been annotated
  if (instr_gpu_config.operation_queue_id() !=
      Thunk::kDefaultExecutionStreamId.value()) {
    return false;
  }
  instr_gpu_config.set_operation_queue_id(channel_id);
  TF_RETURN_IF_ERROR(instr->set_backend_config(instr_gpu_config));
  VLOG(3) << "Add copy-start's backend config: " << channel_id;
  return true;
}

absl::StatusOr<bool> AnnotateStreamAttributesForUsers(
    HloInstruction* instr, GpuBackendConfig& instr_gpu_config) {
  bool changed = false;
  int64_t stream_id = instr_gpu_config.operation_queue_id();
  if (stream_id == Thunk::kDefaultExecutionStreamId.value()) {
    return changed;
  }
  std::vector<HloInstruction*> all_consumers;
  for (auto user : instr->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      user = user->users()[0];
    }
    all_consumers.push_back(user);
  }

  for (auto user : all_consumers) {
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        user->backend_config<GpuBackendConfig>());
    auto it = absl::c_find(gpu_config.wait_on_operation_queues(), stream_id);
    if (it == gpu_config.wait_on_operation_queues().end() &&
        gpu_config.operation_queue_id() != stream_id) {
      gpu_config.mutable_wait_on_operation_queues()->Add(stream_id);
      TF_RETURN_IF_ERROR(user->set_backend_config(gpu_config));
      changed = true;
    }
  }

  return changed;
}
}  // namespace

absl::StatusOr<bool> StreamAttributeAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      5, "StreamAttributeAnnotator::Run(), before:\n" + module->ToString());
  bool changed = false;
  int64_t channel_id = hlo_query::NextChannelId(*module);
  for (const HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      auto instr_gpu_config = instr->backend_config<GpuBackendConfig>();
      if (!instr_gpu_config.ok()) {
        continue;
      }
      // For fusion instruction, only annotate
      // when the root of fusion is a single instruction
      // running on non-default stream.
      if (instr->opcode() == HloOpcode::kFusion) {
        TF_ASSIGN_OR_RETURN(bool comp_result,
                            AnnotateStreamAttributesForInstruction(
                                instr, instr_gpu_config.value()));
        changed |= comp_result;
      } else if (instr->opcode() == HloOpcode::kCopyStart) {
        TF_ASSIGN_OR_RETURN(bool comp_result,
                            AnnotateStreamAttributesForCopyStart(
                                instr, channel_id, instr_gpu_config.value()));
        changed |= comp_result;
        continue;
      }

      TF_ASSIGN_OR_RETURN(
          bool user_result,
          AnnotateStreamAttributesForUsers(instr, instr_gpu_config.value()));
      changed |= user_result;
    }
  }
  XLA_VLOG_LINES(
      5, "StreamAttributeAnnotator::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
