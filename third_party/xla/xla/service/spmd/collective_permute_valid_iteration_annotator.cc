/* Copyright 2024 The OpenXLA Authors.
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

#include "xla/service/spmd/collective_permute_valid_iteration_annotator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<bool> CollectivePermuteValidIterationAnnotator::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *C : module->computations(execution_threads)) {
    for (HloInstruction *I : C->instructions()) {
      if (I->opcode() != HloOpcode::kCollectivePermute) {
        continue;
      }

      if (I->frontend_attributes().map().find(kSendRecvValidationAttr) !=
          I->frontend_attributes().map().end()) {
        continue;
      }
      auto sourceTargetPairs = I->source_target_pairs();
      if (!IsForwardCycle(sourceTargetPairs) &&
          !IsBackwardCycle(sourceTargetPairs)) {
        continue;
      }

      VLOG(2) << "Collective permute with cycle: " << I->ToString();

      int64_t max_device_num = -1;
      for (auto [source, target] : sourceTargetPairs) {
        max_device_num = std::max(std::max(source, target), max_device_num);
      }
      int64_t num_devices = max_device_num + 1;

      HloInstruction *whileOp = I->parent()->WhileCallInstruction();
      if (whileOp == nullptr) {
        VLOG(2) << "No surrounding while op found. Ignoring " << I->name();
        continue;
      }
      if (!whileOp->frontend_attributes().map().contains(
              "is_pipelined_while_loop"))
        continue;
      TF_ASSIGN_OR_RETURN(WhileLoopBackendConfig config,
                          whileOp->backend_config<WhileLoopBackendConfig>());
      if (!config.has_known_trip_count()) {
        VLOG(2) << "Trip count for while loop (" << whileOp->name()
                << "): unknown";
        continue;
      }
      if (!config.known_trip_count().has_step()) {
        VLOG(2) << "Step for while loop (" << whileOp->name() << "): unknown";
        continue;
      }

      int64_t trip_count = config.known_trip_count().n();
      int64_t step = config.known_trip_count().step();
      VLOG(2) << "Trip count for while loop (" << whileOp->name()
              << "): " << trip_count;
      VLOG(2) << "Step for while loop (" << whileOp->name() << "): " << step;
      if (step != 1) {
        VLOG(2) << "Step is not 1. Skipping...";
        continue;
      }

      // For each source i, the send/recv iteration instances are {i, i+offset}
      // where offset is `number of microbatches * CR - 1`. We know that
      // `trip_count = number_of_microbatches * CR + num_devices - 1` So, offset
      // = number_of_microbatches * CR - 1 = trip_count - num_devices.
      int64_t offset = trip_count - num_devices;

      std::vector<std::pair<int64_t, int64_t>> sendRecvValidation(
          sourceTargetPairs.size());

      for (size_t currIdx = 0; currIdx < sourceTargetPairs.size(); currIdx++) {
        sendRecvValidation[currIdx] = {currIdx, currIdx + offset};
      }

      if (IsBackwardCycle(sourceTargetPairs)) {
        std::reverse(sendRecvValidation.begin(), sendRecvValidation.end());
      }

      xla::FrontendAttributes attributes;
      std::string iteration_instances =
          "{" +
          absl::StrJoin(sendRecvValidation, ",",
                        absl::PairFormatter(
                            [](std::string *out, int64_t value) {
                              absl::StrAppend(out, "{", value);
                            },
                            ",",
                            [](std::string *out, int64_t value) {
                              absl::StrAppend(out, value, "}");
                            })) +
          "}";
      (*attributes.mutable_map())[kSendRecvValidationAttr] =
          iteration_instances;

      I->add_frontend_attributes(attributes);
      VLOG(1) << "Adding " << kSendRecvValidationAttr << " to " << I->name()
              << ": " << iteration_instances;
      changed = true;
    }
  }
  return changed;
}
}  // namespace xla
