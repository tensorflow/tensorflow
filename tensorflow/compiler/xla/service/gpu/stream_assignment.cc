/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"

#include "tensorflow/compiler/xla/legacy_flags/stream_assignment_flags.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"

namespace xla {
namespace gpu {

bool StreamAssignment::HasStreamAssigned(const HloInstruction& hlo) const {
  return hlo_to_stream_number_.count(&hlo);
}

int StreamAssignment::StreamNumberForHlo(const HloInstruction& hlo) const {
  return FindOrDie(hlo_to_stream_number_, &hlo);
}

void StreamAssignment::AssignStreamToHlo(const HloInstruction* hlo,
                                         int stream_no) {
  CHECK_GE(stream_no, 0);
  if (stream_no >= stream_count_) {
    stream_count_ = stream_no + 1;
  }
  InsertOrDie(&hlo_to_stream_number_, hlo, stream_no);
  VLOG(2) << "Assign stream #" << stream_no << " to " << hlo->ToString();
}

namespace {

// Returns whether the two HLOs can run concurrently, i.e., neither is a
// transitive consumer of the other.
bool CanRunConcurrently(
    const HloInstruction& a, const HloInstruction& b,
    const HloComputation::ReachabilityMap& transitive_operands) {
  return !transitive_operands.IsConnected(&a, &b);
}

// Returns which existing stream to assign to `hlo`, or -1 if a stream is not
// needed. `stream_assignment` is the existing stream assignment for all
// instructions topologically before `hlo`. `seen_gemms` contains all GEMMs that
// are topologically before `hlo`.
int ComputeStreamToAssign(
    const HloInstruction& hlo, const StreamAssignment& stream_assignment,
    const HloComputation::ReachabilityMap& transitive_operands,
    const std::vector<const HloInstruction*>& seen_gemms) {
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo.opcode() == HloOpcode::kConstant) {
    // kParameter and kConstant do not need a thunk.
    return -1;
  }

  legacy_flags::StreamAssignmentFlags* flags =
      legacy_flags::GetStreamAssignmentFlags();
  if (flags->xla_gpu_disable_multi_streaming) {
    return 0;
  }

  if (!ImplementedAsGemm(hlo)) {
    // If `hlo` is not implemented as a GEMM, keep it close to its operands to
    // avoid excessive synchronization.
    int stream_no = -1;
    for (const auto* operand : hlo.operands()) {
      if (stream_assignment.HasStreamAssigned(*operand)) {
        stream_no =
            std::max(stream_no, stream_assignment.StreamNumberForHlo(*operand));
      }
    }
    if (stream_no == -1) {
      stream_no = 0;
    }
    return stream_no;
  }

  // Assign different streams to concurrent GEMMs. The code below uses a
  // greedy approach. First, we compute as forbidden_stream_numbers the
  // streams assigned to GEMMs that are concurrent with `hlo`. Then, we assign
  // `hlo` a different stream.
  std::set<int> forbidden_stream_numbers;
  for (const auto* seen_gemm : seen_gemms) {
    int stream_no = stream_assignment.StreamNumberForHlo(*seen_gemm);
    if (!forbidden_stream_numbers.count(stream_no) &&
        CanRunConcurrently(*seen_gemm, hlo, transitive_operands)) {
      forbidden_stream_numbers.insert(stream_no);
    }
  }

  for (int stream_no = 0; stream_no < stream_assignment.StreamCount();
       ++stream_no) {
    if (!forbidden_stream_numbers.count(stream_no)) {
      return stream_no;
    }
  }
  return stream_assignment.StreamCount();
}

}  // namespace

std::unique_ptr<StreamAssignment> AssignStreams(const HloModule& module) {
  auto stream_assignment = MakeUnique<StreamAssignment>();
  const HloComputation& computation = *module.entry_computation();
  std::unique_ptr<HloComputation::ReachabilityMap> transitive_operands =
      computation.ComputeTransitiveOperands();
  std::vector<const HloInstruction*> seen_gemms;
  for (const auto* hlo : computation.MakeInstructionPostOrder()) {
    int stream_no = ComputeStreamToAssign(*hlo, *stream_assignment,
                                          *transitive_operands, seen_gemms);
    if (stream_no != -1) {
      stream_assignment->AssignStreamToHlo(hlo, stream_no);
    }
    if (ImplementedAsGemm(*hlo)) {
      seen_gemms.push_back(hlo);
    }
  }
  return stream_assignment;
}

}  // namespace gpu
}  // namespace xla
