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

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/core/platform/random.h"

namespace xla {
namespace gpu {

bool StreamAssignment::HasStreamAssigned(const HloInstruction& hlo) const {
  return hlo_to_stream_number_.contains(&hlo);
}

int StreamAssignment::StreamNumberForHlo(const HloInstruction& hlo) const {
  return FindOrDie(hlo_to_stream_number_, &hlo);
}

void StreamAssignment::AssignStreamToHlo(const HloInstruction* hlo,
                                         int stream_num) {
  CHECK_GE(stream_num, 0);
  if (stream_num >= stream_count_) {
    stream_count_ = stream_num + 1;
  }
  InsertOrDie(&hlo_to_stream_number_, hlo, stream_num);
  VLOG(2) << "Assign stream #" << stream_num << " to " << hlo->ToString();
}

namespace {

// Returns whether the two HLOs can run concurrently, i.e., neither is a
// transitive consumer of the other.
bool CanRunConcurrently(const HloInstruction& a, const HloInstruction& b,
                        const HloReachabilityMap& reachability) {
  return !reachability.IsConnected(&a, &b);
}

constexpr int kInvalidStreamNum = -1;
//  Returns true iff `stream_num` is an invalid stream number.
inline bool IsStreamNumValid(int stream_num) {
  return stream_num != kInvalidStreamNum;
}

// Returns which existing stream to assign to `hlo`, or -1 if a stream is not
// needed. `stream_assignment` is the existing stream assignment for all
// instructions topologically before `hlo`. `seen_gemms` contains all GEMMs that
// are topologically before `hlo`.
int ComputeStreamToAssign(
    const HloInstruction& hlo, const StreamAssignment& stream_assignment,
    const HloReachabilityMap& reachability,
    const std::vector<const HloInstruction*>& seen_gemms) {
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo.opcode() == HloOpcode::kConstant) {
    // kParameter and kConstant do not need a thunk.
    return kInvalidStreamNum;
  }

  const auto& debug_options = hlo.GetModule()->config().debug_options();
  if (debug_options.xla_gpu_disable_multi_streaming()) {
    return 0;
  }

  if (debug_options.xla_gpu_use_random_streams()) {
    // Debug feature: make random stream assignments to try to uncover
    // concurrency bugs.
    return tensorflow::random::New64() % 100;
  }

  if (!(IsCublasGemm(hlo) || IsMatrixMultiplication(hlo))) {
    // If `hlo` is not implemented as a GEMM, keep it close to its operands to
    // avoid excessive synchronization.
    int stream_num = -1;
    for (const auto* operand : hlo.operands()) {
      if (stream_assignment.HasStreamAssigned(*operand)) {
        stream_num = std::max(stream_num,
                              stream_assignment.StreamNumberForHlo(*operand));
      }
    }
    if (!IsStreamNumValid(stream_num)) {
      stream_num = 0;
    }
    return stream_num;
  }

  // Assign different streams to concurrent GEMMs. The code below uses a
  // greedy approach. First, we compute as forbidden_stream_numbers the
  // streams assigned to GEMMs that are concurrent with `hlo`. Then, we assign
  // `hlo` a different stream.
  absl::flat_hash_set<int> forbidden_stream_numbers;
  for (const auto* seen_gemm : seen_gemms) {
    int stream_num = stream_assignment.StreamNumberForHlo(*seen_gemm);
    if (!forbidden_stream_numbers.contains(stream_num) &&
        CanRunConcurrently(*seen_gemm, hlo, reachability)) {
      forbidden_stream_numbers.insert(stream_num);
    }
  }

  for (int stream_num = 0; stream_num < stream_assignment.StreamCount();
       ++stream_num) {
    if (!forbidden_stream_numbers.contains(stream_num)) {
      return stream_num;
    }
  }
  return stream_assignment.StreamCount();
}

}  // namespace

std::unique_ptr<StreamAssignment> AssignStreams(const HloModule& module) {
  auto stream_assignment = absl::make_unique<StreamAssignment>();
  const HloComputation& computation = *module.entry_computation();
  std::unique_ptr<HloReachabilityMap> reachability =
      HloReachabilityMap::Build(&computation);
  std::vector<const HloInstruction*> seen_gemms;
  // The execution of different RNG Hlo instructions in the same module updates
  // a common global variable. To avoid a race condition, we simply assign all
  // RNG kernels to the same stream to make them run sequentially.
  //
  // TODO(b/111791052): If we remove such a common variable, we will need to
  // clean up the code here.
  int stream_num_for_rng = kInvalidStreamNum;
  for (const auto* hlo : computation.MakeInstructionPostOrder()) {
    // If we ever enable fusion of RNG instructions, we will need to extend this
    // code to look inside a fused instruction.
    int stream_num = (hlo->opcode() == HloOpcode::kRng &&
                      IsStreamNumValid(stream_num_for_rng))
                         ? stream_num_for_rng
                         : ComputeStreamToAssign(*hlo, *stream_assignment,
                                                 *reachability, seen_gemms);
    if (IsStreamNumValid(stream_num)) {
      stream_assignment->AssignStreamToHlo(hlo, stream_num);
      if (hlo->opcode() == HloOpcode::kRng &&
          !IsStreamNumValid(stream_num_for_rng)) {
        stream_num_for_rng = stream_num;
      }
    }
    if (IsCublasGemm(*hlo) || IsMatrixMultiplication(*hlo)) {
      seen_gemms.push_back(hlo);
    }
  }
  return stream_assignment;
}

}  // namespace gpu
}  // namespace xla
