/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_METADATA_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_METADATA_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Class for bookkeeping the information on the given modules, in particular on
// the interaction between computations.
//
// Companion instructions are one of the information collected as we build the
// metadata. For example, for each While instruction, companion instructions
// refer to a set of While instructions in other computations that communicate
// with each other.
// In the example below with 3 modules, {While_0, While_2, While_5}, {While_1,
// While_4}, {While_3, While_6} are companion sets.
//
// <Module 0>               <Module 1>                 <Module 2>
// While_0() {              While_2() {                While_5() {
//   While_1() { Send(0) }    While_3() { Send(1) }      While_6() { Recv(1) }
// }                          While_4() { Recv(0) }
//                          }
//
// Companion instructions are used to detect cycles in the graph and also for
// global scheduling.
class HloModuleGroupMetadata {
 public:
  // The kind of companion computation a given instruction can be within.
  enum class ComputationKind {
    kInvalid,
    kWhileCondition,
    kWhileBody,
    kConditionalTrue,
    kConditionalFalse,
  };

  // Tracks the instruction mapped to a given computation, and the computation
  // kind.
  // For example, a body computation of a while instruction, will generate a
  // TrackedInstruction with instruction being the while instruction, and
  // kind being ComputationKind::kWhileBody.
  class TrackedInstruction {
   public:
    TrackedInstruction() = default;
    TrackedInstruction(HloInstruction* instruction, ComputationKind kind)
        : instruction_(instruction), kind_(kind) {}

    bool operator==(const TrackedInstruction& rhs) const {
      return instruction_->opcode() == rhs.instruction_->opcode() &&
             kind_ == rhs.kind_;
    }
    bool operator!=(const TrackedInstruction& rhs) const {
      return !operator==(rhs);
    }

    HloInstruction* instruction() const { return instruction_; }

    string ToString() const;

   private:
    HloInstruction* instruction_ = nullptr;
    ComputationKind kind_ = ComputationKind::kInvalid;
  };

  // Represents a channel and the 4 instructions that form the channel.
  struct Channel {
    int64 id = -1;
    HloInstruction* send = nullptr;
    HloInstruction* recv = nullptr;
    HloInstruction* send_done = nullptr;
    HloInstruction* recv_done = nullptr;
  };

  explicit HloModuleGroupMetadata(const std::vector<HloModule*>& modules)
      : modules_(modules) {}

  ~HloModuleGroupMetadata() = default;

  // Build and return the metadata for the given modules.
  static StatusOr<std::unique_ptr<HloModuleGroupMetadata>> Build(
      const std::vector<HloModule*>& modules);

  // Returns true if the instruction is one of the 4 channel instructions (Send,
  // Recv, SendDone, RecvDone).
  bool IsChannelInstruction(const HloInstruction* instruction) const;

  // Returns true if the instruction is a companion instruction. See the class
  // comment above on companion instructions.
  bool IsCompanionInstruction(HloInstruction* hlo) const;

  // Returns true if the instruction is either a channel instruction or a
  // companion instruction.
  bool InstructionCommunicates(HloInstruction* hlo) const;

  // Returns the Channel instance for the given channel id.
  const Channel& GetChannel(int64 channel_id) const;

  // Returns the computation that contains the peer channel instructions for
  // the given instruction.
  //
  // Precondition: IsChannelInstruction(instruction) is true.
  HloComputation* PeerComputation(const HloInstruction* instruction) const;

  // Returns the path of the nested companion instructions, in terms of HLO
  // instructions. The path goes from inner to outer companions.
  // The returned path does not include the input hlo instruction, in case it
  // is a companion instruction.
  std::vector<TrackedInstruction> GetCompanionsPath(
      const HloInstruction* hlo) const;

  // Checks whether two companion paths (as returned by the GetCompanionsPath()
  // API) are compatible. The two paths are compatible if the sequence of
  // opcodes, and the companion kinds, of the two paths matches.
  bool CheckCompanionPathsCompatibility(
      const std::vector<TrackedInstruction>& path0,
      const std::vector<TrackedInstruction>& path1) const;

  // Returns the unique integer for each module. The returned id is the index of
  // the module in the module vector.
  int64 GetModuleId(const HloModule* module) const;

  // Returns the companion instructions for the given instruction.
  //
  // Precondition: IsCompanionWhile(instruction) is true.
  const std::unordered_set<HloInstruction*>& Companions(
      HloInstruction* instruction) const {
    CHECK_EQ(companion_set_index_.count(instruction), 1);
    return companion_set(companion_set_index_.at(instruction));
  }

  // Returns the companion set at the given index.
  const std::unordered_set<HloInstruction*>& companion_set(int64 index) const {
    CHECK_LT(index, companion_sets_.size());
    return *companion_sets_[index];
  }

  // Returns the companion set index of the given instruction.
  int64 companion_set_index(HloInstruction* instruction) const {
    return companion_set_index_.at(instruction);
  }

  // Returns the list of all companion sets in the HLO module group.
  const std::vector<std::unique_ptr<std::unordered_set<HloInstruction*>>>&
  companion_sets() const {
    return companion_sets_;
  }

  // Returns all channels in the module group.
  const std::vector<Channel>& channels() const { return channels_; }

  // Returns the maximum channel id used in the module group.
  int64 max_channel_id() const { return max_channel_id_; }

 private:
  Status Build();

  // Record all channel instructions and While instructions.
  Status RecordInstructions();

  // Verifies the given HloModules are well-formed and follow the specification,
  // in particular with respect to using channel instructions.
  //
  // * Each channel has all 4 instructions (Send, Recv, SendDone, RecvDone).
  // * The shape of channel instructions match.
  // * The nest level of channel instructions match.
  // * Channel instructions are used in allowed computations; i.e., in the
  //   entry computation of the module or condition/body of While computations.
  //
  // TODO(b/62064342): Currently, HloModuleGroupScheduler checks if there is a
  // cycle in the graph, but it would be good to verify here.
  Status VerifyChannelInstructions();

  // Adds metadata that the given two instructions are companions.
  Status AddCompanion(HloInstruction* instruction1,
                      HloInstruction* instruction2);

  // Retrieves a pointer to the stored TrackedInstruction associated with a
  // tracked computation, or nullptr in case such computation is not tracked.
  const TrackedInstruction* GetTrackedInstruction(
      const HloComputation* computation) const {
    auto it = tracked_instructions_.find(computation);
    return it != tracked_instructions_.end() ? &it->second : nullptr;
  }

  // List of all companion instructions sets in the module.
  std::vector<std::unique_ptr<std::unordered_set<HloInstruction*>>>
      companion_sets_;

  // Map from each companion while instruction to the index into companion_set_.
  tensorflow::gtl::FlatMap<HloInstruction*, int64> companion_set_index_;

  // Map from computation to the instruction using it (a kWhile, kConditional).
  tensorflow::gtl::FlatMap<const HloComputation*, TrackedInstruction>
      tracked_instructions_;

  // All channels in the module.
  std::vector<Channel> channels_;

  // Map from channel ids to the index in channels_.
  tensorflow::gtl::FlatMap<int64, int64> channel_id_map_;

  // The maximum channel id used in the module group.
  int64 max_channel_id_ = -1;

  // The modules that this metadata was built from.
  const std::vector<HloModule*>& modules_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_METADATA_H_
