/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_
#define XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_

#include <string>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

namespace conditional_opt {
// At the conceptual level, a boundary can be thought of as representing a
// single virtual operation, except this virtual operation is conditionally
// instantiated into different concrete operations at each conditional branch.
// So a boundary is mapped to a single concrete operation if it is outside of
// conditional branches, and is mapped to a list of instructions if inside the
// branches. This data structure therefore allows a common data structure
// representation of the instructions to be moved, whether  they are inside or
// outside of the branches. Subsequently, it allows a common implementation
// basis to be used for both moving instructions out of and for moving them
// inside branches.
class Boundary {
 public:
  enum class Position {
    kInsideBranch,
    kOutsideBranchUser,
    kOutsideBranchOperand,
    kUndefined
  };
  Boundary() : position_(Position::kUndefined) {}
  explicit Boundary(Position p) : position_(p) {}
  std::vector<HloInstruction*>& mutable_operands() { return operands_; }
  const std::vector<HloInstruction*>& operands() const { return operands_; }
  bool IsInsideBranch() const { return position_ == Position::kInsideBranch; }
  bool IsOutsideBranchUser() const {
    return position_ == Position::kOutsideBranchUser;
  }
  bool IsOutsideBranchOperand() const {
    return position_ == Position::kOutsideBranchOperand;
  }
  Position GetPosition() const { return position_; }
  bool IsEmpty() const { return operands_.empty(); }
  std::string ToString() const {
    std::string res;
    for (HloInstruction* op : operands_) {
      res += op->ToString() + ";";
    }
    return res;
  }
  bool operator==(const Boundary& that) const {
    return absl::c_equal(operands_, that.operands_);
  }
  template <typename H>
  friend H AbslHashValue(H h, const Boundary& boundary) {
    return H::combine(std::move(h), boundary.operands_);
  }

 private:
  // Boundary instructions in the conditional branches, one from each branch
  // of the conditional; or a single operand from outside the conditional.
  std::vector<HloInstruction*> operands_;
  Position position_;
};

// HLO pass that moves identical ops in/out of conditional.
// - The definition of identical are the shape of the operands are identical
// and their properties are identical.
// - Only the identical ops that won't share operands with other ops will
// be moved out of conditional.
// The cost model of the code motion optimization includes two components:
// represented by the move_config_ and reuse_config_ arrays of the optimization.
// The move_config_ array uses 1 vs 0 to dictate whether each Hlo Opcode, when
// used with its first operand being another given Hlo Opcode, is allowed to
// move across any conditional boundary; the reuse_config_ array uses an integer
// to represent the force between each pair of HloOpcode regarding how
// attractive it is to place these instructions together (both inside or outside
// of a conditional). Both arrays use Hlo Opcode only to drive the
// configuration, regardless of where the operations are located in the
// module.
class ConditionalCodeMotion : public HloModulePass {
 public:
  // If is_layout_sensitive is true, then the hoist process preserves layout
  // during identical comparison. Otherwise, layout is ignored.
  // The search configuration is a single integer but is split into four parts:
  // (sign, n, m, p), where n,m,p each occupy 8 bits and together make the 24
  // bits at the end of the int32_t. For the sign part, if search_config is <0,
  // the reuse_config_ cost model is modified (tuned); if search_config is >0,
  // the move_config_ cost model is modified (tuned); if search_config == 0,
  // the default cost model is used with no tuning. When tuning, the entries in
  // the designated configuration array (move_config_ or reuse_config_) are
  // flipped between 0 and another default integer, starting from the pth entry
  // being queried by the optimization and repeated every nth time a new entry
  // is visited, until a maximal of m entries have been changed. The tuning
  // start over when optimizing a new model.
  explicit ConditionalCodeMotion(bool is_layout_sensitive,
                                 bool pursue_full_conditional_code_motion,
                                 int64_t search_config = 0,
                                 int64_t memory_increase_allowance = 5000)
      : is_layout_sensitive_(is_layout_sensitive),
        pursue_full_conditional_code_motion_(
            /*turn off special case if tuning*/
            pursue_full_conditional_code_motion && search_config == 0),
        search_config_index_(0),
        memory_increase_allowance_(memory_increase_allowance) {
    search_config_.push_back(search_config);
    if (search_config != 0) {
      search_config_map_[0] = search_config_;
    }
  }
  explicit ConditionalCodeMotion(bool is_layout_sensitive,
                                 bool pursue_full_conditional_code_motion,
                                 std::string search_config,
                                 int64_t memory_increase_allowance = 5000)
      : is_layout_sensitive_(is_layout_sensitive),
        pursue_full_conditional_code_motion_(
            /*turn off special case if tuning*/
            pursue_full_conditional_code_motion && search_config.empty()),
        search_config_index_(-1),
        memory_increase_allowance_(memory_increase_allowance) {
    ParseSearchConfiguration(search_config);
  }
  // Parse a given string in the format of a sequence of i,s,m,t into a
  // list of transformation search configurations, each configuration generated
  // by invoking MakeSearchConfig(s,m,t) and will be used for the ith
  // conditional encountered when optimizing a given module.
  void ParseSearchConfiguration(const std::string& search_config);
  // Make a single search configuration for changing transformation decisions:
  // flip the decisions at position n = flip_start + flip_stride * m, and
  // m = 0..max_flip.
  // The following defines how the int64_t search configuration is composed, as
  // flip_start + (flip_max << kMaxPos) + (flip_stride << kStridePos).
  // Position (digit) for maximum number of flips.
  static constexpr int kMaxPos = 16;
  // Position (digit) for the count-down to the first flip.
  static constexpr int kStartPos = 0;
  // Position (digit) for the count-down to the next flip.
  static constexpr int kStridePos = 32;
  // Bit mask for extracting the last digits of value.
  static constexpr int kValueMask = 0xffff;
  static int64_t MakeSearchConfig(int64_t start, int64_t max, int64_t stride) {
    const int64_t config =
        (max << kMaxPos) + (start << kStartPos) + (stride << kStridePos);
    VLOG(2) << "flip stride = " << flip_stride(config) << "\n";
    VLOG(2) << "flig config = " << config << "\n";
    return config;
  }

  static int16_t flip_start(int64_t search_config) {
    return (search_config >> kStartPos) & kValueMask;
  }

  static int16_t flip_stride(int64_t search_config) {
    return (search_config >> kStridePos) & kValueMask;
  }

  static int16_t DecrementMaxFlip(int64_t* search_config) {
    const int16_t max_flip = ((*search_config) >> kMaxPos) & kValueMask;
    // Decrement flip count so we can stop if it reaches 0.
    if (max_flip > 0) {
      *search_config -= (1 << kMaxPos);
    }
    return max_flip;
  }

  absl::string_view name() const override { return "conditional-code-motion"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Optimization decision for each boundary of the conditional instruction.
  class Decision {
   public:
    enum class Direction : uint8_t {
      kMoveOutOfBranch,
      kMoveIntoBranch,
      kNoChange
    };

   public:
    Decision(Direction direction, int benefit)
        : direction_(direction), benefit_(benefit) {}
    Direction GetDirection() const { return direction_; }
    int GetBenefit() const { return benefit_; }

   private:
    Direction direction_;
    int benefit_;
  };
  // If the optimization decision is NO_CHANGE, new_boundary is set to nullptr;
  // otherwise, it is set to the new boundary after proposed optimization.
  virtual Decision ConsiderCodeMotion(
      HloInstruction* conditional, const Boundary& cur_boundary,
      std::vector<Boundary>& to_move, std::vector<Boundary>& new_boundaries,
      absl::flat_hash_map<HloInstruction*, int>& visited_count);

 private:
  const bool is_layout_sensitive_;
  const bool pursue_full_conditional_code_motion_;
  // The following parameterizes the transformation decisions and cost model.
  std::vector<int64_t> search_config_;
  int64_t search_config_index_;
  // Map each conditional to a vector of its search configurations. The key of
  // the map is the index number of the conditional in a module when traversed
  // in post order, and the value of the map is the sequence of search
  // configurations specified with the same index number for the conditional.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> search_config_map_;
  std::vector<std::vector<int64_t>> move_config_, reuse_config_;
  // How much memory increase, calculated using
  // ShapeUtil::ByteSizeOf(hlo->shape(), 1) >> 9, is allowed per instruction
  // moved.
  int64_t memory_increase_allowance_ = 5000;
  int64_t memory_increase_ = 0;
  absl::StatusOr<bool> MoveInstructionOut(
      HloInstruction* conditional, std::vector<Boundary>& to_move_out,
      std::vector<Boundary>& new_boundaries);
  absl::StatusOr<bool> MoveUserInstructionsIn(
      HloInstruction* conditional, std::vector<Boundary>& to_move_in);
  absl::StatusOr<bool> MoveOperandInstructionsIn(
      HloInstruction* conditional, std::vector<Boundary>& to_move_in);
  void SetDefaultMoveConfig();
};
}  // namespace conditional_opt

}  // namespace xla

#endif  // XLA_SERVICE_CONDITIONAL_CODE_MOTION_H_
