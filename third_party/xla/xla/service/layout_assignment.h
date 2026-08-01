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

#ifndef XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
#define XLA_SERVICE_LAYOUT_ASSIGNMENT_H_

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/tuple_points_to_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/map_util.h"
#include "xla/service/computation_layout.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class LayoutAssignment;

// Abstract base class representing a layout constraint in LayoutAssignment.
//
// In XLA, a layout constraint specifies a requirement or preference for how
// tensors/buffers in memory are laid out (e.g., minor-to-major dimension
// ordering, tiling, padding). Hardware backends and specific HLO operations
// (such as convolutions or matrix multiplications) impose layout constraints
// on their inputs, outputs, or computation interfaces to ensure correctness or
// optimize execution speed.
//
// During layout assignment, constraint objects are created, prioritized, and
// propagated across the HLO graph. When conflicting constraints arise, the
// constraint with higher priority takes precedence, and layout copy
// instructions (`kCopy`) are inserted where necessary to bridge incompatible
// layouts.
//
// Specific constraint types derived from LayoutConstraint include:
// - BufferLayoutConstraint: Constrains the layout of a LogicalBuffer produced
//   by an instruction.
// - OperandLayoutConstraint: Constrains the layout expected for an operand by
//   its consumer instruction.
// - ComputationLayoutConstraint: Constrains the parameter and result layouts
//   of an HloComputation interface.
class LayoutConstraint {
 public:
  LayoutConstraint(bool mandatory, bool dfs, int64_t priority)
      : mandatory_(mandatory), dfs_(dfs), priority_(priority) {}
  virtual ~LayoutConstraint() = default;

  // Returns a string representation of the constraint for debugging/logging.
  virtual std::string ToString() const = 0;

  // Returns true if this constraint is mandatory and cannot be overwritten
  // by a lower or equal priority constraint.
  bool mandatory() const { return mandatory_; }

  // Returns true if this constraint should be propagated in DFS (depth-first
  // search) order, or false for BFS (breadth-first search) propagation.
  bool dfs() const { return dfs_; }

  // Returns the priority of this constraint. When conflicting constraints are
  // encountered, higher priority constraints override lower priority ones.
  int64_t priority() const { return priority_; }

  // Returns true if this constraint is set to the default fallback priority.
  bool IsDefaultPriority() const { return priority_ == kDefaultPriority; }

  // Priority of default/fallback layouts when not explicitly specified.
  static constexpr int64_t kDefaultPriority = -2;
  // Beginning priority level used when layout assignment starts.
  static constexpr int64_t kBeginningPriority = 0;
  // Priority assigned to user-specified layouts on the entry computation.
  static constexpr int64_t kGivenPriority = 3;

 protected:
  // Whether this constraint is mandatory (cannot be overridden).
  bool mandatory_;

  // Whether constraint propagation proceeds in DFS (true) or BFS (false)
  // order.
  bool dfs_;

  // Priority of the constraint used for resolving conflicts.
  int64_t priority_;
};

std::ostream& operator<<(std::ostream& out, const LayoutConstraint& constraint);

// Layout constraint on a single LogicalBuffer.
//
// BufferLayoutConstraint specifies the memory layout of an array buffer
// produced by a specific HloInstruction. Unlike OperandLayoutConstraint (which
// constrains a use site and can be satisfied via an inserted copy), a
// BufferLayoutConstraint directly constrains the definition of the logical
// buffer.
//
// Data Structure Usage:
// - `buffer_`: Points to the LogicalBuffer instance whose output layout is
//   constrained.
// - `layout_`: Inlined vector storing the assigned Layout object(s).
// - `from_user_`: Optional pointer to the user HloInstruction that requested
//   or induced this buffer layout constraint.
class BufferLayoutConstraint : public LayoutConstraint {
 public:
  // Constructs a BufferLayoutConstraint for the given LogicalBuffer.
  BufferLayoutConstraint(const Layout& layout, const LogicalBuffer& buffer,
                         bool mandatory, bool dfs, int64_t priority);

  // Returns the constrained LogicalBuffer.
  const LogicalBuffer& buffer() const { return *buffer_; }

  // Returns the target Layout for the buffer.
  const Layout& layout() const { return layout_[0]; }

  // Updates the buffer layout if allowed by priority and mandatory rules,
  // propagating changes back to LayoutAssignment.
  bool UpdateLayout(int64_t priority, const Layout& layout, bool mandatory,
                    bool dfs, LayoutAssignment* assignment,
                    const HloInstruction* from_user = nullptr);

  std::string ToString() const override;

 private:
  // The constrained layout(s) for the buffer.
  absl::InlinedVector<Layout, 2> layout_;

  // The LogicalBuffer being constrained.
  const LogicalBuffer* buffer_;

  // The consumer instruction that induced this constraint, if any.
  const HloInstruction* from_user_ = nullptr;
};

// Layout constraint on an operand of an instruction.
//
// OperandLayoutConstraint specifies the layout required or preferred for an
// operand by a consumer HloInstruction. The constrained operand shape can be
// an array or a tuple.
//
// Unlike BufferLayoutConstraint, this is a constraint on the USE of a shaped
// value rather than a hard constraint on the instruction defining the value.
// If the defining instruction produces a different layout, copy instructions
// (`kCopy`) can be inserted between the definition and use to satisfy this
// constraint.
//
// Data Structure Usage:
// - `instruction_`: Points to the consumer HloInstruction imposing the
//   operand constraint.
// - `operand_no_`: The zero-based index of the operand being constrained.
// - `shape_layout_`: Inlined vector storing the expected ShapeLayout for the
//   operand.
class OperandLayoutConstraint : public LayoutConstraint {
 public:
  // Constructs an OperandLayoutConstraint for the specified operand slot.
  OperandLayoutConstraint(const ShapeLayout& shape_layout,
                          const HloInstruction* instruction, int64_t operand_no,
                          bool mandatory, bool dfs, int64_t priority);

  // Returns the expected ShapeLayout for the operand.
  const ShapeLayout& shape_layout() const { return shape_layout_[0]; }

  // Returns the consumer HloInstruction imposing this constraint.
  const HloInstruction* instruction() const { return instruction_; }

  // Returns the index of the constrained operand.
  int64_t operand_no() const { return operand_no_; }

  // Returns the operand HloInstruction being constrained.
  const HloInstruction* operand() const {
    return instruction_->operand(operand_no_);
  }

  // Updates the operand layout if allowed by priority and mandatory rules.
  bool UpdateLayout(int64_t priority, const Shape& new_shape, bool mandatory,
                    bool dfs, LayoutAssignment* assignment);

  std::string ToString() const override;

 private:
  // The expected shape layout(s) for the operand.
  absl::InlinedVector<ShapeLayout, 2> shape_layout_;

  // The consumer instruction imposing the constraint.
  const HloInstruction* instruction_;

  // The index of the operand in the consumer instruction.
  int64_t operand_no_;
};

// Encapsulates layout constraints on the interface of an HloComputation.
//
// ComputationLayoutConstraint specifies and tracks layout requirements on
// the inputs (parameters) and output (result) of an HloComputation. It wraps
// a ComputationLayout object alongside state flags that record which parts of
// the interface layout have been explicitly set during layout assignment.
//
// Data Structure Usage:
// - `layout_state_`: A bitmask storing layout state flags
//   (kDefaultLayoutIsUsed, kResultLayoutIsSet, kParameterLayoutIsSet, or
//   kComputationLayoutIsSet). Used to track whether parameters, results, or
//   the overall computation interface layout are explicitly constrained vs.
//   using default layouts.
// - `computation_layout_`: An instance of ComputationLayout containing the
//   actual ShapeLayout objects for all parameters and the result of the
//   computation.
class ComputationLayoutConstraint : public LayoutConstraint {
 public:
  // Layout state flags indicating which components of the computation
  // interface layout have been explicitly constrained.
  static constexpr int64_t kDefaultLayoutIsUsed = 0;
  static constexpr int64_t kResultLayoutIsSet = 1;
  static constexpr int64_t kParameterLayoutIsSet = 2;
  static constexpr int64_t kComputationLayoutIsSet = 3;

  // Constructs a ComputationLayoutConstraint for the given computation.
  // If computation_layout is nullptr, initializes a default layout based on
  // the computation's program shape.
  explicit ComputationLayoutConstraint(const HloComputation* computation,
                                       ComputationLayout* computation_layout,
                                       int64_t priority)
      : LayoutConstraint(/*mandatory=*/true, /*dfs=*/true, priority),
        layout_state_((computation_layout == nullptr)
                          ? kDefaultLayoutIsUsed
                          : kComputationLayoutIsSet),
        computation_layout_(
            (computation_layout == nullptr)
                ? ComputationLayout(
                      computation->ComputeProgramShape(),
                      // Computation callers need layout to be set and
                      // computation parameters may miss the layout, so we
                      // cannot rely on them and need to reset/ignore the
                      // layout. Entry computation is special because unset
                      // layouts there are used to indicate that the layout
                      // should be automatically inferred.
                      /*ignore_layouts=*/!computation->IsEntryComputation())
                : *computation_layout) {
    parameter_layouts_set_.resize(computation->num_parameters(), false);
    if (computation_layout != nullptr) {
      for (int i = 0; i < computation->num_parameters(); ++i) {
        parameter_layouts_set_[i] =
            computation_layout->parameter_layout(i).LayoutIsSet();
      }
    }
  }

  // Accessors for the underlying ComputationLayout.
  const ComputationLayout& computation_layout() const {
    return computation_layout_;
  }
  ComputationLayout* mutable_computation_layout() {
    return &computation_layout_;
  }

  // Resets the computation layout and priority, updating state flags for
  // result and parameter layouts according to the propagation flags.
  void ResetComputationLayout(const ComputationLayout& layout, int64_t priority,
                              bool prop_result_layout,
                              bool prop_parameter_layout) {
    if (prop_parameter_layout) {
      for (int i = 0; i < parameter_layouts_set_.size(); ++i) {
        if (layout.parameter_layout(i) !=
            computation_layout_.parameter_layout(i)) {
          parameter_layouts_set_[i] = true;
        }
      }
    }
    computation_layout_ = layout;
    priority_ = priority;
    if (prop_result_layout) {
      layout_state_ |= kResultLayoutIsSet;
    }
    if (prop_parameter_layout) {
      layout_state_ |= kParameterLayoutIsSet;
    }
  }

  // Resets only the result shape layout, marking result_layout_is_set.
  void ResetResultLayout(const ShapeLayout& shape_layout, int64_t priority) {
    *computation_layout_.mutable_result_layout() = shape_layout;
    layout_state_ |= kResultLayoutIsSet;
    priority_ = priority;
  }

  // Returns true if parameter layouts have been explicitly set.
  bool parameter_layout_is_set() const {
    return layout_state_ & kParameterLayoutIsSet;
  }

  // Returns true if specific parameter layout has been explicitly set.
  // We track this per-parameter to allow setting unset parameters even if
  // other parameters in the same computation have already been set with
  // higher priority.
  bool parameter_layout_is_set(int64_t parameter_no) const {
    return parameter_layouts_set_[parameter_no];
  }

  // Returns true if the result layout has been explicitly set.
  bool result_layout_is_set() const {
    return layout_state_ & kResultLayoutIsSet;
  }

  // Returns true if the default layout is currently used.
  bool default_layout_is_used() const {
    return layout_state_ == kDefaultLayoutIsUsed;
  }

  std::string ToString() const override;

 private:
  // Bitmask tracking whether the computation layout is using defaults, or
  // whether parameter/result layouts have been explicitly constrained.
  int64_t layout_state_;

  // The computation interface layout (parameter and result shape layouts).
  ComputationLayout computation_layout_;

  std::vector<bool> parameter_layouts_set_;
};

// Encapsulates layout constraints across communication channels (Send/Recv).
//
// ChannelLayoutConstraints ensures layout consistency across communication
// boundaries (such as Send/Recv instructions matching a channel_id). Any
// unconstrained channels are assigned locally optimal layouts which are then
// registered as channel constraints.
//
// Data Structure Usage:
// - `constraints_`: An absl::flat_hash_map mapping each channel ID (int64_t)
//   to its constrained Layout object.
class ChannelLayoutConstraints {
 public:
  // Constructs an empty channel constraint set.
  ChannelLayoutConstraints() = default;

  // Returns true if channel_id has an associated layout constraint.
  bool IsChannelConstrained(int64_t channel_id) const {
    return constraints_.contains(channel_id);
  }

  // Given `shape`, applies the constrained layout for `channel_id`.
  // `channel_id` must already be constrained.
  Shape LayoutShapeForChannel(Shape shape, int64_t channel_id) const {
    auto it = constraints_.find(channel_id);
    CHECK(it != constraints_.end()) << "Channel " << channel_id;
    *shape.mutable_layout() = it->second;
    return shape;
  }

  // Returns the Layout constraint for `channel_id`, which must already be
  // constrained.
  const Layout& LayoutForChannel(int64_t channel_id) const {
    auto it = constraints_.find(channel_id);
    CHECK(it != constraints_.end()) << "Channel " << channel_id;
    return it->second;
  }

  // Adds a new layout constraint for `channel_id`. Returns nullptr if the
  // channel constraint was successfully added or matches an existing
  // constraint; otherwise returns a pointer to the existing conflicting
  // layout.
  const Layout* ConstrainChannel(int64_t channel_id, const Layout& layout) {
    auto it = constraints_.emplace(std::make_pair(channel_id, layout));
    if (it.second) {
      return nullptr;
    }
    return LayoutUtil::Equal(layout, it.first->second) ? nullptr
                                                       : &it.first->second;
  }

 private:
  // Map from channel ID to its assigned layout.
  absl::flat_hash_map<int64_t, Layout> constraints_;
};

// HLO pass which assigns layouts to all instructions in the HLO module while
// satisfying all necessary invariants and minimizing cost.
class LayoutAssignment : public HloModulePass {
 public:
  // entry_computation_layout is modified to populate a layout for the result in
  // the case that no particular layout is requested.
  //
  // channel_constraints is both an input and output. Any sends or recvs that
  // are present in channel_constraints will be laid out as constrained. Any
  // unconstrained sends or recvs will be laid out as locally optimal and their
  // layout will be added as a constraint to channel_constraints.
  //
  // If channel_constraints is nullptr, no kSend or kRecvs must be contained
  // within any module passed to `Run`.
  explicit LayoutAssignment(
      ComputationLayout* entry_computation_layout,
      ChannelLayoutConstraints* channel_constraints = nullptr,
      bool reverse_computation_order = false);
  ~LayoutAssignment() override {}
  const TuplePointsToAnalysis& points_to_analysis() const {
    return *points_to_analysis_;
  }
  absl::string_view name() const override { return "layout-assignment"; }

  // Encapsulates all layout constraints associated with a single
  // HloComputation.
  //
  // LayoutConstraints tracks two primary categories of constraints scoped to a
  // computation:
  // 1. Computation interface layout constraints (ComputationLayoutConstraint):
  //    Specifies layout constraints on the parameters and result of the
  //    HloComputation interface.
  // 2. Operand layout constraints (OperandLayoutConstraint):
  //    Specifies layout constraints on individual operands consumed by specific
  //    instructions within the computation.
  //
  // Data Structure Usage:
  // - `computation_`: Points to the HloComputation instance whose layouts are
  //   being constrained.
  // - `computation_constraint_`: A ComputationLayoutConstraint object that
  //   wraps the ComputationLayout (parameter and result shape layouts) along
  //   with state metadata (such as priority and whether parameter/result
  //   layouts are explicitly set or defaults).
  // - `operand_constraints_`: An absl::flat_hash_map mapping an
  //   OperandConstraintKey pair (const HloInstruction*, int64_t operand_no) to
  //   a std::unique_ptr<OperandLayoutConstraint>. This lookup structure enables
  //   efficient retrieval, modification, and management of layout constraints
  //   placed on specific operand positions of instructions in the computation.
  class LayoutConstraints {
   public:
    explicit LayoutConstraints(HloComputation* computation,
                               ComputationLayout* computation_layout,
                               int64_t priority)
        : computation_(computation),
          computation_constraint_(computation, computation_layout, priority) {}
    ~LayoutConstraints() = default;

    // Returns the HloComputation associated with these layout constraints.
    const HloComputation* computation() const { return computation_; }
    HloComputation* computation() { return computation_; }

    // Clears all operand layout constraints associated with this computation.
    void ResetOperandConstraints() { operand_constraints_.clear(); }

    // Returns the constrained ShapeLayout for the given instruction operand,
    // or nullptr if the operand is unconstrained.
    const ShapeLayout* OperandLayout(const HloInstruction* instruction,
                                     int64_t operand_no) const;

    // Returns the OperandLayoutConstraint for the given instruction operand,
    // or nullptr if none exists.
    const OperandLayoutConstraint* GetOperandLayoutConstraint(
        const HloInstruction* instruction, int64_t operand_no) const;

    // Returns a mutable reference to the unique_ptr<OperandLayoutConstraint>
    // for the given instruction operand slot, inserting an entry in
    // operand_constraints_ if it does not already exist.
    std::unique_ptr<OperandLayoutConstraint>& MutableOperandLayoutConstraint(
        const HloInstruction* instruction, int64_t operand_no);

    // Returns the ShapeLayout of the computation's result if set (or if this
    // is the entry computation), otherwise returns nullptr.
    const ShapeLayout* ResultLayout() const;

    // Sets the result layout for the computation and registers the updated
    // computation constraint with the LayoutAssignment pass.
    absl::Status SetResultLayout(LayoutAssignment* assignment,
                                 const Shape& shape_with_layout,
                                 int64_t priority);

    // Accessors for the underlying ComputationLayout and
    // ComputationLayoutConstraint.
    const ComputationLayout& computation_layout() const {
      return computation_constraint_.computation_layout();
    }
    const ComputationLayoutConstraint& computation_constraint() const {
      return computation_constraint_;
    }
    ComputationLayoutConstraint* mutable_computation_constraint() {
      return &computation_constraint_;
    }

   private:
    // Key identifying a specific operand position of an instruction within the
    // computation.
    using OperandConstraintKey = std::pair<const HloInstruction*, int64_t>;

    // Map from (instruction, operand_no) pair to its OperandLayoutConstraint.
    absl::flat_hash_map<OperandConstraintKey,
                        std::unique_ptr<OperandLayoutConstraint>>
        operand_constraints_;

    // The computation whose values and interface are constrained.
    HloComputation* computation_;

    // Constraint on the computation interface (parameters and result layouts).
    ComputationLayoutConstraint computation_constraint_;
  };

  // Determines whether an instruction can change layouts. An instruction not
  // being able to change layout means that it requires operands with the same
  // rank as the output to have the same layout as the output.
  static bool InstructionCanChangeLayout(const HloInstruction* instruction);

  const LayoutConstraints& computation_constraints(
      const HloComputation* computation) const {
    return *FindOrDie(computation_layouts_, computation);
  }

  LayoutConstraints& mutable_computation_constraints(
      const HloComputation* computation) {
    return *FindOrDie(computation_layouts_, computation);
  }
  LayoutConstraints* mutable_computation_constraints(
      HloComputation* computation) {
    auto it = computation_layouts_.find(computation);
    LayoutConstraints* constraints = nullptr;
    if (it == computation_layouts_.end()) {
      computation_layouts_.emplace(
          computation,
          constraints = new LayoutConstraints(
              computation, nullptr, LayoutConstraint::kDefaultPriority));
    } else {
      constraints = (*it).second.get();
    }
    return constraints;
  }
  void PushAddedConstraints(const LayoutConstraint* constraint);

  // In case of an array shape returns true iff it is at most rank 1. In case of
  // a tuple shape returns true iff all leaf shapes are at most rank 1.
  static bool IsAtMostRank1(const Shape& shape);
  // Convenience wrapper around SetOperandLayout for setting the layout of a
  // operand using a Layout object. The operand must be array-shaped.
  absl::Status SetArrayOperandLayout(const Layout& layout,
                                     const HloInstruction* instruction,
                                     int64_t operand_no, bool mandatory = true,
                                     bool dfs = true) {
    return SetArrayOperandLayout(layout, instruction, operand_no, mandatory,
                                 dfs, current_priority_);
  }
  absl::Status SetArrayOperandLayout(const Layout& layout,
                                     const HloInstruction* instruction,
                                     int64_t operand_no, bool mandatory,
                                     bool dfs, int64_t priority);
  // Convenience wrapper around SetBufferLayout. Sets the layouts of all buffers
  // created by the instruction to the layouts in the given shape. The
  // instruction must define every logical buffer in its output.
  // If `allow_alias` is false, the function will check that all output buffers
  // are defined by `instruction`, not aliased to an instruction elsewhere.
  absl::Status SetInstructionLayout(const Shape& shape_with_layout,
                                    const HloInstruction* instruction,
                                    bool mandatory = true, bool dfs = true,
                                    bool allow_alias = false,
                                    ShapeIndexView subshape_index = {}) {
    return SetInstructionLayout(shape_with_layout, instruction, mandatory, dfs,
                                allow_alias, current_priority_, subshape_index);
  }
  absl::Status SetInstructionLayout(const Shape& shape_with_layout,
                                    const HloInstruction* instruction,
                                    bool mandatory, bool dfs, bool allow_alias,
                                    int64_t priority,
                                    ShapeIndexView subshape_index = {});
  // Set the same given layout across all components of the instruction output.
  // It works the same as the API above if the output is a single array.
  absl::Status SetInstructionLayout(const Layout& layout,
                                    const HloInstruction* instruction,
                                    bool mandatory = true, bool dfs = true,
                                    bool allow_alias = false,
                                    int64_t priority = -1);
  // Add a constraint on the layout of a LogicalBuffer, the layout of the
  // operand of the instruction, or the layout of the result of the computation,
  // respectively.
  absl::Status SetBufferLayout(const Layout& layout,
                               const LogicalBuffer& buffer,
                               bool mandatory = true, bool dfs = true) {
    return SetBufferLayout(layout, buffer, mandatory, dfs, current_priority_);
  }
  absl::Status SetBufferLayout(const Layout& layout,
                               const LogicalBuffer& buffer, bool mandatory,
                               bool dfs, int64_t priority,
                               const HloInstruction* from_user = nullptr);
  absl::Status SetOperandLayout(const Shape& shape_with_layout,
                                const HloInstruction* instruction,
                                int64_t operand_no, bool mandatory = true,
                                bool dfs = true) {
    return SetOperandLayout(shape_with_layout, instruction, operand_no,
                            mandatory, dfs, current_priority_);
  }
  absl::Status SetOperandLayout(const Shape& shape_with_layout,
                                const HloInstruction* instruction,
                                int64_t operand_no, bool mandatory, bool dfs,
                                int64_t priority);
  bool reverse_computation_order() const { return reverse_computation_order_; }

  ComputationLayout& saved_entry_computation_layout() {
    return saved_entry_computation_layout_;
  }
  virtual bool NegotiateLayout(const HloInstruction* instruction,
                               const Layout& new_layout,
                               const Layout& existing_layout,
                               const HloInstruction* from_user,
                               const HloInstruction* orig_user) {
    return false;
  }
  virtual bool NegotiateOperandLayout(const HloInstruction* instruction,
                                      int64_t operand_no,
                                      const Layout& new_layout,
                                      const Layout& existing_layout) {
    return false;
  }
  // Should be made consistent with the ChooseOperandLayoutFromOutputLayout
  // except that a boolean instead of concrete layout is returned.
  virtual bool OperandLayoutAlwaysPropagateForward(const HloInstruction* user);
  // Controls when all operands of user must have the same layout.
  virtual bool OperandLayoutAlwaysPropagateToSiblings(
      const HloInstruction* user);
  // Controls when all operands of user must have the same layout as the output.
  virtual bool OutputLayoutAlwaysPropagateToOperands(
      const HloInstruction* user);
  // Whether to propagate the reduction layout to the operand by preserving the
  // same relative order of the dimensions that are kept, and making the
  // reduction dims the most minor dimensions.
  virtual bool PropagateReductionLayoutToOperand(const HloInstruction* user) {
    return false;
  }

 protected:
  // These methods, invoked by PropagateConstraints, propagate a layout
  // constraint to its neighbors (i.e. operands and users) in order to minimize
  // the cost of the instructions being constrainted on. New constraints are
  // added to the given constraint set.
  //
  // Backends can override these methods with backend-specific propagation
  // rules.
  virtual absl::Status PropagateBufferConstraint(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);
  virtual absl::Status PropagateOperandConstraint(
      const OperandLayoutConstraint& operand_constraint,
      LayoutConstraints* constraints);
  virtual absl::Status PropagateResultConstraint(
      const ComputationLayoutConstraint& layout_constraint,
      LayoutConstraints* constraints);

  virtual Layout GetUnconstrainedLayout(const LogicalBuffer& buffer) {
    return LayoutUtil::GetDefaultLayoutForShape(buffer.shape());
  }
  // Called after layouts of an instruction have been finalized to allow
  // subclasses to check for platform specific assumptions.
  virtual absl::Status Verify(const HloInstruction* instruction) {
    return absl::OkStatus();
  }

  absl::Status PropagateUnconstraintedBuffers(LayoutConstraints* constraints);
  const BufferLayoutConstraint* GetBufferLayoutConstraint(
      const LogicalBuffer& buffer) const;
  absl::StatusOr<const BufferLayoutConstraint*>
  GetInstructionBufferLayoutConstraint(const HloInstruction* instruction) const;
  // Find a bufferset in the bufferset cache. This is useful since we can
  // currently create the flattened buffer set for the same instruction many
  // times, which is often slow.
  PointsToSet::BufferSet* GetBufferSet(const HloInstruction* instruction) const;
  // Similar to above, but returns true only if all buffers associated with that
  // operand are forwarded.
  bool AllOperandBuffersForwarded(const HloInstruction* instruction,
                                  int64_t operand_no) const;
  // Returns true if any buffer in the given operand is forwarded to the output
  // of the given instruction. For example, the Tuple instruction forwards the
  // buffers of its operands and would return true for each of its operands.
  bool AnyOperandBufferForwarded(const HloInstruction* instruction,
                                 int64_t operand_no) const;
  absl::StatusOr<Layout> InferArrayLayout(const HloInstruction* instruction,
                                          const ShapeIndex& index);

  // Propagates a buffer layout constraint into the operands that use it.
  absl::Status PropagateBufferConstraintToUses(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);

  // Propagates a layout constraint on the use of the result of the given
  // instruction to the definitions of the LogicalBuffers which make up the
  // result.
  absl::Status PropagateUseConstraintToDefs(
      const ShapeLayout& shape_layout, const HloInstruction* instruction,
      LayoutConstraints* constraints, int64_t priority,
      const HloInstruction* user = nullptr);

  // Chooses a layout of operand `operand_no` of `instruction` that minimizes
  // the cost of `instruction`. `output_layout` is the layout of `instruction`.
  // Returns null if it can't decide the best layout.
  // Precondition: `instruction` and the operand are array-shaped.
  virtual std::unique_ptr<Layout> ChooseOperandLayoutFromOutputLayout(
      const Layout& output_layout, const HloInstruction* instruction,
      int64_t operand_no);
  // Given the layout of `user`'s `operand_no`-th operand, chooses a layout of
  // `user` that minimizes its cost on that operand.  Returns null if it can't
  // decide the best layout.
  // Precondition: `user` and the operand are array-shaped.
  virtual std::unique_ptr<Layout> ChooseOutputLayoutFromOperandLayout(
      const Layout& operand_layout, const HloInstruction* user,
      int64_t operand_no);

  // Convenient wrapper for InstructionCanChangeLayout which can be overridden
  // in subclasses.
  virtual bool InstructionCanChangeLayoutInstance(
      const HloInstruction* instruction);

  // The shapes in caller can be different from the shapes in callee. For
  // example, a shape (1024, 128) of an array can be distributed to four threads
  // so the shape for each thread is (256, 128). When verifying the callee's
  // shapes based on the caller, we should use this function to compute the
  // expected shape. The param_id should be the parameter id of the shape or -1
  // for the result output or unknown.
  virtual Shape ShardedShape(const HloInstruction* call, const Shape& shape,
                             int param_id) {
    return shape;
  }
  // When verifying the caller's shapes based on the callee, we should use this
  // function to compute the expected shape.
  // The param_id should be the parameter id of the shape or -1 for the result
  // output or unknown.
  virtual Shape UnShardedShape(const HloInstruction* call, const Shape& shape,
                               int param_id) {
    return shape;
  }

  // The operands of a call must match the layouts of parameters in the
  // ComputationLayout, and the call instruction itself must match the result
  // layout in the ComputationLayout.
  absl::Status CheckCallLayout(HloInstruction* call,
                               const ComputationLayout& computation_layout);
  // For a custom-call user, propagates the operand constraint to the result
  // based on output-to-operand aliasing.
  absl::Status PropagateOperandConstraintToResultForCustomCall(
      const HloInstruction* user,
      const OperandLayoutConstraint& operand_constraint);

  // Assign layouts to the given module. Returns whether the module was changed
  // (any layouts were changed).
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Initializes the layout assignment object for a new Run() call.
  absl::Status Init(HloModule* module);

  // Clones conditional computations with multiple callsites and adds copies
  // for operands of Send and layout-constrained CustomCall instructions.
  absl::Status PrepareHloForLayoutAssignment(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Verifies that the entry computation layout is compatible with the entry
  // computation shape.
  absl::Status VerifyEntryComputationLayout(const HloModule* module) const;

  // Sets up propagation by running points-to analysis, gathering computations
  // to work on, and initializing entry computation constraints.
  absl::StatusOr<std::vector<HloComputation*>> SetupPropagation(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Resolves input-output aliasing by resetting layouts to match if they
  // mismatch. Returns true if any layouts were changed.
  absl::StatusOr<bool> ResolveInputOutputAliasing(
      HloModule* module, ComputationLayout* entry_constraint);

  // Adds constraints which must be satisfied for correctness on all
  // backends. Called once prior to propagating constraints.
  absl::Status AddMandatoryConstraints(
      ChannelLayoutConstraints* channel_constraints,
      LayoutConstraints* constraints);

  // Adds constraints for instructions that define values with pre-existing
  // layouts.
  absl::Status AddInstructionLayoutConstraints(
      ChannelLayoutConstraints* channel_constraints,
      LayoutConstraints* constraints);

  absl::Status AddInfeedConstraints(HloInstruction* instruction);
  absl::Status AddOutfeedConstraints(HloInstruction* instruction);
  absl::Status AddParameterConstraints(HloInstruction* instruction,
                                       LayoutConstraints* constraints);
  absl::Status AddCollectiveConstraints(HloInstruction* instruction);
  absl::Status AddCrossModuleAllReduceConstraints(
      HloInstruction* instruction,
      ChannelLayoutConstraints* channel_constraints);

  // Adds constraints for instructions that call or interact with
  // sub-computations.
  absl::Status AddSubcomputationLayoutConstraints(
      LayoutConstraints* constraints);

  absl::Status AddCallConstraints(HloInstruction* instruction);
  absl::Status AddWhileConstraints(HloInstruction* instruction,
                                   LayoutConstraints* constraints);
  absl::Status AddConditionalConstraints(HloInstruction* instruction);
  absl::Status AddAsyncStartConstraints(HloInstruction* instruction);
  absl::Status AddAsyncUpdateConstraints(HloInstruction* instruction);
  absl::Status AddAsyncDoneConstraints(HloInstruction* instruction,
                                       LayoutConstraints* constraints);
  absl::Status AddAsyncInstructionConstraints(HloInstruction* instruction,
                                              HloComputation* async_comp);

  // Propagates layout constraints from the caller instruction into the inner
  // async sub-computation.
  // This is the forward propagation step: it takes the layouts of the operands
  // and result of the async start/update instruction (which are in the parent
  // computation) and propagates them to the parameters and result of the
  // async sub-computation.
  // If any layout in the sub-computation is updated, it resets the
  // sub-computation layout with an elevated priority to ensure it is respected
  // during the sub-computation's layout assignment. Returns the reconciled
  // layout of the sub-computation.
  ComputationLayout PropagateLayoutsToAsyncSubComputation(
      const HloInstruction* instruction, LayoutConstraints* async_constraint);

  // Propagates the operand array layouts of `instruction` to the parameter
  // layouts defined in `async_layout`.
  // Updates `async_layout` in-place and returns true if any parameter layout
  // was changed.
  bool PropagateOperandLayoutsToAsyncParameters(
      const HloInstruction* instruction, ComputationLayout* async_layout);

  // Propagates array layouts for operand `operand_idx` of `instruction`
  // to the corresponding parameter `param_idx` in `async_layout`.
  // Updates `async_layout` in-place and returns true if the layout was updated.
  bool PropagateOperandLayoutToAsyncParameter(const HloInstruction* instruction,
                                              int64_t operand_idx,
                                              int64_t param_idx,
                                              ComputationLayout* async_layout);

  // Propagates array layouts from the result shape of `instruction` (tuple
  // element 1) to `async_layout`'s result layout.
  // We assume the result shape of the async operation is at index {1} of the
  // `instruction` (async start/update) output tuple.
  // Updates `async_layout` in-place and returns true if the result layout was
  // updated.
  bool PropagateResultLayoutToAsyncSubComputation(
      const HloInstruction* instruction, ComputationLayout* async_layout);

  // Propagates async sub-computation parameter and result layout constraints
  // back onto the caller instruction and its operands in the parent
  // computation. This is the backward propagation step: it takes the resolved
  // layouts from the async sub-computation and applies them as mandatory
  // constraints on the caller instruction's shape (at index {1} for result) and
  // its operands.
  absl::Status PropagateLayoutsFromAsyncSubComputation(
      HloInstruction* instruction, const ComputationLayout& async_layout,
      LayoutConstraints* async_constraint);

  absl::Status ConstrainAsyncOperands(HloInstruction* instruction,
                                      int first_operand_idx,
                                      int start_param_idx,
                                      const ComputationLayout& async_layout);
  absl::Status AddAsyncOpResultLayoutConstraint(
      HloInstruction* instruction, const ComputationLayout& async_layout,
      LayoutConstraints* async_constraint);
  // Sets the computation result layout based on constraints and
  // sub-computations.
  absl::Status AddComputationResultLayoutConstraints(
      LayoutConstraints* constraints);

  // Constrains layouts for custom calls that have specific layout requirements.
  absl::Status AddCustomCallConstraints(LayoutConstraints* constraints);

  // Initializes unconstrained_buffer_ids_ with all array-shaped logical buffers
  // in the given computation.
  void InitUnconstrainedBuffers(HloComputation* computation);

  // Records instructions that lack layout constraints before applying default
  // layouts.
  void RecordUnconstrainedLayoutInstructions();

  // Iteratively assigns layouts to remaining unconstrained buffers and
  // propagates until all buffers are constrained.
  absl::Status AssignLayoutsToUnconstrainedBuffers(
      LayoutConstraints* constraints);

  // Return a vector containing the constraints which have been added to the
  // LayoutConstraints object since the construction of the object or since the
  // last time ConsumeAddedConstraints() has been called. This is used to
  // identify newly added constraints when propagating layouts.
  std::vector<const LayoutConstraint*> ConsumeAddedConstraints() {
    std::vector<const LayoutConstraint*> ret_vec(std::move(added_constraints_));
    added_constraints_.clear();
    return ret_vec;
  }
  void ClearAddedConstraints() { added_constraints_.clear(); }

  // This method can be overridden to add backend-specific constraints to the
  // layout of the instructions of a computation. This method is called after
  // all mandatory constraints have been added via AddMandatoryConstraints
  // and before propagating constraints.
  virtual absl::Status AddBackendConstraints(LayoutConstraints* constraints) {
    return absl::OkStatus();
  }

  // Construct constraints and assign layouts to all instructions in the
  // computation satisfying the given ComputationLayout, if not nullptr.
  // Otherwise the ComputationLayout will be calculated by propagating the
  // computation instruction constraints.
  // Layouts constraints are added, then propagated until all LogicalBuffers in
  // the computation are constrained.
  absl::Status RunOnComputation(LayoutConstraints* constraints,
                                ChannelLayoutConstraints* channel_constraints);

  // Assign layouts to the instructions of a computation which satisfy the given
  // layout constraints. Copies may be added to satisfy the constraints. The
  // given LayoutConstraints must have layout constraints every logical buffer
  // in the computation.
  absl::Status AssignLayouts(LayoutConstraints& constraints);

  // Propagates layout constraints from a set of initial constraints in order to
  // minimize the local cost of the computation. This propagation is *not*
  // required for correctness.
  absl::Status PropagateConstraints(LayoutConstraints* constraints);

  absl::Status PropagateBufferConstraintToOperands(
      const BufferLayoutConstraint& buffer_constraint,
      LayoutConstraints* constraints);

  // Check that all layouts in the module have been set and satisfy all
  // necessary conditions.
  absl::Status CheckLayouts(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Computes the ComputationLayout of the given constraints based of the
  // layouts assigned to parameters and root instruction. Also propagate
  // constraints to computation nested inside.
  absl::Status CalculateComputationLayout(LayoutConstraints* constraints);

  // Clears all the layouts which can be cleared within a computation.
  absl::Status ClearComputationLayouts(HloComputation* computation);

  // Clears the side effects of a previous pass, like added copy instructions.
  absl::Status ClearPreviousPassSideEffects(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Propagates the layouts computed by the layout assignment pass on the given
  // computation, to the computation layout passed in to this API.
  // This API propagates missing layout, and also checks that the caller
  // specified have been respected, by comparing those with the parameters and
  // root computation instruction.
  absl::Status PropagateComputationLayouts(
      HloComputation* computation, ComputationLayout* computation_layout);

  // The pointer to the ComputationLayout passed as constructor parameter.
  ComputationLayout* entry_computation_layout_;

  // A copy of entry_computation_layout_ used to reset it to the initial values
  // during the multiple passes done by the layout assignment operation.
  ComputationLayout saved_entry_computation_layout_;
  // If set true, reverse the computation traversal order when assigning layout.
  bool reverse_computation_order_;

 protected:
  static constexpr int64_t kNumberOfPropagationRounds = 2;
  // Sets up the copy instruction according to the characteristic (sharding,
  // metadata, ...) of the reference instruction. The index argument is used
  // when the instruction is a tuple, and in such case the index represents
  // the location from where the copy instruction was created from.
  // If the index is empty, the whole sharding will be propagated, even in case
  // the instruction has a tuple sharding.
  static void SetupCopiedInstruction(const HloInstruction& instruction,
                                     HloInstruction* copy,
                                     const ShapeIndex& index);

  // Creates and returns a copy of the given instruction with a different
  // layout. Tuple-shaped instructions will be deep-copied, and the last Tuple
  // instruction producing the copy is returned.
  absl::StatusOr<HloInstruction*> CreateCopyWithNewLayout(
      const Shape& shape_with_layout, HloInstruction* instruction);

  // Creates a copy of the given operand if the operand's layout does not match
  // the given layout. This copy replaces the use in the given instruction.
  // Tuple operands will be deep-copied.
  virtual absl::Status CopyOperandIfLayoutsDiffer(
      const ShapeLayout& operand_layout, HloInstruction* instruction,
      int64_t operand_no);

  // Registers a copy instruction added by the layout assignment pass.
  void RegisterAddedCopy(HloInstruction* copy) {
    CHECK_EQ(copy->opcode(), HloOpcode::kCopy);
    added_copies_.insert(copy);
  }

  // Adds a copy for the operand of an instruction, unless such operand is
  // already a copy, and has a single user (which is forcibly the instruction
  // itself).
  absl::Status AddCopyForOperand(HloInstruction* instruction,
                                 int64_t operand_number);

  // Apply the channel layout constraints by populating the channel_constraints
  // data structure passed in at constructor time. Eventually adds copies in
  // case two ends of a channel ended up with a different leyout.
  absl::Status ConstrainChannelLayouts(
      HloComputation* computation,
      ChannelLayoutConstraints* channel_constraints);

  // Resets the input ChannelLayoutConstraints to the original copy received
  // from the constructor input.
  void ResetChannelConstraints() {
    if (channel_layout_constraints_ != nullptr) {
      *channel_layout_constraints_ = channel_constraints_;
    }
  }

  void ResetEntryComputationLayout() {
    *entry_computation_layout_ = saved_entry_computation_layout_;
  }

  // Adds constraints related to host Send/Recv instructions.
  absl::Status BuildHostChannelConstraints(HloComputation* computation);

  // Module points to analysis that can be updated for cloned computations.
  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;

  // The set of HLO instructions which lacked any layout constraint, thus
  // receiving propagated default layouts.
  absl::flat_hash_set<const HloInstruction*> unconstrained_layout_instructions_;

  HloPredicate instruction_can_change_layout_func_;

  std::string ToString(const LayoutConstraints& constraints) const;

  int64_t current_priority() const { return current_priority_; }

 private:
  // Returns whether the given instruction is in a copy-disabled while loop.
  bool IsWhileLoopCopyDisabled(const HloInstruction& instruction) const;

  // Map containing the layouts of all computations assigned so
  // far. Computations are handled in a topological sort where computations are
  // handled before their caller instructions so the layouts of caller
  // instructions can be set to match the computation.
  absl::flat_hash_map<const HloComputation*, std::unique_ptr<LayoutConstraints>>
      computation_layouts_;

  // Map from branch computations to the result layout they should apply.
  absl::flat_hash_map<HloComputation*, ComputationLayout> conditional_mismatch_;

  // Every copy added to the module by the layout assignment pass is registered
  // here.
  absl::flat_hash_set<HloInstruction*> added_copies_;

  // The pointer to the channel layout constraints passed in with the
  // constructor. If not nullptr, this is an input/output argument.
  ChannelLayoutConstraints* channel_layout_constraints_ = nullptr;

  // A copy of the input layout constraints used to reset the above pointer in
  // case we have to undo operations due to the multiple passes over the
  // computations/instructions.
  ChannelLayoutConstraints channel_constraints_;

  // Layout constraints for send/recv instructions which communicate with the
  // host.
  ChannelLayoutConstraints host_channel_constraints_;

  // Array-shaped buffers which have not yet been constrained.
  std::set<LogicalBuffer::Id> unconstrained_buffer_ids_;

  mutable absl::flat_hash_map<const HloInstruction*,
                              std::unique_ptr<PointsToSet::BufferSet>>
      buffer_sets_cache_;

  // The set of BufferLayoutConstraints applied to the computation.
  absl::flat_hash_map<const LogicalBuffer*,
                      std::unique_ptr<BufferLayoutConstraint>>
      buffer_constraints_;

  // A vector which holds constraints as they are added. Can be cleared with
  // ClearAddedConstraints.
  std::vector<const LayoutConstraint*> added_constraints_;
  int64_t current_priority_ = LayoutConstraint::kBeginningPriority;

  // Stores the set of while computations that have copy disabled.
  absl::flat_hash_set<const HloComputation*> copy_disabled_while_computations_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
