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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LAYOUT_ASSIGNMENT_H_

#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Abstract base class for layout constraints. These constraint objects are
// gathered together in LayoutConstraints object.
class LayoutConstraint {
 public:
  LayoutConstraint(bool mandatory) : mandatory_(mandatory) {}
  virtual ~LayoutConstraint() = default;

  virtual string ToString() const = 0;

  // True if this constraint cannot be overwritten by a different constraint.
  bool mandatory() const { return mandatory_; }

 private:
  bool mandatory_;
};

std::ostream& operator<<(std::ostream& out, const LayoutConstraint& constraint);

// Layout constraint on a single LogicalBuffer. This constrains the layout of an
// array produced by a particular instruction.
class BufferLayoutConstraint : public LayoutConstraint {
 public:
  BufferLayoutConstraint(const Layout& layout, const LogicalBuffer& buffer,
                         bool mandatory);

  const LogicalBuffer& buffer() const { return *buffer_; }
  const Layout& layout() const { return layout_; }

  string ToString() const override;

 private:
  Layout layout_;
  const LogicalBuffer* buffer_;
};

// Constraint on the layout of the operand of an instruction. The constrained
// shape can be arbitrarily shaped (array or tuple). This is a constraint on the
// use of a shaped value and is not a hard constraint on the instruction(s)
// which define the value as copies may be inserted between the definition and
// use.
class OperandLayoutConstraint : public LayoutConstraint {
 public:
  OperandLayoutConstraint(const ShapeLayout& shape_layout,
                          const HloInstruction* instruction, int64 operand_no,
                          bool mandatory);

  const ShapeLayout& shape_layout() const { return shape_layout_; }
  const HloInstruction* instruction() const { return instruction_; }
  const int64 operand_no() const { return operand_no_; }
  const HloInstruction* operand() const {
    return instruction_->operand(operand_no_);
  }

  string ToString() const override;

 private:
  ShapeLayout shape_layout_;
  const HloInstruction* instruction_;
  int64 operand_no_;
};

// Constraint on the layout of the result of the entry computation.
class ResultLayoutConstraint : public LayoutConstraint {
 public:
  explicit ResultLayoutConstraint(const ShapeLayout& shape_layout)
      : LayoutConstraint(/*mandatory=*/true), shape_layout_(shape_layout) {}

  const ShapeLayout& shape_layout() const { return shape_layout_; }
  string ToString() const override;

 private:
  const ShapeLayout shape_layout_;
};

// Class encapsulating the layout constraints of the values in a HLO
// computation.
class LayoutConstraints {
 public:
  LayoutConstraints(const TuplePointsToAnalysis& points_to_analysis,
                    HloComputation* computation);
  ~LayoutConstraints() = default;

  const HloComputation* computation() const { return computation_; }
  HloComputation* computation() { return computation_; }
  const TuplePointsToAnalysis& points_to_analysis() const {
    return points_to_analysis_;
  }

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

  // Returns the layout of a LogicalBuffer, the layout of the operand of the
  // instruction, or the layout of the result of the computation, respectively,
  // if it has been constrained. Otherwise return nullptr.
  const Layout* BufferLayout(const LogicalBuffer& buffer) const;
  const BufferLayoutConstraint* GetBufferLayoutConstraint(
      const LogicalBuffer& buffer) const;
  const ShapeLayout* OperandLayout(const HloInstruction* instruction,
                                   int64 operand_no) const;
  const OperandLayoutConstraint* GetOperandLayoutConstraint(
      const HloInstruction* instruction, int64 operand_no) const;
  const ShapeLayout* ResultLayout() const;

  // Add a constraint on the layout of a LogicalBuffer, the layout of the
  // operand of the instruction, or the layout of the result of the computation,
  // respectively.
  Status SetBufferLayout(const Layout& layout, const LogicalBuffer& buffer,
                         bool mandatory = true);
  Status SetOperandLayout(const Shape& shape_with_layout,
                          const HloInstruction* instruction, int64 operand_no,
                          bool mandatory = true);
  Status SetResultLayout(const Shape& shape_with_layout);

  // Convenience wrapper around SetOperandLayout for setting the layout of a
  // operand using a Layout object. The operand must be array-shaped.
  Status SetArrayOperandLayout(const Layout& layout,
                               const HloInstruction* instruction,
                               int64 operand_no, bool mandatory = true);

  // Convenience wrapper around SetBufferLayout. Sets the layouts of all buffers
  // created by the instruction to the layouts in the given shape. The
  // instruction must define every logical buffer in its output.
  Status SetInstructionLayout(const Shape& shape_with_layout,
                              const HloInstruction* instruction);

  // Returns true if any buffer in the given operand is forwarded to the output
  // of the given instruction. For example, the Tuple instruction forwards the
  // buffers of its operands and would return true for each of its operands.
  bool OperandBufferForwarded(const HloInstruction* instruction,
                              int64 operand_no) const;

  // Returns the set of logical buffers (by LogicalBuffer:Id) which do not
  // yet have a layout constraint
  const std::set<LogicalBuffer::Id>& unconstrained_buffer_ids() const {
    return unconstrained_buffer_ids_;
  }

  string ToString() const;

 private:
  // The set of BufferLayoutConstraints applied to the computation.
  std::unordered_map<const LogicalBuffer*, BufferLayoutConstraint>
      buffer_constraints_;

  // The set of OperandLayoutConstraints applied to the computation.
  using OperandConstraintKey = std::pair<const HloInstruction*, int64>;
  std::map<OperandConstraintKey, OperandLayoutConstraint> operand_constraints_;

  // The result constraint for the computation (can be null).
  std::unique_ptr<ResultLayoutConstraint> result_constraint_;

  // A vector which holds constraints as they are added. Can be cleared with
  // ClearAddedConstraints.
  std::vector<const LayoutConstraint*> added_constraints_;

  // Points-to analysis for the module. Used to propagate constraints through
  // the HLO graph.
  const TuplePointsToAnalysis& points_to_analysis_;

  // Array-shaped buffers which have not yet been constrained.
  std::set<LogicalBuffer::Id> unconstrained_buffer_ids_;

  HloComputation* computation_;
};

// Contains constraints on the layout of channels; sends and recvs.
class ChannelLayoutConstraints {
 public:
  // Construct an empty constraint set.
  ChannelLayoutConstraints() {}

  // Returns true if channel_id has a layout constraint.
  bool IsChannelConstrained(int64 channel_id) const {
    return constraints_.count(channel_id) > 0;
  }

  // Given `shape`, apply the layout for `channel_id`. `channel_id` must already
  // be constrained.
  Shape LayoutShapeForChannel(Shape shape, int64 channel_id) const {
    CHECK(IsChannelConstrained(channel_id));
    *shape.mutable_layout() = constraints_.at(channel_id);
    return shape;
  }

  // Returns the layout constraint for `channel_id`, which must already be
  // constrained.
  Layout LayoutForChannel(int64 channel_id) const {
    CHECK(IsChannelConstrained(channel_id));
    return constraints_.at(channel_id);
  }

  // Adds a new layout constraint for `channel_id`. If a constraint for
  // `channel_id` already exists, this operation requires that the new layout is
  // the same as the previously constrained layout.
  void ConstrainChannel(int64 channel_id, const Layout& layout) {
    CHECK(!IsChannelConstrained(channel_id) ||
          LayoutUtil::Equal(layout, constraints_[channel_id]));
    constraints_[channel_id] = layout;
  }

 private:
  std::unordered_map<int64, Layout> constraints_;
};

// HLO pass which assigns layouts to all instructions in the HLO module while
// satisfying all necessary invariants and minimizing cost.
class LayoutAssignment : public HloPassInterface {
 public:
  // entry_computation_layout is modified to populate a layout for the result in
  // the case that no particular layout is requested.
  //
  // channel_constraints is both an input and output. Any sends or recvs that
  // are present in channel_constraints will be layed out as constrained. Any
  // unconstrained sends or recvs will be layed out as locally optimal and their
  // layout will be added as a constraint to channel_constraints.
  //
  // If channel_constraints is nullptr, no kSend or kRecvs must be contained
  // within any module passed to `Run`.
  explicit LayoutAssignment(
      ComputationLayout* entry_computation_layout,
      ChannelLayoutConstraints* channel_constraints = nullptr);
  ~LayoutAssignment() override {}
  tensorflow::StringPiece name() const override { return "layout-assignment"; }

  // Assign layouts to the given module. Returns whether the module was changed
  // (any layouts were changed).
  StatusOr<bool> Run(HloModule* module) override;

 protected:
  // These methods, invoked by PropagateConstraints, propagate a layout
  // constraint to its neighbors (i.e. operands and users) in order to minimize
  // the cost of the instructions being constrainted on. New constraints are
  // added to the given constraint set.
  //
  // Backends can override these methods with backend-specific propagation
  // rules.
  virtual Status PropagateBufferConstraint(
      const BufferLayoutConstraint& layout_constraint,
      LayoutConstraints* constraints);
  virtual Status PropagateOperandConstraint(
      const OperandLayoutConstraint& layout_constraint,
      LayoutConstraints* constraints);
  virtual Status PropagateResultConstraint(
      const ResultLayoutConstraint& layout_constraint,
      LayoutConstraints* constraints);

  // By default LayoutAssignment ensures that inputs and outputs of CustomCalls
  // have the "major-first" layout (i.e.  {n, n-1, ..., 0}).
  //
  // If this function returns true, LayoutAssignment does not set a layout for
  // the given CustomCall.  It's up to the backend to set one in
  // AddBackendConstraints, if necessary.
  //
  // Precondition: instruction->opcode() == HloOpcode::kCustomCall.
  virtual bool CustomCallRequiresMajorFirstLayout(
      const HloInstruction* /*instruction*/) {
    return true;
  }

  // Called after layouts of an instruction have been finalized to allow
  // subclasses to check for platform specific assumptions.
  virtual Status Verify(const HloInstruction* instruction) {
    return Status::OK();
  }

  // Propagates a buffer layout constraint into the operands that use it.
  Status PropagateBufferConstraintToUses(
      const BufferLayoutConstraint& layout_constraint,
      LayoutConstraints* constraints);

  // Propagates a layout constraint on the use of the result of the given
  // instruction to the definitions of the LogicalBuffers which make up the
  // result.
  Status PropagateUseConstraintToDefs(const ShapeLayout& shape_layout,
                                      const HloInstruction* instruction,
                                      LayoutConstraints* constraints);

  // Chooses a layout of operand `operand_no` of `instruction` that minimizes
  // the cost of `instruction`. `output_layout` is the layout of `instruction`.
  // Returns null if it can't decide the best layout.
  // Precondition: `instruction` and the operand are array-shaped.
  std::unique_ptr<Layout> ChooseOperandLayoutFromOutputLayout(
      const Layout& output_layout, const HloInstruction* instruction,
      int64 operand_no);
  // Given the layout of `user`'s `operand_no`-th operand, chooses a layout of
  // `user` that minimizes its cost on that operand.  Returns null if it can't
  // decide the best layout.
  // Precondition: `user` and the operand are array-shaped.
  std::unique_ptr<Layout> ChooseOutputLayoutFromOperandLayout(
      const Layout& operand_layout, const HloInstruction* user,
      int64 operand_no);

 private:
  // Adds constraints which must be satisfied for correctness on all
  // backends. Called once prior to propagating constraints.
  Status AddMandatoryConstraints(
      const ComputationLayout& computation_layout,
      const ChannelLayoutConstraints* channel_constraints,
      HloComputation* computation, LayoutConstraints* constraints);

  // This method can be overridden to add backend-specific constraints to the
  // layout of the instructions of a computation. This method is called after
  // all mandatory constraints have been added via AddMandatoryConstraints
  // and before propagating constraints.
  virtual Status AddBackendConstraints(LayoutConstraints* constraints) {
    return Status::OK();
  }

  // Construct contraints and assign layouts to all instructions in the
  // computation satisfying the given ComputationLayout. Layouts constraints are
  // added, then propagated until all LogicalBuffers in the computation are
  // constrained.
  Status RunOnComputation(const ComputationLayout& computation_layout,
                          const TuplePointsToAnalysis& points_to_analysis,
                          HloComputation* computation,
                          ChannelLayoutConstraints* channel_constraints);

  // Assign layouts to the instructions of a computation which satisfy the given
  // layout constraints. Copies may be added to satisfy the constraints. The
  // given LayoutConstraints must have layout constraints every logical buffer
  // in the computation.
  Status AssignLayouts(const LayoutConstraints& constraints,
                       HloComputation* computation);

  // Propagates layout constraints from a set of initial constraints in order to
  // minimize the local cost of the computation. This propagation is *not*
  // required for correctness.
  Status PropagateConstraints(LayoutConstraints* constraints);

  // Check that all layouts in the module have been set and satisfy all
  // necessary conditions.
  Status CheckLayouts(HloModule* module);

  ComputationLayout* entry_computation_layout_;
  ChannelLayoutConstraints* channel_layout_constraints_;

 protected:
  // Map containing the layouts of all computations assigned so
  // far. Computations are handled in a topological sort where computations are
  // handled before their caller instructions so the layouts of caller
  // instructions can be set to match the computation.
  std::map<HloComputation*, ComputationLayout> computation_layouts_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LAYOUT_ASSIGNMENT_H_
