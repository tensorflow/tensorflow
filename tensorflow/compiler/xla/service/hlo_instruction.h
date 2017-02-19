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

// HLO instructions are in DAG form and represent the computations that the user
// has built up via the XLA service interface. They are ultimately lowered
// in a platform-aware way by traversing the HLO DAG and emitting a lowered
// form; e.g. see DfsHloVisitor.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_H_

#include <functional>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloComputation;

// HLO instructions are the IR used by the high-level compiler.
class HloInstruction {
 public:
  enum class FusionKind {
    kLoop,                // Fused into a loop.
    kInput,               // Fused into a reduction kernel.
    kTransposeDot,        // Fused into a dot with transposed operands.
    kConvBackwardFilter,  // Fused into a backward filter convolution.
    kConvBackwardInput,   // Fused into a backward input convolution.
  };

  // Creates a parameter-retrieving instruction.
  static std::unique_ptr<HloInstruction> CreateParameter(int64 parameter_number,
                                                         const Shape& shape,
                                                         const string& name);

  // Creates a literal constant instruction.
  static std::unique_ptr<HloInstruction> CreateConstant(
      std::unique_ptr<Literal> literal);

  // Creates a get tuple element instruction.
  static std::unique_ptr<HloInstruction> CreateGetTupleElement(
      const Shape& shape, HloInstruction* operand, int64 index);

  // Creates a trace instruction that logs the input operand in the computation.
  static std::unique_ptr<HloInstruction> CreateTrace(const string& tag,
                                                     HloInstruction* operand);

  // Creates a random number generation instruction that fills a shape with
  // random numbers from a given distribution.
  static std::unique_ptr<HloInstruction> CreateRng(
      const Shape& shape, RandomDistribution distribution,
      tensorflow::gtl::ArraySlice<HloInstruction*> parameters);

  // Creates a unary instruction (one operand).
  // Precondition: opcode must be a legitimate unary operation.
  static std::unique_ptr<HloInstruction> CreateUnary(const Shape& shape,
                                                     HloOpcode opcode,
                                                     HloInstruction* operand);

  // Creates a binary instruction (two operands).
  // Precondition: opcode must be a legitimate binary operation.
  static std::unique_ptr<HloInstruction> CreateBinary(const Shape& shape,
                                                      HloOpcode opcode,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs);

  // Creates a ternary instruction (three operands).
  // Precondition: opcode must be a legitimate ternary operation.
  static std::unique_ptr<HloInstruction> CreateTernary(const Shape& shape,
                                                       HloOpcode opcode,
                                                       HloInstruction* lhs,
                                                       HloInstruction* rhs,
                                                       HloInstruction* ehs);

  // Creates a variadic instruction (variable number of operands).
  // Precondition: opcode must be a legitimate variadic operation.
  static std::unique_ptr<HloInstruction> CreateVariadic(
      const Shape& shape, HloOpcode opcode,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands);

  // Creates a map instruction, where the computation (given by the handle) is
  // applied element-wise to every element in operands (across the operands,
  // at a given index) with the same `static_operands`.
  static std::unique_ptr<HloInstruction> CreateMap(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* map_computation,
      tensorflow::gtl::ArraySlice<HloInstruction*> static_operands = {});

  // Creates a convolution op, where rhs is the convolutional filter
  // and window describes how the filter is applied to lhs.
  static std::unique_ptr<HloInstruction> CreateConvolve(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      const Window& window,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Creates a cross replica sum op.
  static std::unique_ptr<HloInstruction> CreateCrossReplicaSum(
      const Shape& shape, HloInstruction* operand);

  // Creates a conversion instruction, where operand is the data to convert and
  // shape is the target shape for the conversion.
  static std::unique_ptr<HloInstruction> CreateConvert(const Shape& shape,
                                                       HloInstruction* operand);

  // Creates an infeed instruction, which reads data of the given shape from the
  // Infeed interface of the device.
  static std::unique_ptr<HloInstruction> CreateInfeed(const Shape& shape,
                                                      const string& config);

  // Creates an outfeed instruction, which outputs data.
  static std::unique_ptr<HloInstruction> CreateOutfeed(
      HloInstruction* operand, tensorflow::StringPiece outfeed_config);

  // Creates a send instruction with the given channel id, which sends the
  // operand data to a unique receive instruction in another computation that
  // has the same channel id.
  static std::unique_ptr<HloInstruction> CreateSend(HloInstruction* operand,
                                                    int64 channel_id);

  // Creates a receive instruction with the given channel id, which receives
  // data of the given shape from a unique send instruction in another
  // computation that has the same channel id.
  static std::unique_ptr<HloInstruction> CreateRecv(const Shape& shape,
                                                    int64 channel_id);

  // Creates a slice instruction, where the operand is sliced by the given
  // start/limit indices.
  static std::unique_ptr<HloInstruction> CreateSlice(
      const Shape& shape, HloInstruction* operand,
      tensorflow::gtl::ArraySlice<int64> start_indices,
      tensorflow::gtl::ArraySlice<int64> limit_indices);

  // Creates a slice instruction, where the first operand is sliced by
  // start indices specified in the second operand, and by size specfied in
  // 'slice_sizes'.
  static std::unique_ptr<HloInstruction> CreateDynamicSlice(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* start_indices,
      tensorflow::gtl::ArraySlice<int64> slice_sizes);

  // Creates a dynamic update slice instruction, which updates a slice
  // of 'operand' with 'update' and 'start_indices'.
  static std::unique_ptr<HloInstruction> CreateDynamicUpdateSlice(
      const Shape& shape, HloInstruction* operand, HloInstruction* update,
      HloInstruction* start_indices);

  // Creates a concatenate instruction, where the operands are concatenated on
  // the provided dimension.
  static std::unique_ptr<HloInstruction> CreateConcatenate(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      int64 dimension);

  // Creates a reduce instruction, where the computation (given by the handle)
  // is applied successively to every element in operand. That is, if f is the
  // function to apply (which either takes 2 [accumulator, value] or 3
  // [accumulator, index, value] arguments) and init is a reduction operator
  // specified initial value (for example, 0 for addition), then this operation
  // will compute:
  //   f(f(init, [index0], value0), [index1], value1), ...)
  static std::unique_ptr<HloInstruction> CreateReduce(
      const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
      HloComputation* reduce_computation);

  // Creates a reduce-window instruction, where the computation (given
  // by the handle) is applied window-wise at each valid window
  // position in the operand.
  static std::unique_ptr<HloInstruction> CreateReduceWindow(
      const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
      const Window& window, HloComputation* reduce_computation);

  // Creates a scatter computation that scatters the `source` array to the
  // selected indices of each window.
  static std::unique_ptr<HloInstruction> CreateSelectAndScatter(
      const Shape& shape, HloInstruction* operand, HloComputation* select,
      const Window& window, HloInstruction* source, HloInstruction* init_value,
      HloComputation* scatter);

  // Creates a broadcast instruction.
  static std::unique_ptr<HloInstruction> CreateBroadcast(
      const Shape& shape, HloInstruction* operand,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions);

  // Creates a pad instruction, where the operand is padded on the edges and
  // between the elements with the given padding value.
  static std::unique_ptr<HloInstruction> CreatePad(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* padding_value, const PaddingConfig& padding_config);

  // Creates a reshape instruction, where the operand is flattened row-major
  // order and then reshaped to the given result shape.
  static std::unique_ptr<HloInstruction> CreateReshape(const Shape& shape,
                                                       HloInstruction* operand);

  // Creates a transpose instruction which permutes the operand dimensions.
  static std::unique_ptr<HloInstruction> CreateTranspose(
      const Shape& shape, HloInstruction* operand,
      tensorflow::gtl::ArraySlice<int64> dimensions);

  // Creates a while instruction, given a condition computation, a body
  // computation, and the initial value for the input of the computations. For
  // example, shape: S32, condition: i -> i < 1000, body: i -> i * 2, init: 1
  // corresponds to the C code below.
  // int32 i = 1; int32 result = while(i < 1000) { i = i * 2 }
  static std::unique_ptr<HloInstruction> CreateWhile(const Shape& shape,
                                                     HloComputation* condition,
                                                     HloComputation* body,
                                                     HloInstruction* init);

  // Creates a fusion instruction. A fusion instruction contains one or more
  // fused instructions forming an expression with a single root
  // "fused_root". Additional instructions can be added to the fusion
  // instruction with the method FuseInstruction.
  static std::unique_ptr<HloInstruction> CreateFusion(
      const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root);

  // Creates a fusion instruction that represents backward convolution. This is
  // similar to CreateFusion, but with extra arguments indicating the window and
  // dimemsion mapping of the backward convolution.
  static std::unique_ptr<HloInstruction> CreateFusionForBackwardConvolution(
      const Shape& shape, FusionKind fusion_kind, const Window& window,
      const ConvolutionDimensionNumbers& conv_dnums,
      HloInstruction* fused_root);

  // Creates a call instruction that applies the given computation on the given
  // operands. "shape" is the resultant shape.
  static std::unique_ptr<HloInstruction> CreateCall(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* computation);

  // Creates a custom call instruction that applies the given custom call target
  // to the given operands. "shape" is the resultant shape.
  static std::unique_ptr<HloInstruction> CreateCustomCall(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      tensorflow::StringPiece custom_call_target);

  // Creates a tuple instruction with the given elements. This is a convenience
  // wrapper around CreateVariadic.
  static std::unique_ptr<HloInstruction> CreateTuple(
      tensorflow::gtl::ArraySlice<HloInstruction*> elements);

  // Creates a reverse instruction, which reverses the order of the elements
  // in the specified dimensions.
  static std::unique_ptr<HloInstruction> CreateReverse(
      const Shape& shape, HloInstruction* operand,
      tensorflow::gtl::ArraySlice<int64> dimensions);

  // Returns the opcode for this instruction.
  HloOpcode opcode() const { return opcode_; }

  // Returns the result shape of this instruction.
  const Shape& shape() const;

  // Returns the (mutable) result shape of this instruction.
  Shape* mutable_shape() { return &shape_; }

  // Returns the ith operand to this instruction.
  const HloInstruction* operand(int64 i) const;

  // Returns the ith operand to this instruction.
  HloInstruction* mutable_operand(int64 i);

  // Returns the number of operands to this instruction.
  int64 operand_count() const { return operands_.size(); }

  // Returns the vector of operands of this instruction.
  const std::vector<HloInstruction*>& operands() const { return operands_; }

  // Returns the index of 'target' in the operands sequence.
  // Precondition: target must be an operand (or a fatal error will occur).
  int64 operand_index(const HloInstruction* target) const;

  // Returns the number of users of this instruction.
  int64 user_count() const { return users_.size(); }

  // Returns the users of this instruction.
  const std::set<HloInstruction*>& users() const { return users_; }

  // Returns the set of control predecessors of this instruction. Control
  // predecessors are the instructions that must be scheduled before the current
  // instruction.
  const std::set<HloInstruction*>& control_predecessors() const {
    return control_predecessors_;
  }

  // Adds the given instruction to the set of control predecessors.
  void AddControlPredecessor(HloInstruction* instruction);

  // Returns the set of control successors of this instruction. Control
  // successors are the instructions that must be scheduled after the current
  // instruction.
  const std::set<HloInstruction*>& control_successors() const {
    return control_successors_;
  }

  // Adds the given instruction to the set of control successors.
  void AddControlSuccessor(HloInstruction* instruction);

  // Returns true if "other" performs the same computation as this instruction.
  // Layout of the instructions' output array is not considered.
  bool Identical(
      const HloInstruction& other,
      std::function<bool(const HloInstruction*, const HloInstruction*)>
          eq_operands = std::equal_to<const HloInstruction*>(),
      std::function<bool(const HloComputation*, const HloComputation*)>
          eq_computations = std::equal_to<const HloComputation*>()) const;

  // Returns whether the instruction has a constant operand.
  bool HasConstantOperand() const;

  // Returns whether this instruction does a rank-2 transposition.
  bool IsRank2Transpose() const;

  // Replaces the use of this instruction in "user" with "new_producer". Note
  // that there might be multiple uses of this instruction in "user"; all will
  // be replaced.
  Status ReplaceUseWith(HloInstruction* user, HloInstruction* new_producer);

  // Replaces the specified operand with new_operand.
  Status ReplaceOperandWith(int64 operand_no, HloInstruction* new_operand);

  // Replaces all uses of this instruction with the new producer. If
  // new_producer is a user of this instruction then new_producer remains a use
  // of this instruction to avoid introducing cycles into the graph.
  Status ReplaceAllUsesWith(HloInstruction* new_producer);

  // Detaches an instruction from its operands. That is, remove the instruction
  // from each operand's user set. This should only be called prior to
  // deallocating the instruction.
  void DetachFromOperands();

  // Performs a postorder DFS visit using this node as the root. If
  // call_finish_visit is true, then DfsHloVisitor::FinishVisit is called when
  // complete.
  Status Accept(DfsHloVisitor* visitor, bool call_finish_visit = true);

  // Performs a postorder DFS visit using this node as the root. Calls the given
  // visitor function at each instruction.
  Status Accept(FunctionVisitor::VisitorFunction visitor_func);

  // Visits all instructions rooted at this instruction using the given visitor
  // in the given order. 'order' must contain at least the set of instructions
  // rooted at this node (ie, those accessible from a DFS traversal from this
  // instruction). Instructions contained in 'order' which are not in the set of
  // instructions rooted at this node are ignored. 'order' must also be a valid
  // topological sort of these instructions (defs appear before uses) though
  // need not be a DFS post-order.
  Status AcceptOrdered(DfsHloVisitor* visitor,
                       const std::vector<const HloInstruction*>& order);

  // Visit this instruction and only this instruction with the given visitor.
  Status Visit(DfsHloVisitor* visitor);

  // Returns the literal associated with this instruction.
  //
  // Note: only constant and parameter opcodes have an associated literal.
  const Literal& literal() const;

  // Returns the parameter number associated with this instruction.
  //
  // Note: only parameter opcodes have an associated parameter number.
  int64 parameter_number() const {
    CHECK_EQ(HloOpcode::kParameter, opcode_);
    return parameter_number_;
  }

  const string& parameter_name() const {
    CHECK_EQ(HloOpcode::kParameter, opcode_);
    return parameter_name_;
  }

  // Returns the dimension sizes or numbers associated with this instruction.
  //
  // Precondition: opcode() is one of: concatenate, reduce, broadcast, reshape,
  // and reverse.
  const std::vector<int64>& dimensions() const;
  int64 dimensions(int64 index) const;

  // Accessor for the dimension in which a concatenate HLO should occur.
  // Precondition: opcode() == HloOpcode::kConcatenate
  int64 concatenate_dimension() const;

  // Returns the tuple index associated with this instruction.
  //
  // Precondition: opcode() == HloOpcode::kGetTupleElement
  int64 tuple_index() const;

  // Gets/sets the to_apply HloComputation for Call, Map, Reduce, etc.
  // The setter should only be called by HloModule or HloComputation methods.
  //
  // Precondition: The instruction has a valid to_apply_ field.
  HloComputation* to_apply() const;
  void set_to_apply(HloComputation* to_apply);

  // Returns the custom_call_target for CustomCall.
  // Precondition: opcode() == HloOpcode::kCustomCall
  const string& custom_call_target() const;

  // Returns the config for the Outfeed instruction.
  // Precondition: opcode() == HloOpcode::kOutfeed
  const string& outfeed_config() const;

  // Gets/sets the while_condition or while_body HloComputation for While. The
  // setters should only be called by HloModule or HloComputation methods.
  //
  // Precondition: The instruction is a While instruction.
  HloComputation* while_condition() const;
  HloComputation* while_body() const;
  void set_while_condition(HloComputation* while_condition);
  void set_while_body(HloComputation* while_body);

  // Gets/sets the select or scatter HloComputation for SelectAndScatter. The
  // setters should only be called by HloModule or HloComputation methods.
  //
  // Precondition: opcode() == HloOpcode::kSelectAndScatter.
  HloComputation* select() const;
  HloComputation* scatter() const;
  void set_select(HloComputation* select);
  void set_scatter(HloComputation* scatter);

  // Returns a string for the signature of this instruction if considered as a
  // function, e.g. the signature of an F32 add is (F32, F32) -> F32.
  string SignatureString() const;

  // Returns a debugging string that represents this instruction.
  string ToString(bool compact_operands = false) const;

  // As ToString, but returns a shorter string.
  string ToShortString() const;

  // Returns a category for the HLO. This could be something like "convolution"
  // or "elementwise".
  string ToCategory() const;

  // Returns the string concatenation of parent name and this instructions name.
  string FullyQualifiedName() const;

  // Returns a logging instruction, if the output of this instruction is logged.
  //
  // Postcondition: retval == nullptr || retval->opcode() == HloOpcode::kTrace
  HloInstruction* tracing() const;
  void set_tracing(HloInstruction* trace_instruction);

  // Returns the channel id associated with the instruction. The id is
  // shared between each Send/Recv pair and is globally unique to identify each
  // channel.
  //
  // Precondition: opcode() == HloOpcode::kSend or HloOpcode::kRecv
  int64 channel_id() const { return channel_id_; }

  // Returns the infeed configuration string. The infeed configuration includes
  // any metadata needed for the backend compiler (e.g., infeed buffer address)
  // and is target-dependent.
  string infeed_config() const { return infeed_config_; }
  void set_infeed_config(const string& config) { infeed_config_ = config; }

  // Returns a tag to be used in tracing.
  //
  // Precondition: opcode() == HloOpcode::kTrace
  const string& tracing_tag() const;

  // Returns whether the instruction is a constant.
  bool IsConstant() const;

  // Returns true if this instruction is fused, ie contained within a fusion
  // instruction.
  bool IsFused() const;

  // Returns true if this instruction can be legally fused into a fusion
  // instruction.
  bool IsFusable() const;

  // Returns the fusion instruction that contains this instruction.
  //
  // Note: only valid if this instruction is fused into a fusion instruction.
  HloInstruction* fusion_instruction() const;

  // Returns the root instruction of the fused expression contained within this
  // fusion instruction.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  HloInstruction* fused_expression_root() const;

  // Returns the vector of fused instructions inside this fusion
  // instruction. The order is a reverse postorder of the fused expression (root
  // is first in the order).
  //
  // Precondition: opcode() == HloOpcode::kFusion
  const std::list<std::unique_ptr<HloInstruction>>& fused_instructions() const;

  // Returns the fused parameter instruction in this fusion instruction
  // corresponding to the given parameter number.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  HloInstruction* fused_parameter(int64 parameter_number) const;

  // Returns the vector of fused parameters inside this fusion instruction.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  const std::vector<HloInstruction*>& fused_parameters() const;

  FusionKind fusion_kind() const {
    CHECK_EQ(HloOpcode::kFusion, opcode_);
    return fusion_kind_;
  }

  // Merges the fused instructions from 'instruction_to_merge' into the
  // fused instruction set of 'this', updating operands as necessary.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  // Predondition: 'instruction_to_merge' must be an operand of 'this'.
  void MergeFusionInstruction(HloInstruction* instruction_to_merge);

  // Fuses the given instruction in this fusion instruction. instruction_to_fuse
  // is cloned and the clone is placed in the fusion
  // instruction. instruction_to_fuse is unchanged. Instruction is cloned rather
  // than moved to cleanly handle the case where the instruction has a use
  // outside the fusion instruction. Moving such an instruction into a fusion
  // instruction would violate the single-result invariant of HLO instructions
  // and significantly complicate code generation.
  //
  // Precondition: this->opcode() == HloOpcode::kFusion
  HloInstruction* FuseInstruction(HloInstruction* instruction_to_fuse);

  // Returns the start index in the given dimension for a slice node.
  //
  // Precondition: opcode() == HloOpcode::kSlice
  int64 slice_starts(int64 dimension) const {
    CHECK_EQ(HloOpcode::kSlice, opcode_);
    return slice_starts_[dimension];
  }
  const std::vector<int64>& slice_starts() const { return slice_starts_; }

  // Returns the (exclusive) limit index in the given dimension for a slice
  // node.
  //
  // Precondition: opcode() == HloOpcode::kSlice
  int64 slice_limits(int64 dimension) const {
    CHECK_EQ(HloOpcode::kSlice, opcode_);
    return slice_limits_[dimension];
  }
  const std::vector<int64>& slice_limits() const {
    CHECK_EQ(HloOpcode::kSlice, opcode_);
    return slice_limits_;
  }

  // Returns the size of the slice in the given dimension for a dynamic
  // slice node.
  //
  // Precondition: opcode() == HloOpcode::kDynamicSlice
  int64 slice_sizes(int64 dimension) const {
    CHECK_EQ(HloOpcode::kDynamicSlice, opcode_);
    return dynamic_slice_sizes_[dimension];
  }
  const std::vector<int64>& dynamic_slice_sizes() const {
    CHECK_EQ(HloOpcode::kDynamicSlice, opcode_);
    return dynamic_slice_sizes_;
  }

  // Returns data on the window in a windowed operation such as
  // convolution.
  const Window& window() const {
    CHECK(window_ != nullptr);
    return *window_;
  }

  // Returns the padding configuration for a pad node.
  //
  // Precondition: opcode() == HloOpcode::kPad
  const PaddingConfig& padding_config() const {
    CHECK(padding_config_ != nullptr);
    return *padding_config_;
  }

  // Returns data on the dimension numbers used for a convolution
  // operation.
  const ConvolutionDimensionNumbers& convolution_dimension_numbers() const {
    CHECK(convolution_dimension_numbers_ != nullptr);
    return *convolution_dimension_numbers_;
  }

  // Returns the random distribution for this rng node.
  //
  // Precondition: opcode() == HloOpcode::kRng
  RandomDistribution random_distribution() const;

  // Clones the HLO instruction. The clone will have the same opcode, shape, and
  // operands. After creation the clone has no uses. "this" (the instruction
  // cloned from) is not changed.
  std::unique_ptr<HloInstruction> Clone();

  // Clones the HLO instruction as above but with new shape and operands.
  std::unique_ptr<HloInstruction> CloneWithNewOperands(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands);

  // Computes and returns the computations this instruction calls (if any). This
  // includes computations called by fused instructions inside of a fusion
  // instruction.
  std::set<HloComputation*> MakeCalledComputationsSet() const;

  // Returns true if this instruction performs an elementwise operation on
  // `operand_idx`-th operand. An instruction is elementwise on an operand iff,
  // after performing necessary implicit broadcast
  // (cs/IrArray::EmitArrayElementAddress), to compute the output at index
  // {i_0,i_1,...,i_n}, the only element required from the operand (if any) is
  // the element at {i_0,i_1,...,i_n}.
  //
  // Note on performance: when this instruction is kFusion, this method, in the
  // worst case, scans all fused instructions. We could speed this up by
  // caching.
  bool IsElementwiseOnOperand(int64 operand_idx) const;

  // Returns true if this instruction is elementwise on all its operands.
  bool IsElementwise() const;

  // Returns whether this instruction may reuse elements of its `i`th operand.
  bool ReusesOperandElements(int64 i) const {
    return OperandElementUse(i) == UseKind::kReuse;
  }

  // Returns the indices that the given operand appear in the operand list of
  // this instruction. Note that an instruction can use the same operand
  // multiple times.
  std::vector<int64> OperandIndices(const HloInstruction* operand) const;

  // Convenience helper for ShapeUtil::InsertedOrDeleted1SizedDimensions. If
  // this reshape merely inserts or deletes 1-sized dimensions, return the input
  // indices of the deleted dimensions and the output indices of the inserted
  // dimensions.
  //
  // Precondition: this op must be a reshape.
  std::tuple<bool, std::vector<int64>, std::vector<int64>>
  ReshapeMerelyInsertsOrDeletes1SizedDimensions() const;

  // Returns a string identifier for this instruction. If no string identifier
  // has been explicitly set, then the identifier is the serialized pointer to
  // this instruction.
  const string& name() const { return name_; }

  // Sets the string identifier for this instruction.
  void set_name(const string& name) { name_ = name; }

  // Set/get the computation containing this instruction. set_parent should only
  // be called by HloComputation methods which add/remove instructions to
  // computations.
  void set_parent(HloComputation* computation) { parent_ = computation; }
  const HloComputation* parent() const { return parent_; }
  HloComputation* parent() { return parent_; }

  // Returns whether we could assign input and output layouts to this
  // instruction to make it a bitcast.
  bool CouldBeBitcast() const;

 private:
  enum class UseKind { kNoUse, kReuse, kUsePermutingElements, kUse };

  // Creates an n-ary elementwise operation.
  static std::unique_ptr<HloInstruction> CreateNary(
      const Shape& shape, HloOpcode opcode,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands);

  // Appends operand to the list of operands and adds this instruction as a user
  // of the operand.
  void AppendOperand(HloInstruction* operand);

  // Adds a user for this instruction.
  void AddUser(HloInstruction* user);

  // Removes a user for this instruction.
  void RemoveUser(HloInstruction* user);

  // Internal constructor for a given opcode/shape, other fields must be filled
  // by factory methods.
  HloInstruction(HloOpcode opcode, const Shape& shape);

  // Clones the given instruction_to_fuse and insert the clone into this fusion
  // instruction.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  HloInstruction* CloneAndFuseInternal(HloInstruction* instruction_to_fuse);

  // Clones a fusion instruction with a new shape and operands.
  std::unique_ptr<HloInstruction> CloneFusionWithNewOperands(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands);

  // Inner DFS traversal function -- this function being called (rather than
  // Accept above) allows us to distinguish the root of the traversal.
  Status AcceptInternal(DfsHloVisitor* visitor);

  // CHECKs various invariants of a fusion instruction.
  void CheckFusionInstruction() const;

  // Returns true if this instruction can legally have the dimensions field
  // set. Used for checking precondition of dimensions field accessors.
  bool CanHaveDimensionsField() const;

  // Returns how this instruction uses elements of its `i`th operand.
  UseKind OperandElementUse(int64 i) const;

  // Result shape of this instruction.
  Shape shape_;

  // Opcode for this instruction.
  HloOpcode opcode_;

  // Literal, only present for kConstant.
  std::unique_ptr<Literal> literal_;

  // Constant index, only present for kGetTupleElement.
  int64 tuple_index_ = 0;

  // Dimensions present for some operations that require reshaping or
  // broadcasting, including Reshape, Reduce, ReduceWindow, and Reverse.
  std::vector<int64> dimensions_;

  // Describes the window in a windowed operation such as convolution.
  std::unique_ptr<Window> window_;

  // Describes the dimension numbers used for a convolution.
  std::unique_ptr<ConvolutionDimensionNumbers> convolution_dimension_numbers_;

  // Describes the [begin, end) index range for a slice.
  std::vector<int64> slice_starts_;
  std::vector<int64> slice_limits_;

  // Describes the [start, start + size) range size for a dynamic slice
  // ('start' is specified dynamically in the second operand of the operation).
  std::vector<int64> dynamic_slice_sizes_;

  // The padding configuration that describes the edge padding and interior
  // padding of this pad instruction. Only set for pad instructions.
  std::unique_ptr<PaddingConfig> padding_config_;

  // The set of instruction fused into this fusion instruction. Only set for
  // fusion instructions.
  std::list<std::unique_ptr<HloInstruction>> fused_instructions_;

  // If this instruction is fused into a fusion instruction, this field points
  // to the fusion instruction.
  HloInstruction* parent_fusion_instruction_ = nullptr;

  // The vector of parameter instructions inside this fusion instruction.  The
  // index of the vector is the parameter_number of the parameter instruction.
  // This vector is non-empty only for fusion instructions.
  std::vector<HloInstruction*> fused_parameters_;

  // The root of the expression fused into this fusion instruction.
  HloInstruction* fused_root_ = nullptr;

  // The type of the fusion. Used by kFusion only.
  FusionKind fusion_kind_;

  // For parameter instructions this field holds the parameter number.
  int64 parameter_number_ = 0;
  string parameter_name_;

  // Computation to apply, only present for kCall, kMap, kReduce and
  // kReduceWindow.
  HloComputation* to_apply_ = nullptr;

  // Name of a global symbol to call, only present for kCustomCall.
  string custom_call_target_;

  // Computation for condition and body of kWhile, only present for kWhile.
  HloComputation* condition_ = nullptr;
  HloComputation* body_ = nullptr;

  // Computation for select and scatter, only present for
  // kSelectAndScatter.
  HloComputation* select_ = nullptr;
  HloComputation* scatter_ = nullptr;

  // Outfeed configuration information, only present for kOutfeed.
  string outfeed_config_;

  // Instruction operands.
  std::vector<HloInstruction*> operands_;

  // The users of this instruction. Users are HLOs where this instruction is an
  // operand.
  std::set<HloInstruction*> users_;

  // The set of control predecessors of this instruction.
  std::set<HloInstruction*> control_predecessors_;

  // The set of control successors of this instruction.
  std::set<HloInstruction*> control_successors_;

  // A trace instruction that consumes this instruction.
  //
  // Invariant: if trace_instruction_ != nullptr, trace_instruction has this as
  // an operand.
  HloInstruction* trace_instruction_ = nullptr;

  // The distribution requested for random number generation.
  // Only present for kRng.
  RandomDistribution distribution_;

  // Represents a unique identifier for each Send/Recv instruction pair.
  // Only present for kSend or kRecv.
  int64 channel_id_ = -1;

  // The string representation of the infeed configuration.
  string infeed_config_;

  // String identifier for instruction.
  string name_;

  // The computation in which this instruction is contained.
  HloComputation* parent_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(HloInstruction);
};

string FusionKindString(HloInstruction::FusionKind kind);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_H_
