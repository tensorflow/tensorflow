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
#include <iosfwd>
#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloComputation;
class HloModule;

// HLO instructions are the IR used by the high-level compiler.
class HloInstruction {
 public:
  enum class FusionKind {
    kLoop,                // Fused into a loop.
    kInput,               // Op's input is fused into the op itself.
    kOutput,              // Op's output is fused into the op itself.
                          // REQUIRES: At least one operand buffer must be able
                          // to alias the output buffer.
    kTransposeDot,        // Fused into a dot with transposed operands.
    kConvBackwardFilter,  // Fused into a backward filter convolution.
    kConvBackwardInput,   // Fused into a backward input convolution.

    kCustom,  // Custom category for backend-specific fusions that
              // do not match any of the more specific ones.
  };

  ~HloInstruction();

  // Creates an instruction from the given proto. Arguments:
  //
  //   module: the module which will contain the instruction. The newly created
  //     instruction is *not* added to the module or any computation, however.
  //   proto: the proto to convert from.
  //   instruction_map: a map from instruction name to HloInstruction*. This map
  //     must contain all operands of the newly constructed instruction.
  //   computation_map: a map from computation name to HloComputation*. This map
  //     must contain all computations which the newly constructed instruction
  //     calls. If the instruction is a fusion instruction, then the fusion
  //     computation is added to this map and the module.
  static StatusOr<std::unique_ptr<HloInstruction>> CreateFromProto(
      HloModule* module, const HloInstructionProto& proto,
      const tensorflow::gtl::FlatMap<string, HloInstruction*>& instruction_map,
      tensorflow::gtl::FlatMap<string, HloComputation*>* computation_map);

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

  // Creates a reduce-precision op, where operand is the data to reduce in
  // precision, and exponent_bits and mantissa_bits describe the precision to
  // reduce it to.
  static std::unique_ptr<HloInstruction> CreateReducePrecision(
      const Shape& shape, HloInstruction* operand, const int exponent_bits,
      const int mantissa_bits);

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
      const Shape& shape, HloInstruction* operand,
      tensorflow::StringPiece outfeed_config);

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
      tensorflow::gtl::ArraySlice<int64> limit_indices,
      tensorflow::gtl::ArraySlice<int64> strides);

  // Creates a slice instruction, where the first operand is sliced by
  // start indices specified in the second operand, and by size specified in
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

  // Creates a batch-norm-training instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormTraining(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, float epsilon, int64 feature_index);

  // Creates a batch-norm-inference instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormInference(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
      float epsilon, int64 feature_index);

  // Creates a batch-norm-grad instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormGrad(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* mean, HloInstruction* variance,
      HloInstruction* grad_output, float epsilon, int64 feature_index);

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

  // Returns true if this instruction has a side effect. An instruction has a
  // side effect if it uses certain opcodes or calls a computation with a side
  // effect.
  bool HasSideEffect() const;

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
  using InstructionVector = tensorflow::gtl::InlinedVector<HloInstruction*, 2>;
  const InstructionVector& operands() const { return operands_; }

  // Returns the index of 'target' in the operands sequence.
  // Precondition: target must be an operand (or a fatal error will occur).
  int64 operand_index(const HloInstruction* target) const;

  // Returns the number of users of this instruction.
  int64 user_count() const { return users_.size(); }

  // Returns the users of this instruction.
  const std::vector<HloInstruction*>& users() const { return users_; }

  // Returns true if this instruction is a user of 'instruction'.
  bool IsUserOf(const HloInstruction* instruction) const {
    return ContainsKey(instruction->user_set_, this);
  }

  // Adds a control dependency from this instruction to the given
  // instruction. This instruction becomes a control predecessor of
  // 'instruction', and 'instruction' becomes a control successor of this
  // instruction. Returns an error status if either of the given instructions
  // does not belong to the same computation.
  //
  // This is used to enforce an additional ordering requirement that is not
  // captured by normal data dependencies, such as ordering among Send or Recv
  // operations to avoid deadlock.
  Status AddControlDependencyTo(HloInstruction* instruction);

  // Removes a previously added control dependency from this instruction to
  // 'instruction'.
  Status RemoveControlDependencyTo(HloInstruction* instruction);

  // Returns the set of control predecessors (successors) of this
  // instruction. Control predecessors (sucessors) must execute before (after)
  // the current instruction.
  const std::vector<HloInstruction*>& control_predecessors() const {
    return control_predecessors_;
  }
  const std::vector<HloInstruction*>& control_successors() const {
    return control_successors_;
  }

  // Returns true if "other" performs the same computation as this instruction.
  // Layout of the instructions' output array is not considered.
  bool Identical(
      const HloInstruction& other,
      std::function<bool(const HloInstruction*, const HloInstruction*)>
          eq_operands = std::equal_to<const HloInstruction*>(),
      std::function<bool(const HloComputation*, const HloComputation*)>
          eq_computations = std::equal_to<const HloComputation*>()) const {
    // An instruction is always identical to itself.
    if (this == &other) {
      return true;
    }

    // Identical instruction must have the same opcode and identical operands.
    // In general, there is no need to check shape because shape is inferred
    // from the shape of the operands.
    if (opcode() != other.opcode() ||
        !ContainersEqual(operands(), other.operands(),
                         std::move(eq_operands))) {
      return false;
    }

    return IdenticalSlowPath(other, eq_computations);
  }

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
  //
  // If this instruction is the root of its computation, sets the computation's
  // root to new_producer.
  Status ReplaceAllUsesWith(HloInstruction* new_producer);

  // Detaches an instruction from its operands. That is, remove the instruction
  // from each operand's user set. This should only be called prior to
  // deallocating the instruction.
  void DetachFromOperands();

  // Performs a postorder DFS visit using this node as the root. If
  // call_finish_visit is true, then DfsHloVisitor::FinishVisit is called when
  // complete. If ignore_control_predecessors is true, instructions only
  // reachable via control dependencies will not be visited, and the postorder
  // will not take control dependencies into account. It is as if the control
  // dependencies didn't exist in the graph at all.
  template <typename HloInstructionPtr>
  Status Accept(DfsHloVisitorBase<HloInstructionPtr>* visitor,
                bool call_finish_visit = true,
                bool ignore_control_predecessors = false);
  Status Accept(ConstDfsHloVisitor* visitor, bool call_finish_visit = true,
                bool ignore_control_predecessors = false) const {
    return const_cast<HloInstruction*>(this)->Accept(
        visitor, call_finish_visit, ignore_control_predecessors);
  }

  // Same as Accept() above, but the order of operand and control predecessor
  // visitation is determined by the given operand order; if compare(A, B) ==
  // true, A is visited before B.
  using CompareFunction =
      std::function<bool(const HloInstruction*, const HloInstruction*)>;
  Status AcceptWithOperandOrder(DfsHloVisitor* visitor,
                                const CompareFunction& operand_order,
                                bool call_finish_visit = true);

  // Performs a postorder DFS visit using this node as the root. Calls the given
  // visitor function at each instruction.
  Status Accept(const std::function<Status(HloInstruction*)>& visitor_func);
  Status Accept(
      const std::function<Status(const HloInstruction*)>& visitor_func) const;

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
  template <typename HloInstructionPtr>
  Status Visit(DfsHloVisitorBase<HloInstructionPtr>* visitor);

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

  void set_parameter_name(const string& str) {
    CHECK_EQ(HloOpcode::kParameter, opcode_);
    parameter_name_ = str;
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

  // Returns the first non-GetTupleElement ancestor instruction of 'hlo'.
  // If the first non-GTE ancestor is tuple-shaped, populates 'index' with the
  // (possibly nested) tuple indices used on the path from ancestor to 'hlo'.
  std::pair<const HloInstruction*, ShapeIndex> LatestNonGteAncestorAndIndex()
      const;

  std::pair<HloInstruction*, ShapeIndex> LatestNonGteAncestorAndIndex() {
    auto rv =
        const_cast<const HloInstruction*>(this)->LatestNonGteAncestorAndIndex();
    return {const_cast<HloInstruction*>(rv.first), rv.second};
  }

  // Same as LatestNonGteAncestorAndIndex, but just returns the HloInstruction.
  const HloInstruction* LatestNonGteAncestor() const;

  HloInstruction* LatestNonGteAncestor() {
    return const_cast<HloInstruction*>(
        const_cast<const HloInstruction*>(this)->LatestNonGteAncestor());
  }

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

  // Returns the shape for the Outfeed instruction.
  // Precondition: opcode() == HloOpcode::kOutfeed
  const Shape& outfeed_shape() const;

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
  string ToString(bool compact_operands = false, bool include_metadata = true,
                  bool include_large_constants = false) const;

  // Components of the ToString() representation:

  // Returns a string representation of the operand list.
  string OperandsToString(bool compact, bool include_large_constants) const;

  // Returns string representation of op-specific attributes.
  std::vector<string> ExtraAttributesToString() const;

  string ToStringNoMetadata() const { return ToString(false, false); }

  // As ToString, but returns a shorter string.
  string ToShortString() const;

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const;

  // Returns a category for the HLO. This could be something like "convolution"
  // or "elementwise".
  string ToCategory() const;

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

  // Returns feature_index field associated with the instruction. The index
  // represents the index of the feature dimension.
  //
  // Precondition: opcode() == HloOpcode::kBatchNormTraining
  int64 feature_index() const { return feature_index_; }

  // Returns a epsilon value associated with the instruction. The is a small
  // number added to the variance to avoid divide-by-zero error.
  //
  // Precondition: opcode() == HloOpcode::kBatchNormTraining
  float epsilon() const { return epsilon_; }

  // Returns the infeed configuration string. The infeed configuration includes
  // any metadata needed for the backend compiler (e.g., infeed buffer address)
  // and is target-dependent.
  string infeed_config() const { return infeed_config_; }
  void set_infeed_config(const string& config) { infeed_config_ = config; }

  // Returns a tag to be used in tracing.
  //
  // Precondition: opcode() == HloOpcode::kTrace
  string TracingTag() const;

  // Returns whether the instruction is a constant.
  bool IsConstant() const;

  // Returns true if this instruction is fused, ie contained within a fusion
  // instruction.
  bool IsFused() const;

  // Returns the computation for this fused instruction.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  HloComputation* fused_instructions_computation() const;

  // Returns true if this instruction can be legally fused into a fusion
  // instruction.
  bool IsFusable() const;

  // Returns the root instruction of the fused expression contained within this
  // fusion instruction.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  HloInstruction* fused_expression_root() const;

  // Returns the list of fused instructions inside this fusion instruction.  The
  // returned type is a range of HloInstruction*s.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  const tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
  fused_instructions() const;

  const tensorflow::gtl::iterator_range<
      UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
  fused_instructions();

  // Gets the number of instructions inside this fusion instruction.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  int64 fused_instruction_count() const;

  // Returns the fused parameter instruction in this fusion instruction
  // corresponding to the given parameter number.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  HloInstruction* fused_parameter(int64 parameter_number) const;

  // Returns the vector of fused parameters inside this fusion instruction.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  const std::vector<HloInstruction*>& fused_parameters() const;

  // Returns true if this instruction is a fusion instruction that generates
  // multiple outputs.
  const bool IsMultiOutputFusion() const {
    return (opcode() == HloOpcode::kFusion &&
            fused_expression_root()->opcode() == HloOpcode::kTuple);
  }

  FusionKind fusion_kind() const {
    CHECK_EQ(HloOpcode::kFusion, opcode_);
    return fusion_kind_;
  }

  void set_fusion_kind(FusionKind kind) {
    CHECK_EQ(HloOpcode::kFusion, opcode_);
    fusion_kind_ = kind;
  }

  // Returns the sharding applied to this operator.
  // REQUIRES: has_sharding() is true.
  const HloSharding& sharding() const {
    CHECK(has_sharding());
    return *sharding_;
  }
  // Returns the sharding applied to this operator, or default_ if none exists.
  const HloSharding& sharding_or_default(const HloSharding& default_) const {
    return sharding_ ? *sharding_ : default_;
  }
  // Sets the sharding of this operator. Should only be called by HloModule or
  // HloComputation methods.
  void set_sharding(const HloSharding& sharding) {
    sharding_ = MakeUnique<HloSharding>(sharding);
  }
  // Remove any sharding from this operator.
  void clear_sharding() { sharding_ = nullptr; }
  // Return true if this operator has a sharding assigned.
  bool has_sharding() const { return sharding_ != nullptr; }

  // Merges the fused instructions from 'instruction_to_merge' into the
  // fused instruction set of 'this', updating operands as necessary.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  // Predondition: 'instruction_to_merge' must be an operand of 'this'.
  void MergeFusionInstruction(HloInstruction* instruction_to_merge);

  // Merges the fused instructions from instruction_to_merge into the fused
  // instruction set of 'this' and generates multioutput fusion instructions.
  // All the users of instruction_to_merge will be redirected to 'this'
  // instruction. instruction_to_merge will be removed from its parent
  // computation.
  //
  // Precondition: opcode() == HloOpcode::kFusion
  void MergeFusionInstructionIntoMultiOutput(
      HloInstruction* instruction_to_merge);

  // Fuses the given instruction in this fusion instruction. instruction_to_fuse
  // is cloned and the clone is placed in the fusion
  // instruction. instruction_to_fuse is unchanged. Instruction is cloned rather
  // than moved to cleanly handle the case where the instruction has a use
  // outside the fusion instruction. Moving such an instruction into a fusion
  // instruction would violate the single-result invariant of HLO instructions
  // and significantly complicate code generation.
  //
  // Precondition: this->opcode() == HloOpcode::kFusion
  HloInstruction* FuseInstruction(HloInstruction* instruction_to_fuse) {
    return FuseInstructionInternal(instruction_to_fuse);
  }

  // Fuses the given instruction in this fusion instruction and generate
  // multioutput fusion instruction. A clone of the instruction_to_fuse will
  // be part of the output of fusion instructions. The users of
  // instruction_to_fuse will be redirected to this fusion instructions.
  // instruction_to_fuse will be removed from its parent computation.
  //
  // Precondition: this->opcode() == HloOpcode::kFusion
  HloInstruction* FuseInstructionIntoMultiOutput(
      HloInstruction* instruction_to_fuse) {
    return FuseInstructionInternal(instruction_to_fuse, /* add_output */ true);
  }

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

  // Returns the stride in the given dimension for a slice node.
  //
  // Precondition: opcode() == HloOpcode::kSlice
  int64 slice_strides(int64 dimension) const {
    CHECK_EQ(HloOpcode::kSlice, opcode_);
    return slice_strides_[dimension];
  }
  const std::vector<int64>& slice_strides() const { return slice_strides_; }

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

  // Returns the number of exponent bits for a reduce-precision node.
  //
  // Precondition: opcode() == HloOpcode::kReducePrecision
  int32 exponent_bits() const {
    CHECK_EQ(HloOpcode::kReducePrecision, opcode_);
    return exponent_bits_;
  }

  // Returns the number of mantissa bits for a reduce-precision node.
  //
  // Precondition: opcode() == HloOpcode::kReducePrecision
  int32 mantissa_bits() const {
    CHECK_EQ(HloOpcode::kReducePrecision, opcode_);
    return mantissa_bits_;
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

  // Returns the dump string of the convolution dimension numbers.
  string ConvolutionDimensionNumbersToString() const;

  // Returns the random distribution for this rng node.
  //
  // Precondition: opcode() == HloOpcode::kRng
  RandomDistribution random_distribution() const;

  // Clones the HLO instruction. The clone will have the same opcode, shape, and
  // operands. After creation the clone has no uses. "this" (the instruction
  // cloned from) is not changed. Suffix is the string to append to the name of
  // the instruction to form the name of the cloned instruction.
  // If the module pointer is not nullptr, it will be the module where
  // the cloned computations will be added to (in order to support deep
  // cloning).
  std::unique_ptr<HloInstruction> Clone(const string& suffix = "clone",
                                        HloModule* module = nullptr) const;

  // Clones the HLO instruction as above but with new shape and operands.
  // If the module pointer is not nullptr, it will be the module where
  // the cloned computations will be added to (in order to support deep
  // cloning).
  std::unique_ptr<HloInstruction> CloneWithNewOperands(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloModule* module = nullptr) const;

  // Returns the computations this instruction directly calls (if any).
  const std::vector<HloComputation*>& called_computations() const {
    return called_computations_;
  }

  // Replaces all called computations based on a map function. This is needed
  // when we clone hlo_computations and want to let the instructions to point
  // to the newly cloned nodes.
  void ReplaceCalledComputations(
      std::function<HloComputation*(HloComputation*)> map_function) {
    for (int64 i = 0; i < called_computations_.size(); ++i) {
      called_computations_[i] = map_function(called_computations_[i]);
    }
  }

  // Clears out the called computations.
  //
  // This is, in particular, necessary when inlining function bodies into their
  // caller. If there were side-effecting operations in the called computations,
  // the call itself is considered side-effecting and thus cannot be removed. By
  // clearing out the computations, we reflect the fact that all side-effecting
  // properties have been reflected in the caller, and make the call HLO
  // removable.
  void ClearCalledComputations() { called_computations_.clear(); }

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

  // Returns true if this elementwise instruction implicitly broadcasts operand
  // `operand_idx`.
  //
  // Precondition: this instruction should be an elementwise operation.
  bool ImplicitlyBroadcastsOperand(int64 operand_idx) const;

  // Returns true if this instruction is binary and elementwise.
  bool IsElementwiseBinary() const;

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

  // Returns the opcode string for this instruction. This is the result from
  // HloOpcodeString plus, for fusion nodes, the fusion kind, separated by a
  // ':'.
  string ExtendedOpcodeStr() const;

  // Returns a string identifier for this instruction. If no string identifier
  // has been explicitly set, then the identifier is the serialized pointer to
  // this instruction.
  const string& name() const { return name_; }

  // Use the given NameUniquer to select a unique name for the instruction based
  // on the instruction's existing name.
  void UniquifyName(NameUniquer* name_uniquer);

  // Set the unique id for this instruction to "id"
  void SetUniqueId(int id) {
    CHECK_EQ(unique_id_, -1);  // Should not be assigned already
    CHECK_GE(id, 0);
    unique_id_ = id;
  }

  // Return the unique ID assigned to this node via SetUniqueId (or -1
  // if no id has been assigned yet).
  int unique_id() const { return unique_id_; }

  // Sets the debug metadata for this instruction.
  void set_metadata(const OpMetadata& metadata) { metadata_ = metadata; }
  const OpMetadata& metadata() const { return metadata_; }

  // Set/get the computation containing this instruction. set_parent should only
  // be called by HloComputation methods which add/remove instructions to
  // computations.
  void set_parent(HloComputation* computation) { parent_ = computation; }
  const HloComputation* parent() const { return parent_; }
  HloComputation* parent() { return parent_; }

  // Returns the module for this instruction.
  HloModule* GetModule() const;

  // Returns whether we could assign input and output layouts to this
  // instruction to make it a bitcast.
  bool CouldBeBitcast() const;

  // Get/Set the number of partitions per outer dimension (in order, starting
  // with outer-most dimension first). Currently used by the parallel cpu
  // backend to partition HLOs into parallel tasks.
  // TODO(b/62783254) Replace these methods with a more general way to
  // annotate HLOs with backend-specific information.
  const std::vector<int64>& outer_dimension_partitions() const {
    return outer_dimension_partitions_;
  }
  void set_outer_dimension_partitions(
      const std::vector<int64>& outer_dimension_partitions);

  // Change the layout for an Constant Hlo instruction to match new_layout.  For
  // tuple shaped constants shape_index is the path to the internal array
  // subshape whose layout needs to be changed.
  void RelayoutConstant(const Layout& new_layout,
                        const ShapeIndex& shape_index = {});

 private:
  enum class UseKind { kNoUse, kReuse, kUsePermutingElements, kUse };

  // Helper class for computing OperandElementUse for kFusion.
  class FusionReusesParamElements;

  // See comments on Identical().
  bool IdenticalSlowPath(
      const HloInstruction& other,
      std::function<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const;

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

  // Fuses the given instruction into this fusion instruction. When add_output
  // is false (which is the default), instruction_to_fuse is cloned and the
  // clone is placed in the fusion instruction. instruction_to_fuse is
  // unchanged.
  //
  // When add_output is true, a clone of the instruction_to_fuse will be part
  // of the output of fusion instructions. The users of instruction_to_fuse
  // will be redirected to this fusion instructions. instruction_to_fuse will
  // be removed from its parent computation.
  //
  // Precondition: this->opcode() == HloOpcode::kFusion
  HloInstruction* FuseInstructionInternal(HloInstruction* instruction_to_fuse,
                                          bool add_output = false);

  // Clones the given instruction_to_fuse and insert the clone into this fusion
  // instruction. If add_output is true, a clone of instruction_to_fuse will
  // be in the output of the this fusion instruction (part of the tuple of the
  // fusion root).
  //
  // Precondition: opcode() == HloOpcode::kFusion
  HloInstruction* CloneAndFuseInternal(HloInstruction* instruction_to_fuse,
                                       bool add_output = false);

  // Clones a fusion instruction with a new shape and operands.
  std::unique_ptr<HloInstruction> CloneFusionWithNewOperands(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloModule* module = nullptr) const;

  // Returns true if this instruction can legally have the dimensions field
  // set. Used for checking precondition of dimensions field accessors.
  bool CanHaveDimensionsField() const;

  // Returns how this instruction uses elements of its `i`th operand.
  UseKind OperandElementUse(int64 i) const;

  int unique_id_;  // Unique to this HloInstruction within a HloModule

  // Opcode for this instruction.
  HloOpcode opcode_;

  // Instruction operands.
  InstructionVector operands_;

  // The set of control predecessors of this instruction.
  std::vector<HloInstruction*> control_predecessors_;

  // The users of this instruction. Users are HLOs where this instruction is an
  // operand. The vector users_ and the set user_set_ contain identical
  // members. The set enables fast membership testing and the vector enables
  // fast, stable iteration.
  std::vector<HloInstruction*> users_;
  std::unordered_set<const HloInstruction*> user_set_;

  // The set of control successors of this instruction.
  std::vector<HloInstruction*> control_successors_;

  // The computation in which this instruction is contained.
  HloComputation* parent_ = nullptr;

  // Shape of outfeed request.
  Shape outfeed_shape_;

  // Result shape of this instruction.
  Shape shape_;

  // Literal, only present for kConstant.
  std::unique_ptr<Literal> literal_;

  // Constant index, only present for kGetTupleElement.
  int64 tuple_index_ = -1;

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
  std::vector<int64> slice_strides_;

  // The bit sizes for a reduce-precision operation.
  int32 exponent_bits_ = 0;
  int32 mantissa_bits_ = 0;

  // Describes the [start, start + size) range size for a dynamic slice
  // ('start' is specified dynamically in the second operand of the operation).
  std::vector<int64> dynamic_slice_sizes_;

  // The padding configuration that describes the edge padding and interior
  // padding of this pad instruction. Only set for pad instructions.
  std::unique_ptr<PaddingConfig> padding_config_;

  // The type of the fusion. Used by kFusion only.
  FusionKind fusion_kind_;

  // The sharding, if one exists.
  std::unique_ptr<HloSharding> sharding_;

  // For parameter instructions this field holds the parameter number.
  int64 parameter_number_ = 0;
  string parameter_name_;

  // Name of a global symbol to call, only present for kCustomCall.
  string custom_call_target_;

  // Computations called by this instruction.
  std::vector<HloComputation*> called_computations_;

  // Indices of computations in called_computations_ for instructions which call
  // multiple computations.
  enum {
    // kWhile computations.
    kBodyComputationIndex = 0,
    kConditionComputationIndex = 1,

    // kSelectAndScatter computations.
    kSelectComputationIndex = 0,
    kScatterComputationIndex = 1,
  };

  // Outfeed configuration information, only present for kOutfeed.
  string outfeed_config_;

  // A trace instruction that consumes this instruction.
  //
  // Invariant: if trace_instruction_ != nullptr, trace_instruction has this as
  // an operand.
  HloInstruction* trace_instruction_ = nullptr;

  // The distribution requested for random number generation.
  // Only present for kRng.
  RandomDistribution distribution_;

  // A small float number added to the variance to avoid divide-by-zero error.
  // Only present for kBatchNormTraining.
  float epsilon_ = 0.0f;

  // An integer value representing the index of the feature dimension.
  // Only present for kBatchNormTraining.
  int64 feature_index_ = -1;

  // Represents a unique identifier for each Send/Recv instruction pair.
  // Only present for kSend or kRecv.
  int64 channel_id_ = -1;

  // The string representation of the infeed configuration.
  string infeed_config_;

  // String identifier for instruction.
  string name_;

  // Metadata for debugging.
  OpMetadata metadata_;

  // The number of partitions per outer dimension (listed in order from
  // outer-most dimension first).
  std::vector<int64> outer_dimension_partitions_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloInstruction);
};

string ToString(HloInstruction::FusionKind kind);
StatusOr<HloInstruction::FusionKind> StringToFusionKind(
    const string& kind_name);

std::ostream& operator<<(std::ostream& os, HloInstruction::FusionKind kind);

// Map classes that guarantee a deterministic iteration order when the key is
// an HloInstruction* or a const HloInstruction*.
// To make the iteration order over the map deterministic, the comparator
// should not be using the pointer values, but rather an intrinsic property of
// the hlo.
struct HloPtrComparator {
  bool operator()(const HloInstruction* const& lhs,
                  const HloInstruction* const& rhs) const {
    return lhs->unique_id() < rhs->unique_id();
  }
};

template <typename ValueT>
using HloInstructionMap = std::map<HloInstruction*, ValueT, HloPtrComparator>;

template <typename ValueT>
using ConstHloInstructionMap =
    std::map<const HloInstruction*, ValueT, HloPtrComparator>;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_H_
