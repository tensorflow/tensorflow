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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_H_

#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloComputation;
class HloInstruction;

// A postorder depth-first HloInstruction visitor. When Handle* is called on an
// instruction, all its operands were already visited. User code can subclass
// this to iterate over an HloInstruction DAG. The Handle* routines have
// operands / data unpacked for ease of use in the visitor subclass.
//
// No instruction will ever be visited twice; however, the root instruction will
// be reported again when the traversal is done via a call to FinishVisit.
//
// A subclass must override at least
// (either HandleElementwiseUnary or all the Handle methods for unary ops) and
// (either HandleElementwiseBinary or all the Handle methods for binary ops)).
// The default Handle methods for (unary, binary) ops call
// (HandleElementwiseUnary, HandleElementwiseBinary).
// The default (HandleElementwiseUnary, HandleElementwiseBinary) return an
// "unimplemented" error status.
//
// Note: this may change to an iterator in the future for flexibility purposes.
//
// TODO(b/26548304): Stop passing in information about the visited
// instruction that is accessible from the instruction object itself.
class DfsHloVisitor {
 public:
  DfsHloVisitor() {}
  virtual ~DfsHloVisitor() {}

  // These routines are self-descriptive, see class comment for usage
  // information.

  virtual Status HandleElementwiseUnary(HloInstruction* hlo, HloOpcode opcode);
  virtual Status HandleElementwiseBinary(HloInstruction* hlo, HloOpcode opcode);
  virtual Status HandleClamp(HloInstruction* clamp, HloInstruction* min,
                             HloInstruction* arg, HloInstruction* max) = 0;
  virtual Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                              HloInstruction* on_true,
                              HloInstruction* on_false) = 0;
  virtual Status HandleMaximum(HloInstruction* maximum) {
    return HandleElementwiseBinary(maximum, HloOpcode::kMaximum);
  }
  virtual Status HandleMinimum(HloInstruction* minimum) {
    return HandleElementwiseBinary(minimum, HloOpcode::kMinimum);
  }
  virtual Status HandleConcatenate(
      HloInstruction* concatenate,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) = 0;
  virtual Status HandleConvert(HloInstruction* convert) {
    return HandleElementwiseUnary(convert, HloOpcode::kConvert);
  }
  virtual Status HandleCopy(HloInstruction* copy) {
    return HandleElementwiseUnary(copy, HloOpcode::kCopy);
  }
  virtual Status HandleMultiply(HloInstruction* multiply, HloInstruction* lhs,
                                HloInstruction* rhs) {
    return HandleElementwiseBinary(multiply, HloOpcode::kMultiply);
  }
  virtual Status HandleDot(HloInstruction* dot, HloInstruction* lhs,
                           HloInstruction* rhs) = 0;
  virtual Status HandlePower(HloInstruction* power, HloInstruction* lhs,
                             HloInstruction* rhs) {
    return HandleElementwiseBinary(power, HloOpcode::kPower);
  }
  virtual Status HandleConvolution(HloInstruction* convolution,
                                   HloInstruction* lhs, HloInstruction* rhs,
                                   const Window& window) = 0;
  virtual Status HandleCrossReplicaSum(HloInstruction* crs) = 0;
  virtual Status HandleCompare(HloInstruction* compare, HloOpcode opcode,
                               HloInstruction* lhs, HloInstruction* rhs) {
    return HandleElementwiseBinary(compare, opcode);
  }
  virtual Status HandleAdd(HloInstruction* add, HloInstruction* lhs,
                           HloInstruction* rhs) {
    return HandleElementwiseBinary(add, HloOpcode::kAdd);
  }
  virtual Status HandleDivide(HloInstruction* divide, HloInstruction* lhs,
                              HloInstruction* rhs) {
    return HandleElementwiseBinary(divide, HloOpcode::kDivide);
  }
  virtual Status HandleRemainder(HloInstruction* remainder, HloInstruction* lhs,
                                 HloInstruction* rhs) {
    return HandleElementwiseBinary(remainder, HloOpcode::kRemainder);
  }
  virtual Status HandleSubtract(HloInstruction* subtract, HloInstruction* lhs,
                                HloInstruction* rhs) {
    return HandleElementwiseBinary(subtract, HloOpcode::kSubtract);
  }
  virtual Status HandleAbs(HloInstruction* abs, HloInstruction* operand) {
    return HandleElementwiseUnary(abs, HloOpcode::kAbs);
  }
  virtual Status HandleSign(HloInstruction* sign, HloInstruction* operand) {
    return HandleElementwiseUnary(sign, HloOpcode::kSign);
  }
  virtual Status HandleNegate(HloInstruction* negate, HloInstruction* operand) {
    return HandleElementwiseUnary(negate, HloOpcode::kNegate);
  }
  virtual Status HandleExp(HloInstruction* exp, HloInstruction* operand) {
    return HandleElementwiseUnary(exp, HloOpcode::kExp);
  }
  virtual Status HandleFloor(HloInstruction* floor, HloInstruction* operand) {
    return HandleElementwiseUnary(floor, HloOpcode::kFloor);
  }
  virtual Status HandleCeil(HloInstruction* ceil, HloInstruction* operand) {
    return HandleElementwiseUnary(ceil, HloOpcode::kCeil);
  }
  virtual Status HandleLog(HloInstruction* log, HloInstruction* operand) {
    return HandleElementwiseUnary(log, HloOpcode::kLog);
  }
  virtual Status HandleCos(HloInstruction* cos, HloInstruction* operand) {
    return HandleElementwiseUnary(cos, HloOpcode::kCos);
  }
  virtual Status HandleSin(HloInstruction* sin, HloInstruction* operand) {
    return HandleElementwiseUnary(sin, HloOpcode::kSin);
  }
  virtual Status HandleTanh(HloInstruction* tanh, HloInstruction* operand) {
    return HandleElementwiseUnary(tanh, HloOpcode::kTanh);
  }
  virtual Status HandleIsFinite(HloInstruction* is_finite,
                                HloInstruction* operand) {
    return HandleElementwiseUnary(is_finite, HloOpcode::kIsFinite);
  }
  virtual Status HandleLogicalAnd(HloInstruction* logical_and,
                                  HloInstruction* lhs, HloInstruction* rhs) {
    return HandleElementwiseBinary(logical_and, HloOpcode::kLogicalAnd);
  }
  virtual Status HandleLogicalNot(HloInstruction* logical_not,
                                  HloInstruction* operand) {
    return HandleElementwiseUnary(logical_not, HloOpcode::kLogicalNot);
  }
  virtual Status HandleLogicalOr(HloInstruction* logical_or,
                                 HloInstruction* lhs, HloInstruction* rhs) {
    return HandleElementwiseBinary(logical_or, HloOpcode::kLogicalOr);
  }
  virtual Status HandleReducePrecision(HloInstruction* reduce_precision) {
    return HandleElementwiseUnary(reduce_precision,
                                  HloOpcode::kReducePrecision);
  }

  virtual Status HandleInfeed(HloInstruction* infeed) = 0;
  virtual Status HandleOutfeed(HloInstruction* outfeed) = 0;
  virtual Status HandleRng(HloInstruction* random,
                           RandomDistribution distribution) = 0;
  virtual Status HandleReverse(HloInstruction* reverse,
                               HloInstruction* operand) = 0;
  virtual Status HandleSort(HloInstruction* sort, HloInstruction* operand) = 0;
  virtual Status HandleConstant(HloInstruction* constant,
                                const Literal& literal) = 0;
  virtual Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                                       HloInstruction* operand) = 0;
  virtual Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                              HloInstruction* init_value,
                              tensorflow::gtl::ArraySlice<int64> dimensions,
                              HloComputation* function) = 0;
  virtual Status HandleBitcast(HloInstruction* bitcast) = 0;
  virtual Status HandleBroadcast(HloInstruction* broadcast) = 0;
  virtual Status HandleReshape(HloInstruction* reshape) = 0;
  virtual Status HandleTranspose(HloInstruction* transpose) = 0;
  virtual Status HandleParameter(HloInstruction* parameter) = 0;
  virtual Status HandleFusion(HloInstruction* fusion) = 0;
  virtual Status HandleCall(HloInstruction* call) = 0;
  virtual Status HandleCustomCall(
      HloInstruction* custom_call,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      tensorflow::StringPiece custom_call_target) = 0;
  virtual Status HandleSlice(HloInstruction* slice,
                             HloInstruction* operand) = 0;
  virtual Status HandleDynamicSlice(HloInstruction* dynamic_slice,
                                    HloInstruction* operand,
                                    HloInstruction* start_indices) = 0;
  virtual Status HandleDynamicUpdateSlice(HloInstruction* dynamic_update_slice,
                                          HloInstruction* operand,
                                          HloInstruction* update,
                                          HloInstruction* start_indices) = 0;
  virtual Status HandleTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) = 0;
  virtual Status HandleMap(
      HloInstruction* map,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* function,
      tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) = 0;
  virtual Status HandleReduceWindow(HloInstruction* reduce_window,
                                    HloInstruction* operand,
                                    const Window& window,
                                    HloComputation* function) = 0;
  virtual Status HandleSelectAndScatter(HloInstruction* instruction) = 0;
  virtual Status HandleWhile(HloInstruction* xla_while) = 0;

  virtual Status HandlePad(HloInstruction* pad) = 0;

  virtual Status HandleSend(HloInstruction* send) = 0;

  virtual Status HandleRecv(HloInstruction* recv) = 0;

  virtual Status HandleBatchNormTraining(HloInstruction* batchNormTraining) = 0;

  virtual Status HandleBatchNormInference(
      HloInstruction* batchNormInference) = 0;

  virtual Status HandleBatchNormGrad(HloInstruction* batchNormGrad) = 0;

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  virtual Status FinishVisit(HloInstruction* root) = 0;

  // 3 possible visitation states of HLO instructions. Each instruction's
  // state only flows one way: kNotVisited -> kVisiting -> kVisited.
  enum VisitState {
    kNotVisited = 0,
    kVisiting = 1,
    kVisited = 2,
  };

  VisitState GetVisitState(int id) { return visit_state_.GetState(id); }
  VisitState GetVisitState(const HloInstruction& instruction);

  // Resize internal state if necessary to hold state for ids <= num.
  // This call is purely a performance hint and can be omitted without
  // affecting correctness.
  void ReserveVisitStates(int num) { visit_state_.Reserve(num); }

  void SetVisitState(int id, VisitState state) {
    visit_state_.SetState(id, state);
  }

  // Sets the visitation state of the given instruction as kVisiting.
  //
  // Precondition: current state must be kNotVisited.
  void SetVisiting(const HloInstruction& instruction);

  // Sets the visitation state of the given instruction as kVisited.
  //
  // Precondition: current state must be either kNotVisited or kVisiting.
  void SetVisited(const HloInstruction& instruction);

  // Returns whether the state of the given instruction is kVisiting.
  bool IsVisiting(const HloInstruction& instruction) {
    return GetVisitState(instruction) == kVisiting;
  }

  // Returns whether the state of the given instruction is kVisited.
  bool DidVisit(const HloInstruction& instruction) {
    return GetVisitState(instruction) == kVisited;
  }

  // Returns whether the state of the given instruction is kNotVisited.
  bool NotVisited(const HloInstruction& instruction) {
    return GetVisitState(instruction) == kNotVisited;
  }

  // This method should be overridden by subclasses that wish to run some
  // operation on an op before its Handle* visitor method is called.
  //
  // For any HLO op, the order of calls is:
  //
  //   Preprocess(op);
  //   Handle/OpType/(op);
  //   Postprocess(op);
  //
  // Overriding methods should call DfsHloVisitor::Preprocess before doing their
  // own preprocessing.
  virtual Status Preprocess(HloInstruction* hlo);

  // This method should be overridden by subclasses that wish to run some
  // operation on an op after its Handle* visitor method is called. See
  // Preprocess for more details.
  //
  // Overriding methods should call DfsHloVisitor::Postprocess after doing their
  // own postprocessing.
  virtual Status Postprocess(HloInstruction* visited);

 private:
  class DFSVisitStates {
   public:
    DFSVisitStates() {}
    void Reserve(uint64 num) {
      states_.reserve((num + kStatesPerWord - 1) / kStatesPerWord);
    }
    VisitState GetState(uint64 id) {
      uint64 word_index = id / kStatesPerWord;
      if (word_index >= states_.size()) {
        return VisitState::kNotVisited;
      }
      static_assert(static_cast<int>(VisitState::kVisited) < 3,
                    "VisitState must fit in two bits");
      uint64 w = states_[word_index];
      uint32 shift = 2 * (id % kStatesPerWord);  // 2 bits per state
      return static_cast<VisitState>((w >> shift) & 0x3);
    }
    void SetState(uint64 id, VisitState state) {
      uint64 word_index = id / kStatesPerWord;
      if (word_index >= states_.size()) {
        states_.resize(word_index + 1, 0);
      }
      uint64* w = &states_[word_index];
      uint32 shift = 2 * (id % kStatesPerWord);  // 2 bits per state
      uint64 mask = 0x3ull << shift;
      *w = (*w & ~mask) | (static_cast<uint64>(state) << shift);
      DCHECK_EQ(GetState(id), state);
    }

   private:
    static const uint32 kStatesPerWord = sizeof(uint64) / 2 /*bits per entry*/;
    // Map from id to two-bit states.  We store 32 such states per 64-bit
    // value
    std::vector<uint64> states_;
  };

  DFSVisitStates visit_state_;

  TF_DISALLOW_COPY_AND_ASSIGN(DfsHloVisitor);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_H_
