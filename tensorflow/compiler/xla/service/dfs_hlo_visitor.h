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

  virtual Status HandleElementwiseUnary(HloInstruction* hlo);
  virtual Status HandleElementwiseBinary(HloInstruction* hlo);
  virtual Status HandleClamp(HloInstruction* clamp) = 0;
  virtual Status HandleSelect(HloInstruction* select) = 0;
  virtual Status HandleMaximum(HloInstruction* maximum) {
    return HandleElementwiseBinary(maximum);
  }
  virtual Status HandleMinimum(HloInstruction* minimum) {
    return HandleElementwiseBinary(minimum);
  }
  virtual Status HandleConcatenate(HloInstruction* concatenate) = 0;
  virtual Status HandleConvert(HloInstruction* convert) {
    return HandleElementwiseUnary(convert);
  }
  virtual Status HandleCopy(HloInstruction* copy) {
    return HandleElementwiseUnary(copy);
  }
  virtual Status HandleComplex(HloInstruction* complex) {
    return HandleElementwiseBinary(complex);
  }
  virtual Status HandleMultiply(HloInstruction* multiply) {
    return HandleElementwiseBinary(multiply);
  }
  virtual Status HandleDot(HloInstruction* dot) = 0;
  virtual Status HandlePower(HloInstruction* power) {
    return HandleElementwiseBinary(power);
  }
  virtual Status HandleConvolution(HloInstruction* convolution) = 0;
  virtual Status HandleCrossReplicaSum(HloInstruction* crs) = 0;
  virtual Status HandleCompare(HloInstruction* compare) {
    return HandleElementwiseBinary(compare);
  }
  virtual Status HandleAdd(HloInstruction* add) {
    return HandleElementwiseBinary(add);
  }
  virtual Status HandleDivide(HloInstruction* divide) {
    return HandleElementwiseBinary(divide);
  }
  virtual Status HandleRemainder(HloInstruction* remainder) {
    return HandleElementwiseBinary(remainder);
  }
  virtual Status HandleSubtract(HloInstruction* subtract) {
    return HandleElementwiseBinary(subtract);
  }
  virtual Status HandleAbs(HloInstruction* abs) {
    return HandleElementwiseUnary(abs);
  }
  virtual Status HandleAtan2(HloInstruction* atan2, HloInstruction* y,
                             HloInstruction* x) {
    return HandleElementwiseBinary(atan2);
  }
  virtual Status HandleRound(HloInstruction* round) {
    return HandleElementwiseUnary(round);
  }
  virtual Status HandleSign(HloInstruction* sign) {
    return HandleElementwiseUnary(sign);
  }
  virtual Status HandleNegate(HloInstruction* negate) {
    return HandleElementwiseUnary(negate);
  }
  virtual Status HandleExp(HloInstruction* exp) {
    return HandleElementwiseUnary(exp);
  }
  virtual Status HandleFloor(HloInstruction* floor) {
    return HandleElementwiseUnary(floor);
  }
  virtual Status HandleCeil(HloInstruction* ceil) {
    return HandleElementwiseUnary(ceil);
  }
  virtual Status HandleLog(HloInstruction* log) {
    return HandleElementwiseUnary(log);
  }
  virtual Status HandleCos(HloInstruction* cos) {
    return HandleElementwiseUnary(cos);
  }
  virtual Status HandleSin(HloInstruction* sin) {
    return HandleElementwiseUnary(sin);
  }
  virtual Status HandleTanh(HloInstruction* tanh) {
    return HandleElementwiseUnary(tanh);
  }
  virtual Status HandleReal(HloInstruction* real) {
    return HandleElementwiseUnary(real);
  }
  virtual Status HandleImag(HloInstruction* imag) {
    return HandleElementwiseUnary(imag);
  }
  virtual Status HandleIsFinite(HloInstruction* is_finite) {
    return HandleElementwiseUnary(is_finite);
  }
  virtual Status HandleAnd(HloInstruction* and_) {
    return HandleElementwiseBinary(and_);
  }
  virtual Status HandleNot(HloInstruction* not_) {
    return HandleElementwiseUnary(not_);
  }
  virtual Status HandleOr(HloInstruction* or_) {
    return HandleElementwiseBinary(or_);
  }
  virtual Status HandleShiftLeft(HloInstruction* shift_left) {
    return HandleElementwiseBinary(shift_left);
  }
  virtual Status HandleShiftRightArithmetic(
      HloInstruction* shift_right_arithmetic) {
    return HandleElementwiseBinary(shift_right_arithmetic);
  }
  virtual Status HandleShiftRightLogical(HloInstruction* shift_right_logical) {
    return HandleElementwiseBinary(shift_right_logical);
  }

  virtual Status HandleReducePrecision(HloInstruction* reduce_precision) {
    return HandleElementwiseUnary(reduce_precision);
  }

  virtual Status HandleInfeed(HloInstruction* infeed) = 0;
  virtual Status HandleOutfeed(HloInstruction* outfeed) = 0;
  virtual Status HandleRng(HloInstruction* random) = 0;
  virtual Status HandleReverse(HloInstruction* reverse) = 0;
  virtual Status HandleSort(HloInstruction* sort) = 0;
  virtual Status HandleConstant(HloInstruction* constant) = 0;
  virtual Status HandleGetTupleElement(HloInstruction* get_tuple_element) = 0;
  virtual Status HandleReduce(HloInstruction* reduce) = 0;
  virtual Status HandleBitcast(HloInstruction* bitcast) = 0;
  virtual Status HandleBroadcast(HloInstruction* broadcast) = 0;
  virtual Status HandleReshape(HloInstruction* reshape) = 0;
  virtual Status HandleTranspose(HloInstruction* transpose) = 0;
  virtual Status HandleParameter(HloInstruction* parameter) = 0;
  virtual Status HandleFusion(HloInstruction* fusion) = 0;
  virtual Status HandleCall(HloInstruction* call) = 0;
  virtual Status HandleCustomCall(HloInstruction* custom_call) = 0;
  virtual Status HandleSlice(HloInstruction* slice) = 0;
  virtual Status HandleDynamicSlice(HloInstruction* dynamic_slice) = 0;
  virtual Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) = 0;
  virtual Status HandleTuple(HloInstruction* tuple) = 0;
  virtual Status HandleMap(HloInstruction* map) = 0;
  virtual Status HandleReduceWindow(HloInstruction* reduce_window) = 0;
  virtual Status HandleSelectAndScatter(HloInstruction* instruction) = 0;
  virtual Status HandleWhile(HloInstruction* xla_while) = 0;

  virtual Status HandlePad(HloInstruction* pad) = 0;

  virtual Status HandleSend(HloInstruction* send) = 0;

  virtual Status HandleRecv(HloInstruction* recv) = 0;

  virtual Status HandleBatchNormTraining(
      HloInstruction* batch_norm_training) = 0;

  virtual Status HandleBatchNormInference(
      HloInstruction* batch_norm_inference) = 0;

  virtual Status HandleBatchNormGrad(HloInstruction* batch_norm_grad) = 0;

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
