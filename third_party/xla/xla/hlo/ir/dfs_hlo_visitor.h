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

#ifndef XLA_HLO_IR_DFS_HLO_VISITOR_H_
#define XLA_HLO_IR_DFS_HLO_VISITOR_H_

#include <cstddef>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/tsl/platform/status.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

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
// If new HloInstructions are added during the traversal (e.g. by replacing an
// instruction), they will also be visited if they are the operand of an
// instruction that has not been visited yet (i.e. the instruction is in state
// kNotVisited). If you want to avoid that a newly added instruction 'hlo' is
// visited, you can call SetVisited(hlo). This may be necessary in normalization
// passes that replace all instructions, otherwise already replaced instructions
// might be visited (and replaced) again.
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
// Users should not use this class directly, but use the type-aliases
// DfsHloVisitor/ConstDfsHloVisitor instead.
template <typename HloInstructionPtr>
class DfsHloVisitorBase {
  static_assert(
      std::is_same<HloInstruction*, HloInstructionPtr>::value ||
          std::is_same<const HloInstruction*, HloInstructionPtr>::value,
      "Template argument expected to be HloInstruction* or const "
      "HloInstruction*");

 public:
  DfsHloVisitorBase() = default;
  virtual ~DfsHloVisitorBase() = default;

  // These routines are self-descriptive, see class comment for usage
  // information.

  virtual absl::Status HandleElementwiseUnary(HloInstructionPtr hlo);
  virtual absl::Status HandleElementwiseBinary(HloInstructionPtr hlo);

  virtual absl::Status HandleClamp(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleSelect(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleMaximum(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleMinimum(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleConcatenate(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleConvert(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleBitcastConvert(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleStochasticConvert(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleCopy(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleComplex(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleMultiply(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleDot(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleRaggedDot(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleScaledDot(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandlePower(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleSqrt(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleRsqrt(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleCbrt(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  /* go/keep-sorted start */
  virtual absl::Status HandleAllGather(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleAllGatherDone(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleAllGatherStart(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleAllReduce(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleAllReduceDone(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleAllReduceStart(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleAllToAll(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleCollectiveBroadcast(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleCollectivePermute(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleCollectivePermuteDone(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleCollectivePermuteStart(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleConvolution(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleOptimizationBarrier(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandlePartitionId(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleRaggedAllToAll(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleReduceScatter(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleReplicaId(HloInstructionPtr hlo) = 0;
  /* go/keep-sorted end */

  /* go/keep-sorted start */
  virtual absl::Status HandleCholesky(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleFft(HloInstructionPtr fft) = 0;
  virtual absl::Status HandleTopK(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleTriangularSolve(HloInstructionPtr hlo) = 0;
  /* go/keep-sorted end */

  virtual absl::Status HandleGetDimensionSize(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleSetDimensionSize(HloInstructionPtr hlo) = 0;

  virtual absl::Status HandleCompare(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleAdd(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleDivide(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleRemainder(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleSubtract(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleAbs(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleAtan2(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleRound(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleRoundNearestEven(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleErf(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleLogistic(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleSign(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleNegate(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleExp(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleExpm1(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleFloor(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleCeil(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleLog(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleClz(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleLog1p(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleCos(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleSin(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleTan(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleTanh(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleReal(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleImag(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleIsFinite(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleAnd(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleNot(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleOr(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleXor(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandlePopulationCount(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }
  virtual absl::Status HandleShiftLeft(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleShiftRightArithmetic(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }
  virtual absl::Status HandleShiftRightLogical(HloInstructionPtr hlo) {
    return HandleElementwiseBinary(hlo);
  }

  virtual absl::Status HandleReducePrecision(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }

  virtual absl::Status HandleDomain(HloInstructionPtr hlo) {
    return HandleElementwiseUnary(hlo);
  }

  /* go/keep-sorted start */
  virtual absl::Status HandleInfeed(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleOutfeed(HloInstructionPtr hlo) = 0;
  /* go/keep-sorted end */

  /* go/keep-sorted start */
  virtual absl::Status HandleBitcast(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleBroadcast(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleCall(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleConditional(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleConstant(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleCustomCall(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleDynamicReshape(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleDynamicSlice(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleDynamicUpdateSlice(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleFusion(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleGather(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleGetTupleElement(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleIota(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleMap(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleParameter(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleReduce(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleReduceWindow(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleReshape(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleReverse(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleRng(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleRngBitGenerator(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleRngGetAndUpdateState(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleScatter(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleSelectAndScatter(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleSlice(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleSort(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleTranspose(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleTuple(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleWhile(HloInstructionPtr hlo) = 0;
  /* go/keep-sorted end */

  virtual absl::Status HandlePad(HloInstructionPtr hlo) = 0;

  virtual absl::Status HandleAsyncStart(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleAsyncUpdate(HloInstructionPtr hlo) = 0;
  virtual absl::Status HandleAsyncDone(HloInstructionPtr hlo) = 0;

  virtual absl::Status HandleCopyStart(HloInstructionPtr copy_start) = 0;
  virtual absl::Status HandleCopyDone(HloInstructionPtr copy_done) = 0;

  virtual absl::Status HandleSend(HloInstructionPtr send) = 0;
  virtual absl::Status HandleSendDone(HloInstructionPtr send_done) = 0;

  virtual absl::Status HandleRecv(HloInstructionPtr recv) = 0;
  virtual absl::Status HandleRecvDone(HloInstructionPtr recv_done) = 0;

  virtual absl::Status HandleBatchNormTraining(HloInstructionPtr hlo) = 0;

  virtual absl::Status HandleBatchNormInference(HloInstructionPtr hlo) = 0;

  virtual absl::Status HandleBatchNormGrad(HloInstructionPtr hlo) = 0;

  virtual absl::Status HandleAddDependency(
      HloInstructionPtr add_dependency) = 0;
  virtual absl::Status HandleAfterAll(HloInstructionPtr token) = 0;

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  virtual absl::Status FinishVisit(HloInstructionPtr root) = 0;

  // 3 possible visitation states of HLO instructions. Each instruction's
  // state only flows one way: kNotVisited -> kVisiting -> kVisited.
  enum VisitState {
    kNotVisited = 0,
    kVisiting = 1,
    kVisited = 2,
  };

  VisitState GetVisitState(int id) {
    auto iter = visit_state_.find(id);
    if (iter == visit_state_.end()) {
      return VisitState::kNotVisited;
    }
    return iter->second;
  }
  VisitState GetVisitState(const HloInstruction& instruction);

  // Resize internal state if necessary to hold state for ids <= num.
  // This call is purely a performance hint and can be omitted without
  // affecting correctness.
  void ReserveVisitStates(int num) { visit_state_.reserve(num); }
  size_t VisitStateCapacity() const { return visit_state_.capacity(); }

  // Useful when we want to visit the same computation more than once with the
  // same visitor.
  void ResetVisitStates() {
    // Clear the map, but don't resize the capacity across uses -- Calculating
    // and reserving space could be expensive, and we always use the same
    // module->instruction_count() as the capacity.
    visit_state_.erase(visit_state_.begin(), visit_state_.end());
  }

  // Useful when we want to free up the memory used by the visit state without
  // destroying the actual visitor subclass.
  void DestroyVisitState() {
    visit_state_ = absl::flat_hash_map<int, VisitState>{};
  }

  void SetVisitState(int id, VisitState state) { visit_state_[id] = state; }

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
  virtual absl::Status Preprocess(HloInstructionPtr hlo);

  // This method should be overridden by subclasses that wish to run some
  // operation on an op after its Handle* visitor method is called. See
  // Preprocess for more details.
  //
  // Overriding methods should call DfsHloVisitor::Postprocess after doing their
  // own postprocessing.
  virtual absl::Status Postprocess(HloInstructionPtr hlo);

  // This method should be overriden by subclasses that wish to skip some ops
  // while traversing the HLO graph. If this method returns false, the calls to
  // Preprocess(op), Handle/OpType/(op) and Postprocess(op) are skipped.
  virtual bool ShouldProcessNode(HloInstructionPtr hlo) { return true; }

 private:
  absl::flat_hash_map<int, VisitState> visit_state_;

  DfsHloVisitorBase(const DfsHloVisitorBase&) = delete;
  DfsHloVisitorBase& operator=(const DfsHloVisitorBase&) = delete;
};

// Explicit instantiations in dfs_hlo_visitor.cc.
extern template class DfsHloVisitorBase<HloInstruction*>;
extern template class DfsHloVisitorBase<const HloInstruction*>;

// Users should use one of these two type aliases, which are the only two valid
// instantiations of DfsHloVisitorBase.
using DfsHloVisitor = DfsHloVisitorBase<HloInstruction*>;
using ConstDfsHloVisitor = DfsHloVisitorBase<const HloInstruction*>;

}  // namespace xla

#endif  // XLA_HLO_IR_DFS_HLO_VISITOR_H_
