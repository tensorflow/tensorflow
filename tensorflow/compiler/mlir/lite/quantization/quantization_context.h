/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONTEXT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONTEXT_H_

#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/device_target.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

namespace mlir {
namespace quant {

static bool EmptyParams(QuantParams p) { return p == quant::QuantizedType(); }

// The state for each op result during the quantization parameters propagation.
struct QuantState {
  // Quantization parameters propagated to an op result.
  QuantParams params;
  // A flag indicates this state (the params) shouldn't be changed after it is
  // initialized. This flag will be set to true if the quantization parameters
  // are from the quantization-aware training.
  const bool immutable;

  bool IsEmpty() { return EmptyParams(params); }
};

// The state for rescaling the propagated quantization parameters. This can be
// on the input side to satisfy the constraint of previous operation, or on the
// output side to satisfy the constraint of the next operation.
struct RequantizeState {
  // Sometimes, we have to "requantize" the quantization result to satisfy all
  // the constraints. The "requantize" can happen either on the input or output
  // of the quantization result.
  enum RequantizePosition {
    NO_REQUANTIZE,
    ON_INPUT,
    ON_OUTPUT
  } pos = NO_REQUANTIZE;

  // Quantization parameters will be used to add the requantize ops.
  QuantParams params;
};

// This class manages all the intermedaite quantization states.
class QuantizeContext {
 public:
  QuantizeContext(FuncOp func, const DeviceTarget &spec);

  // Returns all the quant region ops.
  std::vector<quant::QuantizeRegionOp> GetAllOps();

  // For each quant region op, propagates its quantization parameters according
  // to the kernel specification and also returns the adjcent quant region ops
  // which get the new quantization parameters propagated.
  LogicalResult Handle(quant::QuantizeRegionOp op,
                       llvm::SmallVectorImpl<Operation *> *new_items,
                       bool *changed);

  // Updates the port quantization specifications of all the quant region ops
  // with the propagation results.
  LogicalResult Finalize();

  // Dumps the states stores in the state manager.
  void DumpStates(QuantizeRegionOp current_op = {});

  // Update the quantization parameter for certain result of the op. By this
  // method, the quantization parameter is propagated to all the users of the
  // result as well.
  bool SetResultParams(Operation *op, int index, QuantParams params) {
    return states_manager_.SetResultParams(op, index, params);
  }

  // Update the quantization parameter for certain operand of the op. By this
  // method, the quantization parameter is propagated to the defining op of
  // operand as well.
  bool SetOperandParams(Operation *op, int index, QuantParams params) {
    return states_manager_.SetOperandParams(op, index, params);
  }

  // Return the quantization parameter of certain result of the op.
  QuantParams GetResultParams(Operation *op, int index) {
    return states_manager_.GetResultParams(op, index);
  }

  // Return the quantization parameter of certain operand of the op.
  QuantParams GetOperandParams(Operation *op, int index) {
    return states_manager_.GetOperandParams(op, index);
  }

  // A heuristic to get quantization parameters satisfies the same scale
  // constraints:
  // - If there are immutable states,
  //   - use the single input, or,
  //   - use the single output, or,
  //   - use the first one in the collection,
  // - use the single input if it is ready, or,
  // - use the single output if it is ready, or,
  // - use use the first ready one in the collection.
  QuantParams GetQuantParamsForSameScaleConstraint(Operation *op);

  // Propagate `params` to all the quantizable port of the `op`. The adjcent
  // ops, which have the parameters propagated to, are collected by `new_items`,
  // so they can be added to the working queue. `changed` is set to true if
  // there are any new elements being added to `new_items`.
  LogicalResult PropagateQuantParams(Operation *op, const QuantParams params,
                                     AdjacentOperations *new_items,
                                     bool *changed);

 private:
  class StatesManager {
   public:
    // Sets the quantization parameters of the constant result according to its
    // content.
    //
    // Always returns true.
    bool SetConstantResultParams(Operation *op);

    // Sets the quantization parameters of the result to a fixed value. If any
    // quantization parameters have been propagated, a `requantize` will happen
    // on the input of propagated quantization.
    //
    // Returns true, if the users of the result needs to be added to the
    // worklist.
    bool SetResultParams(Operation *op, int index, QuantParams params);

    // Sets the quantization parameters of the operand to a fixed value. If any
    // quantization parameters have been propagated, a `requantize` will happen
    // on the output of propagated quantization.
    //
    // Returns true, if the defining op of the operand needs to be added to the
    // worklist.
    bool SetOperandParams(Operation *op, int index, QuantParams params);

    // Returns the quantization parameters of the index-th result of the op.
    QuantParams GetResultParams(Operation *op, int index) {
      return states_[result_states_[{op, index}]].params;
    }

    // Returns the quantization parameters of the index-th operand of the op.
    QuantParams GetOperandParams(Operation *op, int index) {
      return states_[operand_states_[{op, index}]].params;
    }

   private:
    friend class QuantizeContext;

    // Uses the type of `val` to set the initial state of the index-th result if
    // `as_result` is true or index-th operand if `as_result` is false. The
    // state is immutable if the type is a quantized type. Returns the index of
    // this new state in the state vector.
    int InitializeState(quant::QuantizeRegionOp op, int index, bool as_result);

    // Sets the state of the index-th operand of the op. If this operand is
    // cached, uses the cached result without creating new entry in the state
    // vector. Otherwise, allocate a new entry in the state vector.
    void InitializeOperandState(quant::QuantizeRegionOp op, int index,
                                llvm::DenseMap<Value, int> *cache);

    // Sets the state of the index-th result of the op. If this result is
    // cached, uses the cached result without creating new entry in the state
    // vector. Otherwise, allocate a new entry in the state vector.
    void InitializeResultState(quant::QuantizeRegionOp op, int index,
                               llvm::DenseMap<Value, int> *cache);

    // Returns the state of the index-th operand of the op.
    QuantState &GetOperandQuantState(Operation *op, int index) {
      return states_[operand_states_[{op, index}]];
    }

    // Returns the state of the index-th result of the op.
    QuantState &GetResultQuantState(Operation *op, int index) {
      return states_[result_states_[{op, index}]];
    }

    // Returns the state of the index-th operand of the op.
    RequantizeState &GetOperandRequantizeState(Operation *op, int index) {
      return rescale_states_[operand_states_[{op, index}]];
    }

    // Returns the state of the index-th result of the op.
    RequantizeState &GetResultRequantizeState(Operation *op, int index) {
      return rescale_states_[result_states_[{op, index}]];
    }

   private:
    // This is used to identify an operand or result of an op. The second
    // element of this pair is the index of the operand or result.
    using OpValue = std::pair<mlir::Operation *, int>;

    // The vector contains all the quantization parameters propagated from the
    // defining operations of the value, or from the quantization aware
    // training.
    std::vector<QuantState> states_;

    // The map contains all the quantization parameters which are required to
    // satisfy the same operands and results constraint. The keys of this map
    // are the values from `operand_states_` and `result_state_`.
    std::unordered_map<int, RequantizeState> rescale_states_;

    // Maps of indexes to the propagation state vector from the ops operands,
    // results and arguments.
    llvm::DenseMap<OpValue, int> operand_states_;
    llvm::DenseMap<OpValue, int> result_states_;
  };

  FuncOp func_;

  DeviceTarget target_spec_;

  StatesManager states_manager_;
};

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_QUANTIZATION_CONTEXT_H_
