/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/core/platform/logging.h"

#define DEBUG_TYPE "quantization-driver"

namespace mlir {
namespace quant {
namespace {
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

  // Avoid clobbering all uses of the value, limit to just these ops.
  SmallVector<std::pair<Operation *, int>> users;
};

using RequantizeStates = SmallVector<RequantizeState>;

// This is a worklist-driven driver for propagating quantization parameters
// across operations.
//
// The initial quantization parameters are extracted from the quantized type
// between adjacent tfl.quantize and tfl.dequantize ops. All these initial
// parameters are marked as immutable because they are from quantization-aware
// training.
//
// The algorithm traverses each op and sets the quantization parameters of its
// operands and results, according to its quantization specification, and then
// adds the operands and results to the worklist. If there are any conflicts
// (for example, there are quantization parameters propagated from the previous
// iteration), this process stops if the existing parameters are the immutable,
// or adding `requantize` op to resolve the conflicts.
//
// After the algorithm is converged, pairs of tfl.quantize and tfl.dequantize
// are inserted to the right position to materialize the propagation and
// requantize results.
//
class QuantizationDriver {
 public:
  explicit QuantizationDriver(func::FuncOp fn, bool is_signed,
                              bool disable_per_channel,
                              OpQuantSpecGetter op_quant_spec_getter,
                              OpQuantScaleSpecGetter op_quant_scale_spec_getter,
                              bool infer_tensor_range, bool legacy_float_scale)
      : fn_(fn),
        builder_(fn.getBody()),
        is_signed_(is_signed),
        disable_per_channel_(disable_per_channel),
        op_quant_spec_getter_(op_quant_spec_getter),
        op_quant_scale_spec_getter_(op_quant_scale_spec_getter),
        infer_tensor_range_(infer_tensor_range),
        legacy_float_scale_(legacy_float_scale) {}

  // The entry point of the quantization parameters propagation.
  void Run();

 private:
  // This is used to identify an operand or result of an op. The second element
  // of this pair is the index of the operand or result.
  using OpValue = std::pair<mlir::Operation *, int>;

  // Sets up the states for all the op results in the function.
  void Initialize();

  // Propagates the quantization parameters across all the ops.
  bool PropagateParams();

  // Duplicates the constant op if it has multiple uses, and replaces
  // target_op->operand[operand_index] with the newly created op. This also
  // replaces corresponsing quantization states.
  arith::ConstantOp DuplicateConstantOpIfNeeded(arith::ConstantOp op,
                                                Operation *target_op,
                                                int operand_index);

  // Adjusts bias scale that is derived from other scales (fc, conv ops) to
  // prevent overflow of quantized bias values. This also changes quantization
  // state of other inputs when needed.
  bool SetBiasParamsWithAdjustments(Operation *op, int bias_index,
                                    const std::vector<int> &input_indices,
                                    QuantParams params);

  // Helper for checking preconditions to adjust bias scale.
  bool ShouldCheckBiasScale(Operation *op, int bias_index,
                            const std::vector<int> &input_indices,
                            QuantParams params, int &input_index,
                            int &filter_index);

  // Inserts the Quantize and Dequantize ops according to the propagation
  // result.
  void Finalize();

  // The quantization parameters of bias operand are usually determined by
  // other operands, so if a constant is used by different ops as bias, it needs
  // to be duplicated, thus each op can assign its own quantization parameter
  // for this bias. Also this method adds all the non-bias constants (weights)
  // to a set for looking up later. This method also adds all the per-channel
  // weights to a set for looking up later.
  void PreprocessConstantOps();

  // Setup all the data structures for quantization propagation.
  void SetupAllStates();

  // Whether the constant is a weight, which shouldn't be shared by different
  // ops.
  bool IsWeight(Operation *cst) { return llvm::is_contained(weights_, cst); }

  // Returns all the related quantization constraints of the op.
  std::unique_ptr<OpQuantSpec> GetQuantSpec(Operation *op);
  std::unique_ptr<OpQuantScaleSpec> GetQuantScaleSpec(Operation *op);

  // Whether Quantization parameters have been propagated to the results of this
  // op.
  bool IsQuantized(Operation *op);

  // Adds all the users of index-th result of op to the work list.
  void AddUserToList(Operation *op, int index) {
    for (auto *user : op->getResult(index).getUsers()) {
      work_list_.push_back(user);
    }
  }

  // Adds the defining op of index-th operand of op to the work list.
  void AddOperandToList(Operation *op, int index) {
    if (auto *inst = op->getOperand(index).getDefiningOp()) {
      work_list_.push_back(inst);
    }
  }

  // Returns the quantization params for the bias input from the non-bias
  // operands which have their indexes in the `non_biases` vector. The returned
  // parameters are calculated by `func`.
  QuantParams GetBiasParams(Operation *op, int bias,
                            const std::vector<int> &non_biases,
                            AccumulatorScaleFunc func);

  // Sets the quantization parameters of the result to a fixed value. If any
  // quantization parameters have been propagated, a `requantize` will happen on
  // the input of propagated quantization.
  bool SetResultParams(Operation *op, int index, QuantParams params);

  // Sets the quantization parameters of the operand to a fixed value. If any
  // quantization parameters have been propagated, a `requantize` will happen on
  // the output of propagated quantization. When `override` is set, quantization
  // state of the value is replaced instead of adding requantization.
  bool SetOperandParams(Operation *op, int index, QuantParams params,
                        bool override = false);

  // Sets the quantization parameters of the constant result according to its
  // content.
  bool SetConstantResultParams(Operation *op);

  // Inserts the Quantize and Dequantize ops for quantizing the index-th result
  // of the op.
  void QuantizeOpResult(Operation *op, int index, QuantParams params);

  void QuantizeArg(BlockArgument arg, QuantParams params);

  // Inserts the Quantize and Dequantize ops to quantize the value and returns
  // the Quantize op.
  void QuantizeValue(Value value, QuantParams params, Location loc);

  // Inserts the Quantize ops for requantizing the index-th result of the op.
  void RequantizeOpResult(Operation *op, int index, RequantizeStates *states);

  // Inserts the Quantize ops for requantizing a block argument.
  void RequantizeArg(BlockArgument arg, RequantizeStates *states);

  // Inserts the Quantize and Dequantize ops to quantize the value and returns
  // the Quantize op.
  void RequantizeValue(Value value, RequantizeStates *states, Location loc);

  // A heuristic to get the quantization parameter satisfies the same scale
  // constraints for the op. Returns an empty option if this quantization
  // parameter doesn't exist.
  QuantParams GetQuantParamsForSameScaleConstraint(Operation *op);

  // Returns the state of the index-th operand of the op.
  QuantState &GetOperandQuantState(Operation *op, int index) {
    return states_[operand_states_[{op, index}]];
  }

  // Returns the state of the index-th result of the op.
  QuantState &GetResultQuantState(Operation *op, int index) {
    return states_[result_states_[{op, index}]];
  }

  // Returns the state of the block argument.
  QuantState &GetArgQuantState(BlockArgument arg) {
    return states_[arg_states_[arg]];
  }

  // Returns the states of the index-th operand of the op.
  RequantizeStates &GetOperandRequantizeStates(Operation *op, int index) {
    return rescale_states_[operand_states_[{op, index}]];
  }

  // Returns the states of the index-th result of the op.
  RequantizeStates &GetResultRequantizeStates(Operation *op, int index) {
    return rescale_states_[result_states_[{op, index}]];
  }

  // Returns the states of the arg.
  RequantizeStates &GetArgRequantizeStates(BlockArgument arg) {
    return rescale_states_[arg_states_[arg]];
  }

  // Uses the type of `val` to set the initial state of the index-th result if
  // `as_result` is true or index-th operand if `as_result` is false. The state
  // is immutable if the type is a quantized type. Returns the index of this
  // new state in the state vector.
  int InitializeState(Operation *op, int index, Value val, bool as_result);

  // Sets the state of an argument. If this value is cached, uses the cached
  // result without creating new entry in the state vector. Otherwise, allocate
  // a new entry in the state vector.
  void InitializeArgState(BlockArgument arg, Value in) {
    auto cached = value_to_state_.insert({in, 0});
    if (!cached.second) {
      arg_states_[arg] = cached.first->second;
      return;
    }
    QuantParams params =
        quant::QuantizedType::getQuantizedElementType(in.getType());
    bool immutable = !EmptyParams(params);
    int next_state_index = states_.size();
    states_.push_back({params, immutable});
    arg_states_[arg] = next_state_index;
    cached.first->second = next_state_index;
  }

  // Sets the state of the index-th operand of the op. If this operand is
  // cached, uses the cached result without creating new entry in the state
  // vector. Otherwise, allocate a new entry in the state vector.
  void InitializeOperandState(Operation *op, int index, Value in) {
    auto cached = value_to_state_.insert({in, 0});
    if (!cached.second) {
      operand_states_[{op, index}] = cached.first->second;
      return;
    }
    cached.first->second = InitializeState(op, index, in, /*as_result=*/false);
  }

  // Sets the state of the index-th result of the op. If this result is cached,
  // uses the cached result without creating new entry in the state vector.
  // Otherwise, allocate a new entry in the state vector.
  void InitializeResultState(Operation *op, int index, Value res) {
    auto cached = value_to_state_.insert({res, 0});
    if (!cached.second) {
      result_states_[{op, index}] = cached.first->second;
      return;
    }
    cached.first->second = InitializeState(op, index, res, /*as_result=*/true);
  }

  // Utility function for debug output for requantize states.
  void DumpRequantizeStates(const RequantizeStates &requantize_states) {
    for (auto &requantize_state : requantize_states) {
      if (requantize_state.pos != RequantizeState::NO_REQUANTIZE) {
        llvm::dbgs() << "+";
        requantize_state.params.print(llvm::dbgs());
      }
    }
  }

  void DumpStates(Operation *current_op) {
    if (current_op) {
      llvm::dbgs() << "\n\n\n" << current_op->getName() << "\n";
    }
    fn_.walk([&](Operation *op) {
      std::unique_ptr<OpQuantScaleSpec> scale_spec = GetQuantScaleSpec(op);
      if (op->hasTrait<OpTrait::IsTerminator>() ||
          (IsOpNotQuantizable(op) && !scale_spec->has_same_scale_requirement) ||
          llvm::isa<quant::QuantizeCastOp, quant::DequantizeCastOp,
                    func::ConstantOp, arith::ConstantOp>(op)) {
        return;
      }
      if (current_op == op) llvm::dbgs() << "===>>>";
      llvm::dbgs() << op->getName() << " : (";
      if (llvm::isa<func::FuncOp>(op)) {
        for (auto &arg : fn_.getArguments()) {
          if (auto params = GetArgQuantState(arg).params) {
            params.print(llvm::dbgs());
            DumpRequantizeStates(GetArgRequantizeStates(arg));
          }
          llvm::dbgs() << ",";
        }
      }
      for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
        if (auto params = GetOperandQuantState(op, i).params) {
          params.print(llvm::dbgs());
          DumpRequantizeStates(GetOperandRequantizeStates(op, i));
        } else {
          op->getOperand(i).getType().cast<ShapedType>().getElementType().print(
              llvm::dbgs());
        }
        llvm::dbgs() << ",";
      }
      llvm::dbgs() << ") -> (";
      for (int i = 0, e = op->getNumResults(); i < e; ++i) {
        if (auto params = GetResultQuantState(op, i).params) {
          params.print(llvm::dbgs());
          DumpRequantizeStates(GetResultRequantizeStates(op, i));
        } else {
          op->getResult(i).getType().cast<ShapedType>().getElementType().print(
              llvm::dbgs());
        }
        llvm::dbgs() << ",";
      }
      llvm::dbgs() << ")\n";
    });
  }

  func::FuncOp fn_;
  OpBuilder builder_;
  bool is_signed_;
  bool disable_per_channel_;

  // We should distinguish weights and bias constants. Biases are specified by
  // the quantization spec or are the operands of ops with same scale spec. The
  // rest are weights.
  llvm::DenseSet<Operation *> weights_;

  // The weights require narrow_range quantization. This map collects all the
  // weight operands defined by the op quant spec. If the value of the entry is
  // positive, per-channel quantization is required.
  llvm::DenseMap<Operation *, int> optimized_weights_;

  // All the ops needs to propagate the quantization parameters to.
  std::vector<Operation *> work_list_;
  std::unordered_set<Operation *> quantized_;

  // The vector contains all the quantization parameters propagated from the
  // defining operations of the value, or from the quantization aware training.
  std::vector<QuantState> states_;

  // The map contains all the quantization parameters which are required to
  // satisfy the same operands and results constraint. The keys of this map are
  // the values from `operand_states_` and `result_state_`.
  std::unordered_map<int, RequantizeStates> rescale_states_;

  // Maps of indexes to the propagation state vector from the ops operands,
  // results and arguments.
  llvm::DenseMap<OpValue, int> operand_states_;
  llvm::DenseMap<OpValue, int> result_states_;
  llvm::DenseMap<BlockArgument, int> arg_states_;
  llvm::DenseMap<Value, int> value_to_state_;

  // This vector is to preserve the arguments order, so the newly inserted
  // quantized ops for the arguments are deterministically ordered.
  llvm::SmallVector<BlockArgument, 4> args_;

  OpQuantSpecGetter op_quant_spec_getter_;
  OpQuantScaleSpecGetter op_quant_scale_spec_getter_;

  // Infer output ranges for activation ops and constants. This is usually
  // required for post-training quantization.
  bool infer_tensor_range_;

  // Calculate scales in float instead of double, so that the scales and
  // quantized values are exactly the same with the TOCO quantizer.
  bool legacy_float_scale_;
};
}  // namespace

std::unique_ptr<OpQuantSpec> QuantizationDriver::GetQuantSpec(Operation *op) {
  return op_quant_spec_getter_(op);
}

std::unique_ptr<OpQuantScaleSpec> QuantizationDriver::GetQuantScaleSpec(
    Operation *op) {
  return op_quant_scale_spec_getter_(op);
}

bool QuantizationDriver::IsQuantized(Operation *op) {
  for (int i = 0, e = op->getNumResults(); i != e; ++i) {
    if (GetResultQuantState(op, i).IsEmpty()) return false;
  }
  return true;
}

int QuantizationDriver::InitializeState(Operation *op, int index, Value val,
                                        bool as_result) {
  QuantParams params =
      quant::QuantizedType::getQuantizedElementType(val.getType());
  bool immutable = !EmptyParams(params);
  int next_state_index = states_.size();
  states_.push_back({params, immutable});
  if (as_result)
    result_states_[{op, index}] = next_state_index;
  else
    operand_states_[{op, index}] = next_state_index;

  return next_state_index;
}

bool QuantizationDriver::SetConstantResultParams(Operation *op) {
  DenseFPElementsAttr attr;
  Value res = op->getResult(0);
  if (!matchPattern(res, m_Constant(&attr))) {
    return false;
  }
  // TODO(fengliuai): make storage_type_width and narrow_range configurable.
  Type final_type;
  auto it = optimized_weights_.find(op);
  bool is_weight = it != optimized_weights_.end();
  bool is_weight_with_per_channel_support =
      is_weight && it->second != -1 && is_signed_;

  if (is_weight_with_per_channel_support && !disable_per_channel_) {
    // When `disable_per_channel_` is false, per-channel symmetric quantization
    // parameters are created from the weights when the ops support per-channel
    // quantization. Otherwise, uses per-tensor asymmetric quantization with
    // narrow range.

    // per-axis quantization weight, with symmetric min/max enforced.
    final_type = GetUniformQuantizedPerAxisTypeForWeight(
        attr, it->second, /*symmetric=*/true, /*num_bits=*/8, is_signed_,
        /*narrow_range=*/true, legacy_float_scale_);
  } else {
    // per-tensor quantization weight
    final_type = GetUniformQuantizedTypeForWeight(
        attr, /*symmetric=*/is_weight && is_signed_,
        /*num_bits=*/8, is_signed_,
        /*narrow_range_=*/is_weight, legacy_float_scale_);
  }
  if (auto quant_type = final_type.dyn_cast_or_null<quant::QuantizedType>()) {
    return SetResultParams(op, 0, quant_type);
  }
  return false;
}

bool QuantizationDriver::SetResultParams(Operation *op, int res_index,
                                         QuantParams params) {
  auto &state = GetResultQuantState(op, res_index);
  if (state.params == params) {
    return false;
  }
  if (!state.IsEmpty()) {
    auto &rescales = GetResultRequantizeStates(op, res_index);
    RequantizeState &rescale = rescales.emplace_back();
    rescale.pos = RequantizeState::ON_INPUT;
    rescale.params = params;
    return true;
  }
  state.params = params;
  AddUserToList(op, res_index);
  return true;
}

QuantParams QuantizationDriver::GetBiasParams(
    Operation *op, int bias, const std::vector<int> &non_biases,
    AccumulatorScaleFunc func) {
  auto &bias_state = GetOperandQuantState(op, bias);
  if (!bias_state.IsEmpty()) {
    return bias_state.params;
  }
  std::vector<QuantParams> op_types;
  op_types.reserve(non_biases.size());
  for (auto non_bias : non_biases) {
    auto &non_bias_type = GetOperandQuantState(op, non_bias);
    op_types.push_back(non_bias_type.params);
  }
  if (op_types.empty()) return {};
  return func(op_types, legacy_float_scale_);
}

bool QuantizationDriver::SetOperandParams(Operation *op, int index,
                                          QuantParams params, bool override) {
  auto &state = GetOperandQuantState(op, index);
  if (state.params == params) {
    return false;
  }

  if (!state.IsEmpty() && !override) {
    auto &rescales = GetOperandRequantizeStates(op, index);
    for (RequantizeState &rescale : rescales) {
      if (rescale.params == params) {
        rescale.users.emplace_back(op, index);
        return true;
      }
    }
    RequantizeState &rescale = rescales.emplace_back();
    rescale.pos = RequantizeState::ON_OUTPUT;
    rescale.params = params;
    rescale.users.emplace_back(op, index);
    return true;
  }

  state.params = params;
  AddOperandToList(op, index);
  return true;
}

void QuantizationDriver::QuantizeOpResult(Operation *op, int index,
                                          QuantParams params) {
  builder_.setInsertionPointAfter(op);
  Value original_result = op->getResult(index);
  QuantizeValue(original_result, params, op->getLoc());
}

void QuantizationDriver::QuantizeArg(BlockArgument arg, QuantParams params) {
  builder_.setInsertionPointToStart(arg.getOwner());
  QuantizeValue(arg, params, builder_.getUnknownLoc());
}

void QuantizationDriver::QuantizeValue(Value value, QuantParams params,
                                       Location loc) {
  Type expressed_type = value.getType();
  Type new_type = params.castFromExpressedType(expressed_type);
  // This value isn't an expressed type (float), skip.
  if (!new_type) return;
  auto quantize = builder_.create<quant::QuantizeCastOp>(loc, new_type, value);
  auto dequantize = builder_.create<quant::DequantizeCastOp>(
      loc, expressed_type, quantize.getResult());

  // This attribute is set to distinguish the quantize ops being added by the
  // quantization pass. These ops can be removed without losing original
  // program accuracy.
  // TODO(fengliuai): make the attribute being part of op definition.
  quantize->setAttr(kVolatileOpAttrName, builder_.getUnitAttr());

  // `original_result` has a use to `quantize`, so this will replace that use
  // by the result of `dequantize`. Remember to reset that use afterwards
  value.replaceAllUsesWith(dequantize);
  quantize.getOperation()->replaceUsesOfWith(dequantize, value);
}

void QuantizationDriver::RequantizeOpResult(Operation *op, int index,
                                            RequantizeStates *states) {
  if (states->empty()) return;

  builder_.setInsertionPointAfter(op);
  Value value = op->getResult(index);
  RequantizeState::RequantizePosition pos = states->front().pos;
  if (pos == RequantizeState::NO_REQUANTIZE) {
    return;
  }
  for (auto &state : *states) {
    // Check that all requantization positions are the same for each state.
    // Unsure if this check is required.
    if (state.pos != pos) {
      return;
    }
  }
  if (pos == RequantizeState::ON_OUTPUT) {
    Operation *user = value.getUses().begin().getUser();
    if (llvm::isa<quant::QuantizeCastOp>(user)) {
      // The requantize op is inserted between `quantize` and `dequantize` ops.
      value = user->getResult(0);
      builder_.setInsertionPointAfter(user);
    }
  }
  RequantizeValue(value, states, op->getLoc());
}

void QuantizationDriver::RequantizeArg(BlockArgument arg,
                                       RequantizeStates *states) {
  Value value = arg;
  builder_.setInsertionPointToStart(arg.getOwner());
  if (value.hasOneUse()) {
    auto user = value.use_begin().getUser();
    if (auto q = llvm::dyn_cast<quant::QuantizeCastOp>(user)) {
      value = q.getResult();
      builder_.setInsertionPoint(arg.getOwner(), ++Block::iterator(user));
    }
  }
  RequantizeValue(value, states, builder_.getUnknownLoc());
}

void QuantizationDriver::RequantizeValue(Value value, RequantizeStates *states,
                                         Location loc) {
  if (states->empty() ||
      states->front().pos == RequantizeState::NO_REQUANTIZE) {
    return;
  }
  if (states->front().pos == RequantizeState::ON_INPUT) {
    auto &state = states->front();
    Type expressed_type = value.getType();
    // The value needs to be requantized. A Quantize op will be created to use
    // it as the operand and replace its uses.
    Type new_type = state.params.castFromExpressedType(expressed_type);
    if (!new_type) return;
    auto requantize_op =
        builder_.create<quant::QuantizeCastOp>(loc, new_type, value);
    value.replaceAllUsesWith(requantize_op);
    requantize_op.getOperation()->replaceUsesOfWith(requantize_op, value);
    // This requantization was defined as required for the result value, so
    // there should be only one requant state.
    return;
  }

  // If this is an operand that requires requantization, then the value should
  // only have one DequantizeCastOp user which produces the operand value.
  if (!value.hasOneUse()) {
    return;
  }
  auto dequant_op = llvm::dyn_cast_or_null<quant::DequantizeCastOp>(
      value.use_begin().getUser());
  if (!dequant_op) {
    return;
  }
  // It is possible that the dequant value is used by a op that doesn't require
  // requant, so only overwrite the first if that is not the case.
  const int num_uses = std::distance(dequant_op.getResult().use_begin(),
                                     dequant_op.getResult().use_end());

  // Whether to replace quantization params of the first dequantize op
  // after the quantized value is produced.
  // If there is a use other than the requantize states, then we can't clobber.
  bool clobber_first = num_uses <= states->size();
  for (auto &state : *states) {
    Type expressed_type =
        quant::QuantizedType::castToExpressedType(value.getType());
    if (!expressed_type) continue;
    // The value needs to be requantized. A Quantize op will be created to use
    // it as the operand and replace its uses.
    Type new_type = state.params.castFromExpressedType(expressed_type);
    // This value isn't an expressed type (float), skip.
    if (!new_type) continue;

    auto requantize_op =
        builder_.create<quant::QuantizeCastOp>(loc, new_type, value);

    if (clobber_first) {
      dequant_op.setOperand(requantize_op.getResult());
      // All ops requiring this value already use the result of dequant.
      clobber_first = false;
    } else {
      auto new_dequant_op = builder_.create<quant::DequantizeCastOp>(
          loc, dequant_op.getResult().getType(), requantize_op.getResult());
      for (auto &op_index : state.users) {
        op_index.first->setOperand(op_index.second, new_dequant_op.getResult());
      }
    }
  }
}

// A heuristic to get quantization parameters satisfies the same scale
// constraints:
// - If there are immutable states,
//   - use the single input, or,
//   - use the single output, or,
//   - use the first one in the collection,
// - use the single input if it is ready, or,
// - use the single output if it is ready, or,
// - use the first ready one in the collection.
QuantParams QuantizationDriver::GetQuantParamsForSameScaleConstraint(
    Operation *op) {
  // Two vector to collect Non-empty operands and results states.
  std::vector<QuantState *> mutable_states, immutable_states;
  for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
    auto &state = GetOperandQuantState(op, i);
    if (state.immutable) {
      immutable_states.push_back(&state);
    } else if (!state.IsEmpty()) {
      mutable_states.push_back(&state);
    }
  }

  int immutable_operands_num = immutable_states.size();
  int mutable_operands_num = mutable_states.size();
  // Use the operand's state if it is immutable and it is the only one
  // operand.
  if (op->getNumOperands() == 1 && immutable_operands_num == 1) {
    return immutable_states.front()->params;
  }

  for (int i = 0, e = op->getNumResults(); i != e; ++i) {
    auto &state = GetResultQuantState(op, i);
    if (state.immutable) {
      immutable_states.push_back(&state);
    } else if (!state.IsEmpty()) {
      mutable_states.push_back(&state);
    }
  }

  int immutable_results_num = immutable_states.size() - immutable_operands_num;
  int mutable_results_num = mutable_states.size() - mutable_operands_num;
  // Use the result's state if it is immutable and it is the only one result.
  if (op->getNumResults() == 1 && immutable_results_num == 1) {
    return immutable_states.back()->params;
  }

  // Use the first immutable state to quantize the rest operands and results.
  if (!immutable_states.empty()) return immutable_states.front()->params;

  // If there are no immutable states, use the operand's state if it is the
  // only one operand and has parameters propagated.
  if (op->getNumOperands() == 1 && mutable_operands_num == 1) {
    return mutable_states.front()->params;
  }

  // If there are no immutable states, use the result's state if it is the
  // only one result and has parameters propagated.
  if (op->getNumResults() == 1 && mutable_results_num == 1) {
    return mutable_states.back()->params;
  }

  // Use the first propagated state to quantize the rest operands and results.
  if (!mutable_states.empty()) return mutable_states.front()->params;

  // None operands/results have parameters propagated, skip this node for now.
  return {};
}

void QuantizationDriver::PreprocessConstantOps() {
  fn_.walk([&](arith::ConstantOp cst) {
    // Non-float tensors are neither weights nor require quantization.
    auto type = cst.getType().dyn_cast<ShapedType>();
    if (!type || !type.getElementType().isa<FloatType>()) return;

    Value value = cst.getResult();
    builder_.setInsertionPoint(cst);

    // The following loop will change the value uses, thus we cache all the uses
    // needs to be changed.
    llvm::SmallVector<std::pair<Operation *, int>, 4> uses;
    for (auto &use : value.getUses()) {
      uses.push_back({use.getOwner(), use.getOperandNumber()});
    }
    for (const auto &indexed_use : llvm::enumerate(uses)) {
      Operation *user = indexed_use.value().first;
      int operand_num = indexed_use.value().second;

      std::unique_ptr<OpQuantSpec> spec = GetQuantSpec(user);
      std::unique_ptr<OpQuantScaleSpec> scale_spec = GetQuantScaleSpec(user);
      BiasParamsMap biases = spec->biases_params;

      // The quantization parameters of a `weight` shouldn't be determined by
      // other values. So any constants which are not bias, an operand of an
      // op with same scale requirements, and haven't been quantized are
      // weights.
      if (biases.find(operand_num) == biases.end() &&
          !scale_spec->has_same_scale_requirement &&
          !llvm::dyn_cast<quant::QuantizeCastOp>(user)) {
        // Needs to scan the content of weights to get the quantization
        // parameters if there are no quantization parameters (FakeQuant ops).
        // For this case, the weight will not be duplicated.
        weights_.insert(cst);
        auto affine_user =
            llvm::dyn_cast<mlir::AffineQuantizedOpInterface>(user);
        if (affine_user && affine_user.GetAffineOperandIndex() == operand_num &&
            affine_user.RequiredNarrowRangeAffineOperand()) {
          optimized_weights_.insert(
              {cst, affine_user.GetQuantizationDimIndex()});
        }
      } else {
        // This is a bias or an operand of an op with same scale requirements,
        // so the quantization parameter are propagated from or determined by
        // other values. Duplicate this constant in case it is shared by
        // different users.
        if (uses.size() > 1) {
          auto new_cst =
              builder_.create<arith::ConstantOp>(cst.getLoc(), cst.getValue());
          user->setOperand(operand_num, new_cst);
        }
      }
    }
  });
}

void QuantizationDriver::SetupAllStates() {
  for (auto arg : fn_.getArguments()) {
    args_.push_back(arg);
    Value value = arg;
    // If the argument is quantized, it should only has one user.
    if (arg.hasOneUse()) {
      auto user = value.use_begin().getUser();
      if (auto q = llvm::dyn_cast<quant::QuantizeCastOp>(user)) {
        value = q.getResult();
      }
    }
    InitializeArgState(arg, value);
  }

  fn_.walk([&](Operation *op) {
    std::unique_ptr<OpQuantScaleSpec> scale_spec = GetQuantScaleSpec(op);
    if (IsOpNotQuantizable(op) && !scale_spec->has_same_scale_requirement) {
      return;
    }
    work_list_.push_back(op);

    for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
      auto operand = op->getOperand(i);
      if (auto *inst = operand.getDefiningOp()) {
        // If the operand comes from a tfl.dequantize op, we use the quantized
        // input of this tfl.dequantize op to set the state.
        if (auto dq = llvm::dyn_cast<quant::DequantizeCastOp>(inst)) {
          operand = dq.getArg();
        }
      }
      InitializeOperandState(op, i, operand);
    }

    for (int res = 0, e = op->getNumResults(); res != e; ++res) {
      Value result = op->getResult(res);
      // If the result has been quantized, it should only be used by a
      // tfl.quantize op. For this case, we uses the quantized result to
      // create the state and mark it immutable.
      if (result.hasOneUse()) {
        auto user = result.use_begin().getUser();
        if (auto q = llvm::dyn_cast<quant::QuantizeCastOp>(user)) {
          result = q.getResult();
        }
      }
      InitializeResultState(op, res, result);
    }
  });
}

// This method scans the operations in the function to setup the initial
// states for quantization parameter propagation.
// TODO(fengliuai): This algorithm assumes there are only one pair of
// tfl.quantize and tfl.dequantize ops between two quantizable ops. A sanity
// check should be applied.
void QuantizationDriver::Initialize() {
  // Duplicate the bias constant, so the states can be setup correctly.
  // TODO(fengliuai): Function definition should also be duplicated if there
  // are multiple call sites.
  PreprocessConstantOps();

  // Setup all the internal states.
  SetupAllStates();
}

bool QuantizationDriver::PropagateParams() {
  // TODO(fengliuai): uses a typed indicator instead of a bool value.
  bool changed = false;
  while (!work_list_.empty()) {
    Operation *op = work_list_.back();
    work_list_.pop_back();

    LLVM_DEBUG(DumpStates(op));

    // This op has been quantized, so we should not consider it again.
    if (llvm::is_contained(quantized_, op)) continue;
    quantized_.insert(op);

    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(op)) {
      // If the workflow requires inferring ranges from the content
      // (post-training quantization) and it is weight (filter) and hasn't
      // been quantized, we infer the quantization parameters from the content.
      if (infer_tensor_range_ && IsWeight(cst) && !IsQuantized(op)) {
        // The quantization parameters are determined by the content of the
        // constant.
        changed |= SetConstantResultParams(op);
      }
      continue;
    }

    std::unique_ptr<OpQuantScaleSpec> scale_spec = GetQuantScaleSpec(op);

    if (scale_spec->has_same_scale_requirement) {
      auto params = GetQuantParamsForSameScaleConstraint(op);
      // The quantization parameters haven't been propagated to any operands
      // or results. Skip this node for now.
      if (!params) {
        quantized_.erase(op);
        continue;
      }

      // Use the final state to set all the operands' parameters.
      for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
        if (auto type = op->getOperand(i).getType().dyn_cast<ShapedType>()) {
          // Without this check, it will accidentally propagate the quantization
          // information by the shared non-float tensors.
          if (type.getElementType().isa<FloatType>())
            changed |= SetOperandParams(op, i, params);
        }
      }

      // Use the final state to set all the results' parameters.
      for (int res = 0, e = op->getNumResults(); res != e; ++res)
        if (auto type = op->getResult(res).getType().dyn_cast<ShapedType>()) {
          // Without this check, it will accidentally propagate the quantization
          // information by the shared non-float-tensors.
          if (type.getElementType().isa<FloatType>())
            changed |= SetResultParams(op, res, params);
        }
    }

    // TODO(fengliuai): make the bit width configurable.
    if (scale_spec->has_fixed_output_range && infer_tensor_range_) {
      // Infer ranges from the activation ops. This is usually required for
      // the post-training quantization workflow.
      // TODO(fengliuai): different result can have different fixed range.
      auto params = scale_spec->fixed_output_range_func(is_signed_,
                                                        /*bit_width=*/8);
      for (auto i = 0; i < op->getNumResults(); ++i) {
        // The range is null if the result has been quantized.
        if (params) {
          changed |= SetResultParams(op, i, params);
        }
      }
    }

    auto spec = GetQuantSpec(op);
    for (auto &it : spec->biases_params) {
      auto params =
          GetBiasParams(op, it.first, it.second.first, it.second.second);
      if (!params) {
        quantized_.erase(op);
        continue;
      }
      changed |=
          SetBiasParamsWithAdjustments(op, it.first, it.second.first, params);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "\n\n\n");
  LLVM_DEBUG(DumpStates(nullptr));

  return changed;
}

arith::ConstantOp QuantizationDriver::DuplicateConstantOpIfNeeded(
    arith::ConstantOp op, Operation *target_op, int operand_index) {
  if (op.getResult().hasOneUse()) {
    return op;
  }
  OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);
  arith::ConstantOp new_op = llvm::cast<arith::ConstantOp>(builder.clone(*op));
  target_op->getOpOperand(operand_index).set(new_op.getResult());
  InitializeOperandState(target_op, operand_index, new_op.getResult());
  InitializeResultState(new_op, 0, new_op.getResult());
  return new_op;
}

bool QuantizationDriver::ShouldCheckBiasScale(
    Operation *op, int bias_index, const std::vector<int> &input_indices,
    QuantParams params, int &input_index, int &filter_index) {
  // For now, restrict scale adjustment to ops with affine quantized weights,
  // and having weights and biases as constants. This currently only applies to
  // FC and Conv* ops. Restriction for the weight can be relaxed if there are
  // needs for adjusting scale of variable weights.
  auto affine_op = llvm::dyn_cast<AffineQuantizedOpInterface>(op);
  auto bias_op = op->getOperand(bias_index).getDefiningOp<arith::ConstantOp>();
  if (!affine_op || !bias_op || input_indices.size() != 2) return false;
  if (!bias_op.getValue().isa<DenseFPElementsAttr>()) return false;
  filter_index = affine_op.GetAffineOperandIndex();
  if (!op->getOperand(filter_index).getDefiningOp<arith::ConstantOp>()) {
    return false;
  }
  if (filter_index == input_indices[0]) {
    input_index = input_indices[1];
  } else if (filter_index == input_indices[1]) {
    input_index = input_indices[0];
  } else {
    return false;
  }

  auto input_state = GetOperandQuantState(op, input_index);
  auto filter_state = GetOperandQuantState(op, filter_index);
  // If quantization paramater for the filter is fixed, should return it as-is.
  // Only checks ops with 8-bit input and weights, and 32-bit biases.
  if (!(input_state.params.getStorageTypeIntegralWidth() == 8 &&
        filter_state.params.getStorageTypeIntegralWidth() == 8 &&
        params.getStorageTypeIntegralWidth() == 32)) {
    return false;
  }
  return true;
}

bool QuantizationDriver::SetBiasParamsWithAdjustments(
    Operation *op, int bias_index, const std::vector<int> &input_indices,
    QuantParams params) {
  bool changed = false;
  int input_index;
  int filter_index;
  if (!ShouldCheckBiasScale(op, bias_index, input_indices, params, input_index,
                            filter_index)) {
    return SetOperandParams(op, bias_index, params);
  }

  quant::QuantState input_state = GetOperandQuantState(op, input_index);
  quant::QuantState filter_state = GetOperandQuantState(op, filter_index);
  auto bias_op = op->getOperand(bias_index).getDefiningOp<arith::ConstantOp>();
  const double input_scale =
      input_state.params.cast<UniformQuantizedType>().getScale();

  auto bias_values = bias_op.getValue().cast<DenseFPElementsAttr>();
  // Restrict maximum absolute value of bias within INT_MAX / 2, to make some
  // room for accumulator.
  const int32_t kBiasMax = std::numeric_limits<int32_t>::max() / 2;
  if (auto bias_params = params.dyn_cast<UniformQuantizedType>()) {
    double bias_half_range = 0.0f;
    for (auto bias : bias_values.getValues<APFloat>()) {
      if (bias_half_range < std::abs(bias.convertToFloat())) {
        bias_half_range = std::abs(bias.convertToFloat());
      }
    }
    if (bias_half_range / bias_params.getScale() < kBiasMax) {
      return SetOperandParams(op, bias_index, params);
    }
    double new_bias_scale = static_cast<double>(bias_half_range) / kBiasMax;

    changed |= SetOperandParams(
        op, bias_index,
        UniformQuantizedType::getChecked(
            bias_op->getLoc(), params.getFlags(), params.getStorageType(),
            params.getExpressedType(), new_bias_scale, 0,
            params.getStorageTypeMin(), params.getStorageTypeMax()));
    auto filter_op = DuplicateConstantOpIfNeeded(
        op->getOperand(filter_index).getDefiningOp<arith::ConstantOp>(), op,
        filter_index);
    if (!filter_op) {
      return SetOperandParams(op, bias_index, params);
    }

    auto filter_param = filter_state.params.cast<UniformQuantizedType>();
    changed |= SetOperandParams(
        op, filter_index,
        UniformQuantizedType::getChecked(
            filter_op->getLoc(), filter_param.getFlags(),
            filter_param.getStorageType(), filter_param.getExpressedType(),
            new_bias_scale / input_scale, 0, filter_param.getStorageTypeMin(),
            filter_param.getStorageTypeMax()),
        /*override=*/true);
  } else if (auto bias_params =
                 params.dyn_cast<UniformQuantizedPerAxisType>()) {
    auto filter_params =
        filter_state.params.cast<UniformQuantizedPerAxisType>();
    std::vector<double> new_bias_scales = bias_params.getScales().vec();
    std::vector<double> new_filter_scales = filter_params.getScales().vec();
    bool needs_adjustment = false;
    for (int i = 0; i < bias_params.getScales().size(); ++i) {
      float abs_bias = std::abs(bias_values.getValues<float>()[i]);
      if (abs_bias / new_bias_scales[i] > kBiasMax) {
        new_bias_scales[i] = static_cast<double>(abs_bias) / kBiasMax;
        new_filter_scales[i] = new_bias_scales[i] / input_scale;
        needs_adjustment = true;
      }
    }
    if (!needs_adjustment) {
      return SetOperandParams(op, bias_index, params);
    }
    changed |= SetOperandParams(
        op, bias_index,
        UniformQuantizedPerAxisType::getChecked(
            bias_op->getLoc(), params.getFlags(), params.getStorageType(),
            params.getExpressedType(), new_bias_scales,
            bias_params.getZeroPoints(), bias_params.getQuantizedDimension(),
            params.getStorageTypeMin(), params.getStorageTypeMax()));

    auto filter_op = DuplicateConstantOpIfNeeded(
        op->getOperand(filter_index).getDefiningOp<arith::ConstantOp>(), op,
        filter_index);
    changed |= SetOperandParams(
        op, filter_index,
        UniformQuantizedPerAxisType::getChecked(
            filter_op->getLoc(), filter_params.getFlags(),
            filter_params.getStorageType(), filter_params.getExpressedType(),
            new_filter_scales, filter_params.getZeroPoints(),
            filter_params.getQuantizedDimension(),
            filter_params.getStorageTypeMin(),
            filter_params.getStorageTypeMax()),
        /*override=*/true);
  }
  return changed;
}

void QuantizationDriver::Finalize() {
  for (auto arg : args_) {
    auto &state = GetArgQuantState(arg);
    auto &requantizes = GetArgRequantizeStates(arg);
    if (state.IsEmpty() || (state.immutable && requantizes.empty())) {
      continue;
    }

    if (!state.immutable) {
      QuantizeArg(arg, state.params);
    }

    if (!requantizes.empty()) {
      RequantizeArg(arg, &requantizes);
    }
  }

  for (auto it : result_states_) {
    Operation *op = it.first.first;
    int res_index = it.first.second;
    auto &state = GetResultQuantState(op, res_index);
    auto &requantizes = GetResultRequantizeStates(op, res_index);
    if (state.IsEmpty() || (state.immutable && requantizes.empty())) {
      continue;
    }

    if (!state.immutable) {
      QuantizeOpResult(op, res_index, state.params);
    }

    if (!requantizes.empty()) {
      RequantizeOpResult(op, res_index, &requantizes);
    }
  }
}

void QuantizationDriver::Run() {
  Initialize();
  if (PropagateParams()) {
    Finalize();
  }
}

void ApplyQuantizationParamsPropagation(mlir::func::FuncOp func, bool is_signed,
                                        bool disable_per_channel,
                                        OpQuantSpecGetter op_quant_spec_getter,
                                        bool infer_tensor_ranges,
                                        bool legacy_float_scale) {
  ApplyQuantizationParamsPropagation(
      func, is_signed, disable_per_channel, op_quant_spec_getter,
      GetDefaultQuantScaleSpec, infer_tensor_ranges, legacy_float_scale);
}

void ApplyQuantizationParamsPropagation(
    mlir::func::FuncOp func, bool is_signed, bool disable_per_channel,
    OpQuantSpecGetter op_quant_spec_getter,
    OpQuantScaleSpecGetter op_quant_scale_spec_getter, bool infer_tensor_ranges,
    bool legacy_float_scale) {
  QuantizationDriver(func, is_signed, disable_per_channel, op_quant_spec_getter,
                     op_quant_scale_spec_getter, infer_tensor_ranges,
                     legacy_float_scale)
      .Run();
}

}  // namespace quant
}  // namespace mlir
