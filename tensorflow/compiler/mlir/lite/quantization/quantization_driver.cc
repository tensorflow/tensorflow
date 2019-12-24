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
#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace TFL {
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
};

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
  explicit QuantizationDriver(FuncOp fn, bool is_signed,
                              bool disable_per_channel,
                              OpQuantSpecGetter op_quant_spec_getter)
      : fn_(fn),
        builder_(fn.getBody()),
        is_signed_(is_signed),
        disable_per_channel_(disable_per_channel),
        op_quant_spec_getter_(op_quant_spec_getter) {}

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

  // Whether Quantization parameters have been propagated to the results of this
  // op.
  bool IsQuantized(Operation *op);

  // Adds all the users of index-th result of op to the work list.
  void AddUserToList(Operation *op, int index) {
    for (auto *user : op->getResult(index)->getUsers()) {
      work_list_.push_back(user);
    }
  }

  // Adds the defining op of index-th operand of op to the work list.
  void AddOperandToList(Operation *op, int index) {
    if (auto *inst = op->getOperand(index)->getDefiningOp()) {
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
  // the output of propagated quantization.
  bool SetOperandParams(Operation *op, int index, QuantParams params);

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
  void RequantizeOpResult(Operation *op, int index, RequantizeState *state);

  void RequantizeArg(BlockArgument arg, RequantizeState *state);

  // Inserts the Quantize and Dequantize ops to quantize the value and returns
  // the Quantize op.
  void RequantizeValue(Value value, RequantizeState *state, Location loc);

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

  QuantState &GetArgQuantState(BlockArgument arg) {
    return states_[arg_states_[arg]];
  }

  // Returns the state of the index-th operand of the op.
  RequantizeState &GetOperandRequantizeState(Operation *op, int index) {
    return rescale_states_[operand_states_[{op, index}]];
  }

  // Returns the state of the index-th result of the op.
  RequantizeState &GetResultRequantizeState(Operation *op, int index) {
    return rescale_states_[result_states_[{op, index}]];
  }

  RequantizeState &GetArgRequantizeState(BlockArgument arg) {
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
  void InitializeArgState(BlockArgument arg, Value in,
                          llvm::DenseMap<Value, int> *cache) {
    auto cached = cache->insert({in, 0});
    if (!cached.second) {
      arg_states_[arg] = cached.first->second;
      return;
    }
    QuantParams params =
        quant::QuantizedType::getQuantizedElementType(in->getType());
    bool immutable = !EmptyParams(params);
    int next_state_index = states_.size();
    states_.push_back({params, immutable});
    arg_states_[arg] = next_state_index;
    cached.first->second = next_state_index;
  }

  // Sets the state of the index-th operand of the op. If this operand is
  // cached, uses the cached result without creating new entry in the state
  // vector. Otherwise, allocate a new entry in the state vector.
  void InitializeOperandState(Operation *op, int index, Value in,
                              llvm::DenseMap<Value, int> *cache) {
    auto cached = cache->insert({in, 0});
    if (!cached.second) {
      operand_states_.insert({{op, index}, cached.first->second});
      return;
    }
    cached.first->second = InitializeState(op, index, in, /*as_result=*/false);
  }

  // Sets the state of the index-th result of the op. If this result is cached,
  // uses the cached result without creating new entry in the state vector.
  // Otherwise, allocate a new entry in the state vector.
  void InitializeResultState(Operation *op, int index, Value res,
                             llvm::DenseMap<Value, int> *cache) {
    auto cached = cache->insert({res, 0});
    if (!cached.second) {
      result_states_.insert({{op, index}, cached.first->second});
      return;
    }
    cached.first->second = InitializeState(op, index, res, /*as_result=*/true);
  }

  FuncOp fn_;
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
  std::unordered_map<int, RequantizeState> rescale_states_;

  // Maps of indexes to the propagation state vector from the ops operands,
  // results and arguments.
  llvm::DenseMap<OpValue, int> operand_states_;
  llvm::DenseMap<OpValue, int> result_states_;
  llvm::DenseMap<BlockArgument, int> arg_states_;

  // This vector is to preserve the arguments order, so the newly inserted
  // quantized ops for the arguments are deterministically ordered.
  llvm::SmallVector<BlockArgument, 4> args_;

  OpQuantSpecGetter op_quant_spec_getter_;
};
}  // namespace

std::unique_ptr<OpQuantSpec> QuantizationDriver::GetQuantSpec(Operation *op) {
  return op_quant_spec_getter_(op);
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
      quant::QuantizedType::getQuantizedElementType(val->getType());
  bool immutable = !EmptyParams(params);
  int next_state_index = states_.size();
  states_.push_back({params, immutable});
  if (as_result)
    result_states_.insert({{op, index}, next_state_index});
  else
    operand_states_.insert({{op, index}, next_state_index});

  return next_state_index;
}

bool QuantizationDriver::SetConstantResultParams(Operation *op) {
  ElementsAttr attr;
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
        /*narrow_range=*/true);
  } else {
    // per-tensor quantization weight
    final_type = GetUniformQuantizedTypeForWeight(
        attr, /*symmetric=*/is_weight && is_signed_,
        /*num_bits=*/8, is_signed_,
        /*narrow_range_=*/is_weight);
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
    auto &rescale = GetResultRequantizeState(op, res_index);
    rescale.params = params;
    rescale.pos = RequantizeState::ON_INPUT;
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
  return func(op_types);
}

bool QuantizationDriver::SetOperandParams(Operation *op, int index,
                                          QuantParams params) {
  auto &state = GetOperandQuantState(op, index);
  if (state.params == params) {
    return false;
  }

  if (!state.IsEmpty()) {
    auto &rescale = GetOperandRequantizeState(op, index);
    rescale.params = params;
    rescale.pos = RequantizeState::ON_OUTPUT;
    return true;
  }

  state.params = params;
  AddOperandToList(op, index);
  return true;
}

void QuantizationDriver::QuantizeOpResult(Operation *op, int index,
                                          QuantParams params) {
  builder_.setInsertionPoint(op->getBlock(), ++Block::iterator(op));
  Value original_result = op->getResult(index);
  QuantizeValue(original_result, params, op->getLoc());
}

void QuantizationDriver::QuantizeArg(BlockArgument arg, QuantParams params) {
  builder_.setInsertionPointToStart(arg->getOwner());
  QuantizeValue(arg, params, builder_.getUnknownLoc());
}

void QuantizationDriver::QuantizeValue(Value value, QuantParams params,
                                       Location loc) {
  Type expressed_type = value->getType();
  Type new_type = params.castFromExpressedType(expressed_type);
  // This value isn't an expressed type (float), skip.
  if (!new_type) return;

  TypeAttr type_attr = TypeAttr::get(new_type);
  auto quantize =
      builder_.create<TFL::QuantizeOp>(loc, new_type, value, type_attr);
  auto dequantize = builder_.create<TFL::DequantizeOp>(loc, expressed_type,
                                                       quantize.output());
  // `original_result` has a use to `quantize`, so this will replace that use
  // by the result of `dequantize`. Remember to reset that use afterwards
  value->replaceAllUsesWith(dequantize);
  quantize.getOperation()->replaceUsesOfWith(dequantize, value);
}

void QuantizationDriver::RequantizeOpResult(Operation *op, int index,
                                            RequantizeState *state) {
  if (state->pos == RequantizeState::NO_REQUANTIZE) return;
  builder_.setInsertionPointAfter(op);
  Value value = op->getResult(index);
  if (state->pos == RequantizeState::ON_OUTPUT) {
    Operation *user = value->getUses().begin().getUser();
    if (llvm::isa<TFL::QuantizeOp>(user)) {
      // The requantize op is inserted between `quantize` and `dequantize` ops.
      value = user->getResult(0);
      builder_.setInsertionPointAfter(user);
    }
  }
  RequantizeValue(value, state, op->getLoc());
}

void QuantizationDriver::RequantizeArg(BlockArgument arg,
                                       RequantizeState *state) {
  Value value = arg;
  builder_.setInsertionPointToStart(arg->getOwner());
  if (value->hasOneUse()) {
    auto user = value->use_begin().getUser();
    if (auto q = llvm::dyn_cast<TFL::QuantizeOp>(user)) {
      value = q.output();
      builder_.setInsertionPoint(arg->getOwner(), ++Block::iterator(user));
    }
  }
  RequantizeValue(value, state, builder_.getUnknownLoc());
}

void QuantizationDriver::RequantizeValue(Value value, RequantizeState *state,
                                         Location loc) {
  Type new_type;
  if (state->pos == RequantizeState::ON_INPUT) {
    Type expressed_type = value->getType();
    // The value needs to be requantized. A Quantize op will be created to use
    // it as the operand and replace its uses.
    new_type = state->params.castFromExpressedType(expressed_type);
  } else {
    Type expressed_type =
        quant::QuantizedType::castToExpressedType(value->getType());
    if (!expressed_type) return;

    // The value needs to be requantized. A Quantize op will be created to use
    // it as the operand and replace its uses.
    new_type = state->params.castFromExpressedType(expressed_type);
  }
  // This value isn't an expressed type (float), skip.
  if (!new_type) return;

  TypeAttr type_attr = TypeAttr::get(new_type);
  auto requantize_op =
      builder_.create<TFL::QuantizeOp>(loc, new_type, value, type_attr);
  value->replaceAllUsesWith(requantize_op);
  requantize_op.getOperation()->replaceUsesOfWith(requantize_op, value);
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
  fn_.walk([&](ConstantOp cst) {
    // Non-float tensors are neither weights nor require quantization.
    auto type = cst.getType().dyn_cast<ShapedType>();
    if (!type || !type.getElementType().isa<FloatType>()) return;

    Value value = cst.getResult();
    SmallVector<std::pair<Operation *, int>, 4> bias_users;
    bool used_as_weight = false;
    for (auto &use : value->getUses()) {
      auto spec = GetQuantSpec(use.getOwner());
      auto biases = spec->biases_params;
      Operation *user = use.getOwner();
      int operand_num = use.getOperandNumber();

      // The user doesn't use this value as a bias operand or require same
      // scale, then this constant is considered to be a weight.
      if (biases.find(operand_num) == biases.end() &&
          !user->hasTrait<OpTrait::quant::SameOperandsAndResultsScale>()) {
        used_as_weight = true;
        auto it = spec->coeff_op_quant_dim.find(operand_num);
        if (it != spec->coeff_op_quant_dim.end()) {
          optimized_weights_.insert({cst, it->second});
        }
      } else {
        bias_users.push_back({user, operand_num});
      }
    }

    // If the constant is used as a weight, this constant will be duplicated
    // for each bias user, so it isn't shared with the weight usage.
    // Otherwise, the first bias user can use the original constant and the
    // rest use the duplications, so we pop bias user from the set.
    if (used_as_weight) {
      // TODO(fengliuai): Looks like there is an assumption that weight has
      // only one user. We should add a check here.
      weights_.insert(cst);
    } else {
      bias_users.pop_back();
      builder_.setInsertionPoint(cst);
    }
    for (auto bias_user : bias_users) {
      auto copied = builder_.create<ConstantOp>(cst.getLoc(), cst.getValue());
      bias_user.first->setOperand(bias_user.second, copied.getResult());
    }
  });
}

void QuantizationDriver::SetupAllStates() {
  llvm::DenseMap<Value, int> value_to_state;

  for (auto arg : fn_.getArguments()) {
    args_.push_back(arg);
    Value value = arg;
    // If the argument is quantized, it should only has one user.
    if (arg->hasOneUse()) {
      auto user = value->use_begin().getUser();
      if (auto q = llvm::dyn_cast<TFL::QuantizeOp>(user)) {
        value = q.output();
      }
    }
    InitializeArgState(arg, value, &value_to_state);
  }

  fn_.walk([&](Operation *op) {
    if (op->isKnownTerminator() ||
        op->hasTrait<OpTrait::quant::NoQuantizableResult>())
      return;
    work_list_.push_back(op);

    for (int i = 0, e = op->getNumOperands(); i != e; ++i) {
      auto operand = op->getOperand(i);
      if (auto *inst = operand->getDefiningOp()) {
        // If the operand comes from a tfl.dequantize op, we use the quantized
        // input of this tfl.dequantize op to set the state.
        if (auto dq = llvm::dyn_cast<TFL::DequantizeOp>(inst)) {
          operand = dq.input();
        }
      }
      InitializeOperandState(op, i, operand, &value_to_state);
    }

    for (int res = 0, e = op->getNumResults(); res != e; ++res) {
      auto result = op->getResult(res);
      // If the result has been quantized, it should only be used by a
      // tfl.quantize op. For this case, we uses the quantized result to
      // create the state and mark it immutable.
      if (result->hasOneUse()) {
        auto user = result->use_begin().getUser();
        if (auto q = llvm::dyn_cast<TFL::QuantizeOp>(user)) {
          result = q.output();
        }
      }
      InitializeResultState(op, res, result, &value_to_state);
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

    // This op has been quantized, so we should not consider it again.
    if (llvm::is_contained(quantized_, op)) continue;
    quantized_.insert(op);

    if (auto cst = llvm::dyn_cast<ConstantOp>(op)) {
      // If it isn't a weight or has been quantized, skip.
      if (!IsWeight(cst) || IsQuantized(op)) continue;

      // The quantization parameters are determined by the content of the
      // constant.
      changed |= SetConstantResultParams(op);
      continue;
    }

    if (op->hasTrait<OpTrait::quant::SameOperandsAndResultsScale>()) {
      auto params = GetQuantParamsForSameScaleConstraint(op);
      // The quantization parameters haven't been propagated to any operands
      // or results. Skip this node for now.
      if (!params) {
        quantized_.erase(op);
        continue;
      }

      // Use the final state to set all the operands' parameters.
      for (int i = 0, e = op->getNumOperands(); i != e; ++i)
        changed |= SetOperandParams(op, i, params);

      // Use the final state to set all the results' parameters.
      for (int res = 0, e = op->getNumResults(); res != e; ++res)
        changed |= SetResultParams(op, res, params);
    }

    // TODO(fengliuai): make the bit width configurable.
    auto spec = GetQuantSpec(op);
    auto key = std::make_pair(8, is_signed_);
    auto &restricted_outputs = spec->restricted_output_params[key];
    for (int i = 0, e = restricted_outputs.size(); i != e; ++i) {
      // The restrict can be nullptr if the result has been quantized.
      if (auto params = restricted_outputs[i]) {
        changed |= SetResultParams(op, i, params);
      }
    }

    for (auto &it : spec->biases_params) {
      auto params =
          GetBiasParams(op, it.first, it.second.first, it.second.second);
      if (!params) {
        quantized_.erase(op);
        continue;
      }
      changed |= SetOperandParams(op, it.first, params);
    }
  }
  return changed;
}

void QuantizationDriver::Finalize() {
  for (auto arg : args_) {
    auto &state = GetArgQuantState(arg);
    auto &requantize = GetArgRequantizeState(arg);
    if (state.IsEmpty() ||
        (state.immutable && requantize.pos == RequantizeState::NO_REQUANTIZE)) {
      continue;
    }

    if (!state.immutable) {
      QuantizeArg(arg, state.params);
    }

    if (requantize.pos != RequantizeState::NO_REQUANTIZE) {
      RequantizeArg(arg, &requantize);
    }
  }

  for (auto it : result_states_) {
    Operation *op = it.first.first;
    int res_index = it.first.second;
    auto &state = GetResultQuantState(op, res_index);
    auto &requantize = GetResultRequantizeState(op, res_index);
    if (state.IsEmpty() ||
        (state.immutable && requantize.pos == RequantizeState::NO_REQUANTIZE)) {
      continue;
    }

    if (!state.immutable) {
      QuantizeOpResult(op, res_index, state.params);
    }

    if (requantize.pos != RequantizeState::NO_REQUANTIZE) {
      RequantizeOpResult(op, res_index, &requantize);
    }
  }
}

void QuantizationDriver::Run() {
  Initialize();
  if (PropagateParams()) {
    Finalize();
  }
}

void ApplyQuantizationParamsPropagation(
    mlir::FuncOp func, bool is_signed, bool disable_per_channel,
    OpQuantSpecGetter op_quant_spec_getter) {
  QuantizationDriver(func, is_signed, disable_per_channel, op_quant_spec_getter)
      .Run();
}

}  // namespace TFL
}  // namespace mlir
