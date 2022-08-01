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

// This file implements logic for lowering TensorFlow dialect's communication
// ops (TF/XLA) to the HLO dialect.

#include <memory>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/tf_xla_passes_detail.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/side_effect_util.h"

namespace mlir {

using func::FuncOp;

namespace mhlo {

namespace {
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";
// TPU core that sends to and receives from host.
constexpr int64_t kShardingTpuCore = 0;

// A pass that legalizes TF/XLA communication ops, propagate their respective
// tokens (for ordering), and rewrite their respective functions and control
// flow ops when necessary.
// Note, this currently does not handle nested modules/functions or region based
// ops other than certain control flow ops (`mhlo.if`, `mhlo.while`).
class LegalizeTFCommunication
    : public LegalizeTFCommunicationPassBase<LegalizeTFCommunication> {
  void runOnOperation() override;
};

// Checks if an op is a TF/XLA communication op.
bool IsCommunicationOp(Operation* op) {
  return isa<TF::_XlaHostComputeMlirOp, TF::XlaSendToHostOp,
             TF::XlaRecvFromHostOp>(op);
}

// Checks if an op is a supported HLO control flow op.
bool IsControlFlowOp(Operation* op) { return isa<IfOp, WhileOp>(op); }

// Collects control flow op ancestors of a given op, up until FuncOp. If any
// ancestor is not a control flow op or a FuncOp, or of a single block region,
// an error will be returned.
LogicalResult GetControlFlowAncestors(
    Operation* op, llvm::SmallPtrSetImpl<Operation*>& control_flow_ops,
    llvm::SmallPtrSetImpl<Block*>& control_flow_blocks) {
  Block* block = op->getBlock();
  Operation* parent = block->getParentOp();
  while (block && parent && !isa<func::FuncOp>(parent)) {
    if (!IsControlFlowOp(parent))
      return op->emitOpError()
             << "expects ancestor(s) to be of ['" << IfOp::getOperationName()
             << "', '" << func::FuncOp::getOperationName() << "']";

    if (!llvm::hasSingleElement(block->getParent()->getBlocks()))
      return op->emitOpError() << "expects single block region ancestor(s)";

    control_flow_ops.insert(parent);
    control_flow_blocks.insert(block);

    parent = block->getParentOp();
    block = parent->getBlock();
  }
  return success();
}

// Finds communication ops in a function. `control_flow_ops` and
// `control_flow_blocks` will be populated with control flow op ancestors for
// every communication op.
LogicalResult FindCommunicationOps(
    func::FuncOp func, llvm::SmallPtrSetImpl<Operation*>& control_flow_ops,
    llvm::SmallPtrSetImpl<Block*>& control_flow_blocks,
    bool& has_communication_ops) {
  auto result = func.walk([&](Operation* op) {
    if (!IsCommunicationOp(op)) return WalkResult::advance();
    has_communication_ops = true;
    if (failed(
            GetControlFlowAncestors(op, control_flow_ops, control_flow_blocks)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

// Helper struct holding a function to be rewritten, it's control flow ops that
// lead to a communication op or function call with a communication op
// (transitively), and an optional clone of itself. If `clone` is set, function
// calls to `original` will be replaced with `clone`.
struct FuncToRewrite {
  func::FuncOp original;
  llvm::SmallPtrSet<Operation*, 4> control_flow_ops;
  llvm::SmallPtrSet<Block*, 4> control_flow_blocks;
  func::FuncOp clone;
};

// Finds all functions that need to be rewritten with communication ops and
// and associated tokens.
LogicalResult GetFunctionsToRewrite(
    ModuleOp module,
    llvm::SmallDenseMap<StringRef, FuncToRewrite>& funcs_to_rewrite) {
  // Find functions containing communication ops.
  SmallVector<func::FuncOp, 4> funcs_to_visit;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    FuncToRewrite func_to_rewrite{/*original=*/func, /*control_flow_ops=*/{},
                                  /*control_flow_blocks=*/{},
                                  /*clone=*/nullptr};
    bool has_communication_ops = false;
    if (failed(FindCommunicationOps(func, func_to_rewrite.control_flow_ops,
                                    func_to_rewrite.control_flow_blocks,
                                    has_communication_ops)))
      return failure();

    if (!has_communication_ops) continue;
    funcs_to_rewrite.insert({func.getName(), func_to_rewrite});
    funcs_to_visit.push_back(func);
  }

  // Find functions that call functions with communication ops, transitively.
  while (!funcs_to_visit.empty()) {
    SmallVector<func::FuncOp, 4> new_funcs_to_visit;
    for (func::FuncOp& func : funcs_to_visit) {
      auto uses = func.getSymbolUses(module);
      if (!uses) continue;
      for (auto& use : *uses) {
        // Only `mlir::func::CallOp` is supported as this requires knowing how
        // to rewrite arguments and results to a function.
        if (!isa<mlir::func::CallOp>(use.getUser())) continue;
        auto caller_parent_func =
            use.getUser()->getParentOfType<func::FuncOp>();
        if (!caller_parent_func) continue;

        FuncToRewrite func_to_rewrite{/*original=*/caller_parent_func,
                                      /*control_flow_ops=*/{},
                                      /*control_flow_blocks=*/{},
                                      /*clone=*/nullptr};
        if (failed(GetControlFlowAncestors(
                use.getUser(), func_to_rewrite.control_flow_ops,
                func_to_rewrite.control_flow_blocks)))
          return failure();

        auto it = funcs_to_rewrite.insert(
            {caller_parent_func.getName(), func_to_rewrite});
        if (it.second) {
          new_funcs_to_visit.push_back(caller_parent_func);
        } else {
          it.first->getSecond().control_flow_ops.insert(
              func_to_rewrite.control_flow_ops.begin(),
              func_to_rewrite.control_flow_ops.end());
          it.first->getSecond().control_flow_blocks.insert(
              func_to_rewrite.control_flow_blocks.begin(),
              func_to_rewrite.control_flow_blocks.end());
        }
      }
    }

    funcs_to_visit.swap(new_funcs_to_visit);
  }

  // Clone public functions that need to be rewritten. Function calls to this
  // function will be replaced with the cloned function.
  SymbolTable symbol_table(module);
  for (auto& func : funcs_to_rewrite) {
    if (func.getSecond().original.isPublic() &&
        !func.getSecond().original.symbolKnownUseEmpty(module)) {
      auto clone = func.getSecond().original.clone();
      clone.setPrivate();
      symbol_table.insert(clone);
      func.getSecond().clone = clone;
    }
  }

  return success();
}

// Assigns op sharding to full tensor on `kShardingTpuCore`.
void SetOpSharding(Operation* op) {
  std::string sharding_serialized =
      ::xla::sharding_builder::AssignDevice(kShardingTpuCore)
          .SerializeAsString();
  op->setAttr(kShardingAttr,
              StringAttr::get(op->getContext(), sharding_serialized));
}

// Assigns frontend attributes holding information about data type and
// TensorFlow rendezvous channel name. The TensorFlow rendezvous channel name is
// handled differently as individual names are used per data send and receive.
void SetFrontendAttributes(Operation* op, int32_t index, StringRef key,
                           Type type, bool device_to_host,
                           StringRef host_handler_name) {
  MLIRContext* context = op->getContext();

  std::string formatted_key =
      device_to_host ? llvm::formatv("{0}_dtoh_{1}", key, index).str()
                     : llvm::formatv("{0}_htod_{1}", key, index).str();

  auto rendezvous_name = StringAttr::get(context, formatted_key);
  auto rendezvous_name_attr = NamedAttribute(
      StringAttr::get(context, xla::kXlaHostTransferRendezvousNameAttr),
      rendezvous_name);

  auto element_type = getElementTypeOrSelf(type);
  auto xla_element_type = ::xla::TypeToPrimitiveType(element_type);
  const std::string& xla_element_type_str =
      ::xla::primitive_util::LowercasePrimitiveTypeName(xla_element_type);
  auto original_type = StringAttr::get(context, xla_element_type_str);
  auto original_type_attr = NamedAttribute(
      StringAttr::get(context, xla::kXlaHostTransferOriginalTypeAttr),
      original_type);

  auto host_handler_name_value =
      StringAttr::get(context, host_handler_name.str());
  auto host_handler_name_attr = NamedAttribute(
      StringAttr::get(context, xla::kXlaHostTransferHandlerNameAttr),
      host_handler_name_value);

  auto frontend_attributes = DictionaryAttr::get(
      context,
      ArrayRef<NamedAttribute>{rendezvous_name_attr, original_type_attr,
                               host_handler_name_attr});
  op->setAttr(kFrontendAttributesAttr, frontend_attributes);
}

// Creates a `mhlo.send` op for sending value `operand`.
Value CreateSendOp(OpBuilder& builder, int64_t& channel_id, Location loc,
                   Value operand, StringRef key, size_t index, Value token,
                   StringRef host_handler_name) {
  // type 2 == DEVICE_TO_HOST
  auto channel_handle = ChannelHandleAttr::get(builder.getContext(),
                                               /*handle=*/channel_id++,
                                               /*type=*/2);
  auto send = builder.create<SendOp>(
      loc, token.getType(), operand, token, channel_handle,
      /*is_host_transfer=*/builder.getBoolAttr(true));

  SetFrontendAttributes(send, index, key, operand.getType(),
                        /*device_to_host=*/true, host_handler_name);

  SetOpSharding(send);

  return send.getResult();
}

// Creates a `mhlo.recv` op for receiving a value.
Value CreateRecvOp(OpBuilder& builder, int64_t& channel_id, Location loc,
                   Value result, StringRef key, size_t index, Value token,
                   StringRef host_handler_name) {
  // type 3 == HOST_TO_DEVICE
  auto channel_handle = ChannelHandleAttr::get(builder.getContext(),
                                               /*handle=*/channel_id++,
                                               /*type=*/3);
  auto result_type = result.getType();
  SmallVector<Type, 2> recv_result_type = {result_type, token.getType()};
  auto recv =
      builder.create<RecvOp>(loc, recv_result_type, token, channel_handle,
                             /*is_host_transfer=*/builder.getBoolAttr(true));

  SetFrontendAttributes(recv, index, key, result_type,
                        /*device_to_host=*/false, host_handler_name);

  SetOpSharding(recv);

  result.replaceAllUsesWith(recv.getResult(0));

  return recv.getResult(1);
}

// Creates a new token if necessary, acting as a sink to previous tokens. If
// there is only one token in `tokens`, the only token is returned. If `tokens`
// is empty, `original_token` is returned instead.
Value CreateSinkToken(OpBuilder& builder, Location loc, ArrayRef<Value> tokens,
                      Value original_token) {
  if (tokens.empty()) {
    return original_token;
  } else if (llvm::hasSingleElement(tokens)) {
    return tokens[0];
  } else {
    return builder.create<AfterAllOp>(loc, original_token.getType(), tokens)
        .getResult();
  }
}

// Replaces `tf._XlaHostComputeMlir` with individual `mhlo.send` and `mhlo.recv`
// ops per operand and result. Unique Channel IDs are assigned per transfer.
// Sink tokens are created across all `mhlo.send` ops first and then by
// all `mhlo.recv` ops.
Value RewriteHostComputeOp(OpBuilder& builder, int64_t& channel_id,
                           TF::_XlaHostComputeMlirOp host_compute,
                           Value token) {
  builder.setInsertionPoint(host_compute);
  Location loc = host_compute.getLoc();

  SmallVector<Value, 4> send_tokens;
  for (auto operand : llvm::enumerate(host_compute.inputs())) {
    auto send_token = CreateSendOp(
        builder, channel_id, loc, operand.value(), host_compute.send_key(),
        operand.index(), token, xla::kXlaHostTransferTfRendezvousHandlerName);
    send_tokens.push_back(send_token);
  }
  token = CreateSinkToken(builder, loc, send_tokens, token);

  SmallVector<Value, 4> recv_tokens;
  for (auto result : llvm::enumerate(host_compute.outputs())) {
    auto recv_token = CreateRecvOp(
        builder, channel_id, loc, result.value(), host_compute.recv_key(),
        result.index(), token, xla::kXlaHostTransferTfRendezvousHandlerName);
    recv_tokens.push_back(recv_token);
  }
  token = CreateSinkToken(builder, loc, recv_tokens, token);

  host_compute.erase();
  return token;
}

// Replaces `tf.XlaSendToHost` with a `mhlo.send`.
Value RewriteSendToHostOp(OpBuilder& builder, int64_t& channel_id,
                          TF::XlaSendToHostOp send_to_host, Value token) {
  builder.setInsertionPoint(send_to_host);
  token = CreateSendOp(builder, channel_id, send_to_host.getLoc(),
                       send_to_host.input(), send_to_host.key(),
                       /*index=*/0, token,
                       xla::kXlaHostTransferTfRendezvousHandlerName);

  send_to_host.erase();
  return token;
}

// Replaces `tf.XlaRecvFromHost` with a `mhlo.recv`.
Value RewriteRecvFromHostOp(OpBuilder& builder, int64_t& channel_id,
                            TF::XlaRecvFromHostOp recv_from_host, Value token) {
  builder.setInsertionPoint(recv_from_host);
  token = CreateRecvOp(builder, channel_id, recv_from_host.getLoc(),
                       recv_from_host.output(), recv_from_host.key(),
                       /*index=*/0, token,
                       xla::kXlaHostTransferTfRendezvousHandlerName);

  recv_from_host.erase();
  return token;
}

// Replaces a `mlir::func::CallOp` with one that has an extra `!mhlo.token`
// operand and `!mhlo.token` result. If `new_symbol` is set, the new call will
// be updated to call the `new_symbol` instead.
Value RewriteCallOp(OpBuilder& builder, func::CallOp call,
                    const Optional<StringRef>& new_symbol, Value token) {
  builder.setInsertionPoint(call);
  auto new_operands = llvm::to_vector(call.getArgOperands());
  new_operands.push_back(token);
  auto new_result_types = llvm::to_vector(call.getResultTypes());
  new_result_types.push_back(token.getType());
  auto new_call = builder.create<func::CallOp>(
      call.getLoc(), new_result_types,
      new_symbol ? *new_symbol : call.getCallee(), new_operands);

  for (auto results : llvm::zip(call.getResults(), new_call.getResults()))
    std::get<0>(results).replaceAllUsesWith(std::get<1>(results));
  call.erase();
  return new_call.getResults().back();
}

// Helper struct holding state of which op to visit to next. If `op` is in a
// control flow op region, `region_idx` will be set with the respective region
// index. `token` will be current token from the last communication op/control
// flow op transitive communication ops.
struct OpVisitorState {
  Optional<unsigned> region_idx;
  Value token;
  Operation* op;
};

// Creates a tuple from a sequence of values.
Value CreateTuple(OpBuilder& builder, Location loc, ArrayRef<Value> operands) {
  return builder.create<TupleOp>(loc, operands).getResult();
}

// Extends `values` with the value `token` attached. If `flatten_tuple` is
// false, `values` will have a single element, say `value`. If `value` is not a
// tuple, a new tuple is formed with `token`. If `values` is a tuple, it is
// extended instead. New tuple values created are cached.
SmallVector<Value> GetValueWithToken(
    OpBuilder& builder, ArrayRef<Value> values, Value token,
    llvm::SmallDenseMap<Value, Value>& rewritten_values, bool flatten_tuple) {
  if (flatten_tuple) {
    auto operands = llvm::to_vector(values);
    operands.push_back(token);
    return operands;
  }

  auto value = values[0];
  // If value with token already exists, reuse it.
  auto it = rewritten_values.find(value);
  if (it != rewritten_values.end()) return {it->getSecond()};

  auto create_tuple = [&](ArrayRef<Value> operands) {
    auto new_result = CreateTuple(builder, value.getLoc(), operands);
    rewritten_values.insert({value, new_result});
    return new_result;
  };

  auto tuple_type = value.getType().dyn_cast<TupleType>();
  // `value` is not a tuple, create a new tuple.
  if (!tuple_type) return {create_tuple({value, token})};

  // Extend tuple if `value` is a tuple.
  // If `value` is an op result and the owner is a `mhlo.tuple`, simply unpack
  // the tuple.
  if (auto tuple_op = value.getDefiningOp<TupleOp>()) {
    auto tuple_operands = llvm::to_vector(tuple_op.getOperands());
    tuple_operands.push_back(token);
    return {create_tuple(tuple_operands)};
  }

  // `value` is not created via a `mhlo.tuple` directly, unpack individual
  // elements directly with `mhlo.get_tuple_element`.
  SmallVector<Value, 4> tuple_operands;
  for (auto idx : llvm::seq<int32_t>(0, tuple_type.getTypes().size()))
    tuple_operands.push_back(
        builder.create<GetTupleElementOp>(value.getLoc(), value, idx)
            .getResult());

  tuple_operands.push_back(token);
  return {create_tuple(tuple_operands)};
}

// Extends the 'types' to include a `mhlo.token` type. If `flatten_tuple` is
// false, `types` will have a single element, say `type`. If `type` is not a
// tuple type, a new tuple type with `type` and `mhlo.token` type is created
// instead.
SmallVector<Type> GetTypeWithToken(OpBuilder& builder, ArrayRef<Type> types,
                                   bool flatten_tuple) {
  SmallVector<Type> new_result_types;
  auto token_type = TokenType::get(builder.getContext());

  if (flatten_tuple) {
    auto result_types = llvm::to_vector(types);
    result_types.push_back(token_type);
    return result_types;
  }

  auto type = types[0];
  if (auto tuple_type = type.dyn_cast<TupleType>()) {
    auto result_types = llvm::to_vector(tuple_type.getTypes());
    result_types.push_back(token_type);
    return {builder.getTupleType(result_types)};
  }

  return {builder.getTupleType({type, token_type})};
}

// Creates a slice of a tuple `value` with `mhlo.get_tuple_element` from index 0
// to `end`, exclusive.
Value CreateSubTuple(OpBuilder& builder, Value value, size_t end) {
  SmallVector<Value, 4> tuple_operands;
  for (auto idx : llvm::seq<int32_t>(0, end))
    tuple_operands.push_back(
        builder.create<GetTupleElementOp>(value.getLoc(), value, idx)
            .getResult());

  return CreateTuple(builder, value.getLoc(), tuple_operands);
}

// Replaces uses of `values` with `replacements`. If `flatten_tuple` is false,
// `values` will have a single element, say `value`. If `value` is not a tuple
// type, an explicit `mhlo.get_tuple_element` is created to unpack the tuple and
// return the first element. Otherwise, `mhlo.get_tuple_element` users are
// simply updated with `replacement`, and all other users are updated with a
// slice of `replacement`.
void ReplaceWithTupleResult(OpBuilder& builder, ArrayRef<Value> values,
                            ArrayRef<Value> replacements, bool flatten_tuple) {
  if (flatten_tuple) {
    for (size_t result_index = 0; result_index < values.size(); result_index++)
      values[result_index].replaceAllUsesWith(replacements[result_index]);
    return;
  }

  auto value = values[0];
  auto replacement = replacements[0];
  auto tuple_type = value.getType().dyn_cast<TupleType>();
  if (!tuple_type) {
    if (!value.use_empty()) {
      auto new_element = builder.create<GetTupleElementOp>(replacement.getLoc(),
                                                           replacement, 0);
      value.replaceAllUsesWith(new_element.getResult());
    }
    return;
  }

  Value sub_tuple;
  for (auto& use : llvm::make_early_inc_range(value.getUses())) {
    if (isa<GetTupleElementOp>(use.getOwner())) {
      use.set(replacement);
      continue;
    }

    if (!sub_tuple)
      sub_tuple = CreateSubTuple(builder, replacement, tuple_type.size());

    use.set(sub_tuple);
  }
}

// Replaces control flow op block arguments with new block arguments
// of types `types`. The last element of the new block argument (token) is
// returned.
Value UpdateControlFlowBlockArgWithToken(OpBuilder& builder, Block& block,
                                         ArrayRef<Type> types) {
  builder.setInsertionPointToStart(&block);

  auto old_args_size = block.getNumArguments();

  block.addArguments(
      types, SmallVector<Location>(types.size(), block.getParent()->getLoc()));

  auto old_args = ArrayRef<Value>(block.getArguments().begin(),
                                  block.getArguments().begin() + old_args_size);
  auto new_args = ArrayRef<Value>(block.getArguments().begin() + old_args_size,
                                  block.getArguments().end());
  assert(!new_args.empty());

  ReplaceWithTupleResult(builder, old_args, new_args, /*flatten_tuple=*/true);
  auto new_arg = new_args[new_args.size() - 1];

  block.eraseArguments(
      llvm::to_vector(llvm::seq((unsigned)0, (unsigned)old_args_size)));

  return new_arg;
}

// Updates control flow op terminator with an extra element `token`.
void RewriteControlFlowTerminator(OpBuilder& builder, Operation* terminator,
                                  Value token, bool flatten_tuple) {
  assert(flatten_tuple || terminator->getNumOperands() == 1);
  assert(flatten_tuple || terminator->getBlock()->getNumArguments() == 1);
  // `mhlo.while` cond terminator does not need to be rewritten as it always
  // returns a tensor<i1> predicate value.
  if (auto while_parent = dyn_cast_or_null<WhileOp>(terminator->getParentOp()))
    if (terminator->getParentRegion() == &while_parent.cond()) return;

  builder.setInsertionPoint(terminator);
  llvm::SmallDenseMap<Value, Value> rewritten_operands;
  auto new_results =
      GetValueWithToken(builder, llvm::to_vector(terminator->getOperands()),
                        token, rewritten_operands, flatten_tuple);
  terminator->setOperands(new_results);
}

// Rewrites a `mhlo.if` op to receive and forward a `mhlo.token`. As If op does
// not have any operands other than the predicate, hence we implicitly capture
// the parent token. Also we use the same implicit token for use in the If op's
// regions.
void RewriteRegionIfOp(OpBuilder& builder, IfOp region_if,
                       SmallVectorImpl<OpVisitorState>& ops_to_visit,
                       Value token) {
  llvm::SmallDenseMap<Value, Value> rewritten_operands;

  auto new_result_types =
      GetTypeWithToken(builder, llvm::to_vector(region_if.getResultTypes()),
                       /*flatten_tuple=*/true);

  // Create new `mhlo.if` op with extra token operands and result.
  auto new_if = builder.create<IfOp>(region_if.getLoc(), new_result_types,
                                     region_if.pred());

  // Move all regions from the old `mhlo.if` op to its replacement.
  new_if.true_branch().takeBody(region_if.true_branch());
  new_if.false_branch().takeBody(region_if.false_branch());

  // Forward result from old `mhlo.if` with replacement.
  SmallVector<Value> old_if_results = region_if.getResults();
  SmallVector<Value> new_if_results = new_if.getResults();

  ReplaceWithTupleResult(builder, old_if_results, new_if_results,
                         /*flatten_tuple=*/true);

  // auto new_token = new_if_results[new_if_results.size() - 1];

  region_if.erase();

  // Next op to visit. The replacement is visited but at its first region.
  // The new region use the same implicit token used by the If op.
  ops_to_visit.push_back({/*region_idx=*/0, token, new_if});
}

// Rewrites a `mhlo.if`/`mhlo.while` region to receive and forward a
// `mhlo.token`. The block argument is updated to have an extra `mhlo.token`
// element. If the region block is to be rewritten, the next op to visit is set
// to the first op in the block. Otherwise the terminator is updated to forward
// `token`.
void RewriteControlFlowOpRegion(
    OpBuilder& builder, Operation* region_op, unsigned region_idx,
    ArrayRef<Type> block_arg_types,
    SmallVectorImpl<OpVisitorState>& ops_to_visit,
    const llvm::SmallPtrSetImpl<Block*>& control_flow_blocks, Value token) {
  ops_to_visit.push_back({region_idx + 1, token, region_op});

  Region& region = region_op->getRegion(region_idx);
  assert(llvm::hasSingleElement(region));

  auto block_token = UpdateControlFlowBlockArgWithToken(builder, region.front(),
                                                        block_arg_types);

  if (control_flow_blocks.contains(&region.front())) {
      ops_to_visit.push_back(
          {/*region_idx=*/llvm::None, block_token, &region.front().front()});
    return;
  }

  RewriteControlFlowTerminator(builder, region.front().getTerminator(),
                               block_token, /*flatten_tuple=*/true);
}

// For mlir::IfOp or mlir::CaseOp, replace the use of their region's block
// argument (of type token) with 'implicit_operand'.
void ReplaceBlockArgumentsWithImplicitOperands(mlir::Operation* op,
                                               unsigned region_idx,
                                               Value implicit_operand) {
  assert((mlir::dyn_cast<mlir::mhlo::IfOp>(*op) ||
          mlir::dyn_cast<mlir::mhlo::CaseOp>(*op)) &&
         "Unexpected mlir op in "
         "HloFunctionImporter::ReplaceBlockArgumentsWithImplicitOperands!");

  auto& region = op->getRegion(region_idx);
  region.getArgument(0).replaceAllUsesWith(implicit_operand);
  region.front().eraseArguments(
      llvm::to_vector(llvm::seq<unsigned>(0, region.getNumArguments())));
}

// Rewrites an `mhlo.if` op or its region. If `region_idx` is not set, the op
// operands and results are rewritten. If `region_idx` is set, region
// `region_idx` is rewritten to take in and return an additional token. Returns
// true if the op or its region was rewritten.
bool ProcessRegionIfOp(OpBuilder& builder, IfOp region_if,
                       Optional<unsigned> region_idx,
                       SmallVectorImpl<OpVisitorState>& ops_to_visit,
                       const llvm::SmallPtrSetImpl<Block*>& control_flow_blocks,
                       Value token) {
  builder.setInsertionPoint(region_if);

  if (!region_idx) {
    RewriteRegionIfOp(builder, region_if, ops_to_visit, token);
    return true;
  }

  if (*region_idx < region_if.getNumRegions()) {
    // For the region-blocks of If op, we create a dummy token argument. Later
    // we replace that block-argument's uses with the same (implicitly captured)
    // token 'token', used for If op, and erase the argument.
    // Note that 'RewriteControlFlowOpRegion' sets the token, used for the first
    // operation of region_idx'th region, to the dummy block-argument. As we
    // erase that argument, we also need to make sure that the token used for
    // the next operation is set to 'token'.
    RewriteControlFlowOpRegion(builder, region_if, *region_idx,
                               {token.getType()}, ops_to_visit,
                               control_flow_blocks, token);

    ReplaceBlockArgumentsWithImplicitOperands(region_if.getOperation(),
                                              *region_idx, token);

    auto next_visitor_state = ops_to_visit.back();
    next_visitor_state.token = token;
    ops_to_visit.pop_back();
    ops_to_visit.push_back(next_visitor_state);
    return true;
  }

  return false;
}

// Rewrites a `mhlo.while` op to receive and forward a `mhlo.token`. Operands to
// the op for all of its regions are extended to have an extra operand `token`.
void RewriteRegionWhileOp(OpBuilder& builder, WhileOp region_while,
                          SmallVectorImpl<OpVisitorState>& ops_to_visit,
                          Value token) {
  llvm::SmallDenseMap<Value, Value> rewritten_operands;

  // Rewrite region operand to have an extra operand `token`.
  auto new_val_operands =
      GetValueWithToken(builder, llvm::to_vector(region_while.getOperands()),
                        token, rewritten_operands,
                        /*flatten_tuple=*/true);

  auto new_result_types =
      GetTypeWithToken(builder, llvm::to_vector(region_while.getResultTypes()),
                       /*flatten_tuple*/ true);

  // Create new `mhlo.while` op with extra token operand and result.
  auto new_while = builder.create<WhileOp>(region_while.getLoc(),
                                           new_result_types, new_val_operands);

  // Move all regions from the old `mhlo.while` op to its replacement.
  new_while.cond().takeBody(region_while.cond());
  new_while.body().takeBody(region_while.body());

  // Forward result from old `mhlo.while` with replacement.
  SmallVector<Value> old_while_results = region_while.getResults();
  SmallVector<Value> new_while_results = new_while.getResults();

  ReplaceWithTupleResult(builder, old_while_results, new_while_results,
                         /*flatten_tuple*/ true);

  auto new_token = new_while_results[new_while_results.size() - 1];

  region_while.erase();

  // Next op to visit. The replacement is visited but at its first region. The
  // token result of the new region if is propagated.
  ops_to_visit.push_back({/*region_idx=*/0, new_token, new_while});
}

// Rewrites an `mhlo.while` op or its region. If `region_idx` is not set, the op
// operands and results are rewritten. If `region_idx` is set, region
// `region_idx` is rewritten to take in and return an additional token. Returns
// true if the op or its region was rewritten.
bool ProcessRegionWhileOp(
    OpBuilder& builder, WhileOp region_while, Optional<unsigned> region_idx,
    SmallVectorImpl<OpVisitorState>& ops_to_visit,
    const llvm::SmallPtrSetImpl<Block*>& control_flow_blocks, Value token) {
  builder.setInsertionPoint(region_while);

  if (!region_idx) {
    RewriteRegionWhileOp(builder, region_while, ops_to_visit, token);
    return true;
  }

  if (*region_idx < region_while.getNumRegions()) {
    SmallVector<Type> operand_types;
    for (auto operand : region_while.operand())
      operand_types.push_back(operand.getType());
    RewriteControlFlowOpRegion(builder, region_while, *region_idx,
                               operand_types, ops_to_visit, control_flow_blocks,
                               token);
    return true;
  }

  return false;
}

// Updates function type based on current function body block arguments and
// terminator operand types.
void UpdateFunctionType(OpBuilder& builder, func::FuncOp func,
                        Block& func_body) {
  auto new_argument_types = llvm::to_vector(func_body.getArgumentTypes());
  auto new_result_types =
      llvm::to_vector(func_body.getTerminator()->getOperandTypes());
  func.setType(FunctionType::get(builder.getContext(), new_argument_types,
                                 new_result_types));
}

// Replaces a function terminator `return` with another `return` that has an
// extra `mhlo.token` operand.
void RewriteFunctionTerminator(OpBuilder& builder,
                               mlir::func::ReturnOp terminator, Value token) {
  auto new_results = llvm::to_vector(terminator.getOperands());
  new_results.push_back(token);
  builder.setInsertionPoint(terminator);
  builder.create<mlir::func::ReturnOp>(terminator.getLoc(), new_results);
  terminator.erase();
}

// Rewrites a function body and communication ops inside. Region control flow
// are updated when necessary, to propagate tokens. The function may either be
// rewritten to create a token or take in and return a token, depending on its
// visibility and if there are any callers.
LogicalResult RewriteFunction(
    OpBuilder& builder, int64_t& channel_id, ModuleOp module, FuncOp func,
    const llvm::SmallDenseMap<StringRef, FuncToRewrite>& funcs,
    const llvm::SmallPtrSetImpl<Operation*>& control_flow_ops,
    const llvm::SmallPtrSetImpl<Block*>& control_flow_blocks, bool is_clone) {
  MLIRContext* context = module.getContext();
  if (!llvm::hasSingleElement(func.getBody()))
    return func.emitError()
           << "'" << FuncOp::getOperationName()
           << "' ops with more than one block are not supported";

  bool rewrite_block =
      is_clone || (!func.isPublic() && !func.symbolKnownUseEmpty(module));
  Block& func_body = func.front();

  builder.setInsertionPointToStart(&func_body);
  auto token_type = TokenType::get(context);
  // If a function is public, it's signature should not be modified, and instead
  // a token will be created. Otherwise a token block argument is inserted.
  Value init_token =
      rewrite_block ? func_body.addArgument(token_type, func.getLoc())
                    : builder.create<CreateTokenOp>(func.getLoc(), token_type)
                          .getResult();

  // Stack to keep track of region based control flow op nesting and current
  // op to visit.
  SmallVector<OpVisitorState, 4> ops_to_visit{
      {/*region_idx=*/llvm::None, init_token, &func_body.front()}};

  while (!ops_to_visit.empty()) {
    OpVisitorState op_to_visit = ops_to_visit.pop_back_val();
    Operation* curr_op = op_to_visit.op;

    Value token = op_to_visit.token;
    // Ops may be removed, so the next op is kept track of beforehand.
    Operation* next_op = curr_op->getNextNode();

    if (auto host_compute = dyn_cast<TF::_XlaHostComputeMlirOp>(curr_op)) {
      token = RewriteHostComputeOp(builder, channel_id, host_compute, token);
    } else if (auto send_to_host = dyn_cast<TF::XlaSendToHostOp>(curr_op)) {
      token = RewriteSendToHostOp(builder, channel_id, send_to_host, token);
    } else if (auto recv_from_host = dyn_cast<TF::XlaRecvFromHostOp>(curr_op)) {
      token = RewriteRecvFromHostOp(builder, channel_id, recv_from_host, token);
    } else if (auto call = dyn_cast<mlir::func::CallOp>(curr_op)) {
      // Only `mlir::func::CallOp` is supported as this requires knowing how to
      // rewrite arguments and results to a function.
      auto it = funcs.find(call.getCallee());
      if (it != funcs.end()) {
        func::FuncOp clone = it->getSecond().clone;
        Optional<StringRef> symbol_name =
            clone ? Optional<StringRef>(clone.getName()) : llvm::None;
        // If the function being called is to be cloned, update the call to also
        // point to the cloned function.
        token = RewriteCallOp(builder, call, symbol_name, token);
      }
    } else if (auto region_if = dyn_cast<IfOp>(curr_op)) {
      if (op_to_visit.region_idx || control_flow_ops.contains(region_if)) {
        auto exist_unprocessed_region =
            ProcessRegionIfOp(builder, region_if, op_to_visit.region_idx,
                              ops_to_visit, control_flow_blocks, token);

        // Once all the IfOp regions are processed (i.e.
        // 'exist_unprocessed_region' == false), select returned token-value
        // from IfOp as the token to be used for the following op.
        if (!exist_unprocessed_region) {
          token = curr_op->getResult(curr_op->getNumResults() - 1);
        } else {
          continue;
        }
      }
    } else if (auto region_while = dyn_cast<WhileOp>(curr_op)) {
      if (op_to_visit.region_idx || control_flow_ops.contains(region_while))
        if (ProcessRegionWhileOp(builder, region_while, op_to_visit.region_idx,
                                 ops_to_visit, control_flow_blocks, token))
          continue;
    } else if (auto region_terminator = dyn_cast<mhlo::ReturnOp>(curr_op)) {
      bool flatten_tuple = isa<mhlo::WhileOp, mhlo::IfOp, mhlo::CaseOp>(
          region_terminator->getParentOp());
      RewriteControlFlowTerminator(builder, region_terminator, token,
                                   flatten_tuple);
      // There is no next op after the control flow op terminator, simply let
      // stack have one less element.
      continue;
    } else if (auto func_terminator = dyn_cast<mlir::func::ReturnOp>(curr_op)) {
      if (rewrite_block)
        RewriteFunctionTerminator(builder, func_terminator, token);

      // There is no next op after the function terminator, simply let stack
      // have one less element/be empty.
      continue;
    }

    // Visit next op.
    ops_to_visit.push_back({/*region_idx=*/llvm::None, token, next_op});
  }

  if (rewrite_block) UpdateFunctionType(builder, func, func_body);

  return success();
}

// Checks if a function call is pointing to a function with communication ops.
bool IsFunctionCallWithCommunication(
    Operation* op,
    const llvm::SmallDenseMap<StringRef, FuncToRewrite>& funcs_to_rewrite) {
  if (auto call = dyn_cast<mlir::func::CallOp>(op))
    return funcs_to_rewrite.count(call.getCallee());

  return false;
}

// Collects all control flow op ancestors of communication ops or function calls
// with communication ops (transitively).
void GetCommunicationControlFlowOps(
    func::FuncOp func,
    const llvm::SmallDenseMap<StringRef, FuncToRewrite>& funcs_to_rewrite,
    llvm::SmallPtrSetImpl<Operation*>& control_flow_ops,
    llvm::SmallPtrSetImpl<Block*>& control_flow_blocks) {
  func.walk([&](Operation* op) {
    if (IsCommunicationOp(op) ||
        IsFunctionCallWithCommunication(op, funcs_to_rewrite))
      if (failed(GetControlFlowAncestors(op, control_flow_ops,
                                         control_flow_blocks)))
        llvm_unreachable(
            "checking original function for control flow ancestors should have "
            "errored first");
  });
}

void LegalizeTFCommunication::runOnOperation() {
  auto module = getOperation();
  llvm::SmallDenseMap<StringRef, FuncToRewrite> funcs_to_rewrite;
  if (failed(GetFunctionsToRewrite(module, funcs_to_rewrite)))
    return signalPassFailure();

  // Module level counter to make sure Channel IDs are unique.
  int64_t channel_id = 1;
  OpBuilder builder(&getContext());
  for (const auto& func_and_name : funcs_to_rewrite) {
    const auto& func_to_rewrite = func_and_name.getSecond();
    func::FuncOp func = func_to_rewrite.original;
    if (failed(RewriteFunction(builder, channel_id, module, func,
                               funcs_to_rewrite,
                               func_to_rewrite.control_flow_ops,
                               func_to_rewrite.control_flow_blocks,
                               /*is_clone=*/false)))
      return signalPassFailure();

    func::FuncOp clone = func_and_name.getSecond().clone;
    if (!clone) continue;
    llvm::SmallPtrSet<Operation*, 4> clone_control_flow_ops;
    llvm::SmallPtrSet<Block*, 4> clone_control_flow_blocks;
    GetCommunicationControlFlowOps(clone, funcs_to_rewrite,
                                   clone_control_flow_ops,
                                   clone_control_flow_blocks);
    if (failed(RewriteFunction(builder, channel_id, module, clone,
                               funcs_to_rewrite, clone_control_flow_ops,
                               clone_control_flow_blocks,
                               /*is_clone=*/true)))
      llvm_unreachable(
          "rewriting of original function should have errored first");
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFCommunicationPass() {
  return std::make_unique<LegalizeTFCommunication>();
}

}  // namespace mhlo
}  // namespace mlir
