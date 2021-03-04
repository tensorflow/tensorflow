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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"

namespace mlir {
namespace mhlo {

namespace {
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";
const char kXlaHostTransferRendezvousNameAttr[] =
    "_xla_host_transfer_rendezvous";
const char kXlaHostTransferOriginalTypeAttr[] =
    "_xla_host_transfer_original_type";

// A pass that legalizes TF/XLA communication ops, propagate their respective
// tokens (for ordering), and rewrite their respective functions and control
// flow ops when necessary.
// Note, this currently does not handle nested modules/functions or region based
// ops other than certain control flow ops (`mhlo.if`, `mhlo.while`).
class LegalizeTFCommunication
    : public PassWrapper<LegalizeTFCommunication, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
  }

 public:
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
  while (block && parent && !isa<FuncOp>(parent)) {
    if (!IsControlFlowOp(parent))
      return op->emitOpError()
             << "expects ancestor(s) to be of ['" << IfOp::getOperationName()
             << "', '" << FuncOp::getOperationName() << "']";

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
    FuncOp func, llvm::SmallPtrSetImpl<Operation*>& control_flow_ops,
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
  FuncOp original;
  llvm::SmallPtrSet<Operation*, 4> control_flow_ops;
  llvm::SmallPtrSet<Block*, 4> control_flow_blocks;
  FuncOp clone;
};

// Finds all functions that need to be rewritten with communication ops and
// and associated tokens.
LogicalResult GetFunctionsToRewrite(
    ModuleOp module,
    llvm::SmallDenseMap<StringRef, FuncToRewrite>& funcs_to_rewrite) {
  // Find functions containing communication ops.
  SmallVector<FuncOp, 4> funcs_to_visit;
  for (FuncOp func : module.getOps<FuncOp>()) {
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
    SmallVector<FuncOp, 4> new_funcs_to_visit;
    for (FuncOp& func : funcs_to_visit) {
      auto uses = func.getSymbolUses(module);
      if (!uses) continue;
      for (auto& use : *uses) {
        // Only `mlir::CallOp` is supported as this requires knowing how to
        // rewrite arguments and results to a function.
        if (!isa<mlir::CallOp>(use.getUser())) continue;
        auto caller_parent_func = use.getUser()->getParentOfType<FuncOp>();
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

// Assigns op sharding to an op for a given device core.
void SetOpSharding(Operation* op, int64_t tpu_core) {
  std::string sharding_serialized =
      ::xla::sharding_builder::AssignDevice(tpu_core).SerializeAsString();
  op->setAttr(kShardingAttr,
              StringAttr::get(op->getContext(), sharding_serialized));
}

// Assigns frontend attributes holding information about data type and
// TensorFlow rendezvous channel name. The TensorFlow rendezvous channel name is
// handled differently as individual names are used per data send and receive.
void SetFrontendAttributes(Operation* op, int32_t index, StringRef key,
                           Type type, bool device_to_host) {
  MLIRContext* context = op->getContext();

  std::string formatted_key =
      device_to_host ? llvm::formatv("{0}_dtoh_{1}", key, index).str()
                     : llvm::formatv("{0}_htod_{1}", key, index).str();

  auto rendezvous_name = StringAttr::get(context, formatted_key);
  auto rendezvous_name_attr = NamedAttribute(
      Identifier::get(kXlaHostTransferRendezvousNameAttr, context),
      rendezvous_name);

  auto element_type = getElementTypeOrSelf(type);
  auto xla_element_type = ::xla::TypeToPrimitiveType(element_type);
  const std::string& xla_element_type_str =
      ::xla::primitive_util::LowercasePrimitiveTypeName(xla_element_type);
  auto original_type = StringAttr::get(context, xla_element_type_str);
  auto original_type_attr =
      NamedAttribute(Identifier::get(kXlaHostTransferOriginalTypeAttr, context),
                     original_type);

  auto frontend_attributes = DictionaryAttr::get(
      context,
      ArrayRef<NamedAttribute>{rendezvous_name_attr, original_type_attr});
  op->setAttr(kFrontendAttributesAttr, frontend_attributes);
}

// Creates a `mhlo.send` op for sending value `operand`. If `tpu_core` is set,
// op sharding for the respective device will be set.
Value CreateSendOp(OpBuilder& builder, int64_t& channel_id, Location loc,
                   Value operand, StringRef key, size_t index,
                   const Optional<int64_t>& tpu_core, Value token) {
  // type 2 == DEVICE_TO_HOST
  auto channel_handle = ChannelHandle::get(
      /*handle=*/builder.getI64IntegerAttr(channel_id++),
      /*type=*/builder.getI64IntegerAttr(2), builder.getContext());
  auto send = builder.create<SendOp>(
      loc, token.getType(), operand, token, channel_handle,
      /*is_host_transfer=*/builder.getBoolAttr(true));

  SetFrontendAttributes(send, index, key, operand.getType(),
                        /*device_to_host=*/true);

  if (tpu_core) SetOpSharding(send, *tpu_core);

  return send.getResult();
}

// Creates a `mhlo.recv` op for receiving a value. If `tpu_core` is set, op
// sharding for the respective device will be set.
Value CreateRecvOp(OpBuilder& builder, int64_t& channel_id, Location loc,
                   Value result, StringRef key, size_t index,
                   const Optional<int64_t>& tpu_core, Value token) {
  // type 3 == HOST_TO_DEVICE
  auto channel_handle = ChannelHandle::get(
      /*handle=*/builder.getI64IntegerAttr(channel_id++),
      /*type=*/builder.getI64IntegerAttr(3), builder.getContext());
  auto result_type = result.getType();
  auto recv_result_type =
      TupleType::get(builder.getContext(), {result_type, token.getType()});
  auto recv =
      builder.create<RecvOp>(loc, recv_result_type, token, channel_handle,
                             /*is_host_transfer=*/builder.getBoolAttr(true));

  SetFrontendAttributes(recv, index, key, result_type,
                        /*device_to_host=*/false);

  if (tpu_core) SetOpSharding(recv, *tpu_core);

  auto get_tuple_element =
      builder.create<GetTupleElementOp>(loc, recv.getResult(), /*index=*/0);
  if (tpu_core) SetOpSharding(get_tuple_element, *tpu_core);

  result.replaceAllUsesWith(get_tuple_element);

  auto new_token = builder.create<GetTupleElementOp>(loc, recv.getResult(),
                                                     /*index=*/1);
  if (tpu_core) SetOpSharding(new_token, *tpu_core);

  return new_token.getResult();
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
// ops per operand and result. Unique Channel Id's are assigned per transfer.
// Sink tokens are created across all `mhlo.send` ops first and then by
// all `mhlo.recv` ops.
Value RewriteHostComputeOp(OpBuilder& builder, int64_t& channel_id,
                           TF::_XlaHostComputeMlirOp host_compute,
                           Value token) {
  builder.setInsertionPoint(host_compute);
  Location loc = host_compute.getLoc();
  int64_t tpu_core = host_compute.tpu_coreAttr().getInt();

  SmallVector<Value, 4> send_tokens;
  for (auto operand : llvm::enumerate(host_compute.inputs())) {
    auto send_token =
        CreateSendOp(builder, channel_id, loc, operand.value(),
                     host_compute.send_key(), operand.index(), tpu_core, token);
    send_tokens.push_back(send_token);
  }
  token = CreateSinkToken(builder, loc, send_tokens, token);

  SmallVector<Value, 4> recv_tokens;
  for (auto result : llvm::enumerate(host_compute.outputs())) {
    auto recv_token =
        CreateRecvOp(builder, channel_id, loc, result.value(),
                     host_compute.recv_key(), result.index(), tpu_core, token);
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
                       /*index=*/0, /*tpu_core=*/llvm::None, token);

  send_to_host.erase();
  return token;
}

// Replaces `tf.XlaRecvFromHost` with a `mhlo.recv`.
Value RewriteRecvFromHostOp(OpBuilder& builder, int64_t& channel_id,
                            TF::XlaRecvFromHostOp recv_from_host, Value token) {
  builder.setInsertionPoint(recv_from_host);
  token = CreateRecvOp(builder, channel_id, recv_from_host.getLoc(),
                       recv_from_host.output(), recv_from_host.key(),
                       /*index=*/0, /*tpu_core=*/llvm::None, token);

  recv_from_host.erase();
  return token;
}

// Replaces a `mlir::CallOp` with one that has an extra `!mhlo.token` operand
// and `!mhlo.token` result. If `new_symbol` is set, the new call will be
// updated to call the `new_symbol` instead.
Value RewriteCallOp(OpBuilder& builder, CallOp call,
                    const Optional<StringRef>& new_symbol, Value token) {
  builder.setInsertionPoint(call);
  auto new_operands = llvm::to_vector<4>(call.getArgOperands());
  new_operands.push_back(token);
  auto new_result_types = llvm::to_vector<4>(call.getResultTypes());
  new_result_types.push_back(token.getType());
  auto new_call = builder.create<CallOp>(
      call.getLoc(), new_result_types, new_symbol ? *new_symbol : call.callee(),
      new_operands);

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

// Replaces a value `value` with a new value but the token attached. If `value`
// is not a tuple, a new tuple is formed with `token`. If `value` is a tuple,
// `value` is extended instead. New tuple values created are cached.
Value GetValueWithToken(OpBuilder& builder, Value value, Value token,
                        llvm::SmallDenseMap<Value, Value>& rewritten_values) {
  // If value with token already exists, reuse it.
  auto it = rewritten_values.find(value);
  if (it != rewritten_values.end()) return it->getSecond();

  auto create_tuple = [&](ArrayRef<Value> operands) {
    auto new_result = CreateTuple(builder, value.getLoc(), operands);
    rewritten_values.insert({value, new_result});
    return new_result;
  };

  auto tuple_type = value.getType().dyn_cast<TupleType>();
  // `value` is not a tuple, create a new tuple.
  if (!tuple_type) return create_tuple({value, token});

  // Extend tuple if `value` is a tuple.
  // If `value` is an op result and the owner is a `mhlo.tuple`, simply unpack
  // the tuple.
  if (auto tuple_op = value.getDefiningOp<TupleOp>()) {
    auto tuple_operands = llvm::to_vector<4>(tuple_op.getOperands());
    tuple_operands.push_back(token);
    return create_tuple(tuple_operands);
  }

  // `value` is not created via a `mhlo.tuple` directly, unpack individual
  // elements directly with `mhlo.get_tuple_element`.
  SmallVector<Value, 4> tuple_operands;
  for (auto idx : llvm::seq<int32_t>(0, tuple_type.getTypes().size()))
    tuple_operands.push_back(
        builder.create<GetTupleElementOp>(value.getLoc(), value, idx)
            .getResult());

  tuple_operands.push_back(token);
  return create_tuple(tuple_operands);
}

// Extends a type to include a `mhlo.token` type. If `type` is not a tuple type,
// a new tuple type with `type` and `mhlo.token` type is created instead.
TupleType GetTypeWithToken(OpBuilder& builder, Type type) {
  auto token_type = TokenType::get(builder.getContext());
  if (auto tuple_type = type.dyn_cast<TupleType>()) {
    auto result_types = llvm::to_vector<4>(tuple_type.getTypes());
    result_types.push_back(token_type);
    return builder.getTupleType(result_types);
  }

  return builder.getTupleType({type, token_type});
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

// Replaces uses of `value` with `replacement`. If `value` is not a tuple type,
// an explicit `mhlo.get_tuple_element` is created to unpack the tuple and
// return the first element. Otherwise, `mhlo.get_tuple_element` users are
// simply updated with `replacement`, and all other users are updated with a
// slice of `replacement`.
void ReplaceWithTupleResult(OpBuilder& builder, Value value,
                            Value replacement) {
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

// Replaces control flow op block single block argument with new block argument
// of type `new_type` (tuple type). The last element of the new block argument
// (token) is returned.
Value UpdateControlFlowBlockArgWithToken(OpBuilder& builder, Block& block,
                                         Type token_type) {
  assert(block.getNumArguments() == 1);
  builder.setInsertionPointToStart(&block);
  auto new_arg = block.addArgument(token_type);
  ReplaceWithTupleResult(builder, block.getArgument(0), new_arg);
  block.eraseArgument(0);
  return builder
      .create<GetTupleElementOp>(new_arg.getLoc(), new_arg,
                                 token_type.cast<TupleType>().size() - 1)
      .getResult();
}

// Updates control flow op terminator with an extra element `token`. If the
// original return value is not a tuple, a new tuple is formed. Otherwise the
// tuple is extended.
void RewriteControlFlowTerminator(OpBuilder& builder, Operation* terminator,
                                  Value token) {
  assert(terminator->getNumOperands() == 1);
  assert(terminator->getBlock()->getNumArguments() == 1);
  // `mhlo.while` cond terminator does not need to be rewritten as it always
  // returns a tensor<i1> predicate value.
  if (auto while_parent = dyn_cast_or_null<WhileOp>(terminator->getParentOp()))
    if (terminator->getParentRegion() == &while_parent.cond()) return;

  builder.setInsertionPoint(terminator);
  llvm::SmallDenseMap<Value, Value> rewritten_operands;
  Value new_result = GetValueWithToken(builder, terminator->getOperand(0),
                                       token, rewritten_operands);
  terminator->setOperand(0, new_result);
}

// Rewrites a `mhlo.if` op to receive and forward a `mhlo.token`. Operands to
// the op for all of its regions are extended to have an extra operand `token`.
void RewriteRegionIfOp(OpBuilder& builder, IfOp region_if,
                       SmallVectorImpl<OpVisitorState>& ops_to_visit,
                       Value token) {
  llvm::SmallDenseMap<Value, Value> rewritten_operands;

  // Rewrite all region operands to have an extra operand `token`.
  Value new_true_operand = GetValueWithToken(builder, region_if.true_arg(),
                                             token, rewritten_operands);
  Value new_false_operand = GetValueWithToken(builder, region_if.false_arg(),
                                              token, rewritten_operands);

  auto new_result_type = GetTypeWithToken(builder, region_if.getType());

  // Create new `mhlo.if` op with extra token operands and result.
  auto new_if = builder.create<IfOp>(region_if.getLoc(), new_result_type,
                                     region_if.pred(), new_true_operand,
                                     new_false_operand);

  // Move all regions from the old `mhlo.if` op to its replacement.
  new_if.true_branch().takeBody(region_if.true_branch());
  new_if.false_branch().takeBody(region_if.false_branch());

  // Forward result from old `mhlo.if` with replacement, and unpack result when
  // necessary.
  ReplaceWithTupleResult(builder, region_if.getResult(), new_if.getResult());

  auto new_token = builder.create<GetTupleElementOp>(
      new_if.getLoc(), new_if.getResult(),
      new_if.getResult().getType().cast<TupleType>().size() - 1);

  region_if.erase();

  // Remove leftover operands to old `mhlo.if` if they have no uses.
  for (auto& rewritten_operand : rewritten_operands)
    if (auto tuple_op = rewritten_operand.getFirst().getDefiningOp<TupleOp>())
      if (tuple_op.use_empty()) tuple_op.erase();

  // Next op to visit. The replacement is visited but at its first region. The
  // token result of the new region if is propagated.
  ops_to_visit.push_back({/*region_idx=*/0, new_token, new_if});
}

// Rewrites a `mhlo.if`/`mhlo.while` region to receive and forward a
// `mhlo.token`. The block argument is updated to have an extra `mhlo.token`
// element. If the region block is to be rewritten, the next op to visit is set
// to the first op in the block. Otherwise the terminator is updated to forward
// `token`.
void RewriteControlFlowOpRegion(
    OpBuilder& builder, Operation* region_op, unsigned region_idx,
    Type block_arg_type, SmallVectorImpl<OpVisitorState>& ops_to_visit,
    const llvm::SmallPtrSetImpl<Block*>& control_flow_blocks, Value token) {
  ops_to_visit.push_back({region_idx + 1, token, region_op});

  Region& region = region_op->getRegion(region_idx);
  assert(llvm::hasSingleElement(region));

  auto block_token = UpdateControlFlowBlockArgWithToken(builder, region.front(),
                                                        block_arg_type);

  if (control_flow_blocks.contains(&region.front())) {
    ops_to_visit.push_back({/*region_idx=*/llvm::None, block_token,
                            block_token.getDefiningOp()->getNextNode()});
    return;
  }

  RewriteControlFlowTerminator(builder, region.front().getTerminator(),
                               block_token);
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
    RewriteControlFlowOpRegion(builder, region_if, *region_idx,
                               region_if.getOperand(*region_idx + 1).getType(),
                               ops_to_visit, control_flow_blocks, token);
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
  Value new_val_operand =
      GetValueWithToken(builder, region_while.val(), token, rewritten_operands);

  auto new_result_type = GetTypeWithToken(builder, region_while.getType());

  // Create new `mhlo.while` op with extra token operand and result.
  auto new_while = builder.create<WhileOp>(region_while.getLoc(),
                                           new_result_type, new_val_operand);

  // Move all regions from the old `mhlo.while` op to its replacement.
  new_while.cond().takeBody(region_while.cond());
  new_while.body().takeBody(region_while.body());

  // Forward result from old `mhlo.while` with replacement, and unpack result
  // when necessary.
  ReplaceWithTupleResult(builder, region_while.getResult(),
                         new_while.getResult());

  auto new_token = builder.create<GetTupleElementOp>(
      new_while.getLoc(), new_while.getResult(),
      new_while.getResult().getType().cast<TupleType>().size() - 1);

  region_while.erase();

  // Remove leftover operands to old `mhlo.while` if they have no uses.
  for (auto& rewritten_operand : rewritten_operands)
    if (auto tuple_op = rewritten_operand.getFirst().getDefiningOp<TupleOp>())
      if (tuple_op.use_empty()) tuple_op.erase();

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
    RewriteControlFlowOpRegion(builder, region_while, *region_idx,
                               region_while.val().getType(), ops_to_visit,
                               control_flow_blocks, token);
    return true;
  }

  return false;
}

// Updates function type based on current function body block arguments and
// terminator operand types.
void UpdateFunctionType(OpBuilder& builder, FuncOp func, Block& func_body) {
  auto new_argument_types = llvm::to_vector<4>(func_body.getArgumentTypes());
  auto new_result_types =
      llvm::to_vector<4>(func_body.getTerminator()->getOperandTypes());
  func.setType(FunctionType::get(builder.getContext(), new_argument_types,
                                 new_result_types));
}

// Replaces a function terminator `return` with another `return` that has an
// extra `mhlo.token` operand.
void RewriteFunctionTerminator(OpBuilder& builder, mlir::ReturnOp terminator,
                               Value token) {
  auto new_results = llvm::to_vector<4>(terminator.getOperands());
  new_results.push_back(token);
  builder.setInsertionPoint(terminator);
  builder.create<mlir::ReturnOp>(terminator.getLoc(), new_results);
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
      rewrite_block ? func_body.addArgument(token_type)
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
    } else if (auto call = dyn_cast<mlir::CallOp>(curr_op)) {
      // Only `mlir::CallOp` is supported as this requires knowing how to
      // rewrite arguments and results to a function.
      auto it = funcs.find(call.getCallee());
      if (it != funcs.end()) {
        FuncOp clone = it->getSecond().clone;
        Optional<StringRef> symbol_name =
            clone ? Optional<StringRef>(clone.getName()) : llvm::None;
        // If the function being called is to be cloned, update the call to also
        // point to the cloned function.
        token = RewriteCallOp(builder, call, symbol_name, token);
      }
    } else if (auto region_if = dyn_cast<IfOp>(curr_op)) {
      if (op_to_visit.region_idx || control_flow_ops.contains(region_if))
        if (ProcessRegionIfOp(builder, region_if, op_to_visit.region_idx,
                              ops_to_visit, control_flow_blocks, token))
          continue;
    } else if (auto region_while = dyn_cast<WhileOp>(curr_op)) {
      if (op_to_visit.region_idx || control_flow_ops.contains(region_while))
        if (ProcessRegionWhileOp(builder, region_while, op_to_visit.region_idx,
                                 ops_to_visit, control_flow_blocks, token))
          continue;
    } else if (auto region_terminator = dyn_cast<mhlo::ReturnOp>(curr_op)) {
      RewriteControlFlowTerminator(builder, region_terminator, token);
      // There is no next op afer the control flow op terminator, simply let
      // stack have one less element.
      continue;
    } else if (auto func_terminator = dyn_cast<mlir::ReturnOp>(curr_op)) {
      if (rewrite_block)
        RewriteFunctionTerminator(builder, func_terminator, token);

      // There is no next op afer the function terminator, simply let stack have
      // one less element/be empty.
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
  if (auto call = dyn_cast<mlir::CallOp>(op))
    return funcs_to_rewrite.count(call.callee());

  return false;
}

// Collects all control flow op ancestors of communication ops or function calls
// with communication ops (transitively).
void GetCommunicationControlFlowOps(
    FuncOp func,
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

  // Module level counter to make sure Channel Id's are unique.
  int64_t channel_id = 1;
  OpBuilder builder(&getContext());
  for (const auto& func_and_name : funcs_to_rewrite) {
    const auto& func_to_rewrite = func_and_name.getSecond();
    FuncOp func = func_to_rewrite.original;
    if (failed(RewriteFunction(builder, channel_id, module, func,
                               funcs_to_rewrite,
                               func_to_rewrite.control_flow_ops,
                               func_to_rewrite.control_flow_blocks,
                               /*is_clone=*/false)))
      return signalPassFailure();

    FuncOp clone = func_and_name.getSecond().clone;
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

static PassRegistration<LegalizeTFCommunication> pass(
    "xla-legalize-tf-communication",
    "Legalize TF/XLA communication ops (TensorFlow dialect) to the HLO "
    "dialect");
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFCommunicationPass() {
  return std::make_unique<LegalizeTFCommunication>();
}

}  // namespace mhlo
}  // namespace mlir
