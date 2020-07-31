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
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
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
// tokens (for ordering), and rewrite their respective functions when necessary.
// Note, this currently does not handle nested modules/functions or region based
// ops (e.g. control flow).
class LegalizeTFCommunication
    : public PassWrapper<LegalizeTFCommunication, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override;
};

// Checks if a function has any communication ops.
bool HasCommunicationOps(FuncOp func) {
  auto result = func.walk([](Operation* op) {
    if (isa<TF::_XlaHostComputeMlirOp, TF::XlaSendToHostOp,
            TF::XlaRecvFromHostOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Helper struct holding a function and optional cloned version. If `clone` is
// set, function calls to `original` will be replaced with `clone`.
struct FuncAndClone {
  FuncOp original;
  FuncOp clone;
};

// Finds all functions that need to be rewritten with communication ops and
// and associated tokens.
llvm::SmallDenseMap<StringRef, FuncAndClone> GetFunctionsToRewrite(
    ModuleOp module) {
  // Find functions containing communication ops.
  llvm::SmallDenseMap<StringRef, FuncAndClone> funcs;
  SmallVector<FuncOp, 4> funcs_to_visit;
  for (FuncOp func : module.getOps<FuncOp>()) {
    if (HasCommunicationOps(func)) {
      funcs.insert({func.getName(), {func, /*clone=*/nullptr}});
      funcs_to_visit.push_back(func);
    }
  }

  // Find functions that call functions with communication ops, transitively.
  while (!funcs_to_visit.empty()) {
    SmallVector<FuncOp, 4> new_funcs_to_visit;
    for (FuncOp& func : funcs_to_visit) {
      auto uses = func.getSymbolUses(module);
      if (!uses) continue;
      for (auto& use : uses.getValue()) {
        // Only `mlir::CallOp` is supported as this requires knowing how to
        // rewrite arguments and results to a function.
        if (!isa<mlir::CallOp>(use.getUser())) continue;
        auto caller_func = use.getUser()->getParentOfType<FuncOp>();
        if (!caller_func) continue;
        if (funcs
                .insert(
                    {caller_func.getName(), {caller_func, /*clone=*/nullptr}})
                .second)
          new_funcs_to_visit.push_back(caller_func);
      }
    }

    funcs_to_visit.swap(new_funcs_to_visit);
  }

  // Clone public functions that need to be rewritten. Function calls to this
  // function will be replaced with the cloned function.
  SymbolTable symbol_table(module);
  for (auto& func : funcs) {
    if (func.getSecond().original.isPublic()) {
      auto clone = func.getSecond().original.clone();
      clone.setVisibility(SymbolTable::Visibility::Private);
      symbol_table.insert(clone);
      func.getSecond().clone = clone;
    }
  }

  return funcs;
}

// Assigns op sharding to an op for a given device core.
void SetOpSharding(Operation* op, int64_t tpu_core) {
  std::string sharding_serialized =
      ::xla::sharding_builder::AssignDevice(tpu_core).SerializeAsString();
  op->setAttr(kShardingAttr,
              StringAttr::get(sharding_serialized, op->getContext()));
}

// Assigns frontend attributes holding information about data type and
// TensorFlow rendezvous channel name.
void SetFrontendAttributes(Operation* op, StringRef key, Type type) {
  MLIRContext* context = op->getContext();

  auto rendezvous_name = StringAttr::get(key, context);
  auto rendezvous_name_attr = NamedAttribute(
      Identifier::get(kXlaHostTransferRendezvousNameAttr, context),
      rendezvous_name);

  auto element_type = getElementTypeOrSelf(type);
  auto xla_element_type = ::xla::TypeToPrimitiveType(element_type);
  const std::string& xla_element_type_str =
      ::xla::primitive_util::LowercasePrimitiveTypeName(xla_element_type);
  auto original_type = StringAttr::get(xla_element_type_str, context);
  auto original_type_attr =
      NamedAttribute(Identifier::get(kXlaHostTransferOriginalTypeAttr, context),
                     original_type);

  auto frontend_attributes = DictionaryAttr::get(
      ArrayRef<NamedAttribute>{rendezvous_name_attr, original_type_attr},
      context);
  op->setAttr(kFrontendAttributesAttr, frontend_attributes);
}

// Assigns frontend attributes holding information about data type and
// TensorFlow rendezvous channel name specific to `tf._XlaHostComputeMlir`.
// TensorFlow rendezvous channel name is handled differently as individual names
// are used per data send and receive.
void SetFrontendAttributes(Operation* op, int32_t index, StringRef key,
                           Type type, bool device_to_host) {
  std::string formatted_key =
      device_to_host ? llvm::formatv("{0}_dtoh_{1}", key, index).str()
                     : llvm::formatv("{0}_htod_{1}", key, index).str();

  return SetFrontendAttributes(op, formatted_key, type);
}

// Creates a `mhlo.send` op for sending value `operand`. If `index` is set,
// `key` will be rewritten with a suffix and index. If `tpu_core` is set, op
// sharding for the respective device will be set.
Value CreateSendOp(OpBuilder& builder, int64_t& channel_id, Location loc,
                   Value operand, StringRef key, const Optional<size_t>& index,
                   const Optional<int64_t>& tpu_core, Value token) {
  // type 2 == DEVICE_TO_HOST
  auto channel_handle = ChannelHandle::get(
      /*handle=*/builder.getI64IntegerAttr(channel_id++),
      /*type=*/builder.getI64IntegerAttr(2), builder.getContext());
  auto send = builder.create<SendOp>(
      loc, token.getType(), operand, token, channel_handle,
      /*is_host_transfer=*/builder.getBoolAttr(true));

  if (index) {
    SetFrontendAttributes(send, index.getValue(), key, operand.getType(),
                          /*device_to_host=*/true);
  } else {
    SetFrontendAttributes(send, key, operand.getType());
  }

  if (tpu_core) SetOpSharding(send, tpu_core.getValue());

  return send.getResult();
}

// Creates a `mhlo.recv` op for receiving a value. If `index` is set, `key` will
// be rewritten with a suffix and index. If `tpu_core` is set, op sharding for
// the respective device will be set.
Value CreateRecvOp(OpBuilder& builder, int64_t& channel_id, Location loc,
                   Value result, StringRef key, const Optional<size_t>& index,
                   const Optional<int64_t>& tpu_core, Value token) {
  // type 3 == HOST_TO_DEVICE
  auto channel_handle = ChannelHandle::get(
      /*handle=*/builder.getI64IntegerAttr(channel_id++),
      /*type=*/builder.getI64IntegerAttr(3), builder.getContext());
  auto result_type = result.getType();
  auto recv_result_type =
      TupleType::get({result_type, token.getType()}, builder.getContext());
  auto recv =
      builder.create<RecvOp>(loc, recv_result_type, token, channel_handle,
                             /*is_host_transfer=*/builder.getBoolAttr(true));
  if (index) {
    SetFrontendAttributes(recv, index.getValue(), key, result_type,
                          /*device_to_host=*/false);
  } else {
    SetFrontendAttributes(recv, key, result.getType());
  }
  if (tpu_core) SetOpSharding(recv, tpu_core.getValue());

  auto get_tuple_element =
      builder.create<GetTupleElementOp>(loc, recv.getResult(), /*index=*/0);
  if (tpu_core) SetOpSharding(get_tuple_element, tpu_core.getValue());

  result.replaceAllUsesWith(get_tuple_element);

  auto new_token = builder.create<GetTupleElementOp>(loc, recv.getResult(),
                                                     /*index=*/1);
  if (tpu_core) SetOpSharding(new_token, tpu_core.getValue());

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
                       /*index=*/llvm::None, /*tpu_core=*/llvm::None, token);

  send_to_host.erase();
  return token;
}

// Replaces `tf.XlaRecvFromHost` with a `mhlo.recv`.
Value RewriteRecvFromHostOp(OpBuilder& builder, int64_t& channel_id,
                            TF::XlaRecvFromHostOp recv_from_host, Value token) {
  builder.setInsertionPoint(recv_from_host);
  token = CreateRecvOp(builder, channel_id, recv_from_host.getLoc(),
                       recv_from_host.output(), recv_from_host.key(),
                       /*index=*/llvm::None, /*tpu_core=*/llvm::None, token);

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
      call.getLoc(), new_result_types,
      new_symbol ? new_symbol.getValue() : call.callee(), new_operands);

  for (auto results : llvm::zip(call.getResults(), new_call.getResults()))
    std::get<0>(results).replaceAllUsesWith(std::get<1>(results));
  call.erase();
  return new_call.getResults().back();
}

// Updates function terminator and type if a token is to be emitted by the
// function.
void RewriteFunctionTerminatorAndUpdateType(OpBuilder& builder, FuncOp func,
                                            Block& func_body, Value token) {
  // If the function signature is changed, update to emit a token and update
  // the function type.
  Operation* terminator = func_body.getTerminator();
  auto new_results = llvm::to_vector<4>(terminator->getOperands());
  new_results.push_back(token);
  builder.setInsertionPoint(terminator);
  auto new_return =
      builder.create<mlir::ReturnOp>(terminator->getLoc(), new_results);
  terminator->erase();

  auto new_argument_types = llvm::to_vector<4>(func_body.getArgumentTypes());
  auto new_result_types = llvm::to_vector<4>(new_return.getOperandTypes());
  func.setType(FunctionType::get(new_argument_types, new_result_types,
                                 builder.getContext()));
}

// Rewrites a function body and communication ops inside. The function may
// either be rewritten to create a token or take in and return a token,
// depending on its visibility and if there are any callers.
LogicalResult RewriteFunction(
    OpBuilder& builder, int64_t& channel_id, ModuleOp module, FuncOp func,
    const llvm::SmallDenseMap<StringRef, FuncAndClone>& funcs) {
  MLIRContext* context = module.getContext();
  if (!llvm::hasSingleElement(func.getBody()))
    return func.emitError()
           << "'" << FuncOp::getOperationName()
           << "' ops with more than one block are not supported";

  bool rewrite_block = !func.isPublic() && !func.symbolKnownUseEmpty(module);
  Block& func_body = func.front();

  builder.setInsertionPointToStart(&func_body);
  auto token_type = mlir::mhlo::TokenType::get(context);
  // If a function is public, it's signature should not be modified, and instead
  // a token will be created. Otherwise a token block argument is inserted.
  Value token = rewrite_block
                    ? func_body.addArgument(token_type)
                    : builder.create<CreateTokenOp>(func.getLoc(), token_type)
                          .getResult();

  for (Operation& op : llvm::make_early_inc_range(func_body)) {
    if (auto host_compute = dyn_cast<TF::_XlaHostComputeMlirOp>(op)) {
      token = RewriteHostComputeOp(builder, channel_id, host_compute, token);
    } else if (auto send_to_host = dyn_cast<TF::XlaSendToHostOp>(op)) {
      token = RewriteSendToHostOp(builder, channel_id, send_to_host, token);
    } else if (auto recv_from_host = dyn_cast<TF::XlaRecvFromHostOp>(op)) {
      token = RewriteRecvFromHostOp(builder, channel_id, recv_from_host, token);
    } else if (auto call = dyn_cast<mlir::CallOp>(op)) {
      // Only `mlir::CallOp` is supported as this requires knowing how to
      // rewrite arguments and results to a function.
      auto it = funcs.find(call.getCallee());
      if (it == funcs.end()) continue;
      FuncOp clone = it->getSecond().clone;
      Optional<StringRef> symbol_name =
          clone ? Optional<StringRef>(clone.getName()) : llvm::None;
      // If the function being called is to be cloned, update the call to also
      // point to the cloned function.
      token = RewriteCallOp(builder, call, symbol_name, token);
    }
  }

  if (rewrite_block)
    RewriteFunctionTerminatorAndUpdateType(builder, func, func_body, token);

  return success();
}

void LegalizeTFCommunication::runOnOperation() {
  auto module = getOperation();
  llvm::SmallDenseMap<StringRef, FuncAndClone> funcs =
      GetFunctionsToRewrite(module);

  // Module level counter to make sure Channel Id's are unique.
  int64_t channel_id = 1;
  OpBuilder builder(&getContext());
  for (const auto& func_and_name : funcs) {
    FuncOp func = func_and_name.getSecond().original;
    if (failed(RewriteFunction(builder, channel_id, module, func, funcs)))
      return signalPassFailure();

    FuncOp clone = func_and_name.getSecond().clone;
    if (!clone) continue;
    if (failed(RewriteFunction(builder, channel_id, module, clone, funcs)))
      return signalPassFailure();
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
