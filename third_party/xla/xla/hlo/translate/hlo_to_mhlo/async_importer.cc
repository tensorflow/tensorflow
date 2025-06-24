/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/translate/hlo_to_mhlo/async_importer.h"

#include <cassert>
#include <functional>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/hlo_to_mhlo/attribute_importer.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";
constexpr char kShardingAttr[] = "mhlo.sharding";

// ============
// Imports an old-style async start op. E.g. an HLO all-gather-start
// instruction is imported as an async-start associated with an all-gather
// computation.
//
// Eventually, old-style async ops (e.g. all-gather-start) and new-style async
// ops (i.e. async-start, async-update and async-done) will converge on the
// HLO side, so we decided to not introduce new MHLO ops for all-gather-start
// and friends.
//
// In the end, there may be new ops added in the old-style because they're not
// compatible with the new-style async semantics, but those should be handled
// on their own, rather than this function which "upgrades" ops to the
// new-style async API.
// ============
template <typename sync_op>
absl::StatusOr<mlir::Operation*> ImportOldStyleAsyncStart(
    mlir::SymbolTable& symbol_table,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    const llvm::SmallVectorImpl<mlir::Value>& operands, mlir::Location loc,
    mlir::Type result_type, mlir::OpBuilder* builder, std::string func_name,
    std::function<absl::Status(sync_op)> mutate_op) {
  auto context = builder->getContext();
  if (!llvm::isa<mlir::TupleType>(result_type)) {
    return tsl::errors::InvalidArgument(
        "expected async_bundle tuple result type");
  }
  auto result_types = mlir::cast<mlir::TupleType>(result_type).getTypes();
  if (result_types.size() < 2) {
    return tsl::errors::InvalidArgument(
        "async_bundle must contain at least two values");
  }
  auto func_type = mlir::FunctionType::get(context, Untuple(result_types[0]),
                                           Untuple(result_types[1]));
  auto function = mlir::func::FuncOp::create(loc, func_name, func_type);

  // The new function doesn't need to be inserted in the beginning but is done
  // to make testing easier and preserve the original behavior.
  mlir::Block& block = symbol_table.getOp()->getRegion(0).front();
  symbol_table.insert(function, mlir::Block::iterator(block.begin()));

  function.setPrivate();
  auto async_builder = mlir::OpBuilder(function.getBody());

  llvm::SmallVector<mlir::NamedAttribute> async_attributes;
  async_attributes.push_back(builder->getNamedAttr(
      "called_computation",
      mlir::FlatSymbolRefAttr::get(builder->getContext(), function.getName())));
  async_attributes.push_back(builder->getNamedAttr(
      "execution_thread", builder->getStringAttr("main")));

  // Attach the frontend_attributes and sharding attributes to the async op
  // instead of the sync op. First, semantically sharding attributes cannot be
  // attached to the sync op since the sync op may not produce the same number
  // of results as the sharding's tuple element count, e.g., `stablehlo.send`
  // vs. HLO `send`. Second, `mlir_hlo_to_hlo.cc` imports these attributes from
  // the `mhlo.async_start` ops, so attaching them to the sync op will make them
  // disappear during StableHLO/MHLO to HLO lowering.
  for (auto it = attributes.begin(); it != attributes.end();) {
    if (it->getName() == kShardingAttr ||
        it->getName() == kFrontendAttributesAttr) {
      async_attributes.push_back(*it);
      it = attributes.erase(it);
    } else {
      ++it;
    }
  }

  llvm::SmallVector<mlir::Location, 1> locs(Untuple(result_types[0]).size(),
                                            loc);
  auto sync_operand =
      async_builder
          .createBlock(&function.getBody(), {}, Untuple(result_types[0]), locs)
          ->getArguments();
  auto sync_operation = async_builder.create<sync_op>(
      loc, Untuple(result_types[1]), sync_operand, attributes);
  async_builder.create<mlir::func::ReturnOp>(loc, sync_operation->getResults());
  TF_RETURN_IF_ERROR(mutate_op(sync_operation));

  function->setAttr("execution_thread", builder->getStringAttr("main"));

  auto bundle_result_type =
      mlir::mhlo::AsyncBundleType::get(context, result_types);
  return builder
      ->create<mlir::mhlo::AsyncStartOp>(loc, bundle_result_type, operands,
                                         async_attributes)
      .getOperation();
}

absl::StatusOr<mlir::Operation*> ImportOldStyleAsyncDone(
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    const llvm::SmallVectorImpl<mlir::Value>& operands, mlir::Location loc,
    mlir::Type result_type, mlir::OpBuilder* builder,
    bool useBundleResult = false) {
  assert(operands.size() == 1 &&
         "*-done ops must take only a single async_bundle operand");
  auto async_start = operands[0].getDefiningOp<mlir::mhlo::AsyncStartOp>();
  if (!async_start) return InvalidArgument("*-start requires *-done as input");
  attributes.push_back(builder->getNamedAttr(
      "called_computation",
      mlir::FlatSymbolRefAttr::get(builder->getContext(),
                                   async_start.getCalledComputation())));
  attributes.push_back(builder->getNamedAttr("execution_thread",
                                             builder->getStringAttr("main")));

  auto async_bundle = llvm::cast<mlir::mhlo::AsyncBundleType>(
      async_start.getResult().getType());

  auto start_tuple =
      llvm::dyn_cast<mlir::TupleType>(async_bundle.getTypes()[1]);
  if (start_tuple && llvm::isa<mlir::TupleType>(start_tuple.getType(0))) {
    auto op = builder->create<mlir::mhlo::AsyncDoneOp>(loc, result_type,
                                                       operands, attributes);
    return {op};
  } else {
    if (useBundleResult) result_type = async_bundle.getTypes()[1];
    auto op = builder->create<mlir::mhlo::AsyncDoneOp>(
        loc, Untuple(result_type), operands, attributes);
    return CreateTupleFromOpResults(builder, loc, op.getOperation(),
                                    result_type);
  }
}

}  // namespace

// Op Converters

absl::StatusOr<mlir::Operation*> ImportSend(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table) {
  auto send_op = Cast<HloSendInstruction>(instruction);
  attributes.push_back(builder->getNamedAttr(
      "is_host_transfer", builder->getBoolAttr(send_op->is_host_transfer())));
  if (send_op->channel_id().has_value()) {
    ChannelHandle channel_handle;
    channel_handle.set_handle(send_op->channel_id().value());
    channel_handle.set_type(send_op->is_host_transfer()
                                ? ChannelHandle::DEVICE_TO_HOST
                                : ChannelHandle::DEVICE_TO_DEVICE);
    attributes.push_back(
        stablehlo::ConvertChannelHandle(channel_handle, builder));
  }

  bool isPipelined =
      instruction->users().front()->opcode() != HloOpcode::kSendDone;
  if (isPipelined) {
    // Consider removing this path and erroring, unclear if support is needed.

    // Return async_start/done for pipelined send.
    //
    // old-style send returns a bundle of (arg, sync flag, token) to be passed
    // along to send-done.
    // However, the new-style async ops have a shared bundle
    // format of (args, results, scratchpad), so to rewrite the `send` and
    // `send-done` ops to use the new-style async API, we need to reorder the
    // arguments to be in (args, token, sync flag) order.
    auto result_types = mlir::cast<mlir::TupleType>(result_type).getTypes();
    if (result_types.size() != 3)
      return InvalidArgument("send should return a 3-tuple");
    auto async_arg_type = mlir::TupleType::get(
        builder->getContext(), {result_types[0], result_types[2]});
    auto async_bundled_tuple = mlir::TupleType::get(
        builder->getContext(),
        {async_arg_type, result_types[2], result_types[1]});
    return ImportOldStyleAsyncStart<mlir::stablehlo::SendOp>(
        symbol_table, attributes, operands, loc, async_bundled_tuple, builder,
        "send_", [](auto) { return absl::OkStatus(); });
  }

  // Otherwise return send op for non-pipelined send.
  // Skip empty data in MLIR send(tuple<>, token) --> stablehlo.send(token)
  auto token = operands[1];
  llvm::ArrayRef<mlir::Value> args = operands;
  if (args.size() == 2 && IsEmptyTuple(args[0].getType())) {
    args = args.drop_front(1);
  }
  auto send = builder
                  ->create<mlir::stablehlo::SendOp>(loc, token.getType(), args,
                                                    attributes)
                  .getOperation();
  if (instruction->has_sharding()) {
    const HloSharding& sharding = instruction->sharding();
    if (sharding.IsTuple() && sharding.tuple_elements().size() == 3) {
      // Here we are returning a 1-tuple, but HLO send returns a 3-tuple. Need
      // to grab a slice of the sharding. All shardings are maximal, so we
      // just need 1 of them.
      send->setAttr(
          kShardingAttr,
          mlir::StringAttr::get(
              builder->getContext(),
              HloSharding::FromProto(sharding.ToProto().tuple_shardings()[0])
                  ->ToString()));
    }
  }
  return send;
}

absl::StatusOr<mlir::Operation*> ImportRecv(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table) {
  auto recv_op = Cast<HloRecvInstruction>(instruction);
  attributes.push_back(builder->getNamedAttr(
      "is_host_transfer", builder->getBoolAttr(recv_op->is_host_transfer())));

  if (recv_op->channel_id().has_value()) {
    ChannelHandle channel_handle;
    channel_handle.set_handle(recv_op->channel_id().value());
    channel_handle.set_type(recv_op->is_host_transfer()
                                ? ChannelHandle::HOST_TO_DEVICE
                                : ChannelHandle::DEVICE_TO_DEVICE);
    attributes.push_back(
        stablehlo::ConvertChannelHandle(channel_handle, builder));
  }

  // Currently only consolidates async recv with result, 0-result recv uses old
  // style, unclear if this support is needed.
  auto result_types = llvm::cast<mlir::TupleType>(result_type).getTypes();
  if (result_types.size() != 3)
    return InvalidArgument("recv should return a 3-tuple");

  bool isPipelined =
      instruction->users().front()->opcode() != HloOpcode::kRecvDone;
  if (isPipelined) {
    // Consider removing this path and erroring, unclear if support is needed.

    // Old-style `recv` returns a bundle of (result, sync flag, token) to be
    // passed along to recv-done.
    // However, the new-style async ops have a shared
    // bundle format of (args, results, scratchpad), so to rewrite the `recv`
    // and `recv-done` ops to use the new-style async API, we need to reorder
    // the arguments to be in (token, (result, token), sync flag) order.
    // OR (token, token, sync flag) if no result is received.
    llvm::SmallVector<mlir::Type> async_result_types = {result_types[0],
                                                        result_types[2]};
    auto async_result_type_tuple = builder->getTupleType(async_result_types);
    auto async_bundled_tuple = builder->getTupleType(
        {result_types[2], async_result_type_tuple, result_types[1]});
    return ImportOldStyleAsyncStart<mlir::stablehlo::RecvOp>(
        symbol_table, attributes, operands, loc, async_bundled_tuple, builder,
        "recv_", [](auto) { return absl::OkStatus(); });
  }

  // Return recv op for non-pipelined send, skip empty tuple result type
  if (!IsEmptyTuple(result_types[0])) {
    auto recv = builder->create<mlir::stablehlo::RecvOp>(
        loc, llvm::SmallVector<mlir::Type>{result_types[0], result_types[2]},
        operands, attributes);
    if (instruction->has_sharding()) {
      const HloSharding& sharding = instruction->sharding();
      if (sharding.IsTuple() && sharding.tuple_elements().size() == 3) {
        // Here we are returning a 2-tuple, but HLO recv returns a 3-tuple. Need
        // to grab a slice of the sharding. All shardings are maximal, so we
        // just need to 2 of them.
        OpSharding sharding_proto = sharding.ToProto();
        auto* tuple_shardings = sharding_proto.mutable_tuple_shardings();
        tuple_shardings->DeleteSubrange(1, 1);
        recv->setAttr(kShardingAttr,
                      mlir::StringAttr::get(
                          builder->getContext(),
                          HloSharding::FromProto(sharding_proto)->ToString()));
      }
    }
    return WrapVariadicResultsInTuple(builder, loc, recv);
  }

  // Recv with no result, only token.
  // To keep parity, if op only returns token, wrap in tuple<tuple<>, token>
  auto recv = builder->create<mlir::stablehlo::RecvOp>(
      loc, llvm::SmallVector<mlir::Type>{result_types[2]}, operands,
      attributes);
  auto empty_tuple = builder->create<mlir::stablehlo::TupleOp>(
      loc, llvm::ArrayRef<mlir::Value>{});

  return builder->create<mlir::stablehlo::TupleOp>(
      loc,
      llvm::ArrayRef<mlir::Value>{empty_tuple.getResult(), recv.getResult(0)});
}

// Async Collectives

absl::StatusOr<mlir::Operation*> ImportAllGatherStart(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table) {
  auto all_gather_start = Cast<HloAllGatherInstruction>(instruction);
  attributes.push_back(builder->getNamedAttr(
      "all_gather_dim",
      builder->getI64IntegerAttr(all_gather_start->all_gather_dimension())));
  attributes.push_back(
      ConvertReplicaGroups(all_gather_start->replica_groups(), builder));
  if (all_gather_start->channel_id().has_value())
    attributes.push_back(stablehlo::ConvertChannelHandle(
        all_gather_start->channel_id().value(), builder));
  if (all_gather_start->use_global_device_ids())
    attributes.push_back(ConvertUseGlobalDeviceIds(builder));
  if (all_gather_start->operands().size() > 1)
    return InvalidArgument(
        "Async tuple all-gather is not supported in StableHLO");

  if (!llvm::isa<mlir::TupleType>(result_type)) {
    // Async AllGather's output type is bundle<input_type,output_type>
    // There are some instances where the output type is not a tuple, this seems
    // to be the more modern case, so we will wrap these in a tuple for
    // StableHLO.
    result_type = mlir::TupleType::get(builder->getContext(),
                                       {operands[0].getType(), result_type});
  }

  return ImportOldStyleAsyncStart<mlir::stablehlo::AllGatherOp>(
      symbol_table, attributes, operands, loc, result_type, builder,
      "all_gather_", [](auto) { return absl::OkStatus(); });
}

absl::StatusOr<mlir::Operation*> ImportAllReduceStart(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    std::function<absl::Status(mlir::stablehlo::AllReduceOp)> mutate_op,
    mlir::SymbolTable& symbol_table) {
  auto all_reduce_start = Cast<HloAllReduceInstruction>(instruction);
  attributes.push_back(
      ConvertReplicaGroups(all_reduce_start->replica_groups(), builder));
  if (all_reduce_start->channel_id().has_value())
    attributes.push_back(stablehlo::ConvertChannelHandle(
        all_reduce_start->channel_id().value(), builder));
  if (all_reduce_start->use_global_device_ids())
    attributes.push_back(ConvertUseGlobalDeviceIds(builder));
  if (all_reduce_start->operands().size() > 1)
    return InvalidArgument(
        "Async tuple all-reduce is not supported in StableHLO");

  if (!llvm::isa<mlir::TupleType>(result_type)) {
    // Async AllReduce's output type is bundle<input_type,output_type>
    // There are some instances where the output type is not a tuple, this seems
    // to be the more modern case, so we will wrap these in a tuple for
    // StableHLO.
    result_type = mlir::TupleType::get(builder->getContext(),
                                       {operands[0].getType(), result_type});
  }

  return ImportOldStyleAsyncStart<mlir::stablehlo::AllReduceOp>(
      symbol_table, attributes, operands, loc, result_type, builder,
      "all_reduce_", mutate_op);
}

// Collective Permute

absl::StatusOr<mlir::Operation*> ImportCollectivePermuteStart(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table) {
  attributes.push_back(
      ConvertSourceTargetPairs(instruction->source_target_pairs(), builder));
  if (!llvm::isa<mlir::TupleType>(result_type)) {
    // Async CollectivePermute's output type is bundle<input_type,output_type>
    // There are some instances where the output type is not a tuple, this seems
    // to be the more modern case, so we will wrap these in a tuple for
    // StableHLO.
    result_type = mlir::TupleType::get(builder->getContext(),
                                       {operands[0].getType(), result_type});
  }
  return ImportOldStyleAsyncStart<mlir::stablehlo::CollectivePermuteOp>(
      symbol_table, attributes, operands, loc, result_type, builder,
      "collective_permute_", [&](auto) { return absl::OkStatus(); });
}

absl::StatusOr<mlir::Operation*> ImportCopyStart(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table) {
  auto context = builder->getContext();
  auto copy_start_instruction = Cast<HloCopyStartInstruction>(instruction);
  if (auto cross_program_prefetch_index =
          copy_start_instruction->cross_program_prefetch_index()) {
    attributes.push_back(builder->getNamedAttr(
        "cross_program_prefetch_index",
        builder->getIntegerAttr(builder->getIntegerType(32),
                                *cross_program_prefetch_index)));
    // Cross-program prefetch allows copy ops to accept tuples, in which
    // case, we need to double-wrap inputs and outputs in tuples.
    if (mlir::isa<mlir::TupleType>(operands[0].getType())) {
      auto result_types = mlir::cast<mlir::TupleType>(result_type).getTypes();
      result_type = mlir::TupleType::get(
          context,
          {mlir::TupleType::get(context, {result_types[0]}),
           mlir::TupleType::get(context, {result_types[1]}), result_types[2]});
    }
  }
  return ImportOldStyleAsyncStart<mlir::mhlo::CopyOp>(
      symbol_table, attributes, operands, loc, result_type, builder, "copy_",
      [](auto) { return absl::OkStatus(); });
}

absl::StatusOr<mlir::Operation*> ImportAsyncOpDone(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    std::optional<HloOpcode> consolidate_if_parent) {
  // Consolidate if the defining op matches `consolidate_if_parent`, ensuring
  // the async communication op is not pipelined.
  if (consolidate_if_parent.has_value() &&
      instruction->operand(0)->opcode() == consolidate_if_parent.value()) {
    return operands[0].getDefiningOp();
  }
  return ImportOldStyleAsyncDone(attributes, operands, loc, result_type,
                                 builder);
}

}  // namespace xla
