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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// TF Device Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  // Returns if its legal to inline 'src' region into the 'dest' region
  // attached to a TF Device operation.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       BlockAndValueMapping& valueMapping) const final {
    return true;
  }

  // Defines the legality of inlining TF Device operations.
  bool isLegalToInline(Operation*, Region*, bool,
                       BlockAndValueMapping&) const final {
    // For now, enable inlining all operations.
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  // This is just re-using the same logic as the TensorFlow dialect right now.
  Operation* materializeCallConversion(OpBuilder& builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
    if (!result_type.isa<TensorType>() || !input.getType().isa<TensorType>())
      return nullptr;
    return builder.create<TF::CastOp>(conversion_loc, result_type, input,
                                      /*truncate=*/builder.getBoolAttr(false));
  }
};

// Checks if a block wraps a single operation and the single operation results
// are perfectly forwarded to the block's terminator.
bool BlockWrapsSingleOp(Block* block) {
  auto body = block->without_terminator();
  if (!hasSingleElement(body)) return false;

  Operation& wrapped_op = *body.begin();
  Operation* terminator = block->getTerminator();
  return wrapped_op.getNumResults() == terminator->getNumOperands() &&
         std::equal(wrapped_op.getResults().begin(),
                    wrapped_op.getResults().end(),
                    terminator->getOperands().begin());
}
}  // end anonymous namespace

TensorFlowDeviceDialect::TensorFlowDeviceDialect(MLIRContext* context)
    : Dialect(/*name=*/"tf_device", context,
              TypeID::get<TensorFlowDeviceDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc.inc"
      >();

  addInterfaces<TFInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// tf_device.launch
//===----------------------------------------------------------------------===//

// Checks if a tf_device.launch wraps a single operation and the single
// operation results are perfectly forwarded to the launch return.
bool LaunchOp::WrapsSingleOp() { return BlockWrapsSingleOp(&GetBody()); }

//===----------------------------------------------------------------------===//
// tf_device.parallel_execute
//===----------------------------------------------------------------------===//

namespace {

LogicalResult Verify(ParallelExecuteOp op) {
  const auto& regions = op.getOperation()->getRegions();
  if (regions.size() < 2) {
    return op.emitOpError() << "must have at least two regions.";
  }

  int output_index = 0;
  for (auto& region_and_index : llvm::enumerate(regions)) {
    auto& region = region_and_index.value();
    auto* region_terminator = region.front().getTerminator();

    // Check that output types of regions match return operand types.
    for (auto result_type : region_terminator->getOperandTypes()) {
      if (result_type !=
          op.getOperation()->getResult(output_index++).getType()) {
        return op.emitOpError() << "output types must be a concatenated "
                                << "list of output types for each regions.";
      }
    }
  }

  // Check that total number of outputs from regions match the output types of
  // the parallel_execute op.
  const int num_output_types = op.getOperation()->getNumResults();
  if (num_output_types != output_index) {
    return op.emitOpError()
           << "number of output types (" << num_output_types << ") "
           << "must match the total number of outputs from all "
           << "regions (" << output_index << ").";
  }

  return success();
}

}  // namespace

// static
void ParallelExecuteOp::build(OpBuilder& builder, OperationState& state,
                              int num_regions, TypeRange output_types) {
  DCHECK_GE(num_regions, 2);
  for (int i = 0; i < num_regions; ++i) {
    Region* region = state.addRegion();
    region->push_back(new Block);
  }
  state.addTypes(output_types);
}

Block& ParallelExecuteOp::GetRegionBlockWithIndex(unsigned index) {
  return getOperation()->getRegion(index).front();
}

Operation::result_range ParallelExecuteOp::GetRegionOutputs(
    unsigned region_index) {
  int num_region_results =
      GetRegionBlockWithIndex(region_index).getTerminator()->getNumOperands();

  int return_value_offset = 0;
  for (int region_id = 0; region_id < region_index; ++region_id)
    return_value_offset +=
        GetRegionBlockWithIndex(region_id).getTerminator()->getNumOperands();

  return getResults().slice(return_value_offset, num_region_results);
}

bool ParallelExecuteOp::RegionWrapsSingleOp(unsigned index) {
  return BlockWrapsSingleOp(&GetRegionBlockWithIndex(index));
}

//===----------------------------------------------------------------------===//
// tf_device.replicate
//===----------------------------------------------------------------------===//

namespace {
ParseResult ParseReplicateOpOperands(
    OpAsmParser* parser, OperationState* state,
    llvm::SmallVectorImpl<llvm::SmallVector<OpAsmParser::OperandType, 8>>*
        replicated_inputs,
    llvm::SmallVectorImpl<OpAsmParser::OperandType>* packed_inputs,
    llvm::SmallVectorImpl<OpAsmParser::OperandType>* region_args,
    llvm::SmallVectorImpl<Type>* region_arg_types) {
  // No operands or empty operand list.
  bool parsed_l_paren = succeeded(parser->parseOptionalLParen());
  if (!parsed_l_paren || succeeded(parser->parseOptionalRParen()))
    return success();

  // Parse comma separated operands of the following format:
  //   replicated_input
  //     [%a, ...] as %block_arg0: type
  //   packed_input
  //     %b as %block_arg1: type
  //
  // Replicated inputs are placed before packed inputs when forming the op.
  llvm::SmallVector<OpAsmParser::OperandType, 8> replicated_region_args;
  llvm::SmallVector<OpAsmParser::OperandType, 8> packed_region_args;
  llvm::SmallVector<Type, 8> replicated_region_arg_types;
  llvm::SmallVector<Type, 8> packed_region_arg_types;
  do {
    OpAsmParser::OperandType operand_type;
    if (parser->parseOptionalOperand(operand_type).hasValue()) {
      packed_inputs->emplace_back(operand_type);
      if (parser->parseKeyword("as",
                               " between packed input and block argument") ||
          parser->parseRegionArgument(packed_region_args.emplace_back()) ||
          parser->parseColonType(packed_region_arg_types.emplace_back()))
        return failure();
    } else if (parser->parseOperandList(replicated_inputs->emplace_back(),
                                        OpAsmParser::Delimiter::Square) ||
               parser->parseKeyword(
                   "as", " between replicated inputs and block argument") ||
               parser->parseRegionArgument(
                   replicated_region_args.emplace_back()) ||
               parser->parseColonType(
                   replicated_region_arg_types.emplace_back())) {
      return failure();
    }
  } while (succeeded(parser->parseOptionalComma()));

  region_args->reserve(replicated_region_args.size() +
                       packed_region_args.size());
  region_args->append(replicated_region_args.begin(),
                      replicated_region_args.end());
  region_args->append(packed_region_args.begin(), packed_region_args.end());

  region_arg_types->reserve(replicated_region_arg_types.size() +
                            packed_region_arg_types.size());
  region_arg_types->append(replicated_region_arg_types.begin(),
                           replicated_region_arg_types.end());
  region_arg_types->append(packed_region_arg_types.begin(),
                           packed_region_arg_types.end());

  // Parse remaining `)` surrounding operands.
  return parser->parseRParen();
}

ParseResult SetReplicateOpOperands(
    llvm::SMLoc loc, OpAsmParser* parser, OperationState* state,
    llvm::ArrayRef<llvm::SmallVector<OpAsmParser::OperandType, 8>>
        replicated_inputs,
    llvm::ArrayRef<OpAsmParser::OperandType> packed_inputs,
    llvm::ArrayRef<Type> region_arg_types, int32_t* n) {
  for (const auto& attr : state->attributes)
    if (attr.getName().strref() == "n")
      if (auto n_attr = attr.getValue().dyn_cast<IntegerAttr>())
        *n = n_attr.getInt();

  if (*n < 2)
    return parser->emitError(loc) << "expects 'n' to be at least 2, got " << *n;

  if (replicated_inputs.empty() && packed_inputs.empty()) return success();

  for (auto replicated_input_and_idx : llvm::enumerate(replicated_inputs)) {
    const int32_t idx = replicated_input_and_idx.index();
    const auto& replicated_input = replicated_input_and_idx.value();
    // Check if replicated input matches `n`.
    if (replicated_input.size() != *n)
      return parser->emitError(loc)
             << "expects number of operands for replicated input " << idx
             << " to be 'n' (" << *n << "), got " << replicated_input.size();

    // Resolve replicated input and block argument type.
    if (parser->resolveOperands(replicated_input, region_arg_types[idx],
                                state->operands))
      return failure();
  }

  const int32_t num_replicated_block_args = replicated_inputs.size();
  for (auto packed_input_and_idx : llvm::enumerate(packed_inputs)) {
    const int32_t idx = packed_input_and_idx.index();
    const auto& packed_input = packed_input_and_idx.value();

    // Resolve packed input and block argument type.
    if (parser->resolveOperand(
            packed_input, region_arg_types[idx + num_replicated_block_args],
            state->operands))
      return failure();
  }

  return success();
}

constexpr char kOperandSegmentSizesAttr[] = "operand_segment_sizes";

ParseResult ParseReplicateOp(OpAsmParser* parser, OperationState* state) {
  llvm::SMLoc loc = parser->getCurrentLocation();

  // Parse operands, attributes, and region of op.
  llvm::SmallVector<llvm::SmallVector<OpAsmParser::OperandType, 8>, 8>
      replicated_inputs;
  llvm::SmallVector<OpAsmParser::OperandType, 8> packed_inputs;
  llvm::SmallVector<OpAsmParser::OperandType, 8> region_args;
  llvm::SmallVector<Type, 8> region_arg_types;
  int32_t n = 0;
  Region& body = *state->addRegion();
  if (ParseReplicateOpOperands(parser, state, &replicated_inputs,
                               &packed_inputs, &region_args,
                               &region_arg_types) ||
      parser->parseOptionalAttrDict(state->attributes) ||
      SetReplicateOpOperands(loc, parser, state, replicated_inputs,
                             packed_inputs, region_arg_types, &n) ||
      parser->parseRegion(body, region_args, region_arg_types))
    return failure();

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  if (!state->attributes.get(kOperandSegmentSizesAttr)) {
    int32_t num_replicated_inputs = replicated_inputs.size() * n;
    int32_t num_packed_inputs = packed_inputs.size();
    auto attr = DenseIntElementsAttr::get(
        VectorType::get({2}, parser->getBuilder().getI32Type()),
        {num_replicated_inputs, num_packed_inputs});
    state->addAttribute(kOperandSegmentSizesAttr, attr);
  }

  // Ensure that the region is well formed: it contains at least a block with
  // a ReturnOp terminator.
  ReplicateOp::ensureTerminator(body, parser->getBuilder(), state->location);

  if (!llvm::hasSingleElement(body))
    return parser->emitError(loc) << "expects a single block region";

  Operation& terminator = body.front().back();
  if (!isa<ReturnOp>(terminator))
    return parser->emitError(loc) << "expects a tf_device.return terminator";

  // Get the results type from the terminator type inside the replicate,
  // replicated each by `n`.
  state->types.reserve(terminator.getNumOperands() * n);
  for (const auto& type : terminator.getOperandTypes())
    state->types.append(n, type);

  return success();
}

void Print(ReplicateOp op, OpAsmPrinter* p) {
  // Print comma separated operands of the following format:
  //   replicated_input
  //     [%a, ...] as %block_arg0: type
  //   packed_input
  //     %b as %block_arg1: type
  const int32_t n = op.n();
  const int32_t num_replicated_inputs =
      (*op.operand_segment_sizes().value_begin<APInt>()).getSExtValue();
  const int32_t num_replicated_block_args = num_replicated_inputs / n;

  if (op.getNumOperands()) {
    *p << '(';
    Block& block = op.body().front();
    interleaveComma(block.getArguments(), *p, [&](BlockArgument arg) {
      const int block_arg_num = arg.getArgNumber();
      if (block_arg_num < num_replicated_block_args) {
        *p << '[';
        p->printOperands(
            std::next(op.replicated_inputs().begin(), block_arg_num * n),
            std::next(op.replicated_inputs().begin(), (block_arg_num + 1) * n));
        *p << "]";
      } else {
        p->printOperand(*std::next(op.packed_inputs().begin(),
                                   block_arg_num - num_replicated_block_args));
      }
      *p << " as " << arg << ": " << arg.getType();
    });
    *p << ')';
  }

  // Skip derived `operand_segment_sizes` attribute as custom print format of
  // operands holds enough information to calculate these variadic operand list
  // lengths.
  p->printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/ArrayRef<StringRef>{
                               kOperandSegmentSizesAttr});
  p->printRegion(op.body(), /*printEntryBlockArgs=*/false);
}

// Checks if two types are compatible (compatible shapes and same elemental
// type).
LogicalResult VerifyCompatibleTypes(Type a, Type b) {
  if (failed(verifyCompatibleShape(a, b)) ||
      getElementTypeOrSelf(a) != getElementTypeOrSelf(b))
    return failure();

  return success();
}

LogicalResult Verify(ReplicateOp op) {
  int32_t n = op.n();

  // Check number of devices, if set, matches `n`.
  if (op.devices().hasValue()) {
    for (auto device_attr : op.devices().getValue().getValue()) {
      auto device_list = device_attr.getValue().dyn_cast_or_null<ArrayAttr>();
      if (!device_list)
        return op.emitError()
               << "expects 'devices' to be a map alias and device name list.";

      bool is_device_string = llvm::all_of(device_list, [](Attribute attr) {
        return attr.dyn_cast_or_null<StringAttr>();
      });
      if (!is_device_string)
        return op.emitOpError() << "expects 'devices' to be a consists of "
                                   "string list as values.";

      if (device_list.size() != n)
        return op.emitOpError()
               << "expects number of devices (" << device_list.size()
               << ") to be equal to 'n' (" << n << ")";
    }
  }

  Block& block = op.body().front();

  auto operand_segment_sizes = op.operand_segment_sizes();
  const int32_t num_replicated_inputs =
      operand_segment_sizes.getValues<APInt>()[0].getSExtValue();
  const int32_t num_packed_inputs =
      operand_segment_sizes.getValues<APInt>()[1].getSExtValue();

  if (num_replicated_inputs % n != 0)
    return op.emitOpError()
           << "expects number of replicated inputs (" << num_replicated_inputs
           << ") to be evenly divisible by 'n' (" << n << ")";

  const int32_t num_replicated_block_args = num_replicated_inputs / n;
  if (num_replicated_block_args + num_packed_inputs != block.getNumArguments())
    return op.emitOpError()
           << "expects number of block arguments (" << block.getNumArguments()
           << ") to be equal to number of replicated inputs ("
           << num_replicated_inputs << ") / 'n' (" << n
           << ") + number of packed inputs (" << num_packed_inputs << ")";

  // Check input types match block argument types.
  auto verify_operand_types = [&](BlockArgument block_arg,
                                  int32_t op_operand_idx) -> LogicalResult {
    Type op_operand_type = op.getOperand(op_operand_idx).getType();
    if (failed(VerifyCompatibleTypes(block_arg.getType(), op_operand_type)))
      return op.emitOpError()
             << "expects operand " << op_operand_idx << " (" << op_operand_type
             << ") and block argument " << block_arg.getArgNumber() << " ("
             << block_arg.getType() << ") to have compatible types";

    return success();
  };
  for (auto block_arg : block.getArguments()) {
    if (block_arg.getArgNumber() < num_replicated_block_args) {
      for (int32_t i = n * block_arg.getArgNumber(), e = i + n; i < e; ++i)
        if (failed(verify_operand_types(block_arg, i))) return failure();
    } else {
      const int32_t idx = block_arg.getArgNumber() - num_replicated_block_args +
                          num_replicated_inputs;
      if (failed(verify_operand_types(block_arg, idx))) return failure();
    }
  }

  Operation& terminator = block.back();

  // Check number of results matches `n` * number of return operands.
  if (op.getNumResults() != n * terminator.getNumOperands())
    return op.emitOpError()
           << "expects number of results (" << op.getNumResults()
           << ") to be equal to 'n' * number of terminator operands (" << n
           << " * " << terminator.getNumOperands() << ")";

  // Check replicated output types match return operand types.
  for (auto operand_type_and_idx :
       llvm::enumerate(terminator.getOperandTypes())) {
    Type operand_type = operand_type_and_idx.value();
    int32_t operand_idx = operand_type_and_idx.index();
    for (int32_t i = n * operand_idx, e = i + n; i < e; ++i)
      if (failed(VerifyCompatibleTypes(operand_type, op.getType(i))))
        return op.emitOpError() << "incompatible types for result " << i
                                << " and terminator operand " << operand_idx;
  }

  return success();
}

void BuildReplicateOp(
    Builder* builder, OperationState* state, int n,
    llvm::Optional<DictionaryAttr> devices,
    llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs,
    ValueRange packed_inputs, TypeRange replica_output_types) {
  DCHECK_GE(n, 2);
  state->addAttribute("n", builder->getI32IntegerAttr(n));

  if (devices.hasValue()) state->addAttribute("devices", devices.getValue());

  Region* region = state->addRegion();
  region->push_back(new Block);
  Block& block = region->front();

  for (auto& replicated_input : replicated_inputs) {
    DCHECK_EQ(llvm::size(replicated_input.first), n);
    for (auto input : replicated_input.first) {
      DCHECK(succeeded(
          VerifyCompatibleTypes(input.getType(), replicated_input.second)));
      state->addOperands(input);
    }
    block.addArgument(replicated_input.second);
  }

  for (auto packed_input : packed_inputs) {
    state->addOperands(packed_input);
    block.addArgument(packed_input.getType());
  }

  // Add derived `operand_segment_sizes` attribute.
  int32_t num_replicated_inputs = replicated_inputs.size() * n;
  int32_t num_packed_inputs = packed_inputs.size();
  auto operand_segment_sizes =
      DenseIntElementsAttr::get(VectorType::get({2}, builder->getI32Type()),
                                {num_replicated_inputs, num_packed_inputs});
  state->addAttribute(kOperandSegmentSizesAttr, operand_segment_sizes);

  for (const auto& output_type : replica_output_types)
    state->addTypes(llvm::SmallVector<Type, 8>(n, output_type));
}
}  // anonymous namespace

void ReplicateOp::build(
    OpBuilder& builder, OperationState& state, int n,
    const llvm::SmallDenseMap<StringRef, llvm::SmallVector<StringRef, 4>>&
        devices,
    llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs,
    ValueRange packed_inputs, TypeRange replica_output_types) {
  llvm::Optional<DictionaryAttr> devices_attr;
  if (!devices.empty()) {
    llvm::SmallVector<mlir::NamedAttribute, 1> device_list;
    device_list.reserve(devices.size());
    for (auto alias_and_devices : devices) {
      NamedAttribute device_name_attr = builder.getNamedAttr(
          alias_and_devices.getFirst(),
          builder.getStrArrayAttr(alias_and_devices.getSecond()));
      device_list.emplace_back(device_name_attr);
    }
    devices_attr.emplace(builder.getDictionaryAttr(device_list));
  }

  BuildReplicateOp(&builder, &state, n, devices_attr, replicated_inputs,
                   packed_inputs, replica_output_types);
}

void ReplicateOp::build(
    OpBuilder& builder, OperationState& state, int n,
    llvm::Optional<DictionaryAttr> devices,
    llvm::ArrayRef<std::pair<ValueRange, Type>> replicated_inputs,
    ValueRange packed_inputs, TypeRange replica_output_types) {
  BuildReplicateOp(&builder, &state, n, devices, replicated_inputs,
                   packed_inputs, replica_output_types);
}

// Returns the number of packed block arguments.
unsigned ReplicateOp::GetNumPackedBlockArguments() {
  return packed_inputs().size();
}

// Returns the number of replicated block arguments.
unsigned ReplicateOp::GetNumReplicatedBlockArguments() {
  return GetBody().getNumArguments() - GetNumPackedBlockArguments();
}

// Returns the replicated block arguments. A copy should be made if the
// replicate op is being modified.
llvm::ArrayRef<BlockArgument> ReplicateOp::GetReplicatedBlockArguments() {
  return GetBody().getArguments().drop_back(GetNumPackedBlockArguments());
}

// Returns the packed block arguments. A copy should be made if the replicate op
// is being modified.
llvm::ArrayRef<BlockArgument> ReplicateOp::GetPackedBlockArguments() {
  return GetBody().getArguments().take_back(GetNumPackedBlockArguments());
}

// Checks if a block argument is replicated (forwarding replicated inputs).
bool ReplicateOp::IsReplicatedBlockArgument(BlockArgument block_arg) {
  assert(block_arg.getOwner() == &GetBody());
  return block_arg.getArgNumber() < GetNumReplicatedBlockArguments();
}

// Checks if a block argument is packed (forwarding a packed input).
bool ReplicateOp::IsPackedBlockArgument(BlockArgument block_arg) {
  return !IsReplicatedBlockArgument(block_arg);
}

// Returns the operand index of the operand being forwarded as a
// replicated/packed block argument for a given replica. This assumes a valid
// block argument (of the replicate op) and a valid replica is provided.
unsigned ReplicateOp::GetReplicaOperandIndexForBlockArgument(
    BlockArgument block_arg, unsigned replica) {
  MutableArrayRef<OpOperand> operands = GetOperandsForBlockArgument(block_arg);
  if (operands.size() == 1) return operands.front().getOperandNumber();

  return operands[replica].getOperandNumber();
}

// Returns the operand being forwarded as a replicated/packed block argument for
// a given replica. This assumes a valid block argument (of the replicate op)
// and a valid replica is provided.
Value ReplicateOp::GetReplicaOperandForBlockArgument(BlockArgument block_arg,
                                                     unsigned replica) {
  MutableArrayRef<OpOperand> operands = GetOperandsForBlockArgument(block_arg);
  if (operands.size() == 1) return operands.front().get();

  return operands[replica].get();
}

// Returns the list of replica op operands that maps to the given block
// argument. Returns list with num_replicas elements for replicated operands
// and list with a single element for packed operands.
//
// Requires that block argument is of this replicate op.
MutableArrayRef<OpOperand> ReplicateOp::GetOperandsForBlockArgument(
    BlockArgument block_arg) {
  assert(block_arg.getOwner() == &GetBody());

  unsigned arg_number = block_arg.getArgNumber();
  unsigned num_replicated_args = GetNumReplicatedBlockArguments();
  int32_t num_replicas = nAttr().getInt();
  MutableArrayRef<OpOperand> operands = getOperation()->getOpOperands();

  // All replicated arguments are before packed arguments so return replicated
  // operands if the given argument is one of the replicated arguments.
  if (arg_number < num_replicated_args)
    return operands.slice(arg_number * num_replicas, num_replicas);

  operands = operands.drop_front(num_replicated_args * num_replicas);
  arg_number -= num_replicated_args;
  return operands.slice(arg_number, 1);
}

// Checks if a tf_device.replicate wraps a single operation and the single
// operation results are perfectly forwarded to the replicate return.
bool ReplicateOp::WrapsSingleOp() { return BlockWrapsSingleOp(&GetBody()); }

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// tf_device.cluster
//===----------------------------------------------------------------------===//

namespace {

// Eliminates cluster op results that are not defined within the cluster and are
// defined outside. cluster op can be rewritten to remove those results.
static LogicalResult EliminatePassThroughResults(ClusterOp op,
                                                 PatternRewriter& rewriter) {
  mlir::Block& body = op.GetBody();
  Operation* return_op = body.getTerminator();
  int num_results = return_op->getNumOperands();

  // Values defined within the cluster.
  llvm::SmallVector<Value, 4> cluster_vals;
  cluster_vals.reserve(num_results);

  // New results stores values to use while replacing the old cluster op.
  llvm::SmallVector<Value, 4> new_results;
  new_results.reserve(num_results);
  for (OpOperand& operand : return_op->getOpOperands()) {
    // If the corresponding result of the cluster op is used in some resource
    // update op, do not eliminate the result. Such assignment ops could be for
    // device resources and are required during fusing of the execute op and
    // the resource update ops.
    bool is_used_for_resource_write = llvm::any_of(
        op.getResult(operand.getOperandNumber()).getUsers(),
        [](Operation* user) { return isa<TF::AssignVariableOp>(user); });

    // TODO(b/186717563): Eliminate all pass through results once XLA correctly
    // handles empty computations. Another approach could be to drop empty
    // clusters within MLIR but that seems to trigger other failures but can be
    // considered again.
    // Old bridge only removes unsupported TPU types (only string for now)
    // during outside compilation extraction so this should be enough for
    // the parity.
    bool is_unsupported_type = getElementTypeOrSelf(operand.get().getType())
                                   .isa<mlir::TF::StringType>();
    Value result = operand.get();
    if (is_unsupported_type && result.getParentBlock() != &body &&
        !is_used_for_resource_write) {
      // Pass through result.
      new_results.push_back(result);
    } else {
      // This result will be populated with the new result after rewriting the
      // cluster op.
      new_results.push_back(nullptr);
      cluster_vals.push_back(result);
    }
  }

  // Return failure if there are no pass through results and op is already
  // canonical.
  if (cluster_vals.size() == num_results) return failure();

  // Rewrite return op in the cluster.
  rewriter.setInsertionPoint(return_op);
  auto new_return =
      rewriter.replaceOpWithNewOp<tf_device::ReturnOp>(return_op, cluster_vals);

  // Rewrite the cluster op.
  rewriter.setInsertionPoint(op);
  auto new_op = rewriter.create<tf_device::ClusterOp>(
      op->getLoc(), new_return.getOperandTypes(), op->getOperands(),
      op->getAttrs());
  rewriter.inlineRegionBefore(op.getBodyRegion(), new_op.getBodyRegion(),
                              new_op.getBodyRegion().end());

  int idx = 0;
  for (Value& result : new_results) {
    if (result == nullptr) result = new_op.getResult(idx++);
  }
  rewriter.replaceOp(op, new_results);
  return success();
}
}  // anonymous namespace

void ClusterOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                            MLIRContext* context) {
  results.insert(EliminatePassThroughResults);
}

//===----------------------------------------------------------------------===//
// tf_device.launch
//===----------------------------------------------------------------------===//

namespace {
// This pattern matches LaunchOps with only one ReturnOp (empty) and remaps the
// results of the LaunchOp to the operands of the ReturnOp.
struct DropEmptyLaunch : public OpRewritePattern<LaunchOp> {
  using OpRewritePattern<LaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LaunchOp op,
                                PatternRewriter& rewriter) const override {
    Block& block = op.GetBody();
    // Check if launch only has a return.
    if (&block.front() != &block.back()) return failure();

    // Map launch results to return operands.
    rewriter.replaceOp(op, block.front().getOperands());

    return success();
  }
};
}  // anonymous namespace

void LaunchOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                           MLIRContext* context) {
  results.insert<DropEmptyLaunch>(context);
}

}  // namespace tf_device
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc.inc"
