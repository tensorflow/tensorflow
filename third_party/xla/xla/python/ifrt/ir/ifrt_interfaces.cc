/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/ifrt_interfaces.h"

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

// Generated definitions.

#define GET_ATTR_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_attr_interfaces.cc.inc"

#define GET_OP_INTERFACE_CLASSES
#include "xla/python/ifrt/ir/ifrt_op_interfaces.cc.inc"

namespace xla {
namespace ifrt {
namespace {

// Returns true if `operand_num` falls within `inputs`.
bool IsInOperandRange(unsigned operand_num, mlir::OperandRange inputs) {
  return !inputs.empty() && operand_num >= inputs.getBeginOperandIndex() &&
         operand_num < inputs.getBeginOperandIndex() + inputs.size();
}

// Verifies that `io_alias` references valid input/output indices on `op` and
// aliases arrays with matching element types and per-shard shapes.
mlir::LogicalResult VerifyIoAlias(mlir::Operation* op, IoAlias io_alias,
                                  mlir::OperandRange inputs,
                                  mlir::ResultRange outputs) {
  if (io_alias.input_index < 0 || io_alias.input_index >= inputs.size()) {
    return op->emitOpError()
           << "can't alias input #" << io_alias.input_index << " to output #"
           << io_alias.output_index << " as only having " << inputs.size()
           << " inputs";
  }
  if (io_alias.output_index < 0 || io_alias.output_index >= outputs.size()) {
    return op->emitOpError()
           << "can't alias input #" << io_alias.input_index << " to output #"
           << io_alias.output_index << " as only having " << outputs.size()
           << " outputs";
  }
  const IfrtArrayType aliased_input =
      GetArrayType(inputs[io_alias.input_index]);
  const IfrtArrayType aliased_output =
      GetArrayType(outputs[io_alias.output_index]);
  if (aliased_input != aliased_output) {
    // TODO(b/382761415): Relax this aliasing check to allow for different
    // per-shard shapes as long as the byte size is the same. We cannot do this
    // now because we do not have layout information.
    if (aliased_input.getShape().getElementType() !=
        aliased_output.getShape().getElementType()) {
      return op->emitOpError() << "can't alias input #" << io_alias.input_index
                               << " to output #" << io_alias.output_index
                               << " with different dtypes: " << aliased_input
                               << " vs. " << aliased_output;
    }
    if (mlir::failed(
            VerifyPerShardShapesAreEqual(op, aliased_input, aliased_output))) {
      return op->emitOpError() << "can't alias input #" << io_alias.input_index
                               << " to output #" << io_alias.output_index;
    }
  }
  return mlir::success();
}

// Verifies `donated_input_indices` and `io_aliases` on `call_op`.
mlir::LogicalResult VerifyIoAliasesAndDonations(IfrtCallOpInterface call_op,
                                                mlir::OperandRange inputs,
                                                mlir::ResultRange outputs) {
  mlir::Operation* op = call_op.getOperation();
  llvm::SmallSet<int, 4> aliased_or_donated_inputs;
  llvm::SmallSet<int, 4> aliased_outputs;
  for (const int32_t donated_input_index : call_op.getDonatedInputIndices()) {
    if (donated_input_index < 0 || donated_input_index >= inputs.size()) {
      return op->emitOpError()
             << "can't donate input #" << donated_input_index
             << " as only having " << inputs.size() << " inputs";
    }
    if (!aliased_or_donated_inputs.insert(donated_input_index).second) {
      return op->emitOpError() << "can't donate input #" << donated_input_index
                               << " more than once";
    }
  }
  for (const IoAlias& io_alias : call_op.getIoAliasPairs()) {
    if (mlir::failed(VerifyIoAlias(op, io_alias, inputs, outputs))) {
      return mlir::failure();
    }
    if (!aliased_or_donated_inputs.insert(io_alias.input_index).second) {
      return op->emitOpError() << "can't alias or donate input #"
                               << io_alias.input_index << " more than once";
    }
    if (!aliased_outputs.insert(io_alias.output_index).second) {
      return op->emitOpError() << "can't alias output #"
                               << io_alias.output_index << " more than once";
    }
  }
  return mlir::success();
}

}  // namespace

namespace detail {

mlir::LogicalResult VerifyPairwiseArrayOp(mlir::Operation* op) {
  IfrtPairwiseArrayOpInterface pairwise_op =
      llvm::cast<IfrtPairwiseArrayOpInterface>(op);
  mlir::OperandRange inputs = pairwise_op.getInputs();
  mlir::ResultRange outputs = pairwise_op.getOutputs();
  if (inputs.empty()) {
    return op->emitOpError() << "requires at least one input array";
  }
  if (inputs.size() != outputs.size()) {
    return op->emitOpError()
           << "requires the same number of input and output arrays";
  }
  return mlir::success();
}

mlir::LogicalResult VerifyCallOp(mlir::Operation* op) {
  IfrtCallOpInterface call_op = llvm::cast<IfrtCallOpInterface>(op);
  mlir::OperandRange inputs = call_op.getInputs();
  mlir::ResultRange outputs = call_op.getOutputs();

  if (mlir::ArrayAttr arg_attrs = call_op.getArgAttrsAttr();
      arg_attrs != nullptr && arg_attrs.size() != inputs.size()) {
    return op->emitOpError()
           << "requires `arg_attrs` to have the same size as `inputs` ("
           << arg_attrs.size() << " vs. " << inputs.size() << ")";
  }
  if (mlir::ArrayAttr res_attrs = call_op.getResAttrsAttr();
      res_attrs != nullptr && res_attrs.size() != outputs.size()) {
    return op->emitOpError()
           << "requires `res_attrs` to have the same size as `outputs` ("
           << res_attrs.size() << " vs. " << outputs.size() << ")";
  }

  return VerifyIoAliasesAndDonations(call_op, inputs, outputs);
}

}  // namespace detail

llvm::SmallVector<IoAlias, 4> GetIoAliases(mlir::ArrayAttr io_aliases) {
  llvm::SmallVector<IoAlias, 4> aliases;
  if (io_aliases == nullptr) {
    return aliases;
  }
  aliases.reserve(io_aliases.size());
  for (mlir::Attribute attr : io_aliases) {
    auto raw_alias = llvm::dyn_cast<mlir::DenseI32ArrayAttr>(attr);
    CHECK(raw_alias != nullptr);
    llvm::ArrayRef<int32_t> pair = raw_alias.asArrayRef();
    CHECK_EQ(pair.size(), 2);
    aliases.push_back(IoAlias{pair[0], pair[1]});
  }
  return aliases;
}

bool CanForwardControlInputs(mlir::Value control_output,
                             mlir::ValueRange control_inputs) {
  if (llvm::all_of(control_output.getUsers(), [](mlir::Operation* user) {
        return llvm::isa<IfrtControlOpInterface>(user);
      })) {
    return true;
  }
  llvm::SmallDenseSet<mlir::Value, 4> unique(control_inputs.begin(),
                                             control_inputs.end());
  return unique.size() == 1;
}

void ForwardControlInputsToUsers(mlir::Value control_output,
                                 mlir::ValueRange control_inputs,
                                 mlir::RewriterBase& rewriter) {
  if (control_output.use_empty()) {
    return;
  }

  llvm::SmallVector<mlir::Value> unique_control_inputs;
  unique_control_inputs.reserve(control_inputs.size());
  llvm::SmallDenseSet<mlir::Value, 8> seen_inputs;
  for (mlir::Value control_input : control_inputs) {
    if (seen_inputs.insert(control_input).second) {
      unique_control_inputs.push_back(control_input);
    }
  }

  llvm::SmallVector<mlir::Operation*> users;
  llvm::SmallPtrSet<mlir::Operation*, 4> seen_users;
  for (mlir::Operation* user : control_output.getUsers()) {
    if (seen_users.insert(user).second) {
      users.push_back(user);
    }
  }

  for (mlir::Operation* user : users) {
    if (IfrtControlOpInterface control_op =
            llvm::dyn_cast<IfrtControlOpInterface>(user);
        control_op != nullptr) {
      mlir::MutableOperandRange user_control_inputs =
          control_op.getControlInputsMutable();
      mlir::OperandRange current_control_inputs =
          user_control_inputs.getAsOperandRange();
      llvm::SmallVector<mlir::Value> new_control_inputs;
      new_control_inputs.reserve(current_control_inputs.size() +
                                 unique_control_inputs.size());
      llvm::SmallDenseSet<mlir::Value, 8> seen_values;
      for (mlir::Value control_value : current_control_inputs) {
        if (control_value == control_output) {
          for (mlir::Value forwarded_input : unique_control_inputs) {
            if (seen_values.insert(forwarded_input).second) {
              new_control_inputs.push_back(forwarded_input);
            }
          }
        } else if (seen_values.insert(control_value).second) {
          new_control_inputs.push_back(control_value);
        }
      }
      rewriter.modifyOpInPlace(
          user, [&]() { user_control_inputs.assign(new_control_inputs); });
      continue;
    }

    CHECK_EQ(unique_control_inputs.size(), 1)
        << "Cannot forward " << unique_control_inputs.size()
        << " unique control inputs to non-variadic user op: "
        << user->getName().getStringRef().str();
    const mlir::Value single_control_input = unique_control_inputs.front();
    rewriter.modifyOpInPlace(user, [&]() {
      for (mlir::OpOperand& operand : user->getOpOperands()) {
        if (operand.get() == control_output) {
          operand.set(single_control_input);
        }
      }
    });
  }
}

mlir::ArrayAttr FilterDictArrayAttr(mlir::ArrayAttr dict_array,
                                    const llvm::BitVector& erase) {
  if (dict_array == nullptr) {
    return nullptr;
  }
  llvm::SmallVector<mlir::Attribute> kept;
  kept.reserve(dict_array.size());
  bool has_non_empty = false;
  for (const auto& [idx, attr] : llvm::enumerate(dict_array)) {
    if (idx < erase.size() && erase.test(idx)) {
      continue;
    }
    kept.push_back(attr);
    if (mlir::DictionaryAttr dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
        dict != nullptr && !dict.empty()) {
      has_non_empty = true;
    }
  }
  if (kept.empty() || !has_non_empty) {
    return nullptr;
  }
  return mlir::ArrayAttr::get(dict_array.getContext(), kept);
}

UsedOutputsSlice CollectUsedOutputs(IfrtArrayOpInterface op) {
  mlir::ResultRange outputs = op.getOutputs();
  const unsigned num_outputs = outputs.size();
  UsedOutputsSlice slice{llvm::BitVector(num_outputs),
                         llvm::BitVector(num_outputs),
                         llvm::SmallVector<int32_t>(num_outputs, -1),
                         {}};
  slice.kept_types.reserve(num_outputs);
  for (unsigned i = 0; i < num_outputs; ++i) {
    if (outputs[i].use_empty()) {
      slice.erase.set(i);
    } else {
      slice.keep.set(i);
      slice.old_to_new[i] = slice.kept_types.size();
      slice.kept_types.push_back(outputs[i].getType());
    }
  }
  return slice;
}

void ReplaceKeptOutputs(mlir::RewriterBase& rewriter,
                        IfrtArrayOpInterface old_op,
                        IfrtArrayOpInterface new_op,
                        llvm::ArrayRef<int32_t> old_to_new) {
  const mlir::ResultRange old_outputs = old_op.getOutputs();
  const mlir::ResultRange new_outputs = new_op.getOutputs();
  CHECK_EQ(old_to_new.size(), old_outputs.size());
  for (const auto& [i, new_idx] : llvm::enumerate(old_to_new)) {
    if (new_idx == -1) {
      CHECK(old_outputs[i].use_empty());
      continue;
    }
    CHECK_GE(new_idx, 0);
    CHECK_LT(static_cast<size_t>(new_idx), new_outputs.size());
    rewriter.replaceAllUsesWith(old_outputs[i], new_outputs[new_idx]);
  }
  rewriter.replaceAllUsesWith(old_op.getControlOutput(),
                              new_op.getControlOutput());
  rewriter.eraseOp(old_op);
}

llvm::SmallDenseSet<int32_t, 8> GetDonatedOrAliasedInputIndices(
    IfrtCallOpInterface call_op) {
  llvm::SmallVector<IoAlias, 4> io_aliases = call_op.getIoAliasPairs();
  llvm::SmallDenseSet<int32_t, 8> donated_or_aliased_idxs;
  donated_or_aliased_idxs.reserve(call_op.getDonatedInputIndices().size() +
                                  io_aliases.size());
  donated_or_aliased_idxs.insert(call_op.getDonatedInputIndices().begin(),
                                 call_op.getDonatedInputIndices().end());
  for (const IoAlias& io_alias : io_aliases) {
    donated_or_aliased_idxs.insert(io_alias.input_index);
  }
  return donated_or_aliased_idxs;
}

llvm::SmallDenseSet<int32_t, 8> GetAliasedOutputIndices(
    IfrtCallOpInterface call_op) {
  llvm::SmallVector<IoAlias, 4> io_aliases = call_op.getIoAliasPairs();
  llvm::SmallDenseSet<int32_t, 8> aliased_idxs;
  aliased_idxs.reserve(io_aliases.size());
  for (const IoAlias& io_alias : io_aliases) {
    aliased_idxs.insert(io_alias.output_index);
  }
  return aliased_idxs;
}

bool IsDonatedOperand(mlir::OpOperand& operand) {
  mlir::Operation* user = operand.getOwner();
  const unsigned operand_num = operand.getOperandNumber();
  return llvm::TypeSwitch<mlir::Operation*, bool>(user)
      .Case<IfrtCallOpInterface>([&](IfrtCallOpInterface call_op) {
        mlir::OperandRange inputs = call_op.getInputs();
        if (!IsInOperandRange(operand_num, inputs)) {
          return false;
        }
        const int32_t input_idx =
            static_cast<int32_t>(operand_num - inputs.getBeginOperandIndex());
        return llvm::is_contained(call_op.getDonatedInputIndices(), input_idx);
      })
      .Case<IfrtPairwiseArrayOpInterface>(
          [&](IfrtPairwiseArrayOpInterface pairwise_op) {
            return IsInOperandRange(operand_num, pairwise_op.getInputs()) &&
                   pairwise_op.getDonated();
          })
      .Case<RemapArraysOp, AssembleOp>([&](auto op) {
        return IsInOperandRange(operand_num, op.getInputs()) && op.getDonated();
      })
      .Case<DisassembleOp>([&](DisassembleOp disassemble_op) {
        return operand_num == 0 && disassemble_op.getDonated();
      })
      .Default(false);
}

bool IsAliasedOperand(mlir::OpOperand& operand) {
  mlir::Operation* user = operand.getOwner();
  const unsigned operand_num = operand.getOperandNumber();
  return llvm::TypeSwitch<mlir::Operation*, bool>(user)
      .Case<IfrtCallOpInterface>([&](IfrtCallOpInterface call_op) {
        mlir::OperandRange inputs = call_op.getInputs();
        if (!IsInOperandRange(operand_num, inputs)) {
          return false;
        }
        const int32_t input_idx =
            static_cast<int32_t>(operand_num - inputs.getBeginOperandIndex());
        return llvm::any_of(call_op.getIoAliasPairs(),
                            [input_idx](const IoAlias& io_alias) {
                              return io_alias.input_index == input_idx;
                            });
      })
      .Case<CopyArraysOp>([&](CopyArraysOp copy_op) {
        return IsInOperandRange(operand_num, copy_op.getInputs()) &&
               copy_op.getReuse();
      })
      .Case<RemapArraysOp, AssembleOp>([&](auto alias_op) {
        return IsInOperandRange(operand_num, alias_op.getInputs()) &&
               !alias_op.getDonated();
      })
      .Case<DisassembleOp>([&](DisassembleOp disassemble_op) {
        return operand_num == 0 && !disassemble_op.getDonated();
      })
      .Default(false);
}

bool IsDonatedOrAliasedOperand(mlir::OpOperand& operand) {
  return IsDonatedOperand(operand) || IsAliasedOperand(operand);
}

bool IsDonatedOrAliasedByUser(mlir::Value value) {
  return llvm::any_of(value.getUses(), IsDonatedOrAliasedOperand);
}

}  // namespace ifrt
}  // namespace xla

namespace mlir {
namespace OpTrait {
namespace xla {
namespace ifrt {
namespace impl {

LogicalResult verifyNestedInIfrtFunc(Operation* op) {
  func::FuncOp func_op =
      llvm::dyn_cast_or_null<func::FuncOp>(op->getParentOp());
  if (func_op == nullptr ||
      (!func_op->hasAttr(::xla::ifrt::kIfrtFunctionAttrName) &&
       !func_op->hasAttr(::xla::ifrt::kIfrtReshardFunctionAttrName))) {
    return op->emitOpError()
           << "must be in a FuncOp with attr `"
           << ::xla::ifrt::kIfrtFunctionAttrName << "` or attr `"
           << ::xla::ifrt::kIfrtReshardFunctionAttrName << "`";
  }
  return success();
}

}  // namespace impl
}  // namespace ifrt
}  // namespace xla
}  // namespace OpTrait
}  // namespace mlir
