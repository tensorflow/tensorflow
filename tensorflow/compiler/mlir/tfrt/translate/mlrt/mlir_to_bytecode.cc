/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/translate/mlrt/mlir_to_bytecode.h"

#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/function.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/kernel.h"

namespace mlrt {
namespace {

// LINT.IfChange(mlrt_attributes)
bool CanBeInlined(mlir::Attribute attr, absl::string_view data) {
  // FlatSymbolRefAttr is a special case as we are emitting it as integer.
  return mlir::isa<mlir::IntegerAttr, mlir::FloatAttr, mlir::FlatSymbolRefAttr>(
             attr) &&
         data.size() <= sizeof(uint32_t);
}
// LINT.ThenChange(../../../../../core/tfrt/mlrt/interpreter/attribute_span.h:mlrt_attributes)

// Encode integer or float-point numbers as bytes.
template <typename T>
std::string EncodeIntegerOrFloat(T attr) {
  std::string data(sizeof(attr), '\0');
  std::memcpy(data.data(), &attr, sizeof(attr));
  return data;
}

// Encode a list of I64 integers as bytes using bc::Vector<uint64_t>. The bytes
// can be decoded directly using bc::Vector<uint64_t>. If `array` is not a list
// I64 integers, a nullopt will be returned.

template <typename T>
std::optional<std::string> EncodeListOfInteger(mlir::ArrayAttr array) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);
  auto ctor = bc::New<bc::Vector<T>>(&allocator, array.size());

  mlir::Type type;

  for (int i = 0; i < array.size(); ++i) {
    if (auto integer_attr = mlir::dyn_cast<mlir::IntegerAttr>(array[i])) {
      if (type && integer_attr.getType() != type) return std::nullopt;
      type = integer_attr.getType();
      llvm::APInt value = integer_attr.getValue();
      if (value.getBitWidth() != sizeof(T) * 8) return std::nullopt;
      ctor.ConstructAt(i, value.getZExtValue());
    } else {
      return std::nullopt;
    }
  }

  return std::string(buffer.data(), buffer.size());
}

std::optional<std::string> EncodeListOfSymbolRef(
    const ModuleEmitterContext& module_context, mlir::ArrayAttr array) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);
  auto ctor = bc::New<bc::Vector<uint32_t>>(&allocator, array.size());

  for (int i = 0; i < array.size(); ++i) {
    if (auto symbol_ref = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(array[i])) {
      ctor.ConstructAt(i, module_context.GetFunctionId(symbol_ref.getValue()));
    } else {
      return std::nullopt;
    }
  }
  return std::string(buffer.data(), buffer.size());
}

template <typename T>
std::optional<std::string> EncodeDenseArray(llvm::ArrayRef<T> array) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);
  auto ctor = bc::New<bc::Vector<T>>(&allocator, array.size());

  if (!array.empty()) {
    ctor.Place(reinterpret_cast<const char*>(array.data()),
               array.size() * sizeof(T));
  }

  return std::string(buffer.data(), buffer.size());
}

// bool values has special encoding in MLIR. It occupies one bit in MLIR
// but in bytecode it is one byte.
std::optional<std::string> EncodeDenseBoolArray(llvm::ArrayRef<bool> array) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);
  auto ctor = bc::New<bc::Vector<uint8_t>>(&allocator, array.size());

  if (!array.empty()) {
    std::vector<uint8_t> data(array.size());
    int i = 0;
    for (auto v : array) {
      data[i++] = static_cast<uint8_t>(v);
    }
    ctor.Place(reinterpret_cast<const char*>(data.data()), data.size());
  }
  return std::string(buffer.data(), buffer.size());
}

// Encode a list of strings as bytes using bc::Vector<bc::String>. The bytes
// can be decoded directly using bc::Vector<bc::String>. If `array` is not a
// list of strings, a nullopt will be returned.
std::optional<std::string> EncodeListOfString(mlir::ArrayAttr array) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);
  auto ctor = bc::New<bc::Vector<bc::String>>(&allocator, array.size());

  for (int i = 0; i < array.size(); ++i) {
    if (auto string_attr = mlir::dyn_cast<mlir::StringAttr>(array[i])) {
      ctor.ConstructAt(i, string_attr.getValue().str());
    } else {
      return std::nullopt;
    }
  }

  return std::string(buffer.data(), buffer.size());
}

struct FunctionEmitterContext {
  explicit FunctionEmitterContext(const ModuleEmitterContext* module_context)
      : module_context(*module_context) {}

  const ModuleEmitterContext& module_context;

  struct RegInfo {
    int num_uses = 0;
    int id = -1;
  };

  int next_reg_id = 0;
  llvm::DenseMap<mlir::Value, RegInfo> register_table;
  std::vector<int> free_regs;

  int AssignRegId() {
    if (free_regs.empty()) {
      return next_reg_id++;
    }
    int id = free_regs.back();
    free_regs.pop_back();
    return id;
  }

  void FreeRegId(int id) { free_regs.push_back(id); }
};

// Emit the bytecode for a kernel. It uses the information in an MLIR operation
// and populates the bytecode using bc::Kernel::Constructor. For a kernel's
// bytecode format, please refer to kernel.h.
void EmitKernel(FunctionEmitterContext& function_context,
                bc::Kernel::Constructor& constructor, mlir::Operation& op,
                std::vector<uint32_t>& function_output_regs,
                std::vector<uint8_t>& function_output_last_uses) {
  // Assign reg ids for results first to make sure results does not reuse reg
  // ids freed from args in the same operation.
  std::vector<uint32_t> results;
  results.reserve(op.getNumResults());
  for (auto result : op.getResults()) {
    auto iter = function_context.register_table.find(result);
    CHECK(iter != function_context.register_table.end());  // Crash Ok
    CHECK_EQ(iter->second.id, -1);                         // Crash Ok
    iter->second.id = function_context.AssignRegId();
    results.push_back(iter->second.id);
  }
  constructor.construct_results(results.size())
      .Assign(results.begin(), results.end());

  std::vector<uint32_t> arguments;
  std::vector<uint8_t> last_uses;
  arguments.reserve(op.getNumOperands());
  last_uses.reserve(op.getNumOperands());
  for (auto operand : op.getOperands()) {
    auto iter = function_context.register_table.find(operand);
    CHECK(iter != function_context.register_table.end());  // Crash Ok
    int id = iter->second.id;
    CHECK_NE(id, -1);  // Crash Ok
    last_uses.push_back(0);
    if (--iter->second.num_uses == 0) {
      function_context.FreeRegId(id);
      last_uses.back() = 1;
    }
    arguments.push_back(id);
  }

  constructor.construct_arguments(arguments.size())
      .Assign(arguments.begin(), arguments.end());
  constructor.construct_last_uses(last_uses.size())
      .Assign(last_uses.begin(), last_uses.end());

  std::vector<uint32_t> attributes;
  attributes.reserve(op.getAttrs().size());
  for (auto attr : op.getAttrs()) {
    int attr_id =
        function_context.module_context.GetAttributeId(attr.getValue());
    absl::string_view attr_data =
        function_context.module_context.attributes().at(attr_id);

    if (CanBeInlined(attr.getValue(), attr_data)) {
      uint32_t data = 0;
      std::memcpy(&data, attr_data.data(), attr_data.size());
      attributes.push_back(data);
    } else {
      attributes.push_back(attr_id);
    }
  }
  constructor.construct_attributes(attributes.size())
      .Assign(attributes.begin(), attributes.end());

  if (llvm::isa<mlir::func::ReturnOp>(&op)) {
    constructor.set_code(function_context.module_context.GetKernelId("return"));

    function_output_regs = std::move(arguments);
    function_output_last_uses = std::move(last_uses);

  } else if (llvm::isa<mlir::func::CallOp>(&op)) {
    constructor.set_code(function_context.module_context.GetKernelId("call"));
  } else {
    llvm::StringRef op_name = op.getName().getStringRef();
    constructor.set_code(function_context.module_context.GetKernelId(op_name));
  }
}

// Emit the bytecode for a function. It uses information in an MLIR function or
// an MLIR region, and populates the bytecode using bc::Function::Constructor.
// For a function's bytecode format, please refer to function.h.
void EmitFunction(const ModuleEmitterContext& module_context,
                  bc::Function::Constructor& constructor, llvm::StringRef name,
                  mlir::Region& region) {
  FunctionEmitterContext function_context(&module_context);

  constructor.construct_name(name.str());

  DCHECK(llvm::hasSingleElement(region)) << "should have a single block";

  auto& block = region.front();

  auto& register_table = function_context.register_table;

  std::vector<uint32_t> input_regs;
  input_regs.reserve(block.getNumArguments());
  for (auto arg : block.getArguments()) {
    int id = function_context.AssignRegId();
    input_regs.push_back(id);
    register_table[arg] = {static_cast<int>(std::distance(arg.getUses().begin(),
                                                          arg.getUses().end())),
                           id};
  }
  constructor.construct_input_regs(input_regs);

  for (auto& op : block) {
    for (auto result : op.getResults()) {
      register_table[result] = {static_cast<int>(
          std::distance(result.getUses().begin(), result.getUses().end()))};
    }
  }

  auto kernels_constructor =
      constructor.construct_kernels(block.getOperations().size());

  std::vector<uint32_t> output_regs;
  std::vector<uint8_t> output_last_uses;
  for (const auto& iter : llvm::enumerate(block.getOperations())) {
    int i = iter.index();
    mlir::Operation& op = iter.value();
    auto kernel_ctor = kernels_constructor.ConstructAt(i);
    EmitKernel(function_context, kernel_ctor, op, output_regs,
               output_last_uses);
  }

  constructor.set_num_regs(function_context.next_reg_id);
  constructor.construct_output_regs(output_regs);
  constructor.construct_output_last_uses(output_last_uses);
}

// Emit the bytecode for an executable. It converts attributes, kernels, and
// functions in an MLIR module to bytecode using bc::Executable::Constructor.
// For an executable's bytecode format, please refer to executable.h.
absl::Status EmitExecutable(ModuleEmitterContext& module_context,
                            bc::Executable::Constructor& constructor,
                            mlir::ModuleOp module) {
  module.walk(
      [&](mlir::func::FuncOp func) { module_context.AddFunction(func); });

  auto functions = module_context.functions();
  for (auto func : functions) {
    if (!llvm::hasSingleElement(func.getRegion())) {
      return absl::InvalidArgumentError("function should have a single block.");
    }
    auto& block = func.getRegion().front();

    for (auto& op : block) {
      if (llvm::isa<mlir::func::CallOp>(&op)) {
        // Canonicalize the MLIR builtin call op's name to "call".
        module_context.AddKernelName("call");
      } else if (llvm::isa<mlir::func::ReturnOp>(&op)) {
        // Canonicalize the return op's name to "return".
        if (op.getNumResults() != 0) {
          return absl::InvalidArgumentError(
              "Block terminator must be a return op.");
        }
        module_context.AddKernelName("return");
      } else {
        module_context.AddKernelName(op.getName().getStringRef().str());
      }

      for (auto attr : op.getAttrs()) {
        if (auto status = module_context.AddAttribute(&op, attr.getValue());
            !status.ok()) {
          return status;
        }
      }

      // TODO(chky): Support inline regions.
    }
  }

  constructor.construct_kernel_names(module_context.kernels().size())
      .Assign(module_context.kernels().begin(), module_context.kernels().end());

  auto functions_constructor =
      constructor.construct_functions(functions.size());
  for (int i = 0; i < functions.size(); ++i) {
    auto func = functions[i];
    auto function_ctor = functions_constructor.ConstructAt(i);
    EmitFunction(module_context, function_ctor, func.getSymName(),
                 func.getRegion());
  }

  // Emit attributes after emitting functions as attributes might be large.
  // Large attributes may result in large offsets that do not fit into a
  // unit32_t integer. Since functions section should fit into 2GB size limit,
  // so we emit functions first.
  constructor.construct_attributes(module_context.attributes().size())
      .Assign(module_context.attributes().begin(),
              module_context.attributes().end());

  return absl::OkStatus();
}

}  // namespace

absl::Status ModuleEmitterContext::AddAttribute(mlir::Operation* op,
                                                mlir::Attribute attr) {
  absl::StatusOr<std::string> attr_data;
  if (auto* encoder = attribute_encoder_registry_.Get(
          op->getName().getDialectNamespace())) {
    attr_data = (*encoder)(*this, attr);
  } else {
    attr_data = DefaultEncodeAttribute(attr);
  }
  if (!attr_data.ok()) return std::move(attr_data).status();

  int id = AddData(std::move(*attr_data), attributes_, attribute_data_id_map_);
  attribute_id_map_[attr] = id;

  return absl::OkStatus();
}

int ModuleEmitterContext::AddFunction(mlir::func::FuncOp func) {
  int id = functions_.size();
  functions_.push_back(func);
  DCHECK(!function_name_id_map_.contains(func.getSymName()));
  function_name_id_map_[func.getSymName()] = id;
  return id;
}

std::optional<std::string> EncodeSimpleAttribute(
    const ModuleEmitterContext& module_context, mlir::Attribute attr) {
  return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
      .Case<mlir::StringAttr>(
          [](const auto& str_attr) { return str_attr.str(); })
      .Case<mlir::IntegerAttr>(
          [](const auto& integer_attr) -> std::optional<std::string> {
            switch (llvm::APInt value = integer_attr.getValue();
                    value.getBitWidth()) {
              case 1:
                return EncodeIntegerOrFloat<uint8_t>(value.getZExtValue());
              case 32:
                return EncodeIntegerOrFloat<uint32_t>(value.getZExtValue());
              case 64:
                return EncodeIntegerOrFloat<uint64_t>(value.getZExtValue());
              default:
                return std::nullopt;
            }
          })
      .Case<mlir::FloatAttr>(
          [](const auto& float_attr) -> std::optional<std::string> {
            llvm::APFloat value = float_attr.getValue();
            if (float_attr.getType().isF32()) {
              return EncodeIntegerOrFloat<float>(value.convertToFloat());
            }
            return std::nullopt;
          })
      .Case<mlir::ArrayAttr>([&](const auto& array_attr)
                                 -> std::optional<std::string> {
        if (auto encoded_list_i32 = EncodeListOfInteger<uint32_t>(array_attr)) {
          return std::move(*encoded_list_i32);
        } else if (auto encoded_list_i64 =
                       EncodeListOfInteger<uint64_t>(array_attr)) {
          return std::move(*encoded_list_i64);
        } else if (auto encoded_list_string = EncodeListOfString(array_attr)) {
          return std::move(*encoded_list_string);
        } else if (auto encoded_list_symbol_ref =
                       EncodeListOfSymbolRef(module_context, array_attr)) {
          return std::move(*encoded_list_symbol_ref);
        } else {
          return std::nullopt;
        }
      })
      .Case<mlir::DenseI32ArrayAttr>(
          [](const auto& dense_array_i32) -> std::optional<std::string> {
            return EncodeDenseArray<int32_t>(dense_array_i32);
          })
      .Case<mlir::DenseI64ArrayAttr>(
          [](const auto& dense_array_i64) -> std::optional<std::string> {
            return EncodeDenseArray<int64_t>(dense_array_i64);
          })
      .Case<mlir::DenseBoolArrayAttr>(
          [](const auto& dense_array_bool) -> std::optional<std::string> {
            return EncodeDenseBoolArray(dense_array_bool.asArrayRef());
          })
      .Case<mlir::FlatSymbolRefAttr>([&](const auto& symbol_ref) {
        return EncodeIntegerOrFloat<uint32_t>(
            module_context.GetFunctionId(symbol_ref.getValue()));
      })
      .Default([](const auto& attr) { return std::nullopt; });
}

// Encode mlir attributes with a limited support such as I64, string and array
// of I64. Returns an error if the attribute is not supported.
absl::StatusOr<std::string> ModuleEmitterContext::DefaultEncodeAttribute(
    mlir::Attribute attr) {
  if (auto result = EncodeSimpleAttribute(*this, attr)) {
    return std::move(*result);
  }

  // TODO(chky): Add a unit test for the error below. This requires we
  // propagate the error all the way back to the entry point.
  std ::string attr_str;
  llvm::raw_string_ostream os(attr_str);
  attr.print(os);

  return absl::InvalidArgumentError(
      absl::StrCat("Try to encode unsupported attribute: ", attr_str));
}

absl::StatusOr<bc::Buffer> EmitExecutable(
    const AttributeEncoderRegistry& attribute_encoder_registry,
    mlir::ModuleOp module) {
  bc::Buffer buffer;
  bc::Allocator allocator(&buffer);

  ModuleEmitterContext module_context(&attribute_encoder_registry);

  auto executable_ctor = bc::New<bc::Executable>(&allocator);

  if (auto status = EmitExecutable(module_context, executable_ctor, module);
      !status.ok()) {
    return status;
  }

  buffer.shrink_to_fit();

  return buffer;
}

}  // namespace mlrt
