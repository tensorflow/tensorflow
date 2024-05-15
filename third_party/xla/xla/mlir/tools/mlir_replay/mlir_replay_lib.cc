/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/mlir/tools/mlir_replay/mlir_replay_lib.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/gaussian_distribution.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/ParseUtilities.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace_utils.h"
#include "xla/service/hlo.pb.h"
#include "tsl/platform/statusor.h"

namespace mlir {
namespace interpreter {
namespace {

absl::StatusOr<SmallVector<InterpreterValue>> LoadArgs(
    const xla::HloSnapshot& snapshot, TypeRange types) {
  SmallVector<InterpreterValue> result;
  for (const auto& [arg, type] : llvm::zip(snapshot.arguments(), types)) {
    TF_ASSIGN_OR_RETURN(auto converted, LiteralToValue(arg, type));
    result.push_back(std::move(converted));
  }
  return result;
}

namespace {
template <typename T, template <typename _> class rng_t>
mlir::interpreter::InterpreterValue RandomTensor(absl::BitGenRef bitgen,
                                                 mlir::Type type) {
  llvm::SmallVector<int64_t> shape;
  auto shaped_ty = type.dyn_cast<mlir::ShapedType>();
  if (shaped_ty) {
    shape = llvm::to_vector(shaped_ty.getShape());
  }

  auto rng = rng_t<T>{};
  auto result = mlir::interpreter::TensorOrMemref<T>::Empty(shape);
  for (const auto& index : result.view.Indices()) {
    auto& elem = result.at(index) = rng(bitgen);
    // Ints are typically indices, so scale them down to a more reasonable
    // range.
    if constexpr (std::is_same_v<T, int64_t>) {
      elem >>= 60;
    }
  }
  if (shaped_ty) {
    return {result};
  }
  return {result.at({})};
}
}  // namespace

mlir::FailureOr<mlir::interpreter::InterpreterValue> MakeRandomInput(
    absl::BitGenRef bitgen, mlir::Type type) {
  auto elem_ty =
      type.isa<ShapedType>() ? type.cast<ShapedType>().getElementType() : type;
  if (elem_ty.isF32()) {
    return RandomTensor<float, absl::gaussian_distribution>(bitgen, type);
  }
  if (elem_ty.isF64()) {
    return RandomTensor<double, absl::gaussian_distribution>(bitgen, type);
  }
  if (elem_ty.isInteger(32)) {
    return RandomTensor<int32_t, absl::uniform_int_distribution>(bitgen, type);
  }
  if (elem_ty.isInteger(16)) {
    return RandomTensor<int16_t, absl::uniform_int_distribution>(bitgen, type);
  }
  if (elem_ty.isInteger(64)) {
    return RandomTensor<int64_t, absl::uniform_int_distribution>(bitgen, type);
  }
  if (elem_ty.isInteger(1)) {
    return {{TensorOrMemref<bool>::Empty(type.cast<ShapedType>().getShape())}};
  }

  llvm::errs() << "Unsupported type: ";
  type.print(llvm::errs());
  llvm::errs() << "\n";
  return failure();
}

// TODO(jreiffers): Add a flag to intentionally alias as many buffers as
// possible (in particular, all non-variable inputs).
// Extracts a mapping from function arguments to allocated buffers.
// The buffer assignment is only relevant once the program is bufferized and
// memref results were converted to arguments.
std::vector<int64_t> ExtractXlaBufferAssignment(func::FuncOp main) {
  std::vector<int64_t> buffer_assignment(main.getNumArguments());
  auto result_mapping =
      main->getAttrOfType<IntegerAttr>("xla_framework.result_mapping");
  if (!result_mapping) {
    // No attribute, fall back to unique buffers for each argument.
    std::iota(buffer_assignment.begin(), buffer_assignment.end(), 0);
    return buffer_assignment;
  }

  std::vector<int64_t> result_to_buffer;
  if (auto inner_mapping = main->getAttrOfType<ArrayAttr>(
          "xla_framework.result_inner_mapping")) {
    llvm::copy(llvm::map_range(inner_mapping.getAsValueRange<IntegerAttr>(),
                               [](const llvm::APInt& value) {
                                 return value.getSExtValue();
                               }),
               std::back_inserter(result_to_buffer));
  } else {
    result_to_buffer = {result_mapping.getInt()};
  }

  int64_t result_index = 0;
  for (int64_t arg_index : llvm::seq<int64_t>(0, main.getNumArguments())) {
    if (auto input_buffer_index = main.getArgAttrOfType<IntegerAttr>(
            arg_index, "xla_framework.input_mapping")) {
      buffer_assignment[arg_index] = input_buffer_index.getInt();
    } else {
      buffer_assignment[arg_index] = result_to_buffer[result_index++];
    }
  }

  return buffer_assignment;
}

}  // namespace

absl::StatusOr<SmallVector<InterpreterValue>> Run(
    MLIRContext& context, const std::string& mlir_ir,
    const xla::HloSnapshot& snapshot, ExecutionTrace* trace,
    const std::vector<std::string>& entry) {
  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlir_ir),
                                 mlir::SMLoc());
  mlir::OwningOpRef<mlir::Operation*> module =
      mlir::parseSourceFileForTool(source_mgr, &context, false);
  if (!module) {
    return absl::InvalidArgumentError("failed to parse MLIR");
  }

  SymbolTable symbols(*module);
  func::FuncOp main;
  for (const std::string& candidate : entry) {
    main = llvm::dyn_cast_or_null<func::FuncOp>(symbols.lookup(candidate));
    if (main && !main.getBody().empty()) {
      break;
    }
  }

  if (!main) {
    return absl::InvalidArgumentError("failed to find entry point");
  }

  if (trace) {
    llvm::raw_string_ostream os(*trace->mutable_ir());
    (*module)->print(os, OpPrintingFlags().printGenericOpForm());
  }

  auto function_args = main.getBody().getBlocks().front().getArguments();
  if (!llvm::all_of(function_args, [&](Value arg) {
        return isa<ShapedType>(arg.getType());
      })) {
    return absl::InvalidArgumentError(
        "expected all function arguments to be shaped types");
  }

  auto args_to_buffers = ExtractXlaBufferAssignment(main);
  TF_ASSIGN_OR_RETURN(auto args,
                      LoadArgs(snapshot, main.getBody().getArgumentTypes()));
  auto out_args =
      main.getBody().getBlocks().front().getArguments().drop_front(args.size());

  absl::flat_hash_map<int64_t, InterpreterValue> buffer_to_value;
  // None of the input arguments will be statically known to alias.
  for (auto [index, value] : llvm::enumerate(args)) {
    buffer_to_value[args_to_buffers[index]] = value;
  }

  std::seed_seq my_seed_seq({0});
  absl::BitGen bitgen(my_seed_seq);
  llvm::SmallVector<InterpreterValue> out_buffers;
  // Add random inputs for output arguments and unspecified inputs.
  for (auto arg : out_args) {
    auto ty = arg.getType();

    int64_t buffer_index = args_to_buffers[arg.getArgNumber()];
    // If we already have a buffer for this argument, use it.
    if (buffer_to_value.contains(buffer_index)) {
      auto& value = buffer_to_value[buffer_index];
      out_buffers.push_back(value);
      args.push_back(value);
      continue;
    }

    auto arg_or = MakeRandomInput(bitgen, ty);
    if (!succeeded(arg_or)) {
      return absl::InvalidArgumentError("failed to create input");
    }
    out_buffers.push_back(*arg_or);
    args.push_back(*arg_or);
    buffer_to_value[buffer_index] = *arg_or;
  }

  InterpreterOptions options;
  ExecutionTraceListener tracer(trace);
  if (trace) {
    options.listener = &tracer;
  }
  TF_ASSIGN_OR_RETURN(auto results,
                      RunInterpreter(symbols, main, args, options));

  if (results.empty()) {
    return out_buffers;
  }
  return results;
}

}  // namespace interpreter
}  // namespace mlir
