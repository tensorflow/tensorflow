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

#include "xla/python/ifrt/ir/transforms/multi_threaded_atom_program_compiler.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/status_macros.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {

namespace {

// Lazily initialized shared thread pool.
tsl::thread::ThreadPool* thread_pool() {
  static tsl::thread::ThreadPool* thread_pool = []() {
    constexpr int kMaxParallelism = 32;
    return new tsl::thread::ThreadPool(tsl::Env::Default(),
                                       tsl::ThreadOptions(),
                                       "CompileAtomPrograms", kMaxParallelism);
  }();
  return thread_pool;
}

void ScheduleWork(tsl::thread::ThreadPool* pool,
                  absl::AnyInvocable<void()> callee) {
  // ThreadPool expects std::function that must be copyable, but we can avoid
  // this by using AnyInvocable.
  pool->Schedule([ptr = new absl::AnyInvocable<void()>(std::move(callee))]() {
    (*ptr)();
    delete ptr;
  });
}

// Construct a bool vector with a True entry for each input sharding that must
// be inferred.
llvm::SmallVector<bool> GetInputShardingPropagation(
    mlir::func::FuncOp func_op) {
  llvm::SmallVector<bool> sharding_propagation_to_input;
  sharding_propagation_to_input.reserve(func_op.getNumArguments());
  for (int idx = 0; idx < func_op.getNumArguments(); ++idx) {
    const auto hlo_sharding_attr =
        func_op.getArgAttrOfType<mlir::StringAttr>(idx, kHloShardingAttrName);
    if (hlo_sharding_attr == nullptr) {
      sharding_propagation_to_input.push_back(true);
    } else {
      sharding_propagation_to_input.push_back(false);
    }
  }
  return sharding_propagation_to_input;
}

// Construct a bool vector with a True entry for each output sharding that must
// be inferred.
llvm::SmallVector<bool> GetOutputShardingPropagation(
    mlir::func::FuncOp func_op) {
  llvm::SmallVector<bool> sharding_propagation_to_output;
  sharding_propagation_to_output.reserve(func_op.getNumResults());
  for (int idx = 0; idx < func_op.getNumResults(); ++idx) {
    const auto hlo_sharding_attr =
        func_op.getResultAttrOfType<mlir::StringAttr>(idx,
                                                      kHloShardingAttrName);
    if (hlo_sharding_attr == nullptr) {
      sharding_propagation_to_output.push_back(true);
    } else {
      sharding_propagation_to_output.push_back(false);
    }
  }
  return sharding_propagation_to_output;
}

}  // namespace

absl::StatusOr<CompileFuture> MultiThreadedAtomProgramCompiler::CompileModule(
    CallOp call_op, mlir::ModuleOp module_op) {
  auto module_type =
      call_op->getAttrOfType<mlir::StringAttr>(kIfrtModuleTypeAttrName);
  if (module_type == kIfrtModuleTypeXla) {
    return CompileXla(call_op, module_op, thread_pool());
  } else if (module_type == kIfrtModuleTypeMpmdReshard) {
    return CompileMpmdReshard(module_op);
  } else if (module_type == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CallOp requires `", kIfrtModuleTypeAttrName.str(), "` to be set"));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("No compiler for module type: ", module_type.str()));
  }
}

absl::StatusOr<xla::CompileOptions>
MultiThreadedAtomProgramCompiler::GetXlaCompileOptions(
    CallOp call_op, mlir::ModuleOp module_op) {
  xla::CompileOptions compile_options;

  // If the CallOp has a compile options key, then try to use the provided
  // compile options.
  auto compile_options_key =
      call_op->getAttrOfType<mlir::StringAttr>(kIfrtCompileOptionsKey);
  bool has_compile_options = false;
  if (compile_options_overrides_ != nullptr && compile_options_key != nullptr) {
    if (auto compile_options_override =
            compile_options_overrides_->find(compile_options_key.str());
        compile_options_override != compile_options_overrides_->end()) {
      if (auto xla_options = llvm::dyn_cast<XlaCompileOptions>(
              compile_options_override->second.get())) {
        compile_options = xla_options->compile_options;
        has_compile_options = true;
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "The `", kIfrtCompileOptionsKey.str(), "` compile options key `",
            compile_options_key.str(),
            "` has an entry that is not of type `XlaCompileOptions`, but the "
            "atom program is an XLA program."));
      }
    }
  }

  if (!has_compile_options) {
    auto& exec_build_options = compile_options.executable_build_options;
    // Executable build options are constructed using logical ids, which are
    // later converted into real Device ids by using the logical ids as
    // indices into the device list given at compilation invocation time.
    llvm::ArrayRef<int> logical_device_ids = call_op.getDevices();
    if (call_op->hasAttrOfType<mlir::UnitAttr>(kIfrtLocalViewAttrName)) {
      exec_build_options.set_num_replicas(logical_device_ids.size());
      exec_build_options.set_num_partitions(1);
      xla::DeviceAssignment device_assignment(logical_device_ids.size(), 1);
      for (const auto [i, device_id] : llvm::enumerate(logical_device_ids)) {
        device_assignment(i, 0) = device_id;
      }
      exec_build_options.set_device_assignment(device_assignment);
    } else {
      exec_build_options.set_num_replicas(1);
      exec_build_options.set_num_partitions(logical_device_ids.size());
      xla::DeviceAssignment device_assignment(1, logical_device_ids.size());
      for (const auto [i, device_id] : llvm::enumerate(logical_device_ids)) {
        device_assignment(0, i) = device_id;
      }
      exec_build_options.set_device_assignment(device_assignment);
      exec_build_options.set_use_spmd_partitioning(true);
      if (enable_sharding_propagation_) {
        mlir::func::FuncOp main_func = GetMainFunction(module_op);
        exec_build_options.set_allow_spmd_sharding_propagation_to_parameters(
            GetInputShardingPropagation(main_func));
        exec_build_options.set_allow_spmd_sharding_propagation_to_output(
            GetOutputShardingPropagation(main_func));
      }
    }
  }

  return compile_options;
}

absl::StatusOr<CompileFuture> MultiThreadedAtomProgramCompiler::CompileXla(
    CallOp call_op, mlir::ModuleOp module_op,
    tsl::thread::ThreadPool* thread_pool) {
  TF_ASSIGN_OR_RETURN(xla::CompileOptions compile_options,
                      GetXlaCompileOptions(call_op, module_op));

  // We must clone the module in order ensure the module string representation
  // is maintained. This is because MLIR printing takes different paths
  // depending on if a ModuleOp has a parent or not.

  auto hlo_program = std::make_unique<HloProgram>(
      /*context=*/nullptr,  // Shares the same long-living context.
      mlir::OwningOpRef<mlir::ModuleOp>(module_op.clone()));
  Promise<AtomProgramCompileResult> promise = CompileFuture::CreatePromise();
  CompileFuture future(promise);
  ScheduleWork(
      thread_pool, [this, hlo_program = std::move(hlo_program),
                    compile_options = std::move(compile_options),
                    promise = std::move(promise)]() mutable {
        promise.Set(compiler_->CompileXla(std::move(hlo_program),
                                          std::move(compile_options)));
      });
  return future;
}

absl::StatusOr<CompileFuture>
MultiThreadedAtomProgramCompiler::CompileMpmdReshard(mlir::ModuleOp module_op) {
  auto main_func =
      module_op.lookupSymbol<mlir::func::FuncOp>(kCalleeMainFuncName);
  TF_RET_CHECK(main_func) << "requires module to have"
                          << kCalleeMainFuncName.str() << " function";
  std::vector<DType> dtypes;
  std::vector<Shape> shapes;
  std::vector<IfrtArrayType> in_arrays_types;
  std::vector<IfrtArrayType> out_arrays_types;
  dtypes.reserve(main_func.getArgumentTypes().size());
  shapes.reserve(main_func.getArgumentTypes().size());
  in_arrays_types.reserve(main_func.getArgumentTypes().size());
  out_arrays_types.reserve(main_func.getResultTypes().size());
  for (const mlir::Type arg_type : main_func.getArgumentTypes()) {
    auto array_type = mlir::dyn_cast<IfrtArrayType>(arg_type);
    TF_RET_CHECK(array_type != nullptr)
        << "Unsupported argument type `" << mlir::debugString(arg_type) << "`";
    TF_ASSIGN_OR_RETURN(DType dtype,
                        ToIfrtDType(array_type.getShape().getElementType()));
    dtypes.push_back(std::move(dtype));
    shapes.push_back(Shape(array_type.getShape().getShape()));
    in_arrays_types.push_back(array_type);
  }
  for (const mlir::Type result_type : main_func.getResultTypes()) {
    auto array_type = mlir::dyn_cast<IfrtArrayType>(result_type);
    TF_RET_CHECK(array_type != nullptr)
        << "Unsupported return type `" << mlir::debugString(result_type) << "`";
    out_arrays_types.push_back(array_type);
  }
  auto promise = CompileFuture::CreatePromise();
  CompileFuture future(promise);
  // No need to dispatch from a different thread because MpmdReshard uses its
  // own thread pool already.
  auto compile_result = compiler_->CompileMpmdReshard(
      std::move(dtypes), std::move(shapes), in_arrays_types, out_arrays_types);
  promise.Set(std::move(compile_result));
  return future;
}

}  // namespace ifrt
}  // namespace xla
