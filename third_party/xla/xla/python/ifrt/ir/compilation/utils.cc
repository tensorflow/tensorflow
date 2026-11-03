/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/compilation/utils.h"

#include <cstdint>  // IWYU pragma: keep
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/device_assignment.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace ifrt {

namespace {

// Constructs a bool vector with a True entry for each input sharding that must
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

// Constructs a bool vector with a True entry for each output sharding that must
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

#if !defined(PLATFORM_GOOGLE)
absl::Status SetStrictMemoryReservation(absl::string_view program_name,
                                        int64_t device_memory,
                                        xla::CompileOptions& options) {
  return absl::OkStatus();
}
#endif

absl::StatusOr<std::vector<std::string>> ExpandPlatformNames(
    const mlir::Pass::ListOption<std::string>& platform_names) {
  std::vector<std::string> expanded_platform_names;
  for (const auto& platform_entry : platform_names) {
    std::vector<absl::string_view> parts = absl::StrSplit(platform_entry, ':');
    if (parts.size() == 1) {
      expanded_platform_names.push_back(platform_entry);
    } else if (parts.size() == 2) {
      std::string platform_name(parts[0]);
      int num_occurences;
      if (!absl::SimpleAtoi(parts[1], &num_occurences)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid platform name: `", platform_entry,
                         "` in platform_names pass option"));
      }
      for (int i = 0; i < num_occurences; ++i) {
        expanded_platform_names.push_back(platform_name);
      }
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid platform name: `", platform_entry,
                       "` in platform_names pass option"));
    }
  }
  return expanded_platform_names;
}

absl::StatusOr<XlaCompileOptions*> GetModuleXlaCompileOverrides(
    mlir::StringAttr compile_options_key,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides) {
  XlaCompileOptions* compile_options = nullptr;
  if (compile_options_overrides != nullptr && compile_options_key != nullptr) {
    if (auto option_override =
            compile_options_overrides->find(compile_options_key.str());
        option_override != compile_options_overrides->end()) {
      if (auto xla_options = xla::ifrt::dyn_cast<XlaCompileOptions>(
              option_override->second.get())) {
        compile_options = xla_options;
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "The `", kIfrtCompileOptionsKey.str(), "` compile options key `",
            compile_options_key.str(),
            "` has an entry that is not of type `XlaCompileOptions`, but the "
            "atom program is an XLA program."));
      }
    }
  }

  return compile_options;
}

absl::StatusOr<std::optional<xla::CompileOptions>> GetModuleCompileOverrides(
    mlir::StringAttr compile_options_key,
    std::shared_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<CompileOptions>>>
        compile_options_overrides) {
  ABSL_ASSIGN_OR_RETURN(XlaCompileOptions * xla_compile_options,
                   GetModuleXlaCompileOverrides(compile_options_key,
                                                compile_options_overrides));
  if (xla_compile_options == nullptr) {
    return std::nullopt;
  }
  return xla_compile_options->compile_options;
}

xla::CompileOptions GetDefaultCompileOptions(CallOp call_op,
                                             bool enable_sharding_propagation,
                                             bool enable_parameter_tupling) {
  mlir::SymbolTableCollection symbol_table;

  xla::CompileOptions compile_options;
  auto& exec_build_options = compile_options.executable_build_options;
  // Executable build options are constructed using logical ids, which are
  // later converted into real Device ids by using the logical ids as
  // indices into the device list given at compilation invocation time.
  llvm::ArrayRef<int> logical_device_ids = call_op.getDevices();
  mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
  CHECK(callee != nullptr) << "Callee function not found for CallOp";
  if (call_op->hasAttrOfType<mlir::UnitAttr>(kIfrtLocalViewAttrName)) {
    exec_build_options.set_num_replicas(logical_device_ids.size());
    exec_build_options.set_num_partitions(1);
    xla::DeviceAssignment device_assignment(logical_device_ids.size(), 1);
    for (const auto [i, device_id] : llvm::enumerate(logical_device_ids)) {
      device_assignment(i, 0) = device_id;
    }
    exec_build_options.set_device_assignment(device_assignment);
    exec_build_options.set_use_spmd_partitioning(false);
  } else {
    exec_build_options.set_num_replicas(1);
    exec_build_options.set_num_partitions(logical_device_ids.size());
    xla::DeviceAssignment device_assignment(1, logical_device_ids.size());
    for (const auto [i, device_id] : llvm::enumerate(logical_device_ids)) {
      device_assignment(0, i) = device_id;
    }
    exec_build_options.set_device_assignment(device_assignment);
    exec_build_options.set_use_spmd_partitioning(true);
    mlir::ModuleOp callee_module = callee->getParentOfType<mlir::ModuleOp>();
    if (xla::sdy::hasShardyMesh(callee_module)) {
      exec_build_options.set_use_shardy_partitioner(true);
    }
    if (enable_sharding_propagation) {
      exec_build_options.set_allow_spmd_sharding_propagation_to_parameters(
          GetInputShardingPropagation(callee));
      exec_build_options.set_allow_spmd_sharding_propagation_to_output(
          GetOutputShardingPropagation(callee));
    }
  }

  compile_options.parameter_is_tupled_arguments = enable_parameter_tupling;

  return compile_options;
}

}  // namespace ifrt
}  // namespace xla
