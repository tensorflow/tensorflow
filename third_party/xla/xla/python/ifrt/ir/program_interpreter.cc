/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/program_interpreter.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/remap_plan.pb.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

using ArrayRef = ::xla::ifrt::ArrayRef;
using ExecuteOptions = ::xla::ifrt::LoadedExecutable::ExecuteOptions;
using ExecuteResult = ::xla::ifrt::LoadedExecutable::ExecuteResult;

namespace {

// Array with additional metadata (e.g., if it can be donated).
struct ArrayState {
  ArrayRef array;
  bool can_be_donated;
};

// Returns an xla::ifrt::Sharding for the given IFRT array type.
absl::StatusOr<xla::ifrt::ShardingRef> GetSharding(
    xla::ifrt::IfrtArrayType array_type, xla::ifrt::Client* client,
    const xla::ifrt::DeviceListRef& devices) {
  const absl::Span<xla::ifrt::Device* const> in_devices = devices->devices();
  absl::InlinedVector<xla::ifrt::Device*, 1> out_devices;
  out_devices.reserve(array_type.getDevices().size());
  for (int logical_id : array_type.getDevices()) {
    out_devices.push_back(in_devices[logical_id]);
  }
  auto sharding_param_attr =
      mlir::dyn_cast_or_null<xla::ifrt::IfrtShardingParamAttr>(
          array_type.getShardingAttr());
  TF_RET_CHECK(sharding_param_attr != nullptr)
      << "Array type: " << mlir::debugString(array_type)
      << " if not of type `IfrtShardingParamAttr`";
  TF_ASSIGN_OR_RETURN(DeviceListRef device_list,
                      client->MakeDeviceList(std::move(out_devices)));
  TF_ASSIGN_OR_RETURN(auto sharding,
                      xla::ifrt::ShardingParamSharding::Create(
                          sharding_param_attr.getSharding(),
                          std::move(device_list), array_type.MemoryKind()));
  return sharding;
}

std::string PrettyPrintGeneric(mlir::Operation* op) {
  return absl::StrCat(op->getName().getStringRef().str(), " from:\n",
                      GetPrettyLocation(op->getLoc()));
}

// Populates the cache storing a Sharding for each IfrtArrayType.
//
// This cache exists to avoid traversing and creating large device lists at
// execution time.
//
// Note that the cache is only populated for array types returned by CopyArrays
// and RemapArrays ops because they are the only ops that need shardings.
absl::StatusOr<llvm::DenseMap<xla::ifrt::IfrtArrayType, xla::ifrt::ShardingRef>>
PopulateShardingCache(mlir::func::FuncOp main_func, xla::ifrt::Client* client,
                      const xla::ifrt::DeviceListRef& devices) {
  llvm::DenseMap<xla::ifrt::IfrtArrayType, xla::ifrt::ShardingRef>
      array_type_to_sharding;
  for (const mlir::Operation& op : main_func.getOps()) {
    if (auto copy_arrays_op = llvm::dyn_cast<xla::ifrt::CopyArraysOp>(&op);
        copy_arrays_op != nullptr) {
      for (const auto [idx, output] :
           llvm::enumerate(copy_arrays_op.getOutputs())) {
        const auto array_type =
            llvm::cast<xla::ifrt::IfrtArrayType>(output.getType());
        TF_RET_CHECK(array_type != nullptr)
            << "Output array #" << idx << " is not of type `IfrtArrayType`. "
            << PrettyPrintGeneric(copy_arrays_op);
        if (array_type_to_sharding.find(array_type) ==
            array_type_to_sharding.end()) {
          TF_ASSIGN_OR_RETURN(auto sharding,
                              GetSharding(array_type, client, devices));
          array_type_to_sharding[array_type] = std::move(sharding);
        }
      }
    } else if (auto remap_op = llvm::dyn_cast<xla::ifrt::RemapArraysOp>(&op);
               remap_op != nullptr) {
      for (const auto [idx, output] : llvm::enumerate(remap_op.getOutputs())) {
        const auto array_type =
            llvm::cast<xla::ifrt::IfrtArrayType>(output.getType());
        TF_RET_CHECK(array_type != nullptr)
            << "Output array #" << idx << " is not of type `IfrtArrayType`. "
            << PrettyPrintGeneric(remap_op);
        if (array_type_to_sharding.find(array_type) ==
            array_type_to_sharding.end()) {
          TF_ASSIGN_OR_RETURN(auto sharding,
                              GetSharding(array_type, client, devices));
          array_type_to_sharding[array_type] = std::move(sharding);
        }
      }
    }
  }
  return array_type_to_sharding;
}

}  // namespace

struct Environment {
  // Associates array with an MLIR value.
  void AssociateArray(mlir::Value value, ArrayState array) {
    CHECK(value_to_array.try_emplace(value, array).second);
  }

  // Map from MLIR value to IFRT array corresponding to the value.
  llvm::DenseMap<mlir::Value, ArrayState> value_to_array;
  // Outputs of the program.
  std::vector<ArrayRef> outputs;
  // `ExecuteOptions.fill_status` passed to Execute().
  bool fill_status;
  // Contains a future for each ifrt.CallOp that is a leaf (i.e., has no outputs
  // or all its outputs are returned from the program).
  std::vector<tsl::Future<>> leaf_call_op_futures;
};

absl::StatusOr<std::unique_ptr<ProgramInterpreter>> ProgramInterpreter::Create(
    xla::ifrt::Client* client, std::shared_ptr<CompiledIfrtIrProgram> program,
    xla::ifrt::DeviceListRef devices) {
  mlir::func::FuncOp main_func =
      xla::ifrt::GetMainFunction(program->program->mlir_module);
  if (!main_func->hasAttr(xla::ifrt::kIfrtFunctionAttrName)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "`main` function of IFRT IR program: ", program->program_name,
        " is not an IFRT function."));
  }
  TF_ASSIGN_OR_RETURN(auto array_type_to_sharding,
                      PopulateShardingCache(main_func, client, devices));
  return std::unique_ptr<ProgramInterpreter>(new ProgramInterpreter(
      client, std::move(program), std::move(devices), mlir::Liveness(main_func),
      std::move(array_type_to_sharding)));
}

absl::StatusOr<ExecuteResult> ProgramInterpreter::Execute(
    absl::Span<ArrayRef> arrays, const ExecuteOptions& options,
    std::optional<xla::ifrt::DeviceListRef> devices) {
  TraceMe traceme([&]() {
    return TraceMeEncode("DispatchProgram",
                         {
                             {"ifrt_ir_program", program_->program_name},
                         });
  });
  VLOG(2) << "Started interpreting program: " << program_->program_name;
  mlir::func::FuncOp main_func =
      xla::ifrt::GetMainFunction(program_->program->mlir_module);
  if (arrays.size() != main_func.getNumArguments()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "`main` function of IFRT IR program: ", program_->program_name,
        " invoked with ", arrays.size(), " arguments, but it expects ",
        main_func.getNumArguments(), " arguments."));
  }

  for (const auto& [idx, array] : llvm::enumerate(arrays)) {
    if (array->IsDeleted()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input array #", idx, " of program ", program_->program_name,
          " has already been deleted or donated."));
    }
  }

  Environment env;
  env.fill_status = options.fill_status;
  for (const auto [idx, arg] : llvm::enumerate(main_func.getArguments())) {
    // Add to the environment the arrays that are used.
    bool is_donated = main_func.getArgAttr(
                          idx, xla::ifrt::kIfrtDonatedArgAttrName) != nullptr &&
                      !options.non_donatable_input_indices.contains(idx);
    if (!arg.use_empty()) {
      env.AssociateArray(arg, ArrayState{/*array=*/arrays[idx],
                                         /*can_be_donated=*/is_donated});
      if (is_donated) {
        deletable_program_arguments_.insert(arg);
      }
    } else if (is_donated) {
      // If the argument is donated but not used, it can be deleted.
      arrays[idx]->Delete();
    }
  }

  // Walk ops one-by-one in program order, and dispatch atom program and
  // copy arrays.
  for (mlir::Operation& op : main_func.getOps()) {
    auto exec_op_status =
        llvm::TypeSwitch<const mlir::Operation&, absl::Status>(op)
            .Case<xla::ifrt::CallLoadedExecutableOp, xla::ifrt::RemapArraysOp,
                  xla::ifrt::CopyArraysOp, mlir::func::ReturnOp>(
                [&](const auto& op) { return ExecuteOp(op, env); })
            .Default([&](const auto& op) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Interpreter found unexpected op: ", mlir::debugString(op)));
            });
    if (!exec_op_status.ok()) {
      tsl::errors::AppendToMessage(&exec_op_status, PrettyPrint(&op));
      return exec_op_status;
    }
  }

  VLOG(2) << "Finished interpreting program: " << program_->program_name;
  ExecuteResult result;
  if (env.fill_status) {
    result.status = tsl::JoinFutures(absl::MakeSpan(env.leaf_call_op_futures));
  }
  result.outputs = std::move(env.outputs);
  return result;
}

absl::Status ProgramInterpreter::ExecuteOp(
    xla::ifrt::CallLoadedExecutableOp call_loaded_op, Environment& env) {
  xla::ifrt::LoadedExecutableOp loaded_exec_op =
      call_loaded_op.getCalleeOp(symbol_table_);
  std::string atom_program_name = loaded_exec_op.getSymName().str();
  TraceMe traceme([&]() {
    return TraceMeEncode("DispatchLoadedExecutableOp",
                         {
                             {"ifrt_ir_program", program_->program_name},
                             {"atom_program", atom_program_name},
                         });
  });
  std::string op_name = call_loaded_op->getName().getStringRef().str();
  VLOG(3) << PrettyPrint(call_loaded_op);
  // Get the loaded executable for the atom program.
  auto exec_it = program_->atom_program_executables->find(atom_program_name);
  TF_RET_CHECK(exec_it != program_->atom_program_executables->end())
      << "Could not find executable. " << PrettyPrint(call_loaded_op);

  absl::flat_hash_set<int> donated_arg_idxs(
      call_loaded_op.getDonatedInputIndices().begin(),
      call_loaded_op.getDonatedInputIndices().end());
  for (const auto& io_alias :
       call_loaded_op.getIoAliases().getAsRange<mlir::DenseI32ArrayAttr>()) {
    // Insert the aliased input to the set.
    donated_arg_idxs.insert(io_alias.asArrayRef()[0]);
  }
  // Get the inputs of the loaded executable.
  std::vector<ArrayRef> inputs;
  xla::ifrt::LoadedExecutable::ExecuteOptions execute_options;
  execute_options.fill_status = env.fill_status;
  llvm::DenseSet<mlir::Value> array_values_to_gc_from_env;
  for (const auto [idx, input] : llvm::enumerate(call_loaded_op.getInputs())) {
    auto array_it = env.value_to_array.find(input);
    TF_RET_CHECK(array_it != env.value_to_array.end())
        << "Input array #" << idx << " not found. "
        << PrettyPrint(call_loaded_op);
    if (array_it->second.array->IsDeleted()) {
      // We explicitly check here for deletion in order to provide a more
      // informative error message.
      return absl::InvalidArgumentError(absl::StrCat(
          "Input array #", idx, "` has already been deleted or donated. ",
          PrettyPrint(call_loaded_op)));
    }
    inputs.push_back(array_it->second.array);

    bool is_donated = donated_arg_idxs.contains(idx);
    if (is_donated && !array_it->second.can_be_donated) {
      VLOG(2) << "Atom program donates input #" << idx
              << ", but it has not been donated to the IFRT IR program. "
                 "Input will not be donated. \n"
              << PrettyPrint(call_loaded_op);
      is_donated = false;
    }
    if (is_donated || liveness_.isDeadAfter(input, call_loaded_op)) {
      array_values_to_gc_from_env.insert(input);
    }
    if (!is_donated) {
      execute_options.non_donatable_input_indices.insert(idx);
    }
  }

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutable::ExecuteResult result,
      exec_it->second->Execute(absl::MakeSpan(inputs), execute_options,
                               /*devices=*/std::nullopt));
  TF_RET_CHECK(result.outputs.size() == call_loaded_op.getOutputs().size())
      << "Got " << result.outputs.size() << " results, but atom program has "
      << call_loaded_op.getOutputs().size() << ". "
      << PrettyPrint(call_loaded_op);

  // Remove the arrays from the environment after the inputs vector is created.
  // This is because in situations such as `ifrt.Call(%0, %0)` the liveness
  // analysis will return that %0 is dead, but it's used for the second
  // argument.
  for (const auto& array_value : array_values_to_gc_from_env) {
    if (deletable_program_arguments_.erase(array_value)) {
      // Explicitly delete donated program arguments that are not used later.
      env.value_to_array[array_value].array->Delete();
    }
    env.value_to_array.erase(array_value);
  }

  bool is_leaf_op = true;
  for (const auto [output_array, output] :
       llvm::zip(result.outputs, call_loaded_op.getOutputs())) {
    if (!output.use_empty()) {
      // The output array is kept only if it used later. This can happen if
      // an executable has multiple output arrays, but only some of them are
      // used.
      env.AssociateArray(output, ArrayState{/*array=*/std::move(output_array),
                                            /*can_be_donated=*/true});
    }
    if (is_leaf_op) {
      for (mlir::OpOperand& use : output.getUses()) {
        // An ifrt.CallOp is not a leaf if any of its outputs are not returned.
        if (llvm::dyn_cast<mlir::func::ReturnOp>(use.getOwner()) == nullptr) {
          is_leaf_op = false;
          break;
        }
      }
    }
  }
  if (is_leaf_op && env.fill_status) {
    env.leaf_call_op_futures.push_back(std::move(result.status));
  }

  return absl::OkStatus();
}

absl::Status ProgramInterpreter::ExecuteOp(xla::ifrt::RemapArraysOp remap_op,
                                           Environment& env) {
  TraceMe traceme([&]() {
    return TraceMeEncode("DispatchRemapArraysOp",
                         {{"ifrt_ir_program", program_->program_name}});
  });
  std::string op_name = remap_op->getName().getStringRef().str();
  VLOG(3) << PrettyPrint(remap_op);

  // Construct the mappings of the remap plan.
  auto mappings =
      std::make_shared<std::vector<xla::ifrt::RemapPlan::Mapping>>();
  mappings->reserve(remap_op.getMappings().size());
  for (const auto& array_mapping : remap_op.getMappings()) {
    const auto array_mapping_attr =
        llvm::cast<xla::ifrt::IfrtArrayMappingAttr>(array_mapping);
    auto& mapping = mappings->emplace_back();
    mapping.in_array = array_mapping_attr.getInArrayIndex();
    mapping.out_array = array_mapping_attr.getOutArrayIndex();
    mapping.from.reserve(array_mapping_attr.getMappings().size());
    mapping.to.reserve(array_mapping_attr.getMappings().size());
    for (const auto& m : array_mapping_attr.getMappings()) {
      const auto mapping_attr = llvm::cast<xla::ifrt::IfrtMappingAttr>(m);
      auto from_shards = mapping_attr.getFromShards();
      auto to_shards = mapping_attr.getToShards();
      mapping.from.push_back(xla::ifrt::RemapPlan::Interval{
          from_shards.getStart(), from_shards.getEnd(), from_shards.getStep()});
      mapping.to.push_back(xla::ifrt::RemapPlan::Interval{
          to_shards.getStart(), to_shards.getEnd(), to_shards.getStep()});
    }
  };

  std::vector<ArrayRef> inputs;
  std::vector<xla::ifrt::ArraySpec> input_specs;
  inputs.reserve(remap_op.getInputs().size());
  input_specs.reserve(remap_op.getInputs().size());
  // Get the input specs of the remap plan and the input arrays.
  llvm::DenseSet<mlir::Value> array_values_to_gc_from_env;
  std::optional<bool> is_donated;
  for (const auto [idx, input] : llvm::enumerate(remap_op.getInputs())) {
    auto array_it = env.value_to_array.find(input);
    TF_RET_CHECK(array_it != env.value_to_array.end())
        << "Input array #" << idx << " not found. " << PrettyPrint(remap_op);
    if (array_it->second.array->IsDeleted()) {
      // We explicitly check here for deletion in order to provide a more
      // informative error message.
      return absl::InvalidArgumentError(absl::StrCat(
          "Input array #", idx, " has already been deleted or donated. ",
          PrettyPrint(remap_op)));
    }
    inputs.push_back(array_it->second.array);
    input_specs.push_back(xla::ifrt::ArraySpec{
        /*dtype=*/array_it->second.array->dtype(),
        /*shape=*/array_it->second.array->shape(),
        /*sharding=*/array_it->second.array->shared_ptr_sharding()});

    // The default buffer donation semantic is finalized at compilation time.
    // Users can override the donation semantic at runtime. In the meantime, the
    // IFRT client RemapArrays API requires all input arrays have the same
    // donation semantic.
    if (!is_donated.has_value()) {
      is_donated = remap_op.getDonated() && array_it->second.can_be_donated;
    }
    if (*is_donated && !array_it->second.can_be_donated) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Donation semantic must be consistent across all input arrays of "
          "RemapArraysOp. Input array #",
          idx,
          " cannot be donated, but previous input arrays can be donated. It's "
          "likely due to a MPMD program argument is marked as non-donatable. ",
          PrettyPrint(remap_op)));
    }
    if (*is_donated || liveness_.isDeadAfter(input, remap_op)) {
      array_values_to_gc_from_env.insert(input);
    }
  }
  TF_RET_CHECK(is_donated.has_value())
      << "Unable to determine the donation semantic of the remap op. The remap "
         "op has no inputs. "
      << PrettyPrint(remap_op);

  // Get the output specs of the remap plan.
  std::vector<xla::ifrt::ArraySpec> output_specs;
  output_specs.reserve(remap_op.getOutputs().size());
  for (const auto [idx, output] : llvm::enumerate(remap_op.getOutputs())) {
    const auto array_type =
        llvm::cast<xla::ifrt::IfrtArrayType>(output.getType());
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::DType dtype,
        xla::ifrt::ToIfrtDType(array_type.getShape().getElementType()));
    output_specs.push_back(xla::ifrt::ArraySpec{
        /*dtype=*/dtype,
        /*shape=*/xla::ifrt::Shape(array_type.getShape().getShape()),
        /*sharding=*/array_type_to_sharding_.at(array_type)});
  }

  // Apply the remap arrays operation.
  xla::ifrt::ArrayCopySemantics copy_semantics =
      *is_donated ? xla::ifrt::ArrayCopySemantics::kDonateInput
                  : xla::ifrt::ArrayCopySemantics::kReuseInput;
  TF_ASSIGN_OR_RETURN(
      auto out_arrays,
      client_->RemapArrays({
                               /*input_specs=*/std::move(input_specs),
                               /*output_specs=*/std::move(output_specs),
                               /*mappings=*/std::move(mappings),
                           },
                           absl::MakeSpan(inputs), copy_semantics));

  for (const auto& array_value : array_values_to_gc_from_env) {
    // Donated remapped arrays are pro-actively deleted, and aliased arrays
    // cannot be deleted later. Thus, remove the arrays from the deletable
    // program arguments set.
    deletable_program_arguments_.erase(array_value);
    env.value_to_array.erase(array_value);
  }

  // Store the result arrays in the environment.
  TF_RET_CHECK(out_arrays.size() == remap_op.getOutputs().size())
      << "Got " << out_arrays.size() << " results, but op has "
      << remap_op.getOutputs().size() << ". " << PrettyPrint(remap_op);
  for (const auto [output_array, output] :
       llvm::zip(out_arrays, remap_op.getOutputs())) {
    if (!output.use_empty()) {
      env.AssociateArray(output, ArrayState{/*array=*/std::move(output_array),
                                            /*can_be_donated=*/true});
    }
  }
  return absl::OkStatus();
}

absl::Status ProgramInterpreter::ExecuteOp(
    xla::ifrt::CopyArraysOp copy_arrays_op, Environment& env) {
  TraceMe traceme([&]() {
    return TraceMeEncode("DispatchCopyArraysOp",
                         {{"ifrt_ir_program", program_->program_name}});
  });
  std::string op_name = copy_arrays_op->getName().getStringRef().str();
  VLOG(3) << PrettyPrint(copy_arrays_op);

  std::vector<ArrayRef> inputs;
  inputs.reserve(copy_arrays_op.getInputs().size());
  llvm::DenseSet<mlir::Value> array_values_to_gc_from_env;
  std::optional<bool> is_donated;
  for (const auto [idx, input] : llvm::enumerate(copy_arrays_op.getInputs())) {
    auto array_it = env.value_to_array.find(input);
    TF_RET_CHECK(array_it != env.value_to_array.end())
        << "Input array #" << idx << " not found. "
        << PrettyPrint(copy_arrays_op);
    if (array_it->second.array->IsDeleted()) {
      // We explicitly check here for deletion in order to provide a more
      // informative error message.
      return absl::InvalidArgumentError(absl::StrCat(
          "Input array #", idx, " has already been deleted or donated. ",
          PrettyPrint(copy_arrays_op)));
    }
    inputs.push_back(array_it->second.array);

    // The default buffer donation semantic is finalized at compilation time.
    // Users can override the donation semantic at runtime. In the meantime, the
    // IFRT client CopyArrays API requires all input arrays have the same
    // donation semantic.
    if (!is_donated.has_value()) {
      is_donated =
          copy_arrays_op.getDonated() && array_it->second.can_be_donated;
    }
    if (*is_donated && !array_it->second.can_be_donated) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Donation semantic must be consistent across all input arrays of "
          "CopyArraysOp. Input array #",
          idx,
          " cannot be donated, but previous input arrays can be donated. It's "
          "likely due to a MPMD program argument is marked as non-donatable. ",
          PrettyPrint(copy_arrays_op)));
    }
    if (*is_donated || liveness_.isDeadAfter(input, copy_arrays_op)) {
      array_values_to_gc_from_env.insert(input);
    }
  }
  TF_RET_CHECK(is_donated.has_value())
      << "Unable to determine the donation semantic of the copy arrays op. The "
         "copy arrays op has no inputs. "
      << PrettyPrint(copy_arrays_op);

  const auto out_array_type = llvm::cast<xla::ifrt::IfrtArrayType>(
      copy_arrays_op.getOutputs().front().getType());
  TF_RET_CHECK(out_array_type != nullptr)
      << "Output array #0 is not of type `IfrtArrayType`. "
      << PrettyPrint(copy_arrays_op);
  auto new_sharding = array_type_to_sharding_.at(out_array_type);
  auto array_copy_semantics = *is_donated
                                  ? xla::ifrt::ArrayCopySemantics::kDonateInput
                                  : xla::ifrt::ArrayCopySemantics::kAlwaysCopy;
  // It is safe to get the devices and memory kind from the first output
  // because all outputs use the same devices and have the same memory kind.
  TF_ASSIGN_OR_RETURN(
      auto copied_arrays,
      client_->CopyArrays(absl::MakeSpan(inputs), new_sharding->devices(),
                          new_sharding->memory_kind(), array_copy_semantics));

  for (const auto& array_value : array_values_to_gc_from_env) {
    if (deletable_program_arguments_.erase(array_value)) {
      // Explicitly delete donated program arguments that are not used later.
      env.value_to_array[array_value].array->Delete();
    }
    env.value_to_array.erase(array_value);
  }

  // Store the result arrays in the environment.
  TF_RET_CHECK(copied_arrays.size() == copy_arrays_op.getOutputs().size())
      << "Got " << copied_arrays.size() << " results, but op has "
      << copy_arrays_op.getOutputs().size() << ". "
      << PrettyPrint(copy_arrays_op);
  for (const auto [output_array, output] :
       llvm::zip(copied_arrays, copy_arrays_op.getOutputs())) {
    if (!output.use_empty()) {
      env.AssociateArray(output, ArrayState{/*array=*/std::move(output_array),
                                            /*can_be_donated=*/true});
    }
  }
  return absl::OkStatus();
}

absl::Status ProgramInterpreter::ExecuteOp(mlir::func::ReturnOp return_op,
                                           Environment& env) {
  auto func_op = return_op->getParentOfType<mlir::func::FuncOp>();
  CHECK_EQ(func_op.getSymName().str(), "main");
  VLOG(3) << return_op->getName().getStringRef().str() << " of `main` function";
  env.outputs.reserve(return_op->getNumOperands());
  for (const auto& [idx, result] : llvm::enumerate(return_op.getOperands())) {
    auto array_it = env.value_to_array.find(result);
    TF_RET_CHECK(array_it != env.value_to_array.end())
        << "Input array #" << idx << " not found. " << PrettyPrint(return_op);
    env.outputs.push_back(std::move(array_it->second.array));
  }
  env.value_to_array.clear();
  return absl::OkStatus();
}

std::string ProgramInterpreter::PrettyPrint(mlir::Operation* op) {
  if (auto call_op = mlir::dyn_cast<xla::ifrt::CallLoadedExecutableOp>(op)) {
    return absl::StrCat(call_op->getName().getStringRef().str(), " `",
                        call_op.getCalleeOp(symbol_table_).getSymName().str(),
                        "` from:\n", GetPrettyLocation(call_op->getLoc()));
  }
  return PrettyPrintGeneric(op);
}

}  // namespace ifrt
}  // namespace xla
