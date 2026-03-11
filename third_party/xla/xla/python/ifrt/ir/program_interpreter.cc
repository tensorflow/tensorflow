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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/remap_plan.pb.h"
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

using ExecuteOptions = ::xla::ifrt::LoadedExecutable::ExecuteOptions;
using ExecuteResult = ::xla::ifrt::LoadedExecutable::ExecuteResult;

namespace {

// Opaque handle that represents an array. Zero is reserved for null.
using ArrayHandle = uintptr_t;

static MemoryKind kPinnedHostMemoryKind(xla::PinnedHostMemorySpace::kKind);

// Array with additional metadata (e.g., if it can be donated).
struct ArrayState {
  ArrayRef array;
  bool can_be_donated;
};

// Assigns a unique handle to the given MLIR value.
ArrayHandle ToArrayHandle(mlir::Value value) {
  return reinterpret_cast<ArrayHandle>(value.getAsOpaquePointer());
}

std::string PrettyPrintGeneric(mlir::Operation* op) {
  return absl::StrCat(op->getName().getStringRef().str(), " from:\n",
                      GetPrettyLocation(op->getLoc()));
}

}  // namespace

struct Environment {
  // Associates array with an opaque handle.
  void AssociateArray(ArrayHandle handle, ArrayState array) {
    CHECK(handle_to_array.try_emplace(handle, array).second);
  }

  // IFRT client for execution.
  Client* client;
  // Name of the program.
  std::string program_name;
  // Set of donated program arguments, which can be deleted after their last
  // use. Entries are removed upon deletion or if they are aliased.
  absl::flat_hash_set<ArrayHandle> deletable_program_arguments;
  // Map from an opaque handle to IFRT array corresponding to the value.
  absl::flat_hash_map<ArrayHandle, ArrayState> handle_to_array;
  // Outputs of the program.
  std::vector<ArrayRef> outputs;
  // `ExecuteOptions.fill_status` passed to Execute().
  bool fill_status;
  // Contains a future for each ifrt.CallOp that is a leaf (i.e., has no outputs
  // or all its outputs are returned from the program).
  std::vector<tsl::Future<>> leaf_call_op_futures;
};

absl::StatusOr<std::unique_ptr<ProgramInterpreter>> ProgramInterpreter::Create(
    Client* client, absl::string_view program_name, mlir::ModuleOp mlir_module,
    std::shared_ptr<AtomExecutableMap> atom_program_executables,
    DeviceListRef devices) {
  mlir::func::FuncOp main_func = GetMainFunction(mlir_module);
  if (!main_func->hasAttr(kIfrtFunctionAttrName)) {
    return absl::InvalidArgumentError(
        absl::StrCat("`main` function of IFRT IR program: ", program_name,
                     " is not an IFRT function."));
  }
  return std::unique_ptr<ProgramInterpreter>(new ProgramInterpreter(
      client, program_name, mlir_module, std::move(atom_program_executables),
      std::move(devices), mlir::Liveness(main_func)));
}

namespace {

struct ProgramInterpreterState {
  Client* client;
  std::string program_name;

  std::vector<ArrayHandle> input_handles;
  absl::flat_hash_set<int> donated_input_indices;

  std::vector<absl::AnyInvocable<absl::Status(Environment& env) const>> op_fns;

  absl::StatusOr<ExecuteResult> Run(
      absl::Span<ArrayRef> arrays, const ExecuteOptions& options,
      std::optional<DeviceListRef> devices) const {
    TraceMe traceme([&]() {
      return TraceMeEncode("DispatchProgram",
                           {{"ifrt_ir_program", program_name}});
    });
    VLOG(2) << "Started interpreting program: " << program_name;

    if (arrays.size() != input_handles.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "`main` function of IFRT IR program: ", program_name,
          " invoked with ", arrays.size(), " arguments, but it expects ",
          input_handles.size(), " arguments."));
    }

    for (int idx = 0; idx < arrays.size(); ++idx) {
      const ArrayRef& array = arrays[idx];
      if (array->IsDeleted()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Input array #", idx, " of program ", program_name,
                         " has already been deleted or donated."));
      }
    }

    Environment env;
    env.client = client;
    env.fill_status = options.fill_status;
    for (int idx = 0; idx < input_handles.size(); ++idx) {
      // Add to the environment the arrays that are used.
      bool is_donated = donated_input_indices.contains(idx) &&
                        !options.non_donatable_input_indices.contains(idx);
      const ArrayHandle handle = input_handles[idx];
      if (handle != 0) {
        env.AssociateArray(handle, ArrayState{
                                       /*array=*/arrays[idx],
                                       /*can_be_donated=*/is_donated,
                                   });
        if (is_donated) {
          env.deletable_program_arguments.insert(handle);
        }
      } else if (is_donated) {
        // If the argument is donated but not used, it can be deleted.
        arrays[idx]->Delete();
      }
    }

    for (const auto& op_fn : op_fns) {
      TF_RETURN_IF_ERROR(op_fn(env));
    }

    VLOG(2) << "Finished interpreting program: " << program_name;
    ExecuteResult result;
    if (env.fill_status) {
      result.status =
          tsl::JoinFutures(absl::MakeSpan(env.leaf_call_op_futures));
    }
    result.outputs = std::move(env.outputs);
    return result;
  };
};

}  // namespace

absl::StatusOr<ProgramInterpreter::ExecuteFn>
ProgramInterpreter::BuildExecuteFn() {
  tsl::profiler::TraceMe traceme("ProgramInterpreter::BuildExecuteFn");

  ProgramInterpreterState state;
  state.client = client_;
  state.program_name = program_name_;

  mlir::func::FuncOp main_func = GetMainFunction(mlir_module_);

  for (const auto [idx, arg] : llvm::enumerate(main_func.getArguments())) {
    // Add to the environment the arrays that are used.
    const ArrayHandle handle = arg.use_empty() ? 0 : ToArrayHandle(arg);
    state.input_handles.push_back(handle);
    if (main_func.getArgAttr(idx, kIfrtDonatedArgAttrName) != nullptr) {
      state.donated_input_indices.insert(idx);
    }
  }

  // Walk ops one-by-one in program order and create functions that execute each
  // op on a given environment.
  for (mlir::Operation& op : main_func.getOps()) {
    auto op_fn =
        llvm::TypeSwitch<const mlir::Operation&, absl::StatusOr<OpFn>>(op)
            .Case<CallLoadedExecutableOp, RemapArraysOp, CopyArraysOp,
                  mlir::func::ReturnOp>(
                [this](const auto& op) { return HandleOp(op); })
            .Default([](const mlir::Operation& op) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Interpreter found unexpected op: ", mlir::debugString(op)));
            });
    if (!op_fn.ok()) {
      absl::Status status = op_fn.status();
      tsl::errors::AppendToMessage(&status, PrettyPrint(&op));
      return status;
    }
    state.op_fns.push_back(
        [op_fn = *std::move(op_fn),
         pretty_print = PrettyPrint(&op)](Environment& env) -> absl::Status {
          absl::Status status = op_fn(env);
          tsl::errors::AppendToMessage(&status, pretty_print);
          return status;
        });
  }

  return absl::bind_front(&ProgramInterpreterState::Run, std::move(state));
}

namespace {

struct CallLoadedExecutableOpState {
  std::string pretty_print;
  std::string atom_program_name;

  std::vector<ArrayHandle> input_handles;
  absl::flat_hash_set<int> donated_arg_idxs;
  absl::flat_hash_set<ArrayHandle> dead_inputs;

  ExecuteOptions execute_options;
  std::shared_ptr<LoadedExecutable> executable;

  std::vector<ArrayHandle> output_handles;
  bool is_leaf_op;
  // TODO(b/488047351): Remove this attribute once MPMD reshards are outlined in
  // IFRT IR.
  bool is_mpmd_reshard;

  absl::Status Run(Environment& env) const {
    TraceMe traceme([&]() {
      return TraceMeEncode("DispatchLoadedExecutableOp",
                           {
                               {"ifrt_ir_program", env.program_name},
                               {"atom_program", atom_program_name},
                           });
    });
    VLOG(3) << pretty_print;

    ExecuteOptions options = execute_options;
    options.fill_status = env.fill_status;

    std::vector<ArrayHandle> arrays_to_remove;
    {
      std::vector<ArrayRef> non_donatable_pinned_host_inputs;
      std::vector<ArrayHandle> non_donatable_pinned_host_inputs_handles;
      for (int idx = 0; idx < input_handles.size(); ++idx) {
        const ArrayHandle handle = input_handles[idx];

        auto array_it = env.handle_to_array.find(handle);
        TF_RET_CHECK(array_it != env.handle_to_array.end())
            << "Input array #" << idx << " not found. " << pretty_print;
        if (array_it->second.array->IsDeleted()) {
          // We explicitly check here for deletion in order to provide a more
          // informative error message.
          return absl::InvalidArgumentError(absl::StrCat(
              "Input array #", idx, "` has already been deleted or donated. ",
              pretty_print));
        }

        bool is_donated = donated_arg_idxs.contains(idx);
        if (is_donated && !array_it->second.can_be_donated) {
          VLOG(2) << "Atom program donates input #" << idx
                  << ", but it has not been donated to the IFRT IR program. "
                     "Input will not be donated. \n"
                  << pretty_print;
          // TODO(b/401105456): Do not special case pinned host arrays once
          // non-donatable pinned host inputs are supported.
          if (!is_mpmd_reshard &&
              array_it->second.array->sharding().memory_kind() ==
                  kPinnedHostMemoryKind) {
            non_donatable_pinned_host_inputs.push_back(array_it->second.array);
            non_donatable_pinned_host_inputs_handles.push_back(handle);
          } else {
            options.non_donatable_input_indices.insert(idx);
          }
          is_donated = false;
        }
        if (is_donated || dead_inputs.contains(handle)) {
          arrays_to_remove.push_back(handle);
        }
        if (!is_donated && is_mpmd_reshard) {
          options.non_donatable_input_indices.insert(idx);
        }
      }

      // TODO(b/401105456): Remove this CopyArrays call once non-donatable
      // pinned host inputs are supported.
      if (!non_donatable_pinned_host_inputs.empty()) {
        TF_ASSIGN_OR_RETURN(
            std::vector<ArrayRef> copied_pinned_host_inputs,
            env.client->CopyArrays(
                absl::MakeSpan(non_donatable_pinned_host_inputs),
                /*devices=*/std::nullopt,
                /*memory_kind=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy));
        for (int idx = 0; idx < copied_pinned_host_inputs.size(); ++idx) {
          env.handle_to_array[non_donatable_pinned_host_inputs_handles[idx]] =
              ArrayState{
                  /*array=*/std::move(copied_pinned_host_inputs[idx]),
                  /*can_be_donated=*/false,
              };
        }
      }
    }

    // Get the inputs of the loaded executable.
    std::vector<ArrayRef> inputs;
    inputs.reserve(input_handles.size());
    for (int idx = 0; idx < input_handles.size(); ++idx) {
      inputs.push_back(
          env.handle_to_array.find(input_handles[idx])->second.array);
    }

    TF_ASSIGN_OR_RETURN(ExecuteResult result,
                        executable->Execute(absl::MakeSpan(inputs), options,
                                            /*devices=*/std::nullopt));
    TF_RET_CHECK(result.outputs.size() == output_handles.size())
        << "Got " << result.outputs.size() << " results, but atom program has "
        << output_handles.size() << ". " << pretty_print;

    // Remove the arrays from the environment after the inputs vector is
    // created. This is because in situations such as `ifrt.Call(%0, %0)` the
    // liveness analysis will return that %0 is dead, but it's used for the
    // second argument.
    for (const auto handle : arrays_to_remove) {
      if (env.deletable_program_arguments.erase(handle)) {
        // Explicitly delete donated program arguments that are not used later.
        env.handle_to_array[handle].array->Delete();
      }
      env.handle_to_array.erase(handle);
    }

    for (int i = 0; i < output_handles.size(); ++i) {
      const ArrayHandle handle = output_handles[i];
      if (handle != 0) {
        // The output array is kept only if it used later. This can happen if an
        // executable has multiple output arrays, but only some of them are
        // used.
        env.AssociateArray(handle, ArrayState{
                                       /*array=*/std::move(result.outputs[i]),
                                       /*can_be_donated=*/true,
                                   });
      }
    }
    if (is_leaf_op && env.fill_status) {
      env.leaf_call_op_futures.push_back(std::move(result.status));
    }
    return absl::OkStatus();
  }
};

}  // namespace

absl::StatusOr<ProgramInterpreter::OpFn> ProgramInterpreter::HandleOp(
    CallLoadedExecutableOp call_loaded_op) {
  CallLoadedExecutableOpState state;
  state.pretty_print = PrettyPrint(call_loaded_op);

  LoadedExecutableOp loaded_exec_op = call_loaded_op.getCalleeOp(symbol_table_);
  state.atom_program_name = loaded_exec_op.getSymName().str();

  // Get the loaded executable for the atom program.
  auto exec_it = atom_program_executables_->find(state.atom_program_name);
  TF_RET_CHECK(exec_it != atom_program_executables_->end())
      << "Could not find executable. " << state.pretty_print;
  state.executable = exec_it->second;

  state.donated_arg_idxs.insert(call_loaded_op.getDonatedInputIndices().begin(),
                                call_loaded_op.getDonatedInputIndices().end());
  for (const auto& io_alias :
       call_loaded_op.getIoAliases().getAsRange<mlir::DenseI32ArrayAttr>()) {
    // Insert the aliased input to the set.
    state.donated_arg_idxs.insert(io_alias.asArrayRef()[0]);
  }
  for (const auto input : call_loaded_op.getInputs()) {
    state.input_handles.push_back(ToArrayHandle(input));
    if (liveness_.isDeadAfter(input, call_loaded_op)) {
      state.dead_inputs.insert(ToArrayHandle(input));
    }
  }

  if (auto module_type = call_loaded_op->getAttrOfType<mlir::StringAttr>(
          kIfrtModuleTypeAttrName);
      module_type != nullptr && module_type == kIfrtModuleTypeMpmdReshard) {
    state.is_mpmd_reshard = true;
  } else {
    state.is_mpmd_reshard = false;
  }

  state.is_leaf_op = true;
  for (const auto output : call_loaded_op.getOutputs()) {
    const ArrayHandle handle = output.use_empty() ? 0 : ToArrayHandle(output);
    state.output_handles.push_back(handle);

    if (state.is_leaf_op) {
      for (mlir::OpOperand& use : output.getUses()) {
        // An ifrt.CallOp is not a leaf if any of its outputs are not returned.
        if (llvm::dyn_cast<mlir::func::ReturnOp>(use.getOwner()) == nullptr) {
          state.is_leaf_op = false;
          break;
        }
      }
    }
  }

  return absl::bind_front(&CallLoadedExecutableOpState::Run, std::move(state));
}

namespace {

struct RemapArraysOpState {
  std::string pretty_print;

  RemapPlan remap_plan;
  std::vector<ArrayHandle> input_handles;
  absl::flat_hash_set<ArrayHandle> dead_inputs;
  bool remap_is_donated;

  std::vector<ArrayHandle> output_handles;

  absl::Status Run(Environment& env) const {
    TraceMe traceme([&]() {
      return TraceMeEncode("DispatchRemapArraysOp",
                           {{"ifrt_ir_program", env.program_name}});
    });
    VLOG(3) << pretty_print;

    std::vector<ArrayRef> inputs;
    inputs.reserve(remap_plan.input_specs.size());

    std::vector<ArrayHandle> arrays_to_remove;

    for (int idx = 0; idx < input_handles.size(); ++idx) {
      const ArrayHandle handle = input_handles[idx];

      auto array_it = env.handle_to_array.find(handle);
      TF_RET_CHECK(array_it != env.handle_to_array.end())
          << "Input array #" << idx << " not found. " << pretty_print;
      if (array_it->second.array->IsDeleted()) {
        // We explicitly check here for deletion in order to provide a more
        // informative error message.
        return absl::InvalidArgumentError(absl::StrCat(
            "Input array #", idx, "` has already been deleted or donated. ",
            pretty_print));
      }

      // The default buffer donation semantic is finalized at compilation time.
      // Users can override the donation semantic at runtime. In the meantime,
      // the IFRT client RemapArrays API requires all input arrays have the same
      // donation semantic. Insert a CopyArrays op if RemapArrays requires all
      // input arrays to be donated, but some of the input arrays have been
      // marked as non-donatable.
      if (remap_is_donated && !array_it->second.can_be_donated) {
        ArrayRef array = array_it->second.array;
        LOG_FIRST_N(WARNING, 5)
            << "Array is cloned because remapping with more than one "
               "input array requires input donation. array="
            << array->DebugString()
            << " (this warning is logged only at most 5 times). This clone "
               "happens only if the array has been marked as non-donatable at "
               "runtime."
            << pretty_print;
        TF_ASSIGN_OR_RETURN(
            std::vector<ArrayRef> copied_arrays,
            env.client->CopyArrays(
                absl::MakeSpan(&array, 1), /*devices=*/std::nullopt,
                /*memory_kind=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy));
        inputs.push_back(std::move(copied_arrays[0]));
      } else {
        inputs.push_back(array_it->second.array);
      }
      if ((remap_is_donated && array_it->second.can_be_donated) ||
          dead_inputs.contains(handle)) {
        arrays_to_remove.push_back(handle);
      }
    }

    // Apply the remap arrays operation.
    TF_ASSIGN_OR_RETURN(
        auto out_arrays,
        env.client->RemapArrays(remap_plan, absl::MakeSpan(inputs),
                                remap_is_donated
                                    ? ArrayCopySemantics::kDonateInput
                                    : ArrayCopySemantics::kReuseInput));

    for (const auto handle : arrays_to_remove) {
      // Donated remapped arrays are pro-actively deleted, and aliased arrays
      // cannot be deleted later. Thus, remove the arrays from the deletable
      // program arguments set.
      env.deletable_program_arguments.erase(handle);
      env.handle_to_array.erase(handle);
    }

    // Store the result arrays in the environment.
    TF_RET_CHECK(out_arrays.size() == remap_plan.output_specs.size())
        << "Got " << out_arrays.size() << " results, but op has "
        << remap_plan.output_specs.size() << ". " << pretty_print;
    for (int i = 0; i < output_handles.size(); ++i) {
      const ArrayHandle handle = output_handles[i];
      if (handle != 0) {
        env.AssociateArray(handle, ArrayState{
                                       /*array=*/std::move(out_arrays[i]),
                                       /*can_be_donated=*/true,
                                   });
      }
    }

    return absl::OkStatus();
  }
};

}  // namespace

absl::StatusOr<ProgramInterpreter::OpFn> ProgramInterpreter::HandleOp(
    RemapArraysOp remap_op) {
  RemapArraysOpState state;
  state.pretty_print = PrettyPrint(remap_op);

  // Construct the mappings of the remap plan.
  auto mappings = std::make_shared<std::vector<RemapPlan::Mapping>>();
  mappings->reserve(remap_op.getMappings().size());
  for (const auto& array_mapping : remap_op.getMappings()) {
    const auto array_mapping_attr =
        llvm::cast<IfrtArrayMappingAttr>(array_mapping);
    auto& mapping = mappings->emplace_back();
    mapping.in_array = array_mapping_attr.getInArrayIndex();
    mapping.out_array = array_mapping_attr.getOutArrayIndex();
    mapping.from.reserve(array_mapping_attr.getMappings().size());
    mapping.to.reserve(array_mapping_attr.getMappings().size());
    for (const auto& m : array_mapping_attr.getMappings()) {
      const auto mapping_attr = llvm::cast<IfrtMappingAttr>(m);
      auto from_shards = mapping_attr.getFromShards();
      auto to_shards = mapping_attr.getToShards();
      mapping.from.push_back(RemapPlan::Interval{
          from_shards.getStart(), from_shards.getEnd(), from_shards.getStep()});
      mapping.to.push_back(RemapPlan::Interval{
          to_shards.getStart(), to_shards.getEnd(), to_shards.getStep()});
    }
  };

  // Get the input specs of the remap plan and the input arrays.
  std::vector<ArraySpec> input_specs;
  input_specs.reserve(remap_op.getInputs().size());
  for (const mlir::Value input : remap_op.getInputs()) {
    state.input_handles.push_back(ToArrayHandle(input));
    TF_ASSIGN_OR_RETURN(
        ArraySpec spec,
        ArraySpecFromMlirType(input.getType(), client_, devices_));
    input_specs.push_back(std::move(spec));
    if (liveness_.isDeadAfter(input, remap_op)) {
      state.dead_inputs.insert(ToArrayHandle(input));
    }
  }

  // Get the output specs of the remap plan.
  std::vector<ArraySpec> output_specs;
  output_specs.reserve(remap_op.getOutputs().size());
  for (const mlir::Value output : remap_op.getOutputs()) {
    TF_ASSIGN_OR_RETURN(
        ArraySpec spec,
        ArraySpecFromMlirType(output.getType(), client_, devices_));
    output_specs.push_back(std::move(spec));
  }

  state.remap_plan = RemapPlan{
      /*input_specs=*/std::move(input_specs),
      /*output_specs=*/std::move(output_specs),
      /*mappings=*/std::move(mappings),
  };
  state.remap_is_donated = remap_op.getDonated();

  TF_RETURN_IF_ERROR(state.remap_plan.ComputeInputDevicesForOutputMap(client_));
  TF_RETURN_IF_ERROR(state.remap_plan.Validate());

  for (const auto output : remap_op.getOutputs()) {
    const ArrayHandle handle = output.use_empty() ? 0 : ToArrayHandle(output);
    state.output_handles.push_back(handle);
  }

  return absl::bind_front(&RemapArraysOpState::Run, std::move(state));
}

namespace {

struct CopyArraysOpState {
  std::string pretty_print;

  std::vector<ArrayHandle> input_handles;
  absl::flat_hash_set<ArrayHandle> dead_inputs;
  bool copy_is_donated;

  std::vector<ArrayHandle> output_handles;
  ShardingRef new_sharding;

  absl::Status Run(Environment& env) const {
    TraceMe traceme([&]() {
      return TraceMeEncode("DispatchCopyArraysOp",
                           {{"ifrt_ir_program", env.program_name}});
    });
    VLOG(3) << pretty_print;

    std::vector<ArrayRef> inputs;
    inputs.reserve(input_handles.size());
    // Indices of donated arrays that must be copied because they've been marked
    // as non-donatable at runtime.
    std::vector<int> array_idxs_to_copy;
    std::vector<ArrayRef> arrays_to_copy;
    std::vector<ArrayHandle> arrays_to_remove;

    for (int idx = 0; idx < input_handles.size(); ++idx) {
      const ArrayHandle handle = input_handles[idx];

      auto array_it = env.handle_to_array.find(handle);
      TF_RET_CHECK(array_it != env.handle_to_array.end())
          << "Input array #" << idx << " not found. " << pretty_print;
      if (array_it->second.array->IsDeleted()) {
        // We explicitly check here for deletion in order to provide a more
        // informative error message.
        return absl::InvalidArgumentError(absl::StrCat(
            "Input array #", idx, " has already been deleted or donated. ",
            pretty_print));
      }
      inputs.push_back(array_it->second.array);

      if (copy_is_donated && !array_it->second.can_be_donated) {
        array_idxs_to_copy.push_back(idx);
        arrays_to_copy.push_back(array_it->second.array);
      }
      if ((copy_is_donated && array_it->second.can_be_donated) ||
          dead_inputs.contains(handle)) {
        arrays_to_remove.push_back(handle);
      }
    }

    if (!array_idxs_to_copy.empty()) {
      // The default buffer donation semantic is finalized at compilation time.
      // Users can override the donation semantic at runtime. In the meantime,
      // the IFRT client CopyArrays API requires all input arrays have the same
      // donation semantic. Insert another CopyArrays op if CopyArrays requires
      // all input arrays to be donated, but some of the input arrays have been
      // marked as non-donatable.
      LOG_FIRST_N(WARNING, 5)
          << "Arrays are cloned because they have been marked as non-donatable "
             "at runtime, but CopyArrays requires all arrays to be donated. "
             "arrays="
          << absl::StrJoin(arrays_to_copy, ",",
                           [](std::string* out, const ArrayRef& array) {
                             absl::StrAppend(out, array->DebugString());
                           })
          << " (this warning is logged only at most 5 times)." << pretty_print;
      TF_ASSIGN_OR_RETURN(
          std::vector<ArrayRef> copied_arrays,
          env.client->CopyArrays(
              absl::MakeSpan(arrays_to_copy), /*devices=*/std::nullopt,
              /*memory_kind=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy));
      for (int i = 0; i < array_idxs_to_copy.size(); ++i) {
        inputs[array_idxs_to_copy[i]] = std::move(copied_arrays[i]);
      }
    }

    // It is safe to get the devices and memory kind from the first output
    // because all outputs use the same devices and have the same memory kind.
    TF_ASSIGN_OR_RETURN(auto copied_arrays,
                        env.client->CopyArrays(
                            absl::MakeSpan(inputs), new_sharding->devices(),
                            new_sharding->memory_kind(),
                            copy_is_donated ? ArrayCopySemantics::kDonateInput
                                            : ArrayCopySemantics::kAlwaysCopy));

    for (const auto handle : arrays_to_remove) {
      if (env.deletable_program_arguments.erase(handle)) {
        // Explicitly delete donated program arguments that are not used later.
        env.handle_to_array[handle].array->Delete();
      }
      env.handle_to_array.erase(handle);
    }

    TF_RET_CHECK(copied_arrays.size() == inputs.size())
        << "Got " << copied_arrays.size() << " results, but op has "
        << inputs.size() << ". " << pretty_print;
    for (int i = 0; i < output_handles.size(); ++i) {
      const ArrayHandle handle = output_handles[i];
      if (handle != 0) {
        env.AssociateArray(handle, ArrayState{
                                       /*array=*/std::move(copied_arrays[i]),
                                       /*can_be_donated=*/true,
                                   });
      }
    }

    return absl::OkStatus();
  }
};

}  // namespace

absl::StatusOr<ProgramInterpreter::OpFn> ProgramInterpreter::HandleOp(
    CopyArraysOp copy_arrays_op) {
  CopyArraysOpState state;
  state.pretty_print = PrettyPrint(copy_arrays_op);

  for (const auto [idx, input] : llvm::enumerate(copy_arrays_op.getInputs())) {
    state.input_handles.push_back(ToArrayHandle(input));
    if (liveness_.isDeadAfter(input, copy_arrays_op)) {
      state.dead_inputs.insert(ToArrayHandle(input));
    }
  }
  state.copy_is_donated = copy_arrays_op.getDonated();

  const auto out_array_type =
      llvm::cast<IfrtArrayType>(copy_arrays_op.getOutputs().front().getType());
  TF_RET_CHECK(out_array_type != nullptr)
      << "Output array #0 is not of type `IfrtArrayType`. "
      << state.pretty_print;
  TF_ASSIGN_OR_RETURN(
      state.new_sharding,
      ShardingFromIfrtArrayType(out_array_type, client_, devices_));

  for (const auto output : copy_arrays_op.getOutputs()) {
    const ArrayHandle handle = output.use_empty() ? 0 : ToArrayHandle(output);
    state.output_handles.push_back(handle);
  }

  return absl::bind_front(&CopyArraysOpState::Run, std::move(state));
}

namespace {

struct ReturnOpState {
  std::string pretty_print;
  std::vector<ArrayHandle> output_handles;

  absl::Status Run(Environment& env) const {
    VLOG(3) << "func.return of `main` function";
    env.outputs.reserve(output_handles.size());
    for (int idx = 0; idx < output_handles.size(); ++idx) {
      auto array_it = env.handle_to_array.find(output_handles[idx]);
      TF_RET_CHECK(array_it != env.handle_to_array.end())
          << "Input array #" << idx << " not found. " << pretty_print;
      env.outputs.push_back(std::move(array_it->second.array));
    }
    env.handle_to_array.clear();
    return absl::OkStatus();
  }
};

}  // namespace

absl::StatusOr<ProgramInterpreter::OpFn> ProgramInterpreter::HandleOp(
    mlir::func::ReturnOp return_op) {
  ReturnOpState state;
  state.pretty_print = PrettyPrint(return_op);

  auto func_op = return_op->getParentOfType<mlir::func::FuncOp>();
  CHECK_EQ(func_op.getSymName().str(), "main");
  state.output_handles.reserve(return_op->getNumOperands());
  for (const auto& [idx, result] : llvm::enumerate(return_op.getOperands())) {
    state.output_handles.push_back(ToArrayHandle(result));
  }

  return absl::bind_front(&ReturnOpState::Run, std::move(state));
}

std::string ProgramInterpreter::PrettyPrint(mlir::Operation* op) {
  if (auto call_op = mlir::dyn_cast<CallLoadedExecutableOp>(op)) {
    return absl::StrCat(call_op->getName().getStringRef().str(), " `",
                        call_op.getCalleeOp(symbol_table_).getSymName().str(),
                        "` from:\n", GetPrettyLocation(call_op->getLoc()));
  }
  return PrettyPrintGeneric(op);
}

}  // namespace ifrt
}  // namespace xla
