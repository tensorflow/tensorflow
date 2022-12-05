/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"

#include <stdint.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compiler.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_stream.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace cpu {

namespace runtime = ::xla::runtime;

CpuExecutable::CpuExecutable(
    std::unique_ptr<SimpleOrcJIT> jit,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module,
    const std::string& entry_function_name,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      jit_(std::move(jit)),
      assignment_(std::move(assignment)),
      module_name_(entry_function_name) {
  if (assignment_) {
    buffer_assignment_ =
        std::make_shared<BufferAssignmentProto>(assignment_->ToProto());
  }
  if (has_module()) {
    XlaDebugInfoManager::Get()->RegisterModule(
        module().unique_id(), shared_module(), buffer_assignment_);
  }

  // Resolve symbols in the constructor rather than at execution time to avoid
  // races because FindSymbol is not thread safe.
  llvm::Expected<llvm::JITEvaluatedSymbol> sym =
      jit_->FindCompiledSymbol(entry_function_name);
  // We expect to find the symbol provided with entry_function_name; otherwise
  // this is an internal error.
  CHECK(*sym) << "Symbol " << entry_function_name << " not found.";
  // getAddress can do work under the hood in the jit, so it needs to be
  // guarded by the mutex.
  compute_function_ = reinterpret_cast<ComputeFunctionType>(sym->getAddress());
  VLOG(1) << "compute_function_ at address "
          << reinterpret_cast<void*>(compute_function_);
  jit_->DoneCompiling();
}

CpuExecutable::CpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<XlaRuntimeCpuExecutable> xla_runtime_executable)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      assignment_(std::move(assignment)),
      xla_runtime_executable_(std::move(xla_runtime_executable)) {
  if (assignment_) {
    buffer_assignment_ =
        std::make_shared<BufferAssignmentProto>(assignment_->ToProto());
  }
  if (has_module()) {
    XlaDebugInfoManager::Get()->RegisterModule(
        module().unique_id(), shared_module(), buffer_assignment_);
  }
}

CpuExecutable::~CpuExecutable() {
  if (has_module()) {
    XlaDebugInfoManager::Get()->UnregisterModule(module().unique_id());
  }
}

static StatusOr<MaybeOwningDeviceMemory> MemoryForAllocation(
    const BufferAllocation& allocation,
    absl::Span<ExecutionInput const> arguments,
    se::DeviceMemoryAllocator* memory_allocator, int device_ordinal) {
  VLOG(3) << allocation.ToString();
  if (allocation.is_entry_computation_parameter()) {
    se::DeviceMemoryBase out = arguments[allocation.parameter_number()]
                                   .Buffer(allocation.param_shape_index())
                                   .AsDeviceMemoryBase();
    CHECK_LE(allocation.size(), out.size())
        << "Size mismatch on param " << allocation.parameter_number()
        << " at shape index " << allocation.param_shape_index().ToString();
    VLOG(3) << "allocation is a parameter";
    return MaybeOwningDeviceMemory{out};
  } else if (allocation.is_constant()) {
    VLOG(3) << "allocation is a constant";
    return MaybeOwningDeviceMemory{se::DeviceMemoryBase{}};
  } else if (allocation.is_thread_local()) {
    VLOG(3) << "buffer is thread-local";
    return MaybeOwningDeviceMemory{se::DeviceMemoryBase{}};
  }

  int64_t buffer_size = allocation.size();
  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory out,
                      memory_allocator->Allocate(device_ordinal, buffer_size));
  VLOG(3) << "buffer allocated " << buffer_size << " bytes [" << out->opaque()
          << "]";

  // Since the output buffer and all the temporary buffers were written into
  // by the JITed code, msan has no way of knowing their memory was
  // initialized. Mark them initialized so that msan doesn't flag loads from
  // these buffers.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(out->opaque(), buffer_size);
  return MaybeOwningDeviceMemory{std::move(out)};
}

StatusOr<std::vector<MaybeOwningDeviceMemory>> CpuExecutable::CreateBufferTable(
    se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
    absl::Span<ExecutionInput const> arguments) {
  std::vector<MaybeOwningDeviceMemory> buffers(
      assignment_->Allocations().size());
  VLOG(3) << "Allocating " << assignment_->Allocations().size()
          << " allocations for module " << module().name();
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    TF_ASSIGN_OR_RETURN(
        buffers[i], MemoryForAllocation(allocation, arguments, memory_allocator,
                                        device_ordinal));
  }

  if (VLOG_IS_ON(3)) {
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment_->GetUniqueTopLevelOutputSlice());
    VLOG(3) << "result index: " << result_slice.index();
  }
  return std::move(buffers);
}

Status CpuExecutable::ExecuteComputeFunction(
    const ExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceMemory const> buffers,
    HloExecutionProfile* hlo_execution_profile) {
  uint64_t start_micros = tsl::Env::Default()->NowMicros();

  size_t profile_counters_size =
      hlo_execution_profile ? hlo_execution_profile->profile_counters().size()
                            : 0;
  int64_t* profile_counters =
      hlo_execution_profile
          ? hlo_execution_profile->mutable_profile_counters()->data()
          : nullptr;

  // Call the computation function following the calling convention. See the
  // definition of 'ComputeFunctionType' for the details of the calling
  // convention of JITed functions.
  std::vector<void*> buffer_pointers;
  for (auto& buffer : buffers) {
    buffer_pointers.push_back(
        const_cast<void*>(buffer.AsDeviceMemoryBase().opaque()));
  }

  VLOG(3) << "Executing compute function:";
  VLOG(3) << absl::StrFormat("  Number of buffer table entries: %u",
                             buffer_pointers.size());
  auto ptr_printer = [](std::string* out, const void* p) {
    absl::StrAppend(out, absl::StrFormat("%p", p));
  };
  VLOG(3) << absl::StrFormat("  Buffer table: [%s]",
                             absl::StrJoin(buffer_pointers, ", ", ptr_printer));
  VLOG(3) << absl::StrFormat("  Number of profile counters: %u",
                             profile_counters_size);
  VLOG(3) << absl::StrFormat("  Profile counters: %p", profile_counters);

  auto record_profile = [&]() {
    uint64_t end_micros = tsl::Env::Default()->NowMicros();
    if (run_options->execution_profile()) {
      const double nanoseconds = (end_micros - start_micros) * 1000.0;
      run_options->execution_profile()->set_compute_time_ns(
          std::max(nanoseconds, 1.0));
      // If hlo profiling was disabled then the cycle count is left empty.
      if (hlo_execution_profile) {
        run_options->execution_profile()->set_compute_cycle_count(
            hlo_execution_profile->total_cycles_executed(
                *module().entry_computation()));
      }
    }
  };

  if (IsXlaRuntime()) {
    std::vector<BufferDesc> descriptor_table;
    descriptor_table.reserve(buffers.size());
    for (const auto& buffer : buffers) {
      const tensorflow::se::DeviceMemoryBase& base =
          buffer.AsDeviceMemoryBase();
      BufferDesc desc(const_cast<void*>(base.opaque()), base.size());
      descriptor_table.push_back(std::move(desc));
    }
    Status status = ExecuteXlaRuntime(descriptor_table, run_options);
    record_profile();
    if (!status.ok()) {
      return status;
    }
  } else {
    XlaCustomCallStatus status;
    // For the entry computation (like all global computations), all inputs and
    // outputs are in the buffer table, and both the result pointer and args
    // array pointers are unused (so we set them to 'nullptr').
    compute_function_(nullptr, run_options, nullptr, buffer_pointers.data(),
                      &status, profile_counters);
    record_profile();
    std::optional<absl::string_view> error_message =
        CustomCallStatusGetMessage(&status);
    if (error_message) {
      return InternalError("CustomCall failed: %s", *error_message);
    }
  }

  return OkStatus();
}

StatusOr<std::unique_ptr<Executable>> CpuExecutable::LoadFromObjFile(
    std::unique_ptr<HloModule> hlo_module, absl::string_view obj_file,
    absl::string_view mlir_module,
    std::unique_ptr<BufferAssignment> buffer_assignment,
    XlaFrameworkMapping xla_framework_mapping,
    runtime::JitExecutable::Options opts) {
  runtime::DialectRegistry dialects;
  opts.compiler.register_dialects(dialects);
  auto threading = mlir::MLIRContext::Threading::DISABLED;
  auto ctx = std::make_unique<mlir::MLIRContext>(*dialects, threading);
  ctx->loadAllAvailableDialects();

  // Load MLIR module behind the compiled object file.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_module, ctx.get());
  if (!module) return InternalError("Failed to parse AOT compiled module");

  llvm::StringRef data(obj_file.data(), obj_file.size());
  auto buffer = llvm::MemoryBuffer::getMemBuffer(data, hlo_module->name());

  // Recover function signatures using calling convention and type converter.
  auto func = mlir::cast<mlir::func::FuncOp>(module->lookupSymbol("main"));
  mlir::FunctionType func_type = func.getFunctionType();
  absl::StatusOr<runtime::FunctionType> sig =
      opts.compiler.type_converter.Convert(func_type);
  if (!sig.ok())
    return InternalError("Type converter failed to convert function type");

  mlir::FunctionType runtime_type = opts.compiler.calling_convention(func_type);
  if (!runtime_type)
    return InternalError("Calling convention failed to convert function type");

  absl::StatusOr<runtime::FunctionType> runtime_sig =
      opts.compiler.type_converter.Convert(runtime_type);
  if (!runtime_sig.ok())
    return InternalError(
        "Type converter failed to convert runtime function type");

  // Cpu executable has a single exported function.
  std::vector<runtime::Executable::LoadFunction> functions;
  functions.push_back({"main", std::move(*sig), std::move(*runtime_sig)});

  // Load XLA Runtime executable from an object file.
  auto executable = runtime::Executable::LoadFromObjFile(
      hlo_module->name(), std::move(buffer), std::move(functions),
      opts.compiler.symbols_binding);

  if (!executable.ok())
    return InternalError("Failed to load XLA Runtime executable: %s",
                         executable.status().message());

  // Move runtime::Executable ownership to the XlaRuntimeCpuExecutable.
  auto executable_ptr =
      std::make_unique<runtime::Executable>(std::move(executable.value()));
  auto xla_runtime_executable = std::make_unique<XlaRuntimeCpuExecutable>(
      std::move(executable_ptr), xla_framework_mapping);

  return std::unique_ptr<Executable>(new CpuExecutable(
      std::move(hlo_module), nullptr, nullptr, std::move(buffer_assignment),
      std::move(xla_runtime_executable)));
}

StatusOr<ExecutionOutput> CpuExecutable::CreateResultShapedBuffer(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceMemory> buffers,
    absl::Span<ExecutionInput> arguments) {
  se::Stream* stream = run_options->stream();
  ExecutionOutput result(/*on_device_shape=*/result_shape(),
                         run_options->allocator(),
                         stream->parent()->device_ordinal());
  const HloInputOutputAliasConfig& input_output_alias =
      module().input_output_alias_config();
  HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  const Shape& root_shape = root->shape();

  // Move se::OwningDeviceMemory values which contain the array(s) of the result
  // into the respective location in ScopedShapedBuffer which is returned to the
  // caller.
  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    se::DeviceMemoryBase& result_buffer = p.second;
    const HloValueSet& sources = this->GetRootValueSet().element(index);
    // The points to set is unambiguous so the set should be a
    // singleton.
    CHECK_EQ(1, sources.values().size());
    const HloValue* value_source = sources.values()[0];
    HloInstruction* src = value_source->instruction();

    // The source for this result buffer can be a nested buffer such as
    // a tuple element.
    TF_ASSIGN_OR_RETURN(
        const BufferAllocation::Slice slice,
        this->assignment_->GetUniqueSlice(src, value_source->index()));
    const BufferAllocation::Index buffer_index = slice.index();

    // TODO(cheshire): duplication with other backends.
    std::optional<HloInputOutputAliasConfig::Alias> alias =
        input_output_alias.GetAliasedParameter(index);
    if (alias) {
      CHECK_LT(alias->parameter_number, arguments.size());
      ExecutionInput& input = arguments[alias->parameter_number];
      MaybeOwningDeviceMemory* maybe_owning_memory =
          input.MutableBuffer(alias->parameter_index);
      if (alias->must_alias() && !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: %s",
            alias->ToString());
      }
      if (std::optional<se::OwningDeviceMemory> owning =
              maybe_owning_memory->Release()) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        // The caller is giving us the
        // input buffer, but in case of error of the execute call, we should
        // not be releasing it as it contains valid data (for example, it is a
        // parameter which the user wants us to alias, in a gradient update
        // computation). So we store the index into the result in the aliased
        // vactor, which will be fed to the ExecutionOutput, which will be
        // using the indices to drop the addresses from its own
        // ScopedShapedBuffer result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(index);
      } else {
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size =
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(root_shape, index));
        TF_ASSIGN_OR_RETURN(
            se::OwningDeviceMemory allocated_buffer,
            run_options->allocator()->Allocate(
                stream->parent()->device_ordinal(), allocation_size));
        result_buffer = allocated_buffer.Release();
        MaybeOwningDeviceMemory& registered_buffer = buffers[buffer_index];
        CHECK_EQ(result_buffer.size(),
                 registered_buffer.AsDeviceMemoryBase().size());
        std::memcpy(/*dest=*/result_buffer.opaque(),
                    /*src=*/registered_buffer.AsDeviceMemoryBase().opaque(),
                    /*n=*/result_buffer.size());
        registered_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      MaybeOwningDeviceMemory& buffer = buffers[buffer_index];
      if (std::optional<se::OwningDeviceMemory> owned_buffer =
              buffer.Release()) {
        result_buffer = owned_buffer->Release();
        buffer = result_buffer;
      } else {
        result_buffer = buffer.AsDeviceMemoryBase();
        result.AddAliasedIndex(index);
      }
    }
  }
  return std::move(result);
}

// Converts a BufferDesc to a MemrefDesc according to the given 'operand_type',
// which should point to a runtime::MemrefType.
// Note: 'descriptor_index' and 'operand_index' are just used for error
// reporting.
static StatusOr<runtime::MemrefDesc> BufferToMemref(
    const BufferDesc& descriptor, const runtime::Type& operand_type,
    size_t descriptor_index, size_t operand_index) {
  auto* memref = llvm::dyn_cast<runtime::MemrefType>(&operand_type);
  if (!memref) {
    return InternalError(
        "Cannot convert descriptor %zu (operand_index %zu): "
        "the corresponding type in the signature is a %s, "
        "not a MemrefType.",
        descriptor_index, operand_index, operand_type.ToString());
  }

  absl::Span<const int64_t> dims = memref->sizes();

  // Verify that the provided descriptor size matches that of the memref.
  size_t n_elem = absl::c_accumulate(dims, size_t{1}, std::multiplies<>());
  size_t expected_size =
      primitive_util::ByteWidth(memref->element_type()) * n_elem;
  if (LLVM_UNLIKELY(expected_size != descriptor.size())) {
    return InvalidArgument(
        "Cannot convert descriptor %zu (operand_index %zu): "
        "buffer size is not equal to that expected from the element type: "
        "got %zu vs expected %zu.",
        descriptor_index, operand_index, descriptor.size(), expected_size);
  }

  auto fill_sizes_and_strides = [&](auto sizes, auto strides) {
    size_t multiplier = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      size_t size = dims[i];
      sizes[i] = size;
      strides[i] = multiplier;
      multiplier *= size;
    }
  };
  return runtime::MemrefDesc(memref->rank(), memref->element_type(),
                             descriptor.data(), /*offset=*/0,
                             fill_sizes_and_strides);
}

// Executes from an XLA Runtime CPU executable, given a buffer descriptor table.
// Relevant elements of the descriptor table (i.e. arguments and results) are
// converted to MemrefDesc's according to the corresponding operands in the
// runtime signature.
Status XlaRuntimeCpuExecutable::Execute(
    const std::vector<BufferDesc>& descriptor_table,
    const ExecutableRunOptions* run_options) {
  const runtime::FunctionType& signature = GetExecutable().runtime_signature();

  size_t num_arguments = xla_framework_mapping_.inputs.size();
  if (xla_framework_mapping_.output_is_tuple) {
    num_arguments += xla_framework_mapping_.flattened_outputs.size();
  } else if (xla_framework_mapping_.result != -1) {
    num_arguments += 1;
  }

  // Verify that the number of arguments in the mapping matches the signature.
  // Add one to num_arguments to account for the signature's execution context.
  if (num_arguments + 1 != signature.num_operands()) {
    return InternalError(
        "Wrong number of arguments: got %zu via XLA FrameworkMapping, expected "
        "%d.",
        num_arguments, static_cast<int>(signature.num_operands()) - 1);
  }

  std::vector<runtime::MemrefDesc> arguments;
  arguments.reserve(num_arguments);

  auto append_converted_buffer = [&](size_t descriptor_index) -> Status {
    const BufferDesc& descriptor = descriptor_table[descriptor_index];

    // Use 1-based index to account for the execution context.
    size_t operand_index = arguments.size() + 1;
    const runtime::Type* operand_type = signature.operand(operand_index);

    StatusOr<runtime::MemrefDesc> memref = BufferToMemref(
        descriptor, *operand_type, descriptor_index, operand_index);
    if (!memref.ok()) {
      return memref.status();
    }
    arguments.push_back(std::move(*memref));
    return OkStatus();
  };

  // Inputs come first; results come last.
  for (int64_t index : xla_framework_mapping_.inputs) {
    TF_RETURN_IF_ERROR(append_converted_buffer(index));
  }
  // If we have a tuple (possibly empty) as output, then .output_is_tuple
  // is set and .result should be ignored.
  if (xla_framework_mapping_.output_is_tuple) {
    for (int64_t index : xla_framework_mapping_.flattened_outputs) {
      TF_RETURN_IF_ERROR(append_converted_buffer(index));
    }
  } else if (xla_framework_mapping_.result != -1) {
    TF_RETURN_IF_ERROR(append_converted_buffer(xla_framework_mapping_.result));
  }

  runtime::Executable::CallFrame call_frame;
  // Skip verification. The MemrefDesc's we created above come from the runtime
  // signature; verifying them against the same signature would be redundant.
  if (auto status =
          GetExecutable().InitializeCallFrame(arguments, &call_frame,
                                              /*verify_arguments=*/false);
      !status.ok()) {
    return InternalError("Failed to initialize call frame: %s.",
                         status.message());
  }

  // No results to return; they are returned via out params.
  runtime::NoResultConverter converter;

  // Collect all emitted diagnostic messages.
  std::string diagnostic;
  runtime::DiagnosticEngine diagnostic_engine;
  diagnostic_engine.AddHandler([&](runtime::Diagnostic& d) {
    absl::StrAppend(&diagnostic, d.status().message());
    return runtime::success();
  });

  runtime::CustomCall::UserData user_data(run_options);

  runtime::Executable::ExecuteOpts opts;
  opts.custom_call_data = &user_data;
  opts.diagnostic_engine = &diagnostic_engine;

  // We don't expect to see any async tasks in the XLA Runtime executable.
  opts.async_task_runner =
      reinterpret_cast<runtime::AsyncTaskRunner*>(0xdeadbeef);

  // Execute with the prepared call frame.
  GetExecutable().Execute(call_frame, opts);
  if (auto status = GetExecutable().ReturnResults(converter, &call_frame);
      !status.ok()) {
    return InternalError("Failed to execute XLA Runtime executable: %s%s%s.",
                         status.message(), diagnostic.empty() ? "" : ": ",
                         diagnostic);
  }
  return OkStatus();
}

StatusOr<ExecutionOutput> CpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  if (GetRootValueSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  if (hlo_module_) {
    const HloComputation* entry_comp = hlo_module_->entry_computation();
    CHECK_EQ(entry_comp->num_parameters(), arguments.size())
        << "Wrong number of arguments passed when running executable";
    for (int64_t i = 0; i < entry_comp->num_parameters(); ++i) {
      const Shape& expected_shape =
          entry_comp->parameter_instruction(i)->shape();
      const Shape& actual_shape = arguments[i].Buffers().shape();
      TF_RET_CHECK(
          ShapeUtil::DynamicShapeIsCompatible(actual_shape, expected_shape))
          << "Shape mismatch on argument " << i << ", "
          << expected_shape.ToString(/*print_layout=*/true) << " vs. "
          << actual_shape.ToString(/*print_layout=*/true);
    }
  }

  auto* host_stream = dynamic_cast<se::host::HostStream*>(
      run_options->stream()->implementation());
  se::Stream* stream = run_options->stream();
  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  TF_ASSIGN_OR_RETURN(
      std::vector<MaybeOwningDeviceMemory> buffers,
      CreateBufferTable(memory_allocator, stream->parent()->device_ordinal(),
                        arguments));

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput result,
      CreateResultShapedBuffer(run_options, absl::MakeSpan(buffers),
                               absl::MakeSpan(arguments)));

  // Logically we want this lambda to capture `buffers` by move, ultimately our
  // functor needs to be wrapped in an std::function, and that requires its
  // functor to be copyable.  Thus we perpetrate the hack of capturing buffers
  // "by shared pointer".
  //
  // We also need to change the types of some of the variables we capture:
  // run_options needs to change from a pointer to a value type, and arguments
  // needs to change from a Span into a vector.  We use a struct instead
  // of a lambda to make this explicit.
  struct AsyncRunTask {
    CpuExecutable* executable;
    ServiceExecutableRunOptions run_options;
    std::shared_ptr<std::vector<MaybeOwningDeviceMemory>> task_buffers;
    HloExecutionProfile* hlo_execution_profile;

    Status operator()() {
      return executable->ExecuteComputeFunction(
          &run_options.run_options(), *task_buffers, hlo_execution_profile);
    }
  };
  host_stream->EnqueueTaskWithStatus(
      AsyncRunTask{this, *run_options,
                   std::make_shared<std::vector<MaybeOwningDeviceMemory>>(
                       std::move(buffers)),
                   hlo_execution_profile});

  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}

/*static*/ int64_t CpuExecutable::ShapeSizeBytes(const Shape& shape) {
  // On the cpu, opaques are pointers.
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  if (shape.is_static() || shape.IsTuple()) {
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*)) + metadata_size;
}

const InstructionValueSet& CpuExecutable::GetRootValueSet() const {
  return assignment_->dataflow_analysis().GetInstructionValueSet(
      module().entry_computation()->root_instruction());
}

int64_t CpuExecutable::SizeOfGeneratedCodeInBytes() const {
  // TODO(b/233850967): support profiling in XLA:CPU-Next, instead of
  // punting on it as we are doing here.
  if (IsXlaRuntime()) return 0;
  return jit_->SizeOfGeneratedCodeInBytes();
}

}  // namespace cpu
}  // namespace xla
