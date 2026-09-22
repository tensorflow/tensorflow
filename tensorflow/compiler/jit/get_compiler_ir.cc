/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/get_compiler_ir.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/translate/portable_api.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {

static absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>>
BuildExecutable(xla::PjRtClient* pjrt_client,
                const XlaCompiler::CompilationResult& result,
                const XlaCompiler::Options& options,
                const bool xla_embed_ir_in_executable = false) {
  xla::CompileOptions compile_options = GetPjRtCompileOptions(options, result);
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_embed_ir_in_executable(xla_embed_ir_in_executable);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      pjrt_client->CompileAndLoad(*result.computation, compile_options));
  return executable;
}

static absl::StatusOr<std::string> BuildHLOString(
    IrExportStage stage, const XlaCompiler::CompilationResult& result,
    xla::PjRtClient* pjrt_client, const XlaCompiler::Options& options) {
  switch (stage) {
    case IrExportStage::STABLEHLO:
    case IrExportStage::STABLEHLO_SERIALIZED:
    case IrExportStage::HLO:
    case IrExportStage::HLO_NO_METADATA:
    case IrExportStage::HLO_SERIALIZED: {
      TF_ASSIGN_OR_RETURN(xla::ProgramShape program_shape,
                          result.computation->GetProgramShape());
      xla::HloModuleConfig config(program_shape);
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<xla::HloModule> new_module,
          xla::HloModule::CreateFromProto(result.computation->proto(), config));

      if (stage == IrExportStage::STABLEHLO_SERIALIZED) {
        TF_ASSIGN_OR_RETURN(
            std::string stablehlo,
            xla::ConvertHloToStablehlo(*new_module, /*emit_bytecode=*/true));
        return stablehlo;
      }
      if (stage == IrExportStage::STABLEHLO) {
        TF_ASSIGN_OR_RETURN(
            std::string stablehlo,
            xla::ConvertHloToStablehlo(*new_module, /*emit_bytecode=*/false));
        return stablehlo;
      }

      xla::HloPrintOptions opts;
      opts.set_print_large_constants(false);
      opts.set_print_operand_shape(true);
      if (stage == IrExportStage::HLO_NO_METADATA) {
        opts.set_print_metadata(false);
      }
      if (stage == IrExportStage::HLO_SERIALIZED) {
        return new_module->ToProto().SerializeAsString();
      }
      return new_module->ToString(opts);
    }
    case IrExportStage::OPTIMIZED_HLO:
    case IrExportStage::OPTIMIZED_HLO_SERIALIZED:
    case IrExportStage::OPTIMIZED_HLO_PROTO_SERIALIZED:
    case IrExportStage::OPTIMIZED_HLO_DOT: {
      TF_RET_CHECK(pjrt_client != nullptr);
      bool xla_embed_ir_in_executable =
          (stage == IrExportStage::OPTIMIZED_HLO_PROTO_SERIALIZED);
      TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                          BuildExecutable(pjrt_client, result, options,
                                          xla_embed_ir_in_executable));
      TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<xla::HloModule>> modules,
                          executable->GetHloModules());
      TF_RET_CHECK(!modules.empty());
      if (stage == IrExportStage::OPTIMIZED_HLO_SERIALIZED) {
        return modules[0]->ToProto().SerializeAsString();
      } else if (stage == IrExportStage::OPTIMIZED_HLO_PROTO_SERIALIZED) {
        xla::HloProto hlo_proto = xla::MakeHloProto(*modules[0]);
        absl::StatusOr<xla::CompiledMemoryStats> memory_stats =
            executable->GetCompiledMemoryStats();
        if (memory_stats.ok() &&
            !memory_stats->serialized_buffer_assignment.empty()) {
          hlo_proto.mutable_buffer_assignment()->ParseFromString(
              memory_stats->serialized_buffer_assignment);
        }
        return hlo_proto.SerializeAsString();
      } else if (stage == IrExportStage::OPTIMIZED_HLO_DOT) {
        absl::StatusOr<std::string> graph = xla::RenderGraph(
            *modules[0]->entry_computation(), "Visualization",
            /*debug_options=*/{}, xla::RenderedGraphFormat::kDot,
            /*hlo_render_options=*/{});
        TF_RETURN_IF_ERROR(graph.status());
        return *graph;
      } else {
        return modules[0]->ToString();
      }
    }
  }
}

static absl::StatusOr<std::vector<XlaCompiler::Argument>>
BuildXlaCompilerArgumentFromTensorSpec(
    const FunctionBody* fbody, absl::Span<int const> must_be_constant_idxs,
    absl::Span<const Tensor* const> inputs,
    absl::Span<VariableInfo const> variable_args,
    absl::Span<const ArgShapeAndDType> flat_arg_shape_and_dtype) {
  TF_RET_CHECK(fbody != nullptr);
  auto& input_args = fbody->record->fdef().signature().input_arg();
  int input_arg_size = input_args.size();
  std::vector<XlaCompiler::Argument> args;
  args.reserve(input_arg_size);

  for (auto& arg_info : flat_arg_shape_and_dtype) {
    XlaCompiler::Argument arg;
    arg.kind = XlaCompiler::Argument::kParameter;
    arg.type = arg_info.dtype;
    arg.shape = arg_info.shape;
    args.push_back(arg);
  }

  // Build Xla Compiler Arguments from concrete_fn.captured_inputs
  absl::flat_hash_map<int, const VariableInfo*> variable_info_lookup;
  TF_RETURN_IF_ERROR(
      CreateVariableInfoLookup(variable_args, variable_info_lookup));

  for (const VariableInfo& info : variable_args) {
    TF_RET_CHECK(!info.var() || info.lock_held() || info.shared_lock_held())
        << "Need to hold the lock on resource variables "
           "before calling BuildXlaCompilerArguments";
    variable_info_lookup.emplace(info.index(), &info);
  }

  int offset = flat_arg_shape_and_dtype.size();
  // Here it takes in the concrete_fn.captured_inputs and builds the appropriate
  // XLA compiler arguments.
  for (int64_t input_num = offset; input_num < input_arg_size; ++input_num) {
    const Tensor* input = inputs[input_num];

    XlaCompiler::Argument arg;
    if (variable_info_lookup.count(input_num)) {
      // Handles tf.resource variables.
      TF_RET_CHECK(input->dtype() == DT_RESOURCE);
      const VariableInfo& variable = *variable_info_lookup[input_num];
      arg.kind = XlaCompiler::Argument::kResource;
      arg.resource_kind = XlaResource::kVariable;
      arg.definition_stack_trace = variable.definition_stack_trace();
      TF_RET_CHECK(variable.var() && variable.var()->is_initialized);
      const Tensor* value = variable.var()->tensor();
      arg.type = value->dtype();
      arg.shape = value->shape();
      arg.initialized = true;
    } else {
      // Instead of embedding constant into HLO,
      // we handle tf.constant as parameter to reduce size.
      arg.kind = XlaCompiler::Argument::kParameter;
      arg.type = input->dtype();
      arg.shape = input->shape();
    }
    args.push_back(arg);
  }

  for (int64_t i = 0; i < input_arg_size; ++i) {
    args[i].name = input_args[i].name();
  }

  return args;
}

absl::Status ValidateGetCompilerIrTfrtTpu(absl::string_view device_type,
                                          stream_executor::Stream* stream,
                                          IrExportStage stage) {
  auto is_tfrt_tpu_supported_stage = [](IrExportStage stage) {
    return stage == IrExportStage::HLO ||
           stage == IrExportStage::HLO_NO_METADATA ||
           stage == IrExportStage::HLO_SERIALIZED ||
           stage == IrExportStage::STABLEHLO ||
           stage == IrExportStage::STABLEHLO_SERIALIZED;
  };
  // TODO(b/238830423): support GetCompilerIr on TFRT TPU device for stages
  // that requires compilation from HLO to executable.
  if (device_type != DEVICE_CPU && stream == nullptr &&
      !is_tfrt_tpu_supported_stage(stage)) {
    return absl::InternalError(
        "GetCompilerIr with requested stage is not supported on this device.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<XlaCompiler::Argument>> PrepareXlaCompilerArgs(
    FunctionLibraryRuntime* flr, const NameAttrList& function,
    EagerContext* context, Device* dev,
    absl::Span<const ArgShapeAndDType> input_arg_shape_and_dtype,
    absl::Span<const TensorHandle* const> input_handles,
    CompilerArgSource compiler_arg_source) {
  const FunctionBody* fbody = nullptr;
  std::vector<int> constant_arg_indices;
  std::vector<int> resource_arg_indices;
  TF_RETURN_IF_ERROR(GetBodyAndConstantsAndResources(
      flr, function, &fbody, &constant_arg_indices, &resource_arg_indices));
  if (dev == nullptr && !resource_arg_indices.empty()) {
    return absl::InternalError(
        "GetCompilerIr does not support resource arguments when device is "
        "null.");
  }

  // `input_args` includes both concrete_fn input args and captured_input here.
  auto& input_args = fbody->record->fdef().signature().input_arg();
  // Here input_arg_size = len(flat_args) + len(captured_input)
  int input_arg_size = input_args.size();

  std::vector<const Tensor*> inputs(input_arg_size);
  std::deque<Tensor> inputs_storage;
  std::vector<VariableInfo> variable_infos;
  int offset = input_arg_shape_and_dtype.size();

  for (int i = 0; i < input_handles.size(); i++) {
    const TensorHandle* th = input_handles[i];
    const Tensor* t;
    // Handle owns the tensor.
    TF_RETURN_IF_ERROR(th->Tensor(&t));
    if (absl::c_binary_search(constant_arg_indices, i)) {
      // Need to make sure it's on the host.
      inputs_storage.emplace_back(t->dtype(), t->shape());
      TF_RETURN_IF_ERROR(
          th->CopyToDevice(*context, /*d=*/nullptr, &inputs_storage.back()));
      inputs[i + offset] = &inputs_storage.back();
    } else {
      inputs[i + offset] = t;
    }
  }

  if (dev != nullptr) {
    TF_RETURN_IF_ERROR(GetVariableInfosFromInputs(dev->resource_manager(), dev,
                                                  inputs, resource_arg_indices,
                                                  &variable_infos));
    TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(variable_infos)));
  }

  absl::StatusOr<std::vector<XlaCompiler::Argument>> args;

  if (compiler_arg_source == CompilerArgSource::TENSOR_SPEC) {
    args = BuildXlaCompilerArgumentFromTensorSpec(fbody, constant_arg_indices,
                                                  inputs, variable_infos,
                                                  input_arg_shape_and_dtype);
  } else if (compiler_arg_source == CompilerArgSource::CONCRETE_INPUT) {
    args = XlaComputationLaunchContext::BuildXlaCompilerArguments(
        constant_arg_indices, inputs, variable_infos, dev);
  }
  return args;
}

absl::StatusOr<std::string> CompileAndBuildHLOString(
    IrExportStage stage, const XlaCompiler::Options& options,
    xla::PjRtClient* pjrt_client, const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args) {
  XlaCompiler::CompileOptions compile_options;
  compile_options.always_return_tuple = false;
  compile_options.alias_resource_update = true;

  XlaCompiler compiler(options);
  XlaCompiler::CompilationResult result;
  TF_RETURN_IF_ERROR(
      compiler.CompileFunction(compile_options, function, args, &result));

  return BuildHLOString(stage, result, pjrt_client, options);
}

/**
 * Clarifies the different meanings of 'input_arg_shape_and_dtype' and
 * 'input_handles' in different cases.
 *
 * For TENSOR_SPEC case:
 *   - `input_arg_shape_and_dtype`: Contains the shape and dtype of
 * concrete_fn input args.
 *   - `input_handles`: Contains the concrete_fn.captured_input tensors.
 *
 * For CONCRETE_INPUT case:
 *   - `input_arg_shape_and_dtype`: it is empty.
 *   - `input_handles`: Contains all concrete_fn inputs tensors, including
 * captured inputs.
 */
absl::StatusOr<std::string> GetCompilerIr(
    IrExportStage stage, ProcessFunctionLibraryRuntime* pflr,
    absl::string_view func_name, Device* dev, EagerContext* context,
    absl::Span<const ArgShapeAndDType> input_arg_shape_and_dtype,
    absl::Span<const TensorHandle* const> input_handles,
    CompilerArgSource compiler_arg_source) {
  using PjRtDeviceCompiler =
      DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;

  se::Stream* stream = nullptr;
  if (const DeviceBase::AcceleratorDeviceInfo* accelerator_device_info =
          dev->tensorflow_accelerator_device_info()) {
    stream = accelerator_device_info->stream;
  }
  TF_RETURN_IF_ERROR(
      ValidateGetCompilerIrTfrtTpu(dev->device_type(), stream, stage));

  NameAttrList function;
  function.set_name(std::string{func_name});

  FunctionLibraryRuntime* flr = pflr->GetFLR(dev->name());

  TF_ASSIGN_OR_RETURN(
      std::vector<XlaCompiler::Argument> args,
      PrepareXlaCompilerArgs(flr, function, context, dev,
                             input_arg_shape_and_dtype, input_handles,
                             compiler_arg_source));

  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(dev);
  DeviceType compilation_device_type = platform_info.device_type();
  if (platform_info.device_type() == DEVICE_TPU) {
    compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);
  } else {
    TF_ASSIGN_OR_RETURN(compilation_device_type,
                        GetCompilationDeviceType(platform_info.device_type()));
  }

  PjRtDeviceCompiler* pjrt_device_compiler = nullptr;
  DeviceCompilationProfiler* profiler = nullptr;
  TF_RETURN_IF_ERROR(GetOrCreatePjRtDeviceCompilerAndProfiler(
      platform_info, dev->resource_manager(), flr, &pjrt_device_compiler,
      &profiler));
  core::ScopedUnref pjrt_device_compiler_ref(pjrt_device_compiler);
  core::ScopedUnref profiler_ref(profiler);

  XlaCompiler::Options options;
  if (platform_info.device_type() == DEVICE_TPU && stream == nullptr) {
    options = GenerateCompilerOptionsForTfrtTpu(compilation_device_type, *flr);
  } else {
    options = GenerateCompilerOptionsForPjRt(*flr, dev, platform_info,
                                             pjrt_device_compiler);
  }

  return CompileAndBuildHLOString(
      stage, options, pjrt_device_compiler->client(), function, args);
}

absl::StatusOr<std::string> GetCompilerIr(
    IrExportStage stage, ProcessFunctionLibraryRuntime* pflr,
    absl::string_view func_name, absl::string_view platform_name,
    EagerContext* context,
    absl::Span<const ArgShapeAndDType> input_arg_shape_and_dtype,
    absl::Span<const TensorHandle* const> input_handles,
    CompilerArgSource compiler_arg_source) {
  TF_RETURN_IF_ERROR(
      ValidateGetCompilerIrTfrtTpu(platform_name, /*stream=*/nullptr, stage));

  NameAttrList function;
  function.set_name(std::string{func_name});

  std::string device_name;
  if (!platform_name.empty()) {
    device_name = absl::StrCat("/device:", platform_name, ":0");
  }
  FunctionLibraryRuntime* flr = pflr->GetFLR(device_name);
  if (flr == nullptr) {
    // Use CPU as the fallback to get the `FunctionLibraryRuntime`.
    flr = pflr->GetFLR("/device:CPU:0");
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<XlaCompiler::Argument> args,
      PrepareXlaCompilerArgs(flr, function, context, /*dev=*/nullptr,
                             input_arg_shape_and_dtype, input_handles,
                             compiler_arg_source));

  se::Platform::Id platform_id = nullptr;
  if (platform_name == DEVICE_CPU) {
    platform_id = se::host::kHostPlatformId;
  }
  XlaPlatformInfo platform_info(DeviceType(platform_name), platform_id,
                                /*xla_device_metadata=*/nullptr,
                                /*pjrt_device_metadata=*/nullptr,
                                /*device_allocator=*/nullptr);
  DeviceType compilation_device_type = platform_info.device_type();
  if (platform_info.device_type() == DEVICE_TPU) {
    compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);
  } else {
    TF_ASSIGN_OR_RETURN(compilation_device_type,
                        GetCompilationDeviceType(platform_info.device_type()));
  }

  XlaCompiler::Options options;
  if (platform_info.device_type() == DEVICE_TPU) {
    options = GenerateCompilerOptionsForTfrtTpu(compilation_device_type, *flr);
  } else {
    options.device_type = compilation_device_type;
    options.flib_def = flr->GetFunctionLibraryDefinition();
    options.graph_def_version = flr->graph_def_version();
    options.allow_cpu_custom_calls =
        (platform_info.platform_id() == se::host::kHostPlatformId);
    options.alias_passthrough_params = !platform_info.is_on_xla_device();
  }

  return CompileAndBuildHLOString(stage, options, /*pjrt_client=*/nullptr,
                                  function, args);
}

}  // namespace tensorflow
