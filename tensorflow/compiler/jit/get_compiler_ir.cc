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

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

static StatusOr<std::unique_ptr<xla::LocalExecutable>> BuildExecutable(
    xla::LocalClient* local_client,
    const XlaCompiler::CompilationResult& result,
    const XlaCompiler::Options& options,
    const bool xla_embed_ir_in_executable = false) {
  std::vector<const xla::Shape*> argument_layouts(
      result.xla_input_shapes.size());
  for (int i = 0, end = result.xla_input_shapes.size(); i < end; ++i) {
    argument_layouts[i] = &result.xla_input_shapes[i];
  }
  xla::ExecutableBuildOptions build_options;
  if (result.collective_info) {
    build_options.set_num_replicas(result.collective_info->group_size);
  }
  build_options.set_device_ordinal(
      options.device_ordinal != -1 ? options.device_ordinal
                                   : local_client->default_device_ordinal());
  build_options.set_result_layout(result.xla_output_shape);
  build_options.set_device_allocator(options.device_allocator.get());
  build_options.set_alias_passthrough_params(options.alias_passthrough_params);
  build_options.mutable_debug_options()->set_xla_detailed_logging_and_dumping(
      options.detailed_logging);
  // If the embed_ir_in_executable is set, hlo_proto will be dumped in
  // executable. The hlo_proto contains HLO modules and buffer assignment.
  build_options.mutable_debug_options()->set_xla_embed_ir_in_executable(
      xla_embed_ir_in_executable);
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::LocalExecutable>> executables,
      local_client->Compile(*result.computation, argument_layouts,
                            build_options));
  TF_RET_CHECK(executables.size() == 1);
  return std::move(executables[0]);
}

StatusOr<std::string> GetCompilerIr(
    IrExportStage stage, ProcessFunctionLibraryRuntime* pflr,
    absl::string_view func_name, Device* dev, EagerContext* context,
    absl::Span<const TensorHandle* const> inputs_handles) {
  NameAttrList function;
  function.set_name(std::string{func_name});

  FunctionLibraryRuntime* flr = pflr->GetFLR(dev->name());
  ResourceMgr* rmgr = dev->resource_manager();

  const FunctionBody* fbody = nullptr;
  std::vector<int> constant_arg_indices;
  std::vector<int> resource_arg_indices;
  TF_RETURN_IF_ERROR(GetBodyAndConstantsAndResources(
      flr, function, &fbody, &constant_arg_indices, &resource_arg_indices));

  MemoryTypeVector input_memory_types =
      GetInputMemoryTypes(fbody, constant_arg_indices, resource_arg_indices);
  MemoryTypeVector output_memory_types = GetOutputMemoryTypes(fbody);

  std::deque<Tensor> inputs_storage;
  std::vector<const Tensor*> inputs;
  inputs.reserve(inputs_handles.size());
  for (int i = 0; i < inputs_handles.size(); i++) {
    const TensorHandle* th = inputs_handles[i];
    const Tensor* t;
    // Handle owns the tensor.
    TF_RETURN_IF_ERROR(th->Tensor(&t));
    if (absl::c_binary_search(constant_arg_indices, i)) {
      // Need to make sure it's on the host.
      inputs_storage.emplace_back(t->dtype(), t->shape());
      TF_RETURN_IF_ERROR(
          th->CopyToDevice(*context, /*d=*/nullptr, &inputs_storage.back()));
      inputs.push_back(&inputs_storage.back());
    } else {
      inputs.push_back(t);
    }
  }

  std::vector<VariableInfo> variable_infos;
  TF_RETURN_IF_ERROR(GetVariableInfosFromInputs(
      rmgr, dev, inputs, resource_arg_indices, &variable_infos));
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(variable_infos)));

  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(dev);

  XlaCompilationCache* cache;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<XlaCompilationCache>(
      rmgr->default_container(), "xla_cache", &cache,
      [&](XlaCompilationCache** cache_write_into) {
        return BuildXlaCompilationCache(dev, flr, platform_info,
                                        cache_write_into);
      }));
  core::ScopedUnref cache_ref(cache);

  se::Stream* stream = nullptr;
  if (const DeviceBase::AcceleratorDeviceInfo* accelerator_device_info =
          dev->tensorflow_accelerator_device_info()) {
    stream = accelerator_device_info->stream;
  }

  XlaCompiler::Options options =
      GenerateCompilerOptions(*cache, *flr, dev, stream, platform_info,
                              /*has_ref_vars=*/false);

  XlaCompiler::CompileOptions compile_options;
  compile_options.always_return_tuple = false;
  compile_options.alias_resource_update = true;

  XlaCompiler compiler(options);

  StatusOr<std::vector<XlaCompiler::Argument>> args =
      XlaComputationLaunchContext::BuildXlaCompilerArguments(
          constant_arg_indices, inputs, variable_infos, dev);
  TF_RETURN_IF_ERROR(args.status());

  xla::LocalClient* local_client = cache->client();
  XlaCompiler::CompilationResult result;
  TF_RETURN_IF_ERROR(
      compiler.CompileFunction(compile_options, function, *args, &result));

  switch (stage) {
    case IrExportStage::HLO:
    case IrExportStage::HLO_NO_METADATA:
    case IrExportStage::HLO_SERIALIZED: {
      TF_ASSIGN_OR_RETURN(xla::ProgramShape program_shape,
                          result.computation->GetProgramShape());
      xla::HloModuleConfig config(program_shape);
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<xla::HloModule> new_module,
          xla::HloModule::CreateFromProto(result.computation->proto(), config));

      xla::HloPrintOptions opts;
      if (stage == IrExportStage::HLO_NO_METADATA) {
        opts.set_print_metadata(false);
      }

      if (stage == IrExportStage::HLO_SERIALIZED) {
        return new_module->ToProto().SerializeAsString();
      } else {
        return new_module->ToString(opts);
      }
    }
    case IrExportStage::OPTIMIZED_HLO:
    case IrExportStage::OPTIMIZED_HLO_SERIALIZED: {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::LocalExecutable> executable,
                          BuildExecutable(local_client, result, options));
      xla::Executable* new_executable = executable->executable();
      if (stage == IrExportStage::OPTIMIZED_HLO_SERIALIZED) {
        return new_executable->module().ToProto().SerializeAsString();
      } else {
        return new_executable->module().ToString();
      }
    }
    case IrExportStage::OPTIMIZED_HLO_PROTO_SERIALIZED: {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::LocalExecutable> executable,
                          BuildExecutable(local_client, result, options,
                                          /*xla_embed_ir_in_executable=*/true));
      return executable->executable()->hlo_proto()->SerializeAsString();
    }
    case IrExportStage::OPTIMIZED_HLO_DOT: {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::LocalExecutable> executable,
                          BuildExecutable(local_client, result, options));
      StatusOr<std::string> graph = xla::RenderGraph(
          *executable->executable()->module().entry_computation(),
          "Visualization",
          /*debug_options=*/{}, xla::RenderedGraphFormat::kDot,
          /*hlo_execution_profile=*/nullptr,
          /*hlo_render_options=*/{});
      TF_RETURN_IF_ERROR(graph.status());
      return *graph;
    }
  }
}

}  // namespace tensorflow
