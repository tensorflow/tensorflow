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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

xla::StatusOr<std::string> GetCompilerIr(
    IrExportStage stage, ProcessFunctionLibraryRuntime* pflr,
    absl::string_view func_name, Device* dev,
    absl::Span<const Tensor* const> inputs) {
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

  std::vector<VariableInfo> variable_infos;
  TF_RETURN_IF_ERROR(GetVariableInfosFromInputs(
      rmgr, dev, inputs, resource_arg_indices, &variable_infos));
  TF_RETURN_IF_ERROR(LockVariables(absl::MakeSpan(variable_infos)));

  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(dev);

  XlaCompilationCache* cache;
  TF_RETURN_IF_ERROR(rmgr->LookupOrCreate<XlaCompilationCache>(
      rmgr->default_container(), "xla_cache", &cache,
      [&](XlaCompilationCache** cache_write_into) {
        return BuildXlaCompilationCache(dev, platform_info, cache_write_into);
      }));
  core::ScopedUnref cache_ref(cache);

  absl::optional<se::TfAllocatorAdapter> tf_allocator_adapter;

  XlaCompiler::Options options =
      GenerateCompilerOptions(*cache, *flr, dev,
                              /*stream=*/nullptr, platform_info,
                              /*has_ref_vars=*/false, &tf_allocator_adapter);

  XlaCompiler::CompileOptions compile_options;
  compile_options.always_return_tuple = false;
  compile_options.alias_resource_update = true;

  XlaCompiler compiler(options);

  xla::StatusOr<std::vector<XlaCompiler::Argument>> args =
      XlaComputationLaunchContext::BuildXlaCompilerArguments(
          constant_arg_indices, inputs, variable_infos);
  TF_RETURN_IF_ERROR(args.status());

  switch (stage) {
    case IrExportStage::HLO: {
      XlaCompiler::CompilationResult result;
      TF_RETURN_IF_ERROR(
          compiler.CompileFunction(compile_options, function, *args, &result));

      TF_ASSIGN_OR_RETURN(xla::ProgramShape program_shape,
                          result.computation->GetProgramShape());
      xla::HloModuleConfig config(program_shape);
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<xla::HloModule> new_module,
          xla::HloModule::CreateFromProto(result.computation->proto(), config));

      return new_module->ToString();
    }
    case IrExportStage::OPTIMIZED_HLO: {
      const XlaCompiler::CompilationResult* compilation_result = nullptr;
      xla::LocalExecutable* executable = nullptr;
      TF_RETURN_IF_ERROR(
          cache->Compile(options, function, *args, compile_options,
                         XlaCompilationCache::CompileMode::kStrict,
                         &compilation_result, &executable));
      return executable->executable()->module().ToString();
    }
  }
}

}  // namespace tensorflow
