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

#include "tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/backends/cpu/buffer_allocation_info_util.h"
#include "xla/backends/cpu/codegen/compiled_function_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {

namespace {

// Returns the index of the result in the temp buffers.
absl::StatusOr<size_t> ComputeResultIndex(
    const xla::BufferAssignment& buffer_assignment) {
  TF_ASSIGN_OR_RETURN(const xla::BufferAllocation::Slice result_slice,
                      buffer_assignment.GetUniqueTopLevelOutputSlice());
  return result_slice.index();
}

// Returns the number of results.
int CountResults(
    absl::Span<const xla::cpu::BufferAllocationInfo> buffer_infos) {
  int num_results = 0;
  for (const auto& info : buffer_infos) {
    if (info.is_result()) {
      ++num_results;
    }
  }
  return num_results;
}

// Collect names from `entries`, where T is one of
// tf2xla::{Feed,Fetch,Variable}. We hold the actual strings in nonempty_names,
// and hold arrays of pointers in name_ptrs, terminated by a nullptr entry.
template <typename T>
void CollectNames(const T& entries, std::vector<std::string>* nonempty_names,
                  std::vector<const char*>* name_ptrs) {
  // First collect `nonempty_names`, to ensure the underlying strings won't
  // change out from under us.
  for (const auto& entry : entries) {
    const std::string& name = entry.name();
    if (!name.empty()) {
      nonempty_names->push_back(name);
    }
  }
  // Now set `name_ptrs` pointing to the strings in `nonempty_names`.
  name_ptrs->reserve(entries.size() + 1);  // +1 for nullptr array terminator
  size_t nonempty_index = 0;
  for (const auto& entry : entries) {
    const std::string& name = entry.name();
    if (!name.empty()) {
      name_ptrs->push_back(nonempty_names->at(nonempty_index).c_str());
      ++nonempty_index;
    } else {
      name_ptrs->push_back("");
    }
  }
  name_ptrs->push_back(nullptr);  // array terminator
}

}  // namespace

/*static*/ absl::StatusOr<std::unique_ptr<XlaJitCompiledCpuFunction>>
XlaJitCompiledCpuFunction::Compile(
    const GraphDef& graph_def, const tf2xla::Config& config,
    const xla::ExecutableBuildOptions& build_options) {
  TF_ASSIGN_OR_RETURN(xla::PjRtCompiler * compiler,
                      xla::GetDefaultPjRtCompiler(xla::CpuName()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));

  // Convert the graph_def into an xla::XlaComputation.
  xla::XlaComputation computation;
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToXla(
      graph_def, config, compiler, &computation, client.get()));

  // Get and verify the program shape.
  TF_ASSIGN_OR_RETURN(xla::ProgramShape program_shape,
                      computation.GetProgramShape());
  if (program_shape.result().element_type() != xla::TUPLE) {
    // The XlaCompiler we use to build the xla computation always generates a
    // tuple result, and XlaCompiledCpuFunction relies on this for simpler
    // calling semantics.
    return absl::InternalError(
        "XlaJitCompiledCpuFunction requires the XLA result to be a tuple");
  }
  // The parameter names are currently meaningless, and redundant with the rest
  // of our metadata, so clear them out to avoid confusion and save space.
  program_shape.clear_parameter_names();

  // Compute arg shapes, needed to compile the executable.
  std::vector<xla::Shape> arg_shapes;
  arg_shapes.reserve(program_shape.parameters_size());
  for (int i = 0; i < program_shape.parameters_size(); ++i) {
    arg_shapes.push_back(program_shape.parameters(i));
  }

  xla::CompileOptions compile_options;
  compile_options.executable_build_options = build_options;
  compile_options.argument_layouts = std::move(arg_shapes);

  // Compile the executable.
  TF_ASSIGN_OR_RETURN(const xla::PjRtTopologyDescription* topology,
                      client->GetTopologyDescription());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtExecutable> pjrt_executable,
                      compiler->Compile(std::move(compile_options), computation,
                                        *topology, client.get()));

  auto* pjrt_cpu_executable =
      absl::down_cast<xla::PjRtCpuExecutable*>(pjrt_executable.get());
  if (!pjrt_cpu_executable) {
    return absl::InternalError(
        "Expected PjRtCpuExecutable from CPU compilation.");
  }
  xla::cpu::CpuExecutable* cpu_executable =
      static_cast<xla::cpu::CpuExecutable*>(
          pjrt_cpu_executable->cpu_executable().get());
  const xla::BufferAssignment& buffer_assignment =
      cpu_executable->buffer_assignment();

  // Compute buffer infos and the result index, needed to run the raw function.
  std::vector<xla::cpu::BufferAllocationInfo> buffer_infos =
      xla::cpu::CreateBufferAllocationInfos(cpu_executable->module(),
                                            buffer_assignment);

  std::vector<int32_t> arg_index_table =
      xla::cpu::CreateArgIndexTable(buffer_infos);
  std::vector<int32_t> result_index_table =
      xla::cpu::CreateResultIndexTable(buffer_infos);
  TF_ASSIGN_OR_RETURN(size_t result_index,
                      ComputeResultIndex(buffer_assignment));
  const int num_results = CountResults(buffer_infos);

  std::unique_ptr<XlaJitCompiledCpuFunction> jit_unique_ptr(
      new XlaJitCompiledCpuFunction);

  XlaJitCompiledCpuFunction* jit = jit_unique_ptr.get();

  if (!cpu_executable->has_thunks()) {
    return absl::InternalError(
        "JIT compilation supports only thunk execution.");
  }

  {
    // This is here for simplicity, effectively just used to get the thunk
    // information to the XlaCompiledCpuFunction.
    TF_ASSIGN_OR_RETURN(
        auto compilation_result,
        xla::cpu::CpuAotCompilationResult::Create(
            &cpu_executable->module(), &cpu_executable->buffer_assignment(),
            cpu_executable->module_name(),
            // Symbols and object files are not needed since the function
            // library will be backed by the one in the executable which is
            // owned by XlaJitCompiledCpuFunction.
            /*obj_files=*/{}, /*symbols=*/{},
            cpu_executable->thunks().thunk_sequence(),
            /*function_library=*/nullptr));

    const std::optional<size_t> temp_allocation_index =
        compilation_result->temp_allocation_index();

    XlaCompiledCpuFunction::set_static_data_temp_allocation_index(
        &jit->static_data_, temp_allocation_index);

    jit->compilation_result_proto_ =
        std::make_unique<xla::cpu::CompilationResultProto>(
            compilation_result->proto());

    auto compiled_function_library =
        absl::down_cast<xla::cpu::CompiledFunctionLibrary*>(
            cpu_executable->function_library());

    if (!compiled_function_library) {
      return absl::InternalError(
          "Could not downcast FunctionLibrary to CompiledFunctionLibrary");
    }

    // NOTE: This will work because the function library is by the
    // executable and keeps the function pointers alive.
    jit->function_library_symbol_map_ =
        compiled_function_library->GetTypelessSymbolsMap();
  }

  jit->pjrt_executable_ = std::move(pjrt_executable);
  jit->buffer_infos_ = std::move(buffer_infos);
  jit->arg_index_table_ = std::move(arg_index_table);
  jit->result_index_table_ = std::move(result_index_table);
  jit->program_shape_ =
      std::make_unique<xla::ProgramShapeProto>(program_shape.ToProto());
  XlaCompiledCpuFunction::set_static_data_compilation_result_proto(
      &jit->static_data_, jit->compilation_result_proto_.get());
  XlaCompiledCpuFunction::set_static_data_function_library_symbol_map(
      &jit->static_data_, jit->function_library_symbol_map_);

  XlaCompiledCpuFunction::set_static_data_buffer_infos(
      &jit->static_data_, jit->buffer_infos_.data());
  XlaCompiledCpuFunction::set_static_data_num_buffers(
      &jit->static_data_, jit->buffer_infos_.size());
  XlaCompiledCpuFunction::set_static_data_arg_index_table(
      &jit->static_data_, jit->arg_index_table_.data());
  XlaCompiledCpuFunction::set_static_data_result_index_table(
      &jit->static_data_, jit->result_index_table_.data());
  XlaCompiledCpuFunction::set_static_data_num_args(
      &jit->static_data_, jit->arg_index_table_.size());
  XlaCompiledCpuFunction::set_static_data_num_variables(&jit->static_data_,
                                                        config.variable_size());
  XlaCompiledCpuFunction::set_static_data_num_results(&jit->static_data_,
                                                      num_results);
  XlaCompiledCpuFunction::set_static_data_result_index(&jit->static_data_,
                                                       result_index);
  // Optional metadata is collected and set below.
  CollectNames(config.feed(), &jit->nonempty_arg_names_, &jit->arg_names_);

  auto variable_copy = config.variable();
  for (auto& var : variable_copy) {
    if (var.name().empty()) {
      var.set_name(var.node_name());
    }
  }
  CollectNames(variable_copy, &jit->nonempty_variable_names_,
               &jit->variable_names_);

  CollectNames(config.fetch(), &jit->nonempty_result_names_,
               &jit->result_names_);
  XlaCompiledCpuFunction::set_static_data_arg_names(&jit->static_data_,
                                                    jit->arg_names_.data());
  XlaCompiledCpuFunction::set_static_data_variable_names(
      &jit->static_data_, jit->variable_names_.data());
  XlaCompiledCpuFunction::set_static_data_result_names(
      &jit->static_data_, jit->result_names_.data());
  XlaCompiledCpuFunction::set_static_data_program_shape(
      &jit->static_data_, jit->program_shape_.get());

  return std::move(jit_unique_ptr);
}

}  // namespace tensorflow
