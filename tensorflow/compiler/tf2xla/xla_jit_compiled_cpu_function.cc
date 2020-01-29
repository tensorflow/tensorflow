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

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/buffer_info_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/platform.h"

namespace tensorflow {

namespace {
constexpr char kHostPlatform[] = "Host";

// Returns the index of the result in the temp buffers.
xla::StatusOr<size_t> ComputeResultIndex(
    const xla::BufferAssignment& buffer_assignment) {
  TF_ASSIGN_OR_RETURN(const xla::BufferAllocation::Slice result_slice,
                      buffer_assignment.GetUniqueTopLevelOutputSlice());
  return result_slice.index();
}

// Collect names from `entries`, where T is one of tf2xla::{Feed,Fetch}. We hold
// the actual strings in nonempty_names, and hold arrays of pointers in
// name_ptrs, terminated by a nullptr entry.
template <typename T>
void CollectNames(const T& entries, std::vector<string>* nonempty_names,
                  std::vector<const char*>* name_ptrs) {
  // First collect `nonempty_names`, to ensure the underlying strings won't
  // change out from under us.
  for (const auto& entry : entries) {
    const string& name = entry.name();
    if (!name.empty()) {
      nonempty_names->push_back(name);
    }
  }
  // Now set `name_ptrs` pointing to the strings in `nonempty_names`.
  name_ptrs->reserve(entries.size() + 1);  // +1 for nullptr array terminator
  size_t nonempty_index = 0;
  for (const auto& entry : entries) {
    const string& name = entry.name();
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

/*static*/ xla::StatusOr<std::unique_ptr<XlaJitCompiledCpuFunction>>
XlaJitCompiledCpuFunction::Compile(
    const GraphDef& graph_def, const tf2xla::Config& config,
    const xla::ExecutableBuildOptions& build_options) {
  // Convert the graph_def into an xla::XlaComputation.
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      xla::PlatformUtil::GetPlatform(kHostPlatform));
  TF_ASSIGN_OR_RETURN(xla::LocalClient * client,
                      xla::ClientLibrary::GetOrCreateLocalClient(platform));
  xla::XlaComputation computation;
  TF_RETURN_IF_ERROR(tensorflow::ConvertGraphDefToXla(graph_def, config, client,
                                                      &computation));

  // Get and verify the program shape.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::ProgramShape> program_shape,
                      client->GetComputationShape(computation));
  if (program_shape->result().element_type() != xla::TUPLE) {
    // The XlaCompiler we use to build the xla computation always generates a
    // tuple result, and XlaCompiledCpuFunction relies on this for simpler
    // calling semantics.
    return errors::Internal(
        "XlaJitCompiledCpuFunction requires the XLA result to be a tuple");
  }
  // The parameter names are currently meaningless, and redundant with the rest
  // of our metadata, so clear them out to avoid confusion and save space.
  program_shape->clear_parameter_names();

  // Compute arg shapes, needed to compile the executable.
  std::vector<const xla::Shape*> arg_shapes;
  arg_shapes.reserve(program_shape->parameters_size());
  for (int i = 0; i < program_shape->parameters_size(); ++i) {
    arg_shapes.push_back(&program_shape->parameters(i));
  }

  // Compile the executable. The static_cast to the CpuExecutable subclass is
  // necessary since the raw function and buffer assignments are only available
  // there.
  TF_ASSIGN_OR_RETURN(auto executables,
                      client->Compile(computation, arg_shapes, build_options));
  TF_RET_CHECK(executables.size() == 1);
  std::unique_ptr<xla::LocalExecutable> executable = std::move(executables[0]);
  const xla::cpu::CpuExecutable* cpu_executable =
      static_cast<xla::cpu::CpuExecutable*>(executable->executable());
  XlaCompiledCpuFunction::RawFunction raw_function =
      cpu_executable->compute_function();
  const xla::BufferAssignment& buffer_assignment =
      cpu_executable->buffer_assignment();

  // Compute buffer infos and the result index, needed to run the raw function.
  std::vector<xla::cpu_function_runtime::BufferInfo> buffer_infos =
      xla::cpu::CreateBufferInfosFromBufferAssignment(buffer_assignment);
  std::vector<int32> arg_index_table =
      xla::cpu::CreateArgIndexTableFromBufferInfos(buffer_infos);
  TF_ASSIGN_OR_RETURN(size_t result_index,
                      ComputeResultIndex(buffer_assignment));

  std::unique_ptr<XlaJitCompiledCpuFunction> jit_unique_ptr(
      new XlaJitCompiledCpuFunction);
  XlaJitCompiledCpuFunction* jit = jit_unique_ptr.get();
  jit->executable_ = std::move(executable);
  jit->buffer_infos_ = std::move(buffer_infos);
  jit->arg_index_table_ = std::move(arg_index_table);
  jit->program_shape_ =
      absl::make_unique<xla::ProgramShapeProto>(program_shape->ToProto());
  XlaCompiledCpuFunction::set_static_data_raw_function(&jit->static_data_,
                                                       raw_function);
  XlaCompiledCpuFunction::set_static_data_buffer_infos(
      &jit->static_data_, jit->buffer_infos_.data());
  XlaCompiledCpuFunction::set_static_data_num_buffers(
      &jit->static_data_, jit->buffer_infos_.size());
  XlaCompiledCpuFunction::set_static_data_arg_index_table(
      &jit->static_data_, jit->arg_index_table_.data());
  XlaCompiledCpuFunction::set_static_data_num_args(
      &jit->static_data_, jit->arg_index_table_.size());
  XlaCompiledCpuFunction::set_static_data_result_index(&jit->static_data_,
                                                       result_index);
  // Optional metadata is collected and set below.
  CollectNames(config.feed(), &jit->nonempty_arg_names_, &jit->arg_names_);
  CollectNames(config.fetch(), &jit->nonempty_result_names_,
               &jit->result_names_);
  XlaCompiledCpuFunction::set_static_data_arg_names(&jit->static_data_,
                                                    jit->arg_names_.data());
  XlaCompiledCpuFunction::set_static_data_result_names(
      &jit->static_data_, jit->result_names_.data());
  XlaCompiledCpuFunction::set_static_data_program_shape(
      &jit->static_data_, jit->program_shape_.get());

  if (cpu_executable->hlo_profiling_enabled()) {
    XlaCompiledCpuFunction::set_static_data_hlo_profile_printer_data(
        &jit->static_data_, &cpu_executable->hlo_profile_printer_data());
    XlaCompiledCpuFunction::set_static_data_profile_counters_size(
        &jit->static_data_,
        cpu_executable->hlo_profile_printer_data().profile_counters_size());
  }

  return std::move(jit_unique_ptr);
}

}  // namespace tensorflow
