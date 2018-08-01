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
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

// Returns a vector of positional argument buffer sizes.
xla::StatusOr<std::vector<intptr_t>> ComputeArgSizes(
    const xla::ProgramShape& program_shape) {
  std::vector<intptr_t> arg_sizes;
  const size_t num_args = program_shape.parameters_size();
  arg_sizes.reserve(num_args);
  for (int i = 0; i < num_args; ++i) {
    const xla::Shape& arg_shape = program_shape.parameters(i);
    constexpr size_t kPointerSize = sizeof(void*);
    arg_sizes.push_back(xla::ShapeUtil::ByteSizeOf(arg_shape, kPointerSize));
  }
  return std::move(arg_sizes);
}

// Returns a vector of positional temporary buffer sizes.
xla::StatusOr<std::vector<intptr_t>> ComputeTempSizes(
    const xla::BufferAssignment& buffer_assignment) {
  const std::vector<xla::BufferAllocation>& allocations =
      buffer_assignment.Allocations();
  std::vector<intptr_t> temp_sizes;
  temp_sizes.reserve(allocations.size());
  for (const xla::BufferAllocation& allocation : allocations) {
    // Callers don't allocate temporary buffers for parameters. Nor for
    // thread-local buffers, which are lowered to alloca.
    if (allocation.is_entry_computation_parameter() ||
        allocation.is_thread_local()) {
      temp_sizes.push_back(-1);
    } else {
      temp_sizes.push_back(allocation.size());
    }
  }
  return std::move(temp_sizes);
}

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
  TF_ASSIGN_OR_RETURN(xla::LocalClient * client,
                      xla::ClientLibrary::GetOrCreateLocalClient());
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
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::LocalExecutable> executable,
                      client->Compile(computation, arg_shapes, build_options));
  const xla::cpu::CpuExecutable* cpu_executable =
      static_cast<xla::cpu::CpuExecutable*>(executable->executable());
  XlaCompiledCpuFunction::RawFunction raw_function =
      cpu_executable->compute_function();
  const xla::BufferAssignment& buffer_assignment =
      cpu_executable->buffer_assignment();

  // Compute buffer sizes and the result index, needed to run the raw function.
  TF_ASSIGN_OR_RETURN(std::vector<intptr_t> arg_sizes,
                      ComputeArgSizes(*program_shape));
  TF_ASSIGN_OR_RETURN(std::vector<intptr_t> temp_sizes,
                      ComputeTempSizes(buffer_assignment));
  TF_ASSIGN_OR_RETURN(size_t result_index,
                      ComputeResultIndex(buffer_assignment));

  std::unique_ptr<XlaJitCompiledCpuFunction> jit_unique_ptr(
      new XlaJitCompiledCpuFunction);
  XlaJitCompiledCpuFunction* jit = jit_unique_ptr.get();
  jit->executable_ = std::move(executable);
  jit->arg_sizes_ = std::move(arg_sizes);
  jit->temp_sizes_ = std::move(temp_sizes);
  jit->program_shape_ = std::move(program_shape);
  jit->static_data_.raw_function = std::move(raw_function);
  jit->static_data_.arg_sizes = jit->arg_sizes_.data();
  jit->static_data_.num_args = jit->arg_sizes_.size();
  jit->static_data_.temp_sizes = jit->temp_sizes_.data();
  jit->static_data_.num_temps = jit->temp_sizes_.size();
  jit->static_data_.result_index = result_index;
  // Optional metadata is collected and set below.
  CollectNames(config.feed(), &jit->nonempty_arg_names_, &jit->arg_names_);
  CollectNames(config.fetch(), &jit->nonempty_result_names_,
               &jit->result_names_);
  jit->static_data_.arg_names = jit->arg_names_.data();
  jit->static_data_.result_names = jit->result_names_.data();
  jit->static_data_.program_shape = jit->program_shape_.get();

  if (cpu_executable->hlo_profiling_enabled()) {
    jit->static_data_.hlo_profile_printer_data =
        &cpu_executable->hlo_profile_printer_data();
    jit->static_data_.profile_counters_size =
        cpu_executable->hlo_profile_printer_data().profile_counters_size();
  }

  return std::move(jit_unique_ptr);
}

}  // namespace tensorflow
