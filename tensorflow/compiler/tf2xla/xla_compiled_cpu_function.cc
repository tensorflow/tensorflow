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

#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"

#include <cassert>
#include <vector>

#include "tensorflow/compiler/xla/cpu_function_runtime.h"

namespace tensorflow {

XlaCompiledCpuFunction::XlaCompiledCpuFunction(const StaticData& static_data,
                                               AllocMode alloc_mode)
    : raw_function_(static_data.raw_function_),
      run_function_(static_data.run_function_),
      cpu_executable_(static_data.cpu_executable_),
      result_index_(static_data.result_index_),
      buffer_table_(new void*[static_data.num_buffers_]),
      buffer_infos_(static_data.buffer_infos_),
      num_buffers_(static_data.num_buffers_),
      arg_index_table_(static_data.arg_index_table_),
      num_args_(static_data.num_args_),
      num_variables_(static_data.num_variables_),
      arg_names_(static_data.arg_names_),
      variable_names_(static_data.variable_names_),
      result_names_(static_data.result_names_),
      program_shape_(static_data.program_shape_),
      hlo_profile_printer_data_(static_data.hlo_profile_printer_data_) {
  bool allocate_entry_params =
      alloc_mode == AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS;
  // Allocate arg and temp buffers.
  alloc_buffer_table_ = xla::cpu_function_runtime::MallocContiguousBuffers(
      static_data.buffer_infos_, static_data.num_buffers_,
      /*allocate_entry_params=*/allocate_entry_params, buffer_table_,
      /*annotate_initialized=*/true);
  // If Hlo profiling is enabled the generated code expects an appropriately
  // sized buffer to be passed in as the last argument.  If Hlo profiling is
  // disabled the last function argument is still present in the function
  // signature, but it is ignored by the generated code and we pass in null for
  // it.
  if (hlo_profiling_enabled()) {
    profile_counters_ = new int64_t[static_data.profile_counters_size_]();
  }
}

bool XlaCompiledCpuFunction::Run() {
  if (run_function_) {
    std::vector<xla::cpu::BufferDesc> descriptor_table =
        MakeXlaRuntimeDescriptorTable();
    return run_function_(cpu_executable_, descriptor_table, &run_options_);
  }
  XlaCustomCallStatus status;
  raw_function_(buffer_table_[result_index_], &run_options_, nullptr,
                buffer_table_, &status, profile_counters_);
  return !xla::CustomCallStatusGetMessage(&status).has_value();
}

std::vector<xla::cpu::BufferDesc>
XlaCompiledCpuFunction::MakeXlaRuntimeDescriptorTable() {
  std::vector<xla::cpu::BufferDesc> descriptor_table;
  descriptor_table.reserve(num_buffers_);
  for (int32_t i = 0; i < num_buffers_; ++i) {
    void* data = buffer_table_[i];
    uint64_t size = buffer_infos_[i].size();
    descriptor_table.emplace_back(data, size);
  }
  return descriptor_table;
}

XlaCompiledCpuFunction::~XlaCompiledCpuFunction() {
  xla::cpu_function_runtime::FreeContiguous(alloc_buffer_table_);
  delete[] buffer_table_;
  delete[] profile_counters_;
}

namespace {

constexpr int kNotFound = -1;

// Linear search through `names` looking for a match with `name`. Returns -1 if
// the name isn't found, or is empty.
//
// REQUIRES: `names` is a nullptr-terminated array.
int LookupNameIndex(const string& name, const char** names) {
  // Hitting this assert means that there is no name-to-index data available;
  // for AOT try the setting the tfcompile --gen_name_to_index flag.
  assert(names != nullptr);

  if (name.empty()) {
    return kNotFound;
  }
  for (int index = 0; names[index] != nullptr; ++index) {
    if (name == names[index]) {
      return index;
    }
  }
  return kNotFound;
}

}  // namespace

int XlaCompiledCpuFunction::LookupArgIndex(const string& name) const {
  return LookupNameIndex(name, arg_names_);
}

int XlaCompiledCpuFunction::LookupVariableIndex(const string& name) const {
  int index = LookupNameIndex(name, variable_names_);
  if (index == kNotFound) {
    return kNotFound;
  }
  return num_args_ - num_variables_ + index;
}

int XlaCompiledCpuFunction::LookupResultIndex(const string& name) const {
  return LookupNameIndex(name, result_names_);
}

}  // namespace tensorflow
