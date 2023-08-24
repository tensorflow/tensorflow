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
#include <iostream>
#include <vector>

#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/runtime/aot_ffi_execution_context.h"

namespace tensorflow {

namespace {
// MemrefDesc's are part of the XLA Runtime ABI. Redefine them here (with a
// slightly different name to avoid confusion) because we cannot depend on
// XLA Runtime's headers.
// Note: this is an internal type, to be used exclusively in this file.
struct MemrefHolder {
  MemrefHolder(const XlaCompiledCpuFunction::ShapeInfo& shape_info,
               void* data_ptr)
      : rank(shape_info.num_dimensions), data(data_ptr), offset(0) {
    sizes.resize(shape_info.num_dimensions);
    strides.resize(shape_info.num_dimensions);
    int64_t multiplier = 1;
    for (int i = shape_info.num_dimensions - 1; i >= 0; --i) {
      int64_t size = shape_info.dimensions[i];
      sizes[i] = size;
      strides[i] = multiplier;
      multiplier *= size;
    }
  }

  unsigned rank = 0;
  // Note: dtype is not needed here.
  void* data = nullptr;
  int64_t offset = 0;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};
}  // namespace

XlaCompiledCpuFunction::XlaCompiledCpuFunction(const StaticData& static_data,
                                               AllocMode alloc_mode)
    : raw_function_(static_data.raw_function_),
      external_run_function_(static_data.external_run_function_),
      cpu_executable_(static_data.cpu_executable_),
      result_index_(static_data.result_index_),
      buffer_table_(new void*[static_data.num_buffers_]),
      buffer_infos_(static_data.buffer_infos_),
      num_buffers_(static_data.num_buffers_),
      num_results_(static_data.num_results_),
      result_index_table_(static_data.result_index_table_),
      arg_index_table_(static_data.arg_index_table_),
      num_args_(static_data.num_args_),
      num_variables_(static_data.num_variables_),
      arg_shape_infos_(static_data.arg_shape_infos_),
      result_shape_infos_(static_data.result_shape_infos_),
      arg_names_(static_data.arg_names_),
      variable_names_(static_data.variable_names_),
      result_names_(static_data.result_names_),
      program_shape_(static_data.program_shape_),
      hlo_profile_printer_data_(static_data.hlo_profile_printer_data_),
      use_xla_runtime_(static_data.use_xla_runtime_) {
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

bool XlaCompiledCpuFunction::RunXlaRuntime() {
  size_t num_memref_args = num_args_ + num_results_;
  std::vector<MemrefHolder> memref_args;
  memref_args.reserve(num_memref_args);

  size_t num_ptrs = 1;  // execution context.

  // Append arguments.
  for (int i = 0; i < num_args_; ++i) {
    const ShapeInfo& shape_info = arg_shape_infos_[i];
    memref_args.emplace_back(shape_info, buffer_table_[arg_index_table_[i]]);
    num_ptrs += 3 + 2 * shape_info.num_dimensions;
  }

  // Append results.
  for (int i = 0; i < num_results_; ++i) {
    const ShapeInfo& shape_info = result_shape_infos_[i];
    memref_args.emplace_back(shape_info, buffer_table_[result_index_table_[i]]);
    num_ptrs += 3 + 2 * shape_info.num_dimensions;

    // Point to this result from the "result" entry in the buffer table.
    void** results = static_cast<void**>(buffer_table_[result_index_]);
    results[i] = buffer_table_[result_index_table_[i]];
  }

  std::vector<void*> call_frame;
  call_frame.resize(num_ptrs);
  size_t ptr_index = 1;
  for (const MemrefHolder& memref : memref_args) {
    auto cast = [](const void* p) { return const_cast<void*>(p); };
    call_frame[ptr_index + 0] = cast(&memref.data);  // memref.basePtr
    call_frame[ptr_index + 1] = cast(&memref.data);  // memref.data
    call_frame[ptr_index + 2] = cast(&memref.offset);
    unsigned rank = memref.rank;
    for (int64_t d = 0; d < rank; ++d) {
      call_frame[ptr_index + 3 + d] = cast(&memref.sizes[d]);
      call_frame[ptr_index + 3 + d + rank] = cast(&memref.strides[d]);
    }
    ptr_index += 3 + 2 * rank;
  }

  assert(num_ptrs == ptr_index);

  xla::runtime::aot::ExecutionContext execution_context;
  execution_context.custom_call_data = &run_options_;
  xla::runtime::aot::ExecutionContext* execution_context_ptr =
      &execution_context;
  call_frame[0] = &execution_context_ptr;

  auto xla_runtime_func =
      reinterpret_cast<XlaRuntimeRawFunction>(raw_function_);
  xla_runtime_func(call_frame.data());
  if (execution_context.error) {
    // No error support in XLA; dump error message to stderr.
    std::cerr << "XLA AOT error: " << execution_context.error << ".\n";
    return false;
  }
  return true;
}

bool XlaCompiledCpuFunction::Run() {
  if (use_xla_runtime_) {
    return RunXlaRuntime();
  }
  if (external_run_function_) {
    std::vector<xla::cpu::BufferDesc> descriptor_table =
        MakeXlaRuntimeDescriptorTable();
    return external_run_function_(cpu_executable_, descriptor_table,
                                  &run_options_);
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
