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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_H_

#include <cassert>
#include <string>
#include <vector>

#include "xla/cpu_function_runtime.h"
#include "xla/executable_run_options.h"
#include "xla/service/cpu/buffer_desc.h"
#include "xla/service/custom_call_status_internal.h"
#include "tensorflow/core/platform/types.h"

// Forward-declare, rather than include, to reduce code size for users that
// never use this functionality.
namespace xla {
class ProgramShapeProto;
class HloProfilePrinterData;

namespace cpu {
class CpuExecutable;
}  // namespace cpu
}  // namespace xla

namespace tensorflow {

// Represents a function compiled by XLA, produced via either JIT or AOT.
//
// The Run method invokes the actual computation, with inputs read from arg
// buffers, and outputs written to result buffers. Each Run call may also use a
// set of temporary buffers for the computation.
//
// By default each instance of this class manages its own arg, result and temp
// buffers. The AllocMode constructor parameter may be used to modify the buffer
// allocation strategy.
//
// Under the default allocation strategy, this class is thread-compatible:
// o Calls to non-const methods require exclusive access to the object.
// o Concurrent calls to const methods are OK, if those calls are made while it
//   is guaranteed that no thread may call a non-const method.
class XlaCompiledCpuFunction {
 public:
  // Type of the raw XLA Classic function, produced by either JIT or AOT.
  using RawFunction = void (*)(void* result,
                               const xla::ExecutableRunOptions* run_options,
                               const void** args, void** temps,
                               XlaCustomCallStatus*, int64_t* profile_counters);

  // Signature of the XLA Runtime raw function. Used only by XLA Runtime AOT.
  using XlaRuntimeRawFunction = void (*)(void**);

  // Signature of an external run function. Used only by XLA Runtime JIT.
  using ExternalRunFunction =
      bool (*)(const xla::cpu::CpuExecutable* cpu_executable,
               const std::vector<xla::cpu::BufferDesc>& descriptor_table,
               const xla::ExecutableRunOptions* run_options);

  // Simple struct to describe a tensor's shape.
  // Note: this is a poor man's substitute for xla::ShapeProto, but we cannot
  // depend on protobuf's in this library.
  // TODO(ecg): extend ShapeInfo to support tuples, if needed.
  struct ShapeInfo {
    const int32_t* dimensions = nullptr;
    int32_t num_dimensions = 0;
  };

  // StaticData represents the state necessary to run an XLA-compiled
  // function. For JIT this is backed by data in XlaJitCompiledCpuFunction; for
  // AOT this is backed by data compiled into the object file.
  //
  // The contents of StaticData are XLA-internal implementation details and
  // should not be relied on by clients (and therefore are private).
  class StaticData {
   private:
    // The raw function to call.
    RawFunction raw_function_;

    ExternalRunFunction external_run_function_ = nullptr;
    const xla::cpu::CpuExecutable* cpu_executable_ = nullptr;

    // Contains information about the buffers used by the XLA computation.
    const xla::cpu_function_runtime::BufferInfo* buffer_infos_ = nullptr;
    int32_t num_buffers_ = 0;

    // Result parameter i is described by
    // buffer_infos[result_index_table[i]].
    const int32* result_index_table_ = nullptr;

    // There are num_results result parameters.
    int64_t num_results_ = 0;

    // Entry parameter i is described by
    // buffer_infos[arg_index_table[i]].
    const int32* arg_index_table_ = nullptr;

    // There are num_args entry parameters.
    int64_t num_args_ = 0;

    // There are num_variables variables.
    int64_t num_variables_ = 0;

    // The 0-based index of the result tuple, in the temp buffers.
    size_t result_index_ = 0;

    const ShapeInfo* arg_shape_infos_ = nullptr;
    const ShapeInfo* result_shape_infos_ = nullptr;

    // [Optional] Arrays of arg and result names. These are arrays of C-style
    // strings, where the array is terminated by nullptr.
    const char** arg_names_ = nullptr;
    const char** variable_names_ = nullptr;
    const char** result_names_ = nullptr;

    // [Optional] Arg and result shapes.
    const xla::ProgramShapeProto* program_shape_ = nullptr;

    // [Optional] Profile printer data.  Null if profiling is disabled.
    const xla::HloProfilePrinterData* hlo_profile_printer_data_ = nullptr;

    // [Optional] The number of profile counters expected in the profile counter
    // buffer by the generated code and hlo_profile_printer.  0 if profiling is
    // disabled.  This information is already present in
    // hlo_profile_printer_data but xla::HloProfilePrinterData is forward
    // declared so we don't have access to that information here.
    int64_t profile_counters_size_ = 0;

    bool use_xla_runtime_ = false;

    // Only XlaCompiledCpuFunction is allowed to read and write the above
    // fields.
    friend class XlaCompiledCpuFunction;
  };

  // AllocMode controls the buffer allocation mode.
  enum class AllocMode {
    // Allocate all buffers - args, results, profile and temps.
    ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS,

    // Only allocate result, profile and temp buffers.
    // Use set_arg_data to set argument buffers before Run is called.
    RESULTS_PROFILES_AND_TEMPS_ONLY,
  };

  explicit XlaCompiledCpuFunction(
      const StaticData& static_data,
      AllocMode alloc_mode =
          AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS);
  virtual ~XlaCompiledCpuFunction();

  XlaCompiledCpuFunction(const XlaCompiledCpuFunction&) = delete;
  XlaCompiledCpuFunction& operator=(const XlaCompiledCpuFunction&) = delete;

  // Sets the intra-op thread pool used to run individual ops concurrently.
  void set_thread_pool(const Eigen::ThreadPoolDevice* pool) {
    run_options_.set_intra_op_thread_pool(pool);
  }

  // Runs the computation, with inputs read from arg buffers, and outputs
  // written to result buffers. Returns true on success and false on failure.
  bool Run();

  // Returns the error message from the previous failed Run call.
  //
  // TODO(fschneider): For now this always returns an empty string because there
  // is no support for error reporting in XLA. Remove this once all callers are
  // updated.
  string error_msg() const { return {}; }

  // ------------------------------
  // Arg methods for managing input buffers. Buffers are in row-major order.

  // Returns the buffer for the positional argument at the given `index`.
  void* arg_data(size_t index) {
    return buffer_table_[arg_index_table_[index]];
  }
  const void* arg_data(size_t index) const {
    return buffer_table_[arg_index_table_[index]];
  }

  int num_results() const { return num_results_; }

  int num_args() const { return num_args_; }

  int num_variables() const { return num_variables_; }

  // Returns the size of entry parameter `idx`.
  //
  // There is a static version of this method on tfcompile generated subclasses
  // of XlaCompiledCpuFunction, but try to prefer this when possible since it
  // works both for XlaJitCompiledCpuFunction and AOT compiled subclasses.
  int arg_size(int idx) const {
    assert(idx < num_args());
    return buffer_infos_[arg_index_table_[idx]].size();
  }

  // Sets the buffer for the positional argument at the given `index` to `data`.
  // Must be called before Run to have an effect. May be called under any
  // AllocMode; if the AllocMode is RESULTS_AND_TEMPS_ONLY, this method must be
  // called for each positional argument, in order to set the argument buffers.
  //
  // Allocated memory must be aligned to the size specified by
  // xla::cpu_function_runtime::MinAlign(). If possible, use the functions in
  // tensorflow/compiler/tf2xla/cpu_function_runtime.h to ensure correct
  // alignment.
  //
  // Aliasing of argument and result buffers is not allowed, and results in
  // undefined behavior.
  void set_arg_data(size_t index, const void* data) {
    assert((arg_size(index) < xla::cpu_function_runtime::MinAlign() ||
            (uintptr_t)data % xla::cpu_function_runtime::MinAlign() == 0) &&
           "Underaligned pointer!");
    // The const_cast is safe because the generated code does not write to arg
    // buffers.
    //
    // buffer_table_ contains pointers to buffers that _will_ be written to by
    // generated code so it would be misleading to make buffer_table_ a `const
    // void**`.
    buffer_table_[arg_index_table_[index]] = const_cast<void*>(data);
  }

  // ------------------------------
  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. Unlike the arg methods,
  // there is no set_resultN_data method. The result buffers are managed
  // internally, and may change after each call to Run.

  // Returns the underlying array of result buffers, where results()[I] is the
  // buffer for the positional result at index I.
  void** results() { return static_cast<void**>(buffer_table_[result_index_]); }
  const void* const* results() const {
    return static_cast<const void* const*>(buffer_table_[result_index_]);
  }

  // Profile counters for this XLA computation.
  //
  // When Hlo profiling is enabled (`hlo_profiling_enabled()` return true in
  // this case) these counters are non-null and are automatically populated by
  // `Run`.  The counters can then be pretty-printed using
  // `hlo_profile_printer()`.
  //
  // When Hlo profiling is disabled, this accessor returns null.
  const int64_t* profile_counters() const { return profile_counters_; }

  // Returns the buffer for the positional result at the given `index`.
  void* result_data(size_t index) { return results()[index]; }
  const void* result_data(size_t index) const { return results()[index]; }

  // ------------------------------
  // Methods for extracting optional metadata.

  // Returns true iff data is available for the Lookup{Arg,Variable,Result}Index
  // methods. E.g. the data might not be compiled into the binary for AOT.
  bool HasNameIndices() const {
    return arg_names_ != nullptr && variable_names_ != nullptr &&
           result_names_ != nullptr;
  }

  // Returns the 0-based index for the argument with the given `name`.
  // Returns -1 if the name wasn't found, or data isn't available.
  //
  // The index remains constant for every instance of XlaCompiledCpuFunction
  // generated from the same static data, and might not be cheap to determine.
  // Recommended usage is to capture this in a variable for re-use.
  int LookupArgIndex(const string& name) const;

  // Returns the 0-based index for the variable with the given `name`.
  // Returns -1 if the name wasn't found, or data isn't available.
  //
  // The index remains constant for every instance of XlaCompiledCpuFunction
  // generated from the same static data, and might not be cheap to determine.
  // Recommended usage is to capture this in a variable for re-use.
  int LookupVariableIndex(const string& name) const;

  // Returns the 0-based index for the result with the given `name`.
  // Returns -1 if the name wasn't found, or data isn't available.
  //
  // The index remains constant for every instance of XlaCompiledCpuFunction
  // generated from the same static data, and might not be cheap to determine.
  // Recommended usage is to capture this in a variable for re-use.
  int LookupResultIndex(const string& name) const;

  // Returns the shape of the args and results. May return nullptr if the
  // program shape isn't available.
  const xla::ProgramShapeProto* ProgramShape() const { return program_shape_; }

  bool hlo_profiling_enabled() const {
    return hlo_profile_printer_data_ != nullptr;
  }
  const xla::HloProfilePrinterData& hlo_profile_printer_data() const {
    assert(hlo_profiling_enabled());
    return *hlo_profile_printer_data_;
  }

 protected:
  // ---------------------------------------------------------------------------
  // Accessors for reading from and writing to instances of `StaticData`.
  //
  // Classes generated by tfcompile can call these because the generated classes
  // inherit from `XlaCompiledCpuFunction`.  `XlaJitCompiledCpuFunction` can
  // call these because it is explicitly added as a friend.

  static void set_static_data_raw_function(StaticData* static_data,
                                           RawFunction raw_function) {
    static_data->raw_function_ = raw_function;
  }

  static void set_static_data_external_run_function(
      StaticData* static_data, ExternalRunFunction external_run_function) {
    static_data->external_run_function_ = external_run_function;
  }

  static void set_static_data_cpu_executable(
      StaticData* static_data, const xla::cpu::CpuExecutable* cpu_executable) {
    static_data->cpu_executable_ = cpu_executable;
  }

  static void set_static_data_buffer_infos(
      StaticData* static_data,
      const xla::cpu_function_runtime::BufferInfo* buffer_infos) {
    static_data->buffer_infos_ = buffer_infos;
  }

  static void set_static_data_num_buffers(StaticData* static_data,
                                          size_t num_buffers) {
    static_data->num_buffers_ = num_buffers;
  }

  static void set_static_data_result_index_table(
      StaticData* static_data, const int32* result_index_table) {
    static_data->result_index_table_ = result_index_table;
  }

  static void set_static_data_num_results(StaticData* static_data,
                                          int64_t num_results) {
    static_data->num_results_ = num_results;
  }

  static void set_static_data_arg_index_table(StaticData* static_data,
                                              const int32* arg_index_table) {
    static_data->arg_index_table_ = arg_index_table;
  }

  static void set_static_data_num_args(StaticData* static_data,
                                       int64_t num_args) {
    static_data->num_args_ = num_args;
  }

  static void set_static_data_num_variables(StaticData* static_data,
                                            int64_t num_variables) {
    static_data->num_variables_ = num_variables;
  }

  static void set_static_data_result_index(StaticData* static_data,
                                           size_t result_index) {
    static_data->result_index_ = result_index;
  }

  static void set_static_data_arg_shape_infos(StaticData* static_data,
                                              const ShapeInfo* shape_infos) {
    static_data->arg_shape_infos_ = shape_infos;
  }

  static void set_static_data_result_shape_infos(StaticData* static_data,
                                                 const ShapeInfo* shape_infos) {
    static_data->result_shape_infos_ = shape_infos;
  }

  static void set_static_data_arg_names(StaticData* static_data,
                                        const char** arg_names) {
    static_data->arg_names_ = arg_names;
  }

  static void set_static_data_variable_names(StaticData* static_data,
                                             const char** variable_names) {
    static_data->variable_names_ = variable_names;
  }

  static void set_static_data_result_names(StaticData* static_data,
                                           const char** result_names) {
    static_data->result_names_ = result_names;
  }

  static void set_static_data_program_shape(
      StaticData* static_data, const xla::ProgramShapeProto* program_shape) {
    static_data->program_shape_ = program_shape;
  }

  static void set_static_data_hlo_profile_printer_data(
      StaticData* static_data,
      const xla::HloProfilePrinterData* hlo_profile_printer_data) {
    static_data->hlo_profile_printer_data_ = hlo_profile_printer_data;
  }

  static const xla::HloProfilePrinterData*
  get_static_data_hlo_profile_printer_data(StaticData* static_data) {
    return static_data->hlo_profile_printer_data_;
  }

  static void set_static_data_profile_counters_size(
      StaticData* static_data, int64_t profile_counters_size) {
    static_data->profile_counters_size_ = profile_counters_size;
  }

  static void set_static_data_use_xla_runtime(StaticData* static_data,
                                              bool use_xla_runtime) {
    static_data->use_xla_runtime_ = use_xla_runtime;
  }

 private:
  const RawFunction raw_function_;

  // [Optional] External Run() function.
  const ExternalRunFunction external_run_function_;
  // [Maybe Optional] CpuExecutable to be passed to external_run_function_.
  const xla::cpu::CpuExecutable* cpu_executable_;

  const size_t result_index_;

  // Array containing pointers to argument and temp buffers (slots corresponding
  // to constant and on-stack buffers are null).
  void** const buffer_table_;

  // Describes the buffers used by the XLA computation.
  const xla::cpu_function_runtime::BufferInfo* const buffer_infos_;
  const int32 num_buffers_;

  // Indices of expanded result tuple.
  const int32 num_results_;
  const int32* const result_index_table_;

  // Argument i needs to be placed in buffer_table_[arg_index_to_temp_index_[i]]
  // for XLA generated code to be able to find it.
  const int32* const arg_index_table_;

  // The number of incoming arguments.
  const int32 num_args_;

  // The number of incoming variables.
  const int32 num_variables_;

  // Shapes of the input arguments.
  const ShapeInfo* const arg_shape_infos_;

  // Shapes of the results.
  const ShapeInfo* const result_shape_infos_;

  // Backing memory for buffer_table_ and args_, the latter depending on
  // AllocMode.
  void* alloc_buffer_table_ = nullptr;

  // Backing memory for profiling counters.
  int64_t* profile_counters_ = nullptr;

  // Options and context passed to the compiled function.
  xla::ExecutableRunOptions run_options_;

  // Optional metadata.
  const char** arg_names_ = nullptr;
  const char** variable_names_ = nullptr;
  const char** result_names_ = nullptr;
  const xla::ProgramShapeProto* program_shape_ = nullptr;
  const xla::HloProfilePrinterData* hlo_profile_printer_data_ = nullptr;

  const bool use_xla_runtime_ = false;

  // Creates a descriptor table for XLA Runtime.
  std::vector<xla::cpu::BufferDesc> MakeXlaRuntimeDescriptorTable();

  bool RunXlaRuntime();

  // Add `XlaJitCompiledCpuFunction` as a friend so that it can access the
  // `set_static_data_*` static methods above.
  friend class XlaJitCompiledCpuFunction;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_H_
