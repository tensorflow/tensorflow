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

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/core/platform/types.h"

// Forward-declare, rather than include, to reduce code size for users that
// never use this functionality.
namespace xla {
class ProgramShape;
class HloProfilePrinterData;
}

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
  // Type of the raw function, produced by either JIT or AOT.
  using RawFunction = void (*)(void* result,
                               const xla::ExecutableRunOptions* run_options,
                               const void** args, void** temps,
                               int64* profile_counters);

  // StaticData represents the state necessary to run an XLA-compiled
  // function. For JIT this is backed by data in XlaJitCompiledCpuFunction; for
  // AOT this is backed by data compiled into the object file.
  struct StaticData {
    // The raw function to call.
    RawFunction raw_function;

    // Cardinality and size of arg buffers.
    const intptr_t* arg_sizes = nullptr;
    size_t num_args = 0;

    // Cardinality and size of temp buffers.
    //
    // If temp_sizes[i] >= 0 then the i'th temp is a regular temporary buffer.
    //
    // If temp_sizes[i] == -1 then the i'th temp is a constant buffer.  The
    // corresponding entry in the temp buffer array needs to be set to null.
    //
    // If temp_sizes[i] < -1 then the i'th temp is the entry parameter
    // -(temp_sizes[i] + 2).
    const intptr_t* temp_sizes = nullptr;
    size_t num_temps = 0;

    // The 0-based index of the result tuple, in the temp buffers.
    size_t result_index = 0;

    // [Optional] Arrays of arg and result names. These are arrays of C-style
    // strings, where the array is terminated by nullptr.
    const char** arg_names = nullptr;
    const char** result_names = nullptr;

    // [Optional] Arg and result shapes.
    const xla::ProgramShape* program_shape = nullptr;

    // [Optional] Profile printer data.  Null if profiling is disabled.
    const xla::HloProfilePrinterData* hlo_profile_printer_data = nullptr;

    // [Optional] The number of profile counters expected in the profile counter
    // buffer by the generated code and hlo_profile_printer.  0 if profiling is
    // disabled.  This information is already present in
    // hlo_profile_printer_data but xla::HloProfilePrinterData is forward
    // declared so we don't have access to that information here.
    int64 profile_counters_size = 0;
  };

  // AllocMode controls the buffer allocation mode.
  enum class AllocMode {
    // Allocate all buffers - args, results, profile and temps.
    ARGS_RESULTS_PROFILES_AND_TEMPS,

    // Only allocate result, profile and temp buffers.
    // Use set_arg_data to set argument buffers before Run is called.
    RESULTS_PROFILES_AND_TEMPS_ONLY,
  };

  XlaCompiledCpuFunction(
      const StaticData& static_data,
      AllocMode alloc_mode = AllocMode::ARGS_RESULTS_PROFILES_AND_TEMPS);
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

  // Returns the underlying array of argument buffers, where args()[I] is the
  // buffer for the positional argument at index I.
  void** args() { return args_; }
  const void* const* args() const { return args_; }

  // Returns the buffer for the positional argument at the given `index`.
  void* arg_data(size_t index) { return args_[index]; }
  const void* arg_data(size_t index) const { return args_[index]; }

  // Sets the buffer for the positional argument at the given `index` to `data`.
  // Must be called before Run to have an effect. May be called under any
  // AllocMode; if the AllocMode is RESULTS_AND_TEMPS_ONLY, this method must be
  // called for each positional argument, in order to set the argument buffers.
  //
  // Allocated memory must be aligned to the size specified by
  // tensorflow::tfcompile::runtime::kAlign. If possible, use the functions in
  // tensorflow/compiler/aot/runtime.h to ensure correct alignment.
  //
  // Aliasing of argument and result buffers is not allowed, and results in
  // undefined behavior.
  void set_arg_data(size_t index, void* data) { args_[index] = data; }

  // ------------------------------
  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. Unlike the arg methods,
  // there is no set_resultN_data method. The result buffers are managed
  // internally, and may change after each call to Run.

  // Returns the underlying array of result buffers, where results()[I] is the
  // buffer for the positional result at index I.
  void** results() { return static_cast<void**>(temps_[result_index_]); }
  const void* const* results() const {
    return static_cast<const void* const*>(temps_[result_index_]);
  }

  // Profile counters for this XLA computation.
  //
  // When Hlo profiling is enabled (`hlo_profiling_enabled()` return true in
  // this case) these counters are non-null and are automatically populated by
  // `Run`.  The counters can then be pretty-printed using
  // `hlo_profile_printer()`.
  //
  // When Hlo profiling is disabled, this accessor returns null.
  const int64* profile_counters() const { return profile_counters_; }

  // Returns the buffer for the positional result at the given `index`.
  void* result_data(size_t index) { return results()[index]; }
  const void* result_data(size_t index) const { return results()[index]; }

  // ------------------------------
  // Methods for extracting optional metadata.

  // Returns true iff data is available for the Lookup{Arg,Result}Index methods.
  // E.g. the data might not be compiled into the binary for AOT.
  bool HasNameIndices() const {
    return arg_names_ != nullptr && result_names_ != nullptr;
  }

  // Returns the 0-based index for the argument with the given `name`.
  // Returns -1 if the name wasn't found, or data isn't available.
  //
  // The index remains constant for every instance of XlaCompiledCpuFunction
  // generated from the same static data, and might not be cheap to determine.
  // Recommended usage is to capture this in a variable for re-use.
  int LookupArgIndex(const string& name) const;

  // Returns the 0-based index for the result with the given `name`.
  // Returns -1 if the name wasn't found, or data isn't available.
  //
  // The index remains constant for every instance of XlaCompiledCpuFunction
  // generated from the same static data, and might not be cheap to determine.
  // Recommended usage is to capture this in a variable for re-use.
  int LookupResultIndex(const string& name) const;

  // Returns the shape of the args and results. May return nullptr if the
  // program shape isn't available.
  const xla::ProgramShape* ProgramShape() const { return program_shape_; }

  bool hlo_profiling_enabled() const {
    return hlo_profile_printer_data_ != nullptr;
  }
  const xla::HloProfilePrinterData& hlo_profile_printer_data() const {
    assert(hlo_profiling_enabled());
    return *hlo_profile_printer_data_;
  }

 private:
  const RawFunction raw_function_;
  const size_t result_index_;

  // Arrays of argument and temp buffers; entries in args_ may be overwritten by
  // the user.
  void** args_ = nullptr;
  void** temps_ = nullptr;

  // Argument i needs to be placed in temps_[arg_index_to_temp_index_[i]] for
  // XLA generated code to be able to find it.
  //
  // For now we need to keep around the args_ array because there is code that
  // depends on args() returning a void**.  However, in the future we may remove
  // args_ in favor of using temps_ as the sole storage for the arguments.
  int32* arg_index_to_temp_index_;

  // The number of incoming arguments.
  int32 num_args_;

  // Backing memory for individual arg and temp buffers.
  void* alloc_args_ = nullptr;
  void* alloc_temps_ = nullptr;

  // Backing memory for profiling counters.
  int64* profile_counters_ = nullptr;

  // Options and context passed to the compiled function.
  xla::ExecutableRunOptions run_options_;

  // Optional metadata.
  const char** arg_names_ = nullptr;
  const char** result_names_ = nullptr;
  const xla::ProgramShape* program_shape_ = nullptr;
  const xla::HloProfilePrinterData* hlo_profile_printer_data_ = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_H_
