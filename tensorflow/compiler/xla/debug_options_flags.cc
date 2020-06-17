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

#include "tensorflow/compiler/xla/debug_options_flags.h"

#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/debug_options_parsers.h"
#include "tensorflow/compiler/xla/parse_flags_from_env.h"

namespace xla {

DebugOptions DefaultDebugOptionsIgnoringFlags() {
  DebugOptions opts;
  opts.set_xla_llvm_enable_alias_scope_metadata(true);
  opts.set_xla_llvm_enable_noalias_metadata(true);
  opts.set_xla_llvm_enable_invariant_load_metadata(true);
  opts.set_xla_llvm_disable_expensive_passes(false);
  opts.set_xla_backend_optimization_level(3);
  opts.set_xla_gpu_autotune_level(4);
  opts.set_xla_cpu_multi_thread_eigen(true);
  opts.set_xla_gpu_cuda_data_dir("./cuda_sdk_lib");
  opts.set_xla_gpu_asm_extra_flags("");
  opts.set_xla_eliminate_hlo_implicit_broadcast(true);
  opts.set_xla_dump_hlo_as_html(false);
  opts.set_xla_dump_include_timestamp(true);
  opts.set_xla_dump_max_hlo_modules(-1);
#ifdef INTEL_MKL
  opts.set_xla_cpu_use_mkl_dnn(true);
#endif  // INTEL_MKL
  opts.set_xla_gpu_max_kernel_unroll_factor(4);
  // Set cudnn batchnorm off by default; it does not provide a performance win
  // on average.
  opts.set_xla_gpu_use_cudnn_batchnorm(false);

  // Run all GPU work on one stream by default.  Using multiple streams
  // increases memory usage and we lack strong motivating benchmarks for tuning
  // the heuristics needed to decide when to run on multiple streams.  See
  // b/77879207.
  opts.set_xla_gpu_disable_multi_streaming(true);

  // Disable forms of fast math that have caused users problems in the past.
  opts.set_xla_cpu_enable_fast_math(true);
  opts.set_xla_cpu_fast_math_honor_nans(true);
  opts.set_xla_cpu_fast_math_honor_infs(true);
  opts.set_xla_cpu_fast_math_honor_functions(true);
  opts.set_xla_cpu_fast_math_honor_division(true);

  // By default, copy TF's Eigen style min_max behavior with nans.
  opts.set_xla_cpu_enable_fast_min_max(false);

  opts.set_xla_gpu_enable_fast_min_max(true);

  opts.set_xla_allow_excess_precision(true);
  opts.set_xla_force_host_platform_device_count(1);
  opts.set_xla_gpu_deterministic_reductions(false);
  opts.set_xla_cpu_enable_xprof_traceme(true);
  opts.set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(false);

  return opts;
}

static absl::once_flag flags_init;
static DebugOptions* flag_values;
static std::vector<tensorflow::Flag>* flag_objects;

// Maps pass -> initial fuel values (parsed when AllocateFlags was run).
static absl::flat_hash_map<string, int64>* initial_fuel;

// Maps pass -> whether fuel was ever consumed for that pass.
static absl::node_hash_map<string, std::atomic<bool>>* fuel_ever_consumed;

// Maps pass -> remaining fuel.
//
// All threads start off using this global fuel pool, but ResetThreadLocalFuel()
// switches them to a thread-local fuel pool.
static absl::node_hash_map<string, std::atomic<int64>>* global_fuel;

// If we're using thread-local fuel, this stores it.
static thread_local std::unique_ptr<
    absl::node_hash_map<string, std::atomic<int64>>>
    thread_fuel;  // NOLINT (global variable with nontrivial destructor)

// Logs a warning if a pass's fuel was never consumed, on the theory that this
// may be a typo in the flag value.  Called atexit.
static void WarnIfFuelWasNeverConsumed() {
  CHECK(fuel_ever_consumed != nullptr);
  for (const auto& kv : *fuel_ever_consumed) {
    absl::string_view pass = kv.first;
    bool was_consumed = kv.second;
    if (!was_consumed) {
      LOG(ERROR) << absl::StreamFormat(
          "Compiler fuel for \"%s\" was never consumed. This may be a typo in "
          "the --xla_fuel flag you passed.",
          pass);
    }
  }
}

// Allocates flag_values and flag_objects; this function must not be called more
// than once - its call done via call_once.
static void AllocateFlags() {
  flag_values = new DebugOptions(DefaultDebugOptionsIgnoringFlags());

  // Returns a lambda that calls "member_setter" on "flag_values" with the
  // argument passed in to the lambda.
  auto bool_setter_for = [](void (DebugOptions::*member_setter)(bool)) {
    return [member_setter](bool value) {
      (flag_values->*member_setter)(value);
      return true;
    };
  };

  // Returns a lambda that calls "member_setter" on "flag_values" with the
  // argument passed in to the lambda.
  auto int32_setter_for = [](void (DebugOptions::*member_setter)(int32)) {
    return [member_setter](int32 value) {
      (flag_values->*member_setter)(value);
      return true;
    };
  };

  auto string_setter_for =
      [](void (DebugOptions::*member_setter)(const string& value)) {
        return [member_setter](const string& value) {
          (flag_values->*member_setter)(value);
          return true;
        };
      };

  // Custom "sub-parser" lambda for xla_disable_hlo_passes.
  auto setter_for_xla_disable_hlo_passes = [](string comma_separated_values) {
    for (const auto& passname :
         std::vector<string>(absl::StrSplit(comma_separated_values, ','))) {
      flag_values->add_xla_disable_hlo_passes(passname);
    }
    return true;
  };

  // Custom "sub-parser" lambda for xla_enable_hlo_passes_only.
  auto setter_for_xla_enable_hlo_passes_only =
      [](string comma_separated_values) {
        for (const auto& passname :
             std::vector<string>(absl::StrSplit(comma_separated_values, ','))) {
          flag_values->add_xla_enable_hlo_passes_only(passname);
        }
        return true;
      };

  // Custom "sub-parser" lambda for xla_gpu_ptx_file.
  auto setter_for_xla_gpu_ptx_file = [](string value) {
    flag_values->add_xla_gpu_ptx_file(value);
    return true;
  };

  // Custom "sub-parser" lambda for xla_backend_extra_options.
  auto setter_for_xla_backend_extra_options =
      [](string comma_separated_values) {
        auto* extra_options_map =
            flag_values->mutable_xla_backend_extra_options();
        parse_xla_backend_extra_options(extra_options_map,
                                        comma_separated_values);
        return true;
      };

  // Custom "sub-parser" for xla_fuel.  Note that ConsumeFuel does not do any
  // locking on the fuel global variables.  This means that it's
  // illegal/undefined behavior to modify this flag value while the compiler is
  // running.
  initial_fuel = new absl::flat_hash_map<string, int64>();
  fuel_ever_consumed = new absl::node_hash_map<string, std::atomic<bool>>();
  global_fuel = new absl::node_hash_map<string, std::atomic<int64>>();
  auto setter_for_xla_fuel = [](string xla_fuel_value) {
    initial_fuel->clear();
    global_fuel->clear();
    fuel_ever_consumed->clear();

    for (const auto& kv : absl::StrSplit(xla_fuel_value, ',')) {
      std::vector<string> pass_and_fuel = absl::StrSplit(kv, '=');
      if (pass_and_fuel.size() != 2) {
        LOG(ERROR) << absl::StreamFormat(
            "Illegal value for --xla_fuel. Saw %s, but expected token %s to "
            "have format X=INTEGER.",
            xla_fuel_value, kv);
        return false;
      }
      const auto& pass = pass_and_fuel[0];
      const auto& fuel_str = pass_and_fuel[1];
      int64 fuel;
      if (!absl::SimpleAtoi(fuel_str, &fuel)) {
        LOG(ERROR) << absl::StreamFormat(
            "Illegal value for --xla_fuel. Saw %s, but expected token %s to be "
            "an integer.",
            xla_fuel_value, fuel_str);
        return false;
      }
      initial_fuel->emplace(pass, fuel);
      global_fuel->emplace(pass, fuel);
      fuel_ever_consumed->emplace(pass, false);
    }

    // If --xla_fuel was specified, register an atexit handler which logs a
    // warning if a pass was specified but never consumed any fuel, on the
    // theory that this is may be a typo.
    if (!initial_fuel->empty()) {
      static absl::once_flag register_atexit_once;
      absl::call_once(
          register_atexit_once,
          +[] { std::atexit(WarnIfFuelWasNeverConsumed); });
    }
    return true;
  };

  flag_objects = new std::vector<tensorflow::Flag>();
  flag_objects->reserve(55);
  // Don't use an initializer list for initializing the vector; this would
  // create a temporary copy, and exceeds the stack space when compiling with
  // certain configurations.
  flag_objects->push_back(tensorflow::Flag(
      "xla_cpu_enable_fast_math",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_fast_math),
      flag_values->xla_cpu_enable_fast_math(),
      "Enable unsafe fast-math optimizations in the CPU compiler; this may "
      "produce faster code at the expense of some accuracy."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_cpu_fast_math_honor_nans",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_nans),
      flag_values->xla_cpu_fast_math_honor_nans(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "allow operations to produce NaNs.  Ignored when "
      "xla_cpu_enable_fast_math is false."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_cpu_fast_math_honor_infs",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_infs),
      flag_values->xla_cpu_fast_math_honor_infs(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "allow operations to produce infinites.  Ignored when "
      "xla_cpu_enable_fast_math is false."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_cpu_fast_math_honor_division",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_division),
      flag_values->xla_cpu_fast_math_honor_division(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "forbid to use multiplication by the reciprocal instead of division. "
      "Ignored when xla_cpu_enable_fast_math is false."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_cpu_fast_math_honor_functions",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_functions),
      flag_values->xla_cpu_fast_math_honor_functions(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "forbid to approximate calculations for functions. Ignored when "
      "xla_cpu_enable_fast_math is false."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_cpu_enable_fast_min_max",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_fast_min_max),
      flag_values->xla_cpu_enable_fast_min_max(),
      "Enable fast floating point min/max lowering that always propagates "
      "NaNs."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_enable_fast_min_max",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_fast_min_max),
      flag_values->xla_gpu_enable_fast_min_max(),
      "Enable fast floating point min/max lowering that does not propagate "
      "NaNs."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_llvm_enable_alias_scope_metadata",
      bool_setter_for(&DebugOptions::set_xla_llvm_enable_alias_scope_metadata),
      flag_values->xla_llvm_enable_alias_scope_metadata(),
      "In LLVM-based backends, enable the emission of !alias.scope metadata in "
      "the generated IR."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_llvm_enable_noalias_metadata",
      bool_setter_for(&DebugOptions::set_xla_llvm_enable_noalias_metadata),
      flag_values->xla_llvm_enable_noalias_metadata(),
      "In LLVM-based backends, enable the emission of !noalias metadata in the "
      "generated IR."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_llvm_enable_invariant_load_metadata",
      bool_setter_for(
          &DebugOptions::set_xla_llvm_enable_invariant_load_metadata),
      flag_values->xla_llvm_enable_invariant_load_metadata(),
      "In LLVM-based backends, enable the emission of !invariant.load metadata "
      "in the generated IR."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_llvm_disable_expensive_passes",
      bool_setter_for(&DebugOptions::set_xla_llvm_disable_expensive_passes),
      flag_values->xla_llvm_disable_expensive_passes(),
      "In LLVM-based backends, disable a custom set of expensive optimization "
      "passes."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_backend_optimization_level",
      int32_setter_for(&DebugOptions::set_xla_backend_optimization_level),
      flag_values->xla_backend_optimization_level(),
      "Numerical optimization level for the XLA compiler backend."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_disable_hlo_passes", setter_for_xla_disable_hlo_passes, "",
      "Comma-separated list of hlo passes to be disabled. These names must "
      "exactly match the passes' names; no whitespace around commas."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_enable_hlo_passes_only", setter_for_xla_enable_hlo_passes_only, "",
      "Comma-separated list of hlo passes to be enabled. These names must "
      "exactly match the passes' names; no whitespace around commas. The "
      "unspecified passes are all disabled."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_disable_all_hlo_passes",
      bool_setter_for(&DebugOptions::set_xla_disable_all_hlo_passes), false,
      "Disables all HLO passes.  Notes that some passes are necessary for "
      "correctness and the invariants that must be satisfied by 'fully "
      "optimized' HLO are different for different devices and may change "
      "over time.  The only 'guarantee', such as it is, is that if you compile "
      "XLA and dump the optimized HLO for some graph, you should be able to "
      "run it again on the same device with the same build of XLA."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_embed_ir_in_executable",
      bool_setter_for(&DebugOptions::set_xla_embed_ir_in_executable),
      flag_values->xla_embed_ir_in_executable(),
      "Embed the compiler IR as a string in the executable."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_eliminate_hlo_implicit_broadcast",
      bool_setter_for(&DebugOptions::set_xla_eliminate_hlo_implicit_broadcast),
      flag_values->xla_eliminate_hlo_implicit_broadcast(),
      "Eliminate implicit broadcasts when lowering user computations to HLO "
      "instructions; use explicit broadcast instead."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_cpu_multi_thread_eigen",
      bool_setter_for(&DebugOptions::set_xla_cpu_multi_thread_eigen),
      flag_values->xla_cpu_multi_thread_eigen(),
      "When generating calls to Eigen in the CPU backend, use multi-threaded "
      "Eigen mode."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_cuda_data_dir", flag_values->mutable_xla_gpu_cuda_data_dir(),
      "If non-empty, specifies a local directory containing ptxas and nvvm "
      "libdevice files; otherwise we use those from runfile directories."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_ftz", bool_setter_for(&DebugOptions::set_xla_gpu_ftz),
      flag_values->xla_gpu_ftz(),
      "If true, flush-to-zero semantics are enabled in the code generated for "
      "GPUs."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_disable_multi_streaming",
      bool_setter_for(&DebugOptions::set_xla_gpu_disable_multi_streaming),
      flag_values->xla_gpu_disable_multi_streaming(),
      "If true, multi-streaming in the GPU backend is disabled."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_max_kernel_unroll_factor",
      int32_setter_for(&DebugOptions::set_xla_gpu_max_kernel_unroll_factor),
      flag_values->xla_gpu_max_kernel_unroll_factor(),
      "Specify the maximum kernel unroll factor for the GPU backend."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_ptx_file", setter_for_xla_gpu_ptx_file, "",
      "If non-empty, specifies a file containing ptx to use. The filename "
      "prefix must have the same pattern as PTX dumped by XLA. This allows to "
      "match one specific module. General workflow. Get the generated module "
      "ptx from XLA. Modify it. Then pass it back via this option."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_test_all_output_layouts",
      bool_setter_for(&DebugOptions::set_xla_test_all_output_layouts),
      flag_values->xla_test_all_output_layouts(),
      "Let ClientLibraryTestBase::ComputeAndCompare* test all permutations of "
      "output layouts. For example, with a 3D shape, all permutations of the "
      "set {0, 1, 2} are tried."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_test_all_input_layouts",
      bool_setter_for(&DebugOptions::set_xla_test_all_input_layouts),
      flag_values->xla_test_all_input_layouts(),
      "Let ClientLibraryTestBase::ComputeAndCompare* test all permutations of "
      "*input* layouts. For example, for 2 input arguments with 2D shape and "
      "4D shape, the computation will run 2! * 4! times for every possible "
      "layouts"));
  flag_objects->push_back(tensorflow::Flag(
      "xla_hlo_profile", bool_setter_for(&DebugOptions::set_xla_hlo_profile),
      flag_values->xla_hlo_profile(),
      "Instrument the computation to collect per-HLO cycle counts"));
  flag_objects->push_back(tensorflow::Flag(
      "xla_backend_extra_options", setter_for_xla_backend_extra_options, "",
      "Extra options to pass to a backend; comma-separated list of 'key=val' "
      "strings (=val may be omitted); no whitespace around commas."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_use_cudnn_batchnorm",
      bool_setter_for(&DebugOptions::set_xla_gpu_use_cudnn_batchnorm),
      flag_values->xla_gpu_use_cudnn_batchnorm(),
      "Allows the GPU backend to implement batchnorm HLOs using cudnn, rather "
      "than expanding them to a soup of HLOs."));
  flag_objects->push_back(
      tensorflow::Flag("xla_cpu_use_mkl_dnn",
                       bool_setter_for(&DebugOptions::set_xla_cpu_use_mkl_dnn),
                       flag_values->xla_cpu_use_mkl_dnn(),
                       "Generate calls to MKL-DNN in the CPU backend."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_crash_on_verification_failures",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_crash_on_verification_failures),
      flag_values->xla_gpu_crash_on_verification_failures(),
      "Crashes the program on extra verification failures, e.g. cuDNN cross "
      "checking failures"));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_autotune_level",
      int32_setter_for(&DebugOptions::set_xla_gpu_autotune_level),
      flag_values->xla_gpu_autotune_level(),
      "Set GEMM and Convolution auto-tuning level. 0 = off; 1 = on; 2 = "
      "on+init; 3 = on+init+reinit; 4 = on+init+reinit+check."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_force_host_platform_device_count",
      int32_setter_for(&DebugOptions::set_xla_force_host_platform_device_count),
      flag_values->xla_force_host_platform_device_count(),
      "Force the host platform to pretend that there are these many host "
      "\"devices\". All of these host devices are backed by the same "
      "threadpool. Setting this to anything other than 1 can increase overhead "
      "from context switching but we let the user override this behavior to "
      "help run tests on the host that run models in parallel across multiple "
      "devices."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_disable_gpuasm_optimizations",
      bool_setter_for(&DebugOptions::set_xla_gpu_disable_gpuasm_optimizations),
      flag_values->xla_gpu_disable_gpuasm_optimizations(),
      "In XLA:GPU run ptxas in -O0 (default is -O3)."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_asm_extra_flags",
      string_setter_for(&DebugOptions::set_xla_gpu_asm_extra_flags), "",
      "Pass extra parameters to the GPU assembler tool (i.e., ptxas for CUDA). "
      "If multiple parameters, separate them by comma."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_fuel", setter_for_xla_fuel, /*default_value_for_display=*/"",
      "Sets compiler fuel, useful for bisecting bugs in passes.  Format "
      "--xla_fuel=PASS1=NUM1,PASS2=NUM2,..."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_to", string_setter_for(&DebugOptions::set_xla_dump_to),
      flag_values->xla_dump_to(),
      "Directory into which debugging data is written. If not specified but "
      "another dumping flag is passed, data will be written to stdout. To "
      "explicitly write to stdout, set this to \"-\". The values \"sponge\" "
      "and \"test_undeclared_outputs_dir\" have a special meaning: They cause "
      "us to dump into the directory specified by the environment variable "
      "TEST_UNDECLARED_OUTPUTS_DIR."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_hlo_as_text",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_text),
      flag_values->xla_dump_hlo_as_text(),
      "Dumps HLO modules as text before and after optimizations. Results are "
      "written to the --xla_dump_to dir, or, if no dir is specified, to "
      "stdout."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_hlo_as_proto",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_proto),
      flag_values->xla_dump_hlo_as_proto(),
      "Dumps HLO modules as HloProtos to the directory specified by "
      "--xla_dump_to."));
  flag_objects->push_back(
      tensorflow::Flag("xla_dump_hlo_as_dot",
                       bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_dot),
                       flag_values->xla_dump_hlo_as_dot(),
                       "Dumps HLO modules rendered as dot files to the "
                       "directory specified by --xla_dump_to."));
  flag_objects->push_back(
      tensorflow::Flag("xla_dump_hlo_as_html",
                       bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_html),
                       flag_values->xla_dump_hlo_as_html(),
                       "Dumps HLO modules rendered as HTML files to the "
                       "directory specified by --xla_dump_to."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_hlo_as_url",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_url),
      flag_values->xla_dump_hlo_as_url(),
      "Tries to dump HLO modules rendered as URLs to stdout (and also to the "
      "directory specified by --xla_dump_to). This is not implemented by "
      "default; you need to add a plugin which calls "
      "RegisterGraphToURLRenderer()."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_hlo_snapshots",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_snapshots),
      flag_values->xla_dump_hlo_snapshots(),
      "Every time an HLO module is run, dumps an HloSnapshot to the directory "
      "specified by --xla_dump_to."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_hlo_module_re",
      string_setter_for(&DebugOptions::set_xla_dump_hlo_module_re),
      flag_values->xla_dump_hlo_module_re(),
      "Limits dumping only to modules which match this regular expression. "
      "Default is to dump all modules."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_hlo_pass_re",
      string_setter_for(&DebugOptions::set_xla_dump_hlo_pass_re),
      flag_values->xla_dump_hlo_pass_re(),
      "If specified, dumps HLO before and after optimization passes which "
      "match this regular expression, in addition to dumping at the very "
      "beginning and end of compilation."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_include_timestamp",
      bool_setter_for(&DebugOptions::set_xla_dump_include_timestamp),
      flag_values->xla_dump_include_timestamp(),
      "If specified, includes a timestamp in the dumped filenames."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_dump_max_hlo_modules",
      int32_setter_for(&DebugOptions::set_xla_dump_max_hlo_modules),
      flag_values->xla_dump_max_hlo_modules(),
      "Max number of hlo module dumps in a directory. Set to < 0 for "
      "unbounded."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_hlo_graph_addresses",
      bool_setter_for(&DebugOptions::set_xla_hlo_graph_addresses),
      flag_values->xla_hlo_graph_addresses(),
      "When rendering graphs (--xla_dump_hlo_as_{dot,html,url}), displays "
      "the address in memory of each HloInstruction object."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_hlo_graph_sharding_color",
      bool_setter_for(&DebugOptions::set_xla_hlo_graph_sharding_color),
      flag_values->xla_hlo_graph_sharding_color(),
      "Assign colors based on sharding assignments when generating the HLO "
      "graphs."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_allow_excess_precision",
      bool_setter_for(&DebugOptions::set_xla_allow_excess_precision),
      flag_values->xla_allow_excess_precision(),
      "Allow xla to increase the output precision of an instruction."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_force_conv_nchw",
      bool_setter_for(&DebugOptions::set_xla_gpu_force_conv_nchw),
      flag_values->xla_gpu_force_conv_nchw(),
      "For cuDNN convolutions, always NCHW layouts."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_algorithm_blacklist_path",
      string_setter_for(&DebugOptions::set_xla_gpu_algorithm_blacklist_path),
      flag_values->xla_gpu_algorithm_blacklist_path(),
      "An AlgorithmBlacklist text proto file as a blacklist of convolutions to "
      "avoid to use."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_deterministic_reductions",
      bool_setter_for(&DebugOptions::set_xla_gpu_deterministic_reductions),
      flag_values->xla_gpu_deterministic_reductions(),
      "Always run deterministic reductions on GPU"));
  flag_objects->push_back(tensorflow::Flag(
      "xla_tpu_detect_nan",
      bool_setter_for(&DebugOptions::set_xla_tpu_detect_nan),
      flag_values->xla_tpu_detect_nan(),
      "Trigger error on execution on TPU if a NAN value is detected"));
  flag_objects->push_back(tensorflow::Flag(
      "xla_tpu_detect_inf",
      bool_setter_for(&DebugOptions::set_xla_tpu_detect_inf),
      flag_values->xla_tpu_detect_inf(),
      "Trigger error on execution on TPU if a INF value is detected"));
  flag_objects->push_back(tensorflow::Flag(
      "xla_cpu_enable_xprof_traceme",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_xprof_traceme),
      flag_values->xla_cpu_enable_xprof_traceme(),
      "If true, XLA CPU generates code to call "
      "TraceMe::Activity{Start|End} around HLO operations."));
  flag_objects->push_back(tensorflow::Flag(
      "xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found),
      flag_values->xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(),
      "If true, XLA GPU falls back to the driver if ptxas is not found. Note "
      "that falling back to the driver can have drawbacks like using more "
      "memory and/or other bugs during compilation, so we recommend setting "
      "this flag to false."));
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", *flag_objects);
}

void AppendDebugOptionsFlags(std::vector<tensorflow::Flag>* flag_list) {
  absl::call_once(flags_init, &AllocateFlags);
  flag_list->insert(flag_list->end(), flag_objects->begin(),
                    flag_objects->end());
}

xla::DebugOptions GetDebugOptionsFromFlags() {
  absl::call_once(flags_init, &AllocateFlags);
  return *flag_values;
}

void ResetThreadLocalFuel() {
  absl::call_once(flags_init, &AllocateFlags);

  thread_fuel.reset(new absl::node_hash_map<string, std::atomic<int64>>());
  CHECK(initial_fuel != nullptr);
  for (const auto& kv : *initial_fuel) {
    thread_fuel->emplace(kv.first, kv.second);
  }
}

bool ConsumeFuel(absl::string_view pass, bool* just_ran_out) {
  absl::call_once(flags_init, &AllocateFlags);
  if (just_ran_out != nullptr) {
    *just_ran_out = false;
  }
  auto* fuel_pool = thread_fuel ? thread_fuel.get() : global_fuel;
  if (fuel_pool->empty()) {
    return true;
  }
  auto it = fuel_pool->find(pass);
  if (it == fuel_pool->end()) {
    return true;
  }
  std::atomic<int64>& remaining_fuel = it->second;
  std::atomic<bool>& fuel_has_been_consumed = fuel_ever_consumed->at(pass);
  fuel_has_been_consumed = true;

  int64 remaining = remaining_fuel.fetch_sub(1);
  if (just_ran_out != nullptr) {
    *just_ran_out = remaining == 0;
  }
  return remaining > 0;
}

}  // namespace xla
