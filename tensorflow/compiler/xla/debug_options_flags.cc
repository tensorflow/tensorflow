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

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/debug_options_parsers.h"
#include "tensorflow/compiler/xla/parse_flags_from_env.h"
#include "tensorflow/compiler/xla/xla.pb.h"

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
  opts.set_xla_gpu_use_runtime_fusion(true);
  opts.set_xla_eliminate_hlo_implicit_broadcast(true);
  opts.set_xla_dump_hlo_as_html(false);
  opts.set_xla_dump_fusion_visualization(false);
  opts.set_xla_dump_include_timestamp(false);
  opts.set_xla_dump_max_hlo_modules(-1);
  opts.set_xla_dump_module_metadata(false);
  opts.set_xla_dump_hlo_as_long_text(false);
#ifdef ENABLE_MKL
  opts.set_xla_cpu_use_mkl_dnn(true);
#endif  // ENABLE_MKL
#ifdef XLA_CPU_USE_ACL
  opts.set_xla_cpu_use_acl(true);
#endif
  opts.set_xla_cpu_use_xla_runtime(false);

  opts.set_xla_cpu_enable_fast_math(false);
  // Disable forms of fast math that have caused users problems in the past.
  opts.set_xla_cpu_fast_math_honor_nans(true);
  opts.set_xla_cpu_fast_math_honor_infs(true);
  opts.set_xla_cpu_fast_math_honor_functions(true);
  opts.set_xla_cpu_fast_math_honor_division(true);

  // By default, copy TF's Eigen style min_max behavior with nans.
  opts.set_xla_cpu_enable_fast_min_max(true);

  opts.set_xla_gpu_enable_cudnn_frontend(true);

  opts.set_xla_gpu_enable_cublaslt(false);

  // TODO(b/258036887): Remove this flag once CUDA Graphs are fully supported.
  opts.set_xla_gpu_enable_cuda_graphs(false);

  // Despite the name, fast min/max on GPUs does not seem to be any faster, and
  // adds very counter-intuitive "NaN-swallowing" behavior.
  opts.set_xla_gpu_enable_fast_min_max(false);
  opts.set_xla_gpu_strict_conv_algorithm_picker(true);

  opts.set_xla_allow_excess_precision(true);
  opts.set_xla_force_host_platform_device_count(1);
  opts.set_xla_gpu_all_reduce_combine_threshold_bytes(30 * 1024 * 1024);
  opts.set_xla_gpu_enable_async_all_reduce(true);
  opts.set_xla_cpu_enable_xprof_traceme(false);
  opts.set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(false);
  opts.set_xla_multiheap_size_constraint_per_heap(-1);
  opts.set_xla_detailed_logging_and_dumping(true);

  opts.set_xla_gpu_enable_xla_runtime_executable(true);
  opts.set_xla_gpu_nccl_termination_timeout_seconds(-1);
  opts.set_xla_gpu_enable_shared_constants(true);

  // Set 4GB space limit for redzone scratch allocator.
  opts.set_xla_gpu_redzone_scratch_max_megabytes(1LL << 12);
  opts.set_xla_gpu_shape_checks(DebugOptions::RUNTIME);
  opts.set_xla_cpu_enable_mlir_lowering(false);
  opts.set_xla_gpu_enable_mlir_lowering(true);
  opts.set_xla_gpu_enable_softmax_fusion(false);
  opts.set_xla_gpu_normalize_layouts(true);
  opts.set_xla_gpu_simplify_all_fp_conversions(true);
  return opts;
}

static absl::once_flag flags_init;
static DebugOptions* flag_values;
static std::vector<tsl::Flag>* flag_objects;

// Maps pass -> initial fuel values (parsed when AllocateFlags was run).
static absl::flat_hash_map<std::string, int64_t>* initial_fuel;

// Maps pass -> whether fuel was ever consumed for that pass.
static absl::node_hash_map<std::string, std::atomic<bool>>* fuel_ever_consumed;

// Maps pass -> remaining fuel.
//
// All threads start off using this global fuel pool, but ResetThreadLocalFuel()
// switches them to a thread-local fuel pool.
static absl::node_hash_map<std::string, std::atomic<int64_t>>* global_fuel;

// If we're using thread-local fuel, this stores it.
static thread_local std::unique_ptr<
    absl::node_hash_map<std::string, std::atomic<int64_t>>>
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

void MakeDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                           DebugOptions* debug_options) {
  // Returns a lambda that calls "member_setter" on "debug_options" with the
  // argument passed in to the lambda.
  auto bool_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(bool)) {
        return [debug_options, member_setter](bool value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  // Returns a lambda that calls "member_setter" on "debug_options" with the
  // argument passed in to the lambda.
  auto int32_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(int32_t)) {
        return [debug_options, member_setter](int32_t value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  auto int64_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(int64_t)) {
        return [debug_options, member_setter](int64_t value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  auto string_setter_for = [debug_options](void (DebugOptions::*member_setter)(
                               const std::string& value)) {
    return [debug_options, member_setter](const std::string& value) {
      (debug_options->*member_setter)(value);
      return true;
    };
  };

  // Custom "sub-parser" lambda for xla_gpu_shape_checks.
  auto setter_for_xla_gpu_shape_checks =
      [debug_options](const std::string& value) {
        DebugOptions::ShapeChecks shape_checks;
        if (!DebugOptions::ShapeChecks_Parse(value, &shape_checks)) {
          return false;
        }
        debug_options->set_xla_gpu_shape_checks(shape_checks);
        return true;
      };

  // Custom "sub-parser" lambda for xla_disable_hlo_passes.
  auto setter_for_xla_disable_hlo_passes =
      [debug_options](std::string comma_separated_values) {
        for (const auto& passname : std::vector<std::string>(
                 absl::StrSplit(comma_separated_values, ','))) {
          debug_options->add_xla_disable_hlo_passes(passname);
        }
        return true;
      };

  // Custom "sub-parser" lambda for xla_enable_hlo_passes_only.
  auto setter_for_xla_enable_hlo_passes_only =
      [debug_options](std::string comma_separated_values) {
        for (const auto& passname : std::vector<std::string>(
                 absl::StrSplit(comma_separated_values, ','))) {
          debug_options->add_xla_enable_hlo_passes_only(passname);
        }
        return true;
      };

  // Custom "sub-parser" lambda for xla_gpu_ptx_file.
  auto setter_for_xla_gpu_ptx_file = [debug_options](std::string value) {
    debug_options->add_xla_gpu_ptx_file(value);
    return true;
  };

  // Custom "sub-parser" lambda for xla_gpu_llvm_ir_file.
  auto setter_for_xla_gpu_llvm_ir_file =
      [debug_options](const std::string& value) {
        debug_options->add_xla_gpu_llvm_ir_file(value);
        return true;
      };

  // Custom "sub-parser" lambda for xla_backend_extra_options.
  auto setter_for_xla_backend_extra_options =
      [debug_options](std::string comma_separated_values) {
        auto* extra_options_map =
            debug_options->mutable_xla_backend_extra_options();
        parse_xla_backend_extra_options(extra_options_map,
                                        comma_separated_values);
        return true;
      };

  auto setter_for_xla_gpu_enable_softmax_fusion = [debug_options](bool value) {
    // It is only possible to enable softmax fusion if
    // xla_gpu_enable_mlir_lowering is also enabled.
    if (value && !debug_options->xla_gpu_enable_mlir_lowering()) {
      LOG(ERROR) << "xla_gpu_enable_softmax_fusion can only be enabled if "
                    "xla_gpu_enable_mlir_lowering is enabled as well";
      return false;
    }
    debug_options->set_xla_gpu_enable_softmax_fusion(value);
    return true;
  };

  auto setter_for_xla_gpu_enable_mlir_lowering = [debug_options](bool value) {
    // It is only possible to disable mlir lowering if
    // xla_gpu_enable_softmax_fusion is also disabled.
    if (!value && debug_options->xla_gpu_enable_softmax_fusion()) {
      LOG(ERROR) << "xla_gpu_enable_mlir_lowering can only be disabled if "
                    "xla_gpu_enable_softmax_fusion is disabled as well";
      return false;
    }
    debug_options->set_xla_gpu_enable_mlir_lowering(value);
    return true;
  };

  // Custom "sub-parser" for xla_fuel.  Note that ConsumeFuel does not do any
  // locking on the fuel global variables.  This means that it's
  // illegal/undefined behavior to modify this flag value while the compiler is
  // running.
  initial_fuel = new absl::flat_hash_map<std::string, int64_t>();
  fuel_ever_consumed =
      new absl::node_hash_map<std::string, std::atomic<bool>>();
  global_fuel = new absl::node_hash_map<std::string, std::atomic<int64_t>>();
  auto setter_for_xla_fuel = [](std::string xla_fuel_value) {
    initial_fuel->clear();
    global_fuel->clear();
    fuel_ever_consumed->clear();

    for (const auto& kv : absl::StrSplit(xla_fuel_value, ',')) {
      std::vector<std::string> pass_and_fuel = absl::StrSplit(kv, '=');
      if (pass_and_fuel.size() != 2) {
        LOG(ERROR) << absl::StreamFormat(
            "Illegal value for --xla_fuel. Saw %s, but expected token %s to "
            "have format X=INTEGER.",
            xla_fuel_value, kv);
        return false;
      }
      const auto& pass = pass_and_fuel[0];
      const auto& fuel_str = pass_and_fuel[1];
      int64_t fuel;
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

  // Don't use an initializer list for initializing the vector; this would
  // create a temporary copy, and exceeds the stack space when compiling with
  // certain configurations.
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_fast_math",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_fast_math),
      debug_options->xla_cpu_enable_fast_math(),
      "Enable unsafe fast-math optimizations in the CPU compiler; this may "
      "produce faster code at the expense of some accuracy."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_fast_math_honor_nans",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_nans),
      debug_options->xla_cpu_fast_math_honor_nans(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "allow operations to produce NaNs.  Ignored when "
      "xla_cpu_enable_fast_math is false."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_fast_math_honor_infs",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_infs),
      debug_options->xla_cpu_fast_math_honor_infs(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "allow operations to produce infinites.  Ignored when "
      "xla_cpu_enable_fast_math is false."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_fast_math_honor_division",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_division),
      debug_options->xla_cpu_fast_math_honor_division(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "forbid to use multiplication by the reciprocal instead of division. "
      "Ignored when xla_cpu_enable_fast_math is false."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_fast_math_honor_functions",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_functions),
      debug_options->xla_cpu_fast_math_honor_functions(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "forbid to approximate calculations for functions. Ignored when "
      "xla_cpu_enable_fast_math is false."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_fast_min_max",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_fast_min_max),
      debug_options->xla_cpu_enable_fast_min_max(),
      "Enable fast floating point min/max lowering that always propagates "
      "NaNs."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_fast_min_max",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_fast_min_max),
      debug_options->xla_gpu_enable_fast_min_max(),
      "Enable fast floating point min/max lowering that does not propagate "
      "NaNs."));
  flag_list->push_back(tsl::Flag(
      "xla_llvm_enable_alias_scope_metadata",
      bool_setter_for(&DebugOptions::set_xla_llvm_enable_alias_scope_metadata),
      debug_options->xla_llvm_enable_alias_scope_metadata(),
      "In LLVM-based backends, enable the emission of !alias.scope metadata in "
      "the generated IR."));
  flag_list->push_back(tsl::Flag(
      "xla_llvm_enable_noalias_metadata",
      bool_setter_for(&DebugOptions::set_xla_llvm_enable_noalias_metadata),
      debug_options->xla_llvm_enable_noalias_metadata(),
      "In LLVM-based backends, enable the emission of !noalias metadata in the "
      "generated IR."));
  flag_list->push_back(tsl::Flag(
      "xla_llvm_enable_invariant_load_metadata",
      bool_setter_for(
          &DebugOptions::set_xla_llvm_enable_invariant_load_metadata),
      debug_options->xla_llvm_enable_invariant_load_metadata(),
      "In LLVM-based backends, enable the emission of !invariant.load metadata "
      "in the generated IR."));
  flag_list->push_back(tsl::Flag(
      "xla_llvm_disable_expensive_passes",
      bool_setter_for(&DebugOptions::set_xla_llvm_disable_expensive_passes),
      debug_options->xla_llvm_disable_expensive_passes(),
      "In LLVM-based backends, disable a custom set of expensive optimization "
      "passes."));
  flag_list->push_back(tsl::Flag(
      "xla_backend_optimization_level",
      int32_setter_for(&DebugOptions::set_xla_backend_optimization_level),
      debug_options->xla_backend_optimization_level(),
      "Numerical optimization level for the XLA compiler backend."));
  flag_list->push_back(tsl::Flag(
      "xla_disable_hlo_passes", setter_for_xla_disable_hlo_passes, "",
      "Comma-separated list of hlo passes to be disabled. These names must "
      "exactly match the passes' names; no whitespace around commas."));
  flag_list->push_back(tsl::Flag(
      "xla_enable_hlo_passes_only", setter_for_xla_enable_hlo_passes_only, "",
      "Comma-separated list of hlo passes to be enabled. These names must "
      "exactly match the passes' names; no whitespace around commas. The "
      "unspecified passes are all disabled."));
  flag_list->push_back(tsl::Flag(
      "xla_disable_all_hlo_passes",
      bool_setter_for(&DebugOptions::set_xla_disable_all_hlo_passes), false,
      "Disables all HLO passes.  Notes that some passes are necessary for "
      "correctness and the invariants that must be satisfied by 'fully "
      "optimized' HLO are different for different devices and may change "
      "over time.  The only 'guarantee', such as it is, is that if you compile "
      "XLA and dump the optimized HLO for some graph, you should be able to "
      "run it again on the same device with the same build of XLA."));
  flag_list->push_back(
      tsl::Flag("xla_embed_ir_in_executable",
                bool_setter_for(&DebugOptions::set_xla_embed_ir_in_executable),
                debug_options->xla_embed_ir_in_executable(),
                "Embed the compiler IR as a string in the executable."));
  flag_list->push_back(tsl::Flag(
      "xla_eliminate_hlo_implicit_broadcast",
      bool_setter_for(&DebugOptions::set_xla_eliminate_hlo_implicit_broadcast),
      debug_options->xla_eliminate_hlo_implicit_broadcast(),
      "Eliminate implicit broadcasts when lowering user computations to HLO "
      "instructions; use explicit broadcast instead."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_multi_thread_eigen",
      bool_setter_for(&DebugOptions::set_xla_cpu_multi_thread_eigen),
      debug_options->xla_cpu_multi_thread_eigen(),
      "When generating calls to Eigen in the CPU backend, use multi-threaded "
      "Eigen mode."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_cuda_data_dir", debug_options->mutable_xla_gpu_cuda_data_dir(),
      "If non-empty, specifies a local directory containing ptxas and nvvm "
      "libdevice files; otherwise we use those from runfile directories."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_ftz", bool_setter_for(&DebugOptions::set_xla_gpu_ftz),
      debug_options->xla_gpu_ftz(),
      "If true, flush-to-zero semantics are enabled in the code generated for "
      "GPUs."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_ptx_file", setter_for_xla_gpu_ptx_file, "",
      "If non-empty, specifies a file containing ptx to use. The filename "
      "prefix must have the same pattern as PTX dumped by XLA. This allows to "
      "match one specific module. General workflow. Get the generated module "
      "ptx from XLA, modify it, then pass it back via this option."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_llvm_ir_file", setter_for_xla_gpu_llvm_ir_file, "",
      "If non-empty, specifies a file containing textual LLVM IR to use. The "
      "filename prefix must have the same pattern as LLVM dumped by XLA "
      "(i.e. module_0001.ir-no-opt.ll -> module_0001.MY_NEW_FILE.ll). This "
      "allows to match one specific module. General workflow. Get the not "
      "optimized LLVM IR from XLA, modify it, then pass it back via this "
      "option."));
  flag_list->push_back(tsl::Flag(
      "xla_test_all_output_layouts",
      bool_setter_for(&DebugOptions::set_xla_test_all_output_layouts),
      debug_options->xla_test_all_output_layouts(),
      "Let ClientLibraryTestBase::ComputeAndCompare* test all permutations of "
      "output layouts. For example, with a 3D shape, all permutations of the "
      "set {0, 1, 2} are tried."));
  flag_list->push_back(tsl::Flag(
      "xla_test_all_input_layouts",
      bool_setter_for(&DebugOptions::set_xla_test_all_input_layouts),
      debug_options->xla_test_all_input_layouts(),
      "Let ClientLibraryTestBase::ComputeAndCompare* test all permutations of "
      "*input* layouts. For example, for 2 input arguments with 2D shape and "
      "4D shape, the computation will run 2! * 4! times for every possible "
      "layouts"));
  flag_list->push_back(tsl::Flag(
      "xla_hlo_profile", bool_setter_for(&DebugOptions::set_xla_hlo_profile),
      debug_options->xla_hlo_profile(),
      "Instrument the computation to collect per-HLO cycle counts"));
  flag_list->push_back(tsl::Flag(
      "xla_backend_extra_options", setter_for_xla_backend_extra_options, "",
      "Extra options to pass to a backend; comma-separated list of 'key=val' "
      "strings (=val may be omitted); no whitespace around commas."));
  flag_list->push_back(
      tsl::Flag("xla_cpu_use_mkl_dnn",
                bool_setter_for(&DebugOptions::set_xla_cpu_use_mkl_dnn),
                debug_options->xla_cpu_use_mkl_dnn(),
                "Generate calls to MKL-DNN in the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_use_acl", bool_setter_for(&DebugOptions::set_xla_cpu_use_acl),
      debug_options->xla_cpu_use_acl(),
      "Generate calls to ACL (Arm Compute Library) in the CPU backend."));
  flag_list->push_back(
      tsl::Flag("xla_cpu_use_xla_runtime",
                bool_setter_for(&DebugOptions::set_xla_cpu_use_xla_runtime),
                debug_options->xla_cpu_use_xla_runtime(),
                "Enable XLA Runtime in the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_crash_on_verification_failures",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_crash_on_verification_failures),
      debug_options->xla_gpu_crash_on_verification_failures(),
      "Crashes the program on extra verification failures, e.g. cuDNN cross "
      "checking failures"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_strict_conv_algorithm_picker",
      bool_setter_for(&DebugOptions::set_xla_gpu_strict_conv_algorithm_picker),
      debug_options->xla_gpu_strict_conv_algorithm_picker(),
      "Upgrades warnings to failures when all algorithms fail conv "
      "autotuning."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_autotune_level",
      int32_setter_for(&DebugOptions::set_xla_gpu_autotune_level),
      debug_options->xla_gpu_autotune_level(),
      "Set GEMM and Convolution auto-tuning level. 0 = off; 1 = on; 2 = "
      "on+init; 3 = on+init+reinit; 4 = on+init+reinit+check."));
  flag_list->push_back(tsl::Flag(
      "xla_force_host_platform_device_count",
      int32_setter_for(&DebugOptions::set_xla_force_host_platform_device_count),
      debug_options->xla_force_host_platform_device_count(),
      "Force the host platform to pretend that there are these many host "
      "\"devices\". All of these host devices are backed by the same "
      "threadpool. Setting this to anything other than 1 can increase overhead "
      "from context switching but we let the user override this behavior to "
      "help run tests on the host that run models in parallel across multiple "
      "devices."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_disable_gpuasm_optimizations",
      bool_setter_for(&DebugOptions::set_xla_gpu_disable_gpuasm_optimizations),
      debug_options->xla_gpu_disable_gpuasm_optimizations(),
      "In XLA:GPU run ptxas in -O0 (default is -O3)."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_asm_extra_flags",
      string_setter_for(&DebugOptions::set_xla_gpu_asm_extra_flags), "",
      "Pass extra parameters to the GPU assembler tool (i.e., ptxas for CUDA). "
      "If multiple parameters, separate them by comma."));
  flag_list->push_back(tsl::Flag(
      "xla_fuel", setter_for_xla_fuel, /*default_value_for_display=*/"",
      "Sets compiler fuel, useful for bisecting bugs in passes.  Format "
      "--xla_fuel=PASS1=NUM1,PASS2=NUM2,..."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_to", string_setter_for(&DebugOptions::set_xla_dump_to),
      debug_options->xla_dump_to(),
      "Directory into which debugging data is written. If not specified but "
      "another dumping flag is passed, data will be written to stdout. To "
      "explicitly write to stdout, set this to \"-\". The values \"sponge\" "
      "and \"test_undeclared_outputs_dir\" have a special meaning: They cause "
      "us to dump into the directory specified by the environment variable "
      "TEST_UNDECLARED_OUTPUTS_DIR."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_hlo_as_text",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_text),
      debug_options->xla_dump_hlo_as_text(),
      "Dumps HLO modules as text before and after optimizations. debug_options "
      "are "
      "written to the --xla_dump_to dir, or, if no dir is specified, to "
      "stdout."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_hlo_as_long_text",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_long_text),
      debug_options->xla_dump_hlo_as_long_text(),
      "Dumps HLO modules as long text before and after optimizations. "
      "debug_options "
      "are written to the --xla_dump_to dir, or, if no dir is specified, to "
      "stdout. Ignored unless xla_dump_hlo_as_text is true."));
  flag_list->push_back(
      tsl::Flag("xla_dump_hlo_as_proto",
                bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_proto),
                debug_options->xla_dump_hlo_as_proto(),
                "Dumps HLO modules as HloProtos to the directory specified by "
                "--xla_dump_to."));
  flag_list->push_back(
      tsl::Flag("xla_dump_hlo_as_dot",
                bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_dot),
                debug_options->xla_dump_hlo_as_dot(),
                "Dumps HLO modules rendered as dot files to the "
                "directory specified by --xla_dump_to."));
  flag_list->push_back(
      tsl::Flag("xla_dump_hlo_as_html",
                bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_html),
                debug_options->xla_dump_hlo_as_html(),
                "Dumps HLO modules rendered as HTML files to the "
                "directory specified by --xla_dump_to."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_hlo_as_url",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_as_url),
      debug_options->xla_dump_hlo_as_url(),
      "Tries to dump HLO modules rendered as URLs to stdout (and also to the "
      "directory specified by --xla_dump_to). This is not implemented by "
      "default; you need to add a plugin which calls "
      "RegisterGraphToURLRenderer()."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_fusion_visualization",
      bool_setter_for(&DebugOptions::set_xla_dump_fusion_visualization),
      debug_options->xla_dump_fusion_visualization(),
      "Tries to generate HLO fusion visualization as an HTML page to the "
      "directory specified by --xla_dump_to). This is not implemented by "
      "default; you need to add a plugin which calls "
      "RegisterGraphToURLRenderer(). Generates a file per computation. "
      "Currently only implemented for the GPU backend."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_hlo_snapshots",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_snapshots),
      debug_options->xla_dump_hlo_snapshots(),
      "Every time an HLO module is run, dumps an HloSnapshot to the directory "
      "specified by --xla_dump_to."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_hlo_module_re",
      string_setter_for(&DebugOptions::set_xla_dump_hlo_module_re),
      debug_options->xla_dump_hlo_module_re(),
      "Limits dumping only to modules which match this regular expression. "
      "Default is to dump all modules."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_hlo_pass_re",
      string_setter_for(&DebugOptions::set_xla_dump_hlo_pass_re),
      debug_options->xla_dump_hlo_pass_re(),
      "If specified, dumps HLO before and after optimization passes which "
      "match this regular expression, in addition to dumping at the very "
      "beginning and end of compilation."));
  flag_list->push_back(
      tsl::Flag("xla_dump_include_timestamp",
                bool_setter_for(&DebugOptions::set_xla_dump_include_timestamp),
                debug_options->xla_dump_include_timestamp(),
                "If specified, includes a timestamp in the dumped filenames."));
  flag_list->push_back(
      tsl::Flag("xla_dump_max_hlo_modules",
                int32_setter_for(&DebugOptions::set_xla_dump_max_hlo_modules),
                debug_options->xla_dump_max_hlo_modules(),
                "Max number of hlo module dumps in a directory. Set to < 0 for "
                "unbounded."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_module_metadata",
      bool_setter_for(&DebugOptions::set_xla_dump_module_metadata),
      debug_options->xla_dump_module_metadata(),
      "Dumps HloModuleMetadata as text protos to the directory specified "
      "by --xla_dump_to."));
  flag_list->push_back(
      tsl::Flag("xla_dump_compress_protos",
                bool_setter_for(&DebugOptions::set_xla_dump_compress_protos),
                debug_options->xla_dump_compress_protos(),
                "Gzip-compress protos dumped by --xla_dump_hlo_as_proto."));
  flag_list->push_back(tsl::Flag(
      "xla_hlo_graph_addresses",
      bool_setter_for(&DebugOptions::set_xla_hlo_graph_addresses),
      debug_options->xla_hlo_graph_addresses(),
      "When rendering graphs (--xla_dump_hlo_as_{dot,html,url}), displays "
      "the address in memory of each HloInstruction object."));
  flag_list->push_back(tsl::Flag(
      "xla_hlo_graph_sharding_color",
      bool_setter_for(&DebugOptions::set_xla_hlo_graph_sharding_color),
      debug_options->xla_hlo_graph_sharding_color(),
      "Assign colors based on sharding assignments when generating the HLO "
      "graphs."));
  flag_list->push_back(tsl::Flag(
      "xla_allow_excess_precision",
      bool_setter_for(&DebugOptions::set_xla_allow_excess_precision),
      debug_options->xla_allow_excess_precision(),
      "Allow xla to increase the output precision of an instruction."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_force_conv_nchw",
                bool_setter_for(&DebugOptions::set_xla_gpu_force_conv_nchw),
                debug_options->xla_gpu_force_conv_nchw(),
                "For cuDNN convolutions, always use NCHW layouts."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_force_conv_nhwc",
                bool_setter_for(&DebugOptions::set_xla_gpu_force_conv_nhwc),
                debug_options->xla_gpu_force_conv_nhwc(),
                "For cuDNN convolutions, always use NHWC layouts."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_algorithm_denylist_path",
      string_setter_for(&DebugOptions::set_xla_gpu_algorithm_denylist_path),
      debug_options->xla_gpu_algorithm_denylist_path(),
      "An AlgorithmDenylist text proto file as a denylist of convolutions to "
      "avoid to use."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_use_runtime_fusion",
                bool_setter_for(&DebugOptions::set_xla_gpu_use_runtime_fusion),
                debug_options->xla_gpu_use_runtime_fusion(),
                "For using cuDNN runtime compiled fusion kernels."));
  flag_list->push_back(tsl::Flag(
      "xla_tpu_detect_nan",
      bool_setter_for(&DebugOptions::set_xla_tpu_detect_nan),
      debug_options->xla_tpu_detect_nan(),
      "Trigger error on execution on TPU if a NAN value is detected"));
  flag_list->push_back(tsl::Flag(
      "xla_tpu_detect_inf",
      bool_setter_for(&DebugOptions::set_xla_tpu_detect_inf),
      debug_options->xla_tpu_detect_inf(),
      "Trigger error on execution on TPU if a INF value is detected"));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_xprof_traceme",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_xprof_traceme),
      debug_options->xla_cpu_enable_xprof_traceme(),
      "If true, XLA CPU generates code to call "
      "TraceMe::Activity{Start|End} around HLO operations."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found),
      debug_options->xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(),
      "If true, XLA GPU falls back to the driver if ptxas is not found. Note "
      "that falling back to the driver can have drawbacks like using more "
      "memory and/or other bugs during compilation, so we recommend setting "
      "this flag to false."));
  flag_list->push_back(tsl::Flag(
      "xla_multiheap_size_constraint_per_heap",
      int32_setter_for(
          &DebugOptions::set_xla_multiheap_size_constraint_per_heap),
      debug_options->xla_multiheap_size_constraint_per_heap(),
      "Generates multiple heaps (i.e., temp buffers) with a size "
      "constraint on each heap to avoid Out-of-Memory due to memory "
      "fragmentation. The constraint is soft, so it works with tensors "
      "larger than the given constraint size. -1 corresponds to no "
      "constraints."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_force_compilation_parallelism",
      int32_setter_for(
          &DebugOptions::set_xla_gpu_force_compilation_parallelism),
      debug_options->xla_gpu_force_compilation_parallelism(),
      "Overrides normal multi-threaded compilation settting to use this many "
      "threads. Setting to 0 (the default value) means no enforcement."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_deterministic_ops",
                bool_setter_for(&DebugOptions::set_xla_gpu_deterministic_ops),
                debug_options->xla_gpu_deterministic_ops(),
                "Guarantees run-to-run determinism on GPU."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_async_all_reduce",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_async_all_reduce),
      debug_options->xla_gpu_enable_async_all_reduce(),
      "Converts synchronous all-reduce ops into asynchronous."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_all_reduce_combine_threshold_bytes",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_all_reduce_combine_threshold_bytes),
      debug_options->xla_gpu_all_reduce_combine_threshold_bytes(),
      "Size threshold (in bytes) for the GPU all-reduce combiner."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_all_reduce_contiguous",
      bool_setter_for(&DebugOptions::set_xla_gpu_all_reduce_contiguous),
      debug_options->xla_gpu_all_reduce_contiguous(),
      "Combine all-reduces into a single operation over a contiguous buffer."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_all_reduce_blueconnect_num_devices_per_host",
      int32_setter_for(
          &DebugOptions::
              set_xla_gpu_all_reduce_blueconnect_num_devices_per_host),
      debug_options->xla_gpu_all_reduce_blueconnect_num_devices_per_host(),
      "Number of devices per host for first stage of BlueConnect decomposition "
      "pass. The pass will attempt to decompose all-reduces ops into a "
      "ReduceScatter-AllReduce-AllGather sequence, with the initial "
      "ReduceScatter being performed over all of the devices in the same host. "
      "Set to < 1 to disable all-reduce decomposition."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_dump_llvmir",
                bool_setter_for(&DebugOptions::set_xla_gpu_dump_llvmir),
                debug_options->xla_gpu_dump_llvmir(), "Dump LLVM IR."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_cudnn_frontend",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_cudnn_frontend),
      debug_options->xla_gpu_enable_cudnn_frontend(),
      "Use the cuDNN frontend API for convolutions when possible."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_cublaslt",
                bool_setter_for(&DebugOptions::set_xla_gpu_enable_cublaslt),
                debug_options->xla_gpu_enable_cublaslt(),
                "Use cuBLASLt for GEMMs when possible."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_cuda_graphs",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_cuda_graphs),
      debug_options->xla_gpu_enable_cuda_graphs(),
      "Use CUDA graphs to execute XLA GPU executables when possible."));
  flag_list->push_back(
      tsl::Flag("xla_dump_disable_metadata",
                bool_setter_for(&DebugOptions::set_xla_dump_disable_metadata),
                debug_options->xla_dump_disable_metadata(),
                "Disable dumping HLO metadata in HLO dumps."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_hlo_pipeline_re",
      string_setter_for(&DebugOptions::set_xla_dump_hlo_pipeline_re),
      debug_options->xla_dump_hlo_pipeline_re(),
      "If specified, dumps HLO before and after optimization passes in the "
      "pass pipelines that match this regular expression."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_xla_runtime_executable",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_xla_runtime_executable),
      debug_options->xla_gpu_enable_xla_runtime_executable(),
      "Whether to enable XLA runtime for XLA:GPU backend"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_nccl_termination_timeout_seconds",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_nccl_termination_timeout_seconds),
      debug_options->xla_gpu_nccl_termination_timeout_seconds(),
      "Timeout in seconds before terminating jobs stuck in NCCL Rendezvous."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_shared_constants",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_shared_constants),
      debug_options->xla_gpu_enable_shared_constants(),
      "Enable constant sharing between GPU executables"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_redzone_scratch_max_megabytes",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_redzone_scratch_max_megabytes),
      debug_options->xla_gpu_redzone_scratch_max_megabytes(),
      "Max size (in megabytes) for the GPU redzone scratch allocator."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_simplify_all_fp_conversions",
      bool_setter_for(&DebugOptions::set_xla_gpu_simplify_all_fp_conversions),
      debug_options->xla_gpu_simplify_all_fp_conversions(),
      "Allows any chain of floating-point conversions to be simplified."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_shape_checks", setter_for_xla_gpu_shape_checks,
      DebugOptions::ShapeChecks_Name(debug_options->xla_gpu_shape_checks()),
      "When to perform shape checks in XLA:GPU."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_mlir_lowering",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_mlir_lowering),
      debug_options->xla_cpu_enable_mlir_lowering(),
      "Enable MLIR-based lowering in XLA:CPU instead of LLVM emitters."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_mlir_lowering", setter_for_xla_gpu_enable_mlir_lowering,
      debug_options->xla_gpu_enable_mlir_lowering(),
      "Enable MLIR-based lowering in XLA:GPU instead of LLVM emitters."));
  flag_list->push_back(tsl::Flag("xla_gpu_enable_softmax_fusion",
                                 setter_for_xla_gpu_enable_softmax_fusion,
                                 debug_options->xla_gpu_enable_softmax_fusion(),
                                 "Enable MLIR-based softmax fusion."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_normalize_layouts",
                bool_setter_for(&DebugOptions::set_xla_gpu_normalize_layouts),
                debug_options->xla_gpu_normalize_layouts(),
                "An experimental option to force all layouts present in the "
                "after-optimizations HLO to be descending"));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_strict_dot_conv_math",
      bool_setter_for(&DebugOptions::set_xla_cpu_strict_dot_conv_math),
      debug_options->xla_cpu_strict_dot_conv_math(),
      "By default, XLA:CPU will run fp16 dot/conv as fp32, as this is "
      "generally (much) faster on our hardware.  Set this flag to true to "
      "disable this behavior."));
}  // NOLINT(readability/fn_size)

// Allocates flag_values and flag_objects; this function must not be called more
// than once - its call done via call_once.
static void AllocateFlags(DebugOptions* defaults) {
  if (defaults == nullptr) {
    defaults = new DebugOptions(DefaultDebugOptionsIgnoringFlags());
  }
  flag_values = defaults;
  flag_objects = new std::vector<tsl::Flag>();
  MakeDebugOptionsFlags(flag_objects, flag_values);
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", *flag_objects);
}

void AppendDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                             DebugOptions* debug_options) {
  absl::call_once(flags_init, &AllocateFlags, debug_options);
  flag_list->insert(flag_list->end(), flag_objects->begin(),
                    flag_objects->end());
}

xla::DebugOptions GetDebugOptionsFromFlags() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  return *flag_values;
}

void ResetThreadLocalFuel() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);

  thread_fuel = std::make_unique<
      absl::node_hash_map<std::string, std::atomic<int64_t>>>();
  CHECK(initial_fuel != nullptr);
  for (const auto& kv : *initial_fuel) {
    thread_fuel->emplace(kv.first, kv.second);
  }
}

bool ConsumeFuel(absl::string_view pass, bool* just_ran_out) {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
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
  std::atomic<int64_t>& remaining_fuel = it->second;
  std::atomic<bool>& fuel_has_been_consumed = fuel_ever_consumed->at(pass);
  fuel_has_been_consumed = true;

  int64_t remaining = remaining_fuel.fetch_sub(1);
  if (just_ran_out != nullptr) {
    *just_ran_out = remaining == 0;
  }
  return remaining > 0;
}

}  // namespace xla
