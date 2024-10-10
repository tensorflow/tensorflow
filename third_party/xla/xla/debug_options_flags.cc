/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/debug_options_flags.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_parsers.h"
#include "xla/parse_flags_from_env.h"
#include "xla/service/collective_utils.h"
#include "xla/stream_executor/cuda/nvjitlink_support.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace xla {

DebugOptions DefaultDebugOptionsIgnoringFlags() {
  DebugOptions opts;
  opts.set_xla_llvm_enable_alias_scope_metadata(true);
  opts.set_xla_llvm_enable_noalias_metadata(true);
  opts.set_xla_llvm_enable_invariant_load_metadata(true);
  opts.set_xla_llvm_disable_expensive_passes(false);
  opts.set_xla_backend_optimization_level(3);
  opts.set_xla_gpu_autotune_level(4);
  opts.set_xla_gpu_autotune_max_solutions(0);
  opts.set_xla_cpu_multi_thread_eigen(true);
  opts.set_xla_gpu_cuda_data_dir("./cuda_sdk_lib");
  opts.set_xla_gpu_asm_extra_flags("");

  // As of cudnn 8.9.0, runtime fusion creates convolutions that take about 7s
  // seconds to run the first time we call them, at least on Ampere.  In
  // contrast, non-runtime-fusion convs usually run in about 50ms.  Thus runtime
  // fusion can cause a 100x slowdown when compiling models that have convs that
  // use runtime fusion.  We therefore can't enable this by default today.
  // Additional details in b/237009940#comment46.
  opts.set_xla_gpu_use_runtime_fusion(false);

  opts.set_xla_eliminate_hlo_implicit_broadcast(true);
  opts.set_xla_dump_hlo_as_html(false);
  opts.set_xla_dump_fusion_visualization(false);
  opts.set_xla_dump_include_timestamp(false);
  opts.set_xla_dump_max_hlo_modules(-1);
  opts.set_xla_dump_module_metadata(false);
  opts.set_xla_dump_hlo_as_long_text(false);
  opts.set_xla_dump_large_constants(false);
  opts.set_xla_dump_enable_mlir_pretty_form(true);
  opts.set_xla_debug_buffer_assignment_show_max(15);
#ifdef ENABLE_MKL
  opts.set_xla_cpu_use_mkl_dnn(true);
#endif  // ENABLE_MKL
#ifdef XLA_CPU_USE_ACL
  opts.set_xla_cpu_use_acl(true);
#endif
  opts.set_xla_cpu_use_thunk_runtime(true);
  opts.set_xla_cpu_parallel_codegen_split_count(32);
  opts.set_xla_cpu_enable_concurrency_optimized_scheduler(false);
  opts.set_xla_cpu_prefer_vector_width(256);
  opts.set_xla_cpu_max_isa("");

  opts.set_xla_cpu_enable_fast_math(false);
  // Disable forms of fast math that have caused users problems in the past.
  opts.set_xla_cpu_fast_math_honor_nans(true);
  opts.set_xla_cpu_fast_math_honor_infs(true);
  opts.set_xla_cpu_fast_math_honor_functions(true);
  opts.set_xla_cpu_fast_math_honor_division(true);

  // TODO(AyanmoI): Remove this flag when cuDNN FMHA is fully supported.
  //
  // cuDNN FMHA currently rewrites attention layers to use FlashAttention by
  // default. This reassociation is not semantics-preserving, and the user
  // should explicitly opt in if they wish to use this feature. cuDNN FMHA can
  // not be turned on by default.
  opts.set_xla_gpu_enable_cudnn_fmha(false);

  opts.set_xla_gpu_fused_attention_use_cudnn_rng(false);

  // By default, copy TF's Eigen style min_max behavior with nans.
  opts.set_xla_cpu_enable_fast_min_max(true);

  opts.set_xla_gpu_enable_cudnn_frontend(true);

  opts.set_xla_gpu_enable_cublaslt(false);

  opts.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  opts.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  opts.add_xla_gpu_enable_command_buffer(DebugOptions::CUSTOM_CALL);
  opts.add_xla_gpu_enable_command_buffer(DebugOptions::CUDNN);
  opts.set_xla_gpu_graph_min_graph_size(5);
  opts.set_xla_gpu_graph_enable_concurrent_region(false);
  opts.set_xla_cmd_buffer_trace_cache_size(16);

  // Despite the name, fast min/max on GPUs does not seem to be any faster, and
  // adds very counter-intuitive "NaN-swallowing" behavior.
  opts.set_xla_gpu_enable_fast_min_max(false);
  opts.set_xla_gpu_strict_conv_algorithm_picker(true);

  opts.set_xla_allow_excess_precision(true);
  opts.set_xla_force_host_platform_device_count(1);
  opts.set_xla_gpu_all_reduce_combine_threshold_bytes(
      kDefaultAllReduceCombineThreshold);
  opts.set_xla_gpu_all_gather_combine_threshold_bytes(
      kDefaultAllGatherCombineThreshold);
  opts.set_xla_gpu_reduce_scatter_combine_threshold_bytes(
      kDefaultReduceScatterCombineThreshold);
  opts.set_xla_gpu_enable_all_gather_combine_by_dim(false);
  opts.set_xla_gpu_enable_reduce_scatter_combine_by_dim(true);
  opts.set_xla_gpu_enable_approx_costly_collectives(false);

  opts.set_xla_gpu_enable_reassociation_for_converted_ar(true);

  opts.set_xla_cpu_enable_xprof_traceme(false);
  opts.set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(false);
  opts.set_xla_multiheap_size_constraint_per_heap(-1);
  opts.set_xla_detailed_logging(true);
  opts.set_xla_enable_dumping(true);

  opts.set_xla_gpu_enable_custom_fusions(false);
  opts.set_xla_gpu_enable_dynamic_slice_fusion(false);
  opts.set_xla_gpu_nccl_termination_timeout_seconds(-1);
  opts.set_xla_gpu_enable_shared_constants(true);
  opts.set_xla_gpu_enable_nccl_user_buffers(false);
  opts.set_xla_gpu_enable_nccl_comm_splitting(true);
  opts.set_xla_gpu_enable_nccl_per_stream_comms(false);

  opts.set_xla_gpu_temp_buffer_use_separate_color(false);

  // Set 4GB space limit for redzone scratch allocator.
  opts.set_xla_gpu_redzone_scratch_max_megabytes(1LL << 12);
  opts.set_xla_gpu_redzone_padding_bytes(8 * 1024 * 1024);
  opts.set_xla_gpu_shape_checks(DebugOptions::RUNTIME);
  opts.set_xla_dump_latency_hiding_schedule(false);
  opts.set_xla_gpu_enable_latency_hiding_scheduler(false);
  opts.set_xla_gpu_lhs_enable_gpu_async_tracker(true);
  opts.set_xla_gpu_enable_analytical_latency_estimator(false);
  opts.set_xla_gpu_pgle_profile_file_or_directory_path("");
  opts.set_xla_gpu_memory_limit_slop_factor(95);
  opts.set_xla_gpu_enable_highest_priority_async_stream(true);

  opts.set_xla_gpu_enable_pipelined_collectives(false);
  opts.set_xla_gpu_enable_pipelined_all_reduce(false);
  opts.set_xla_gpu_enable_pipelined_all_gather(true);
  opts.set_xla_gpu_enable_pipelined_reduce_scatter(true);
  opts.set_xla_gpu_enable_pipelined_p2p(false);

  opts.set_xla_gpu_run_post_layout_collective_pipeliner(false);

  opts.set_xla_gpu_collective_permute_decomposer_threshold(
      std::numeric_limits<int64_t>::max());

  opts.set_xla_cpu_enable_mlir_tiling_and_fusion(true);
  opts.set_xla_cpu_enable_custom_matmul_tiling(false);
  opts.set_xla_cpu_matmul_tiling_m_dim(8);
  opts.set_xla_cpu_matmul_tiling_n_dim(8);
  opts.set_xla_cpu_matmul_tiling_k_dim(8);
  opts.set_xla_cpu_enable_mlir_fusion_outlining(true);
  opts.set_xla_cpu_enable_experimental_deallocation(true);

  opts.set_xla_partitioning_algorithm(
      DebugOptions::PARTITIONING_ALGORITHM_NOOP);

  opts.set_xla_gpu_enable_triton_gemm(true);
  opts.set_xla_gpu_enable_cudnn_int8x32_convolution_reordering(true);
  opts.set_xla_gpu_triton_gemm_any(false);
  opts.set_xla_gpu_triton_fusion_level(2);
  opts.set_xla_gpu_verify_triton_fusion_numerics(false);

  // Moving reduce-scatter out of while loops can increase memory footprint, so
  // turning it off by default.
  opts.set_xla_gpu_enable_while_loop_reduce_scatter_code_motion(false);

  opts.set_xla_gpu_collective_inflation_factor(1);
  opts.set_xla_llvm_force_inline_before_split(true);

  opts.set_xla_gpu_exhaustive_tiling_search(false);

  opts.set_xla_gpu_enable_priority_fusion(true);
  opts.set_xla_gpu_experimental_enable_triton_softmax_priority_fusion(false);

  opts.set_xla_gpu_auto_spmd_partitioning_memory_budget_gb(0);
  opts.set_xla_gpu_auto_spmd_partitioning_memory_budget_ratio(1.1);
  opts.set_xla_gpu_unsafe_pipelined_loop_annotator(false);

  opts.set_xla_gpu_copy_insertion_use_region_analysis(false);
  opts.set_xla_gpu_collect_cost_model_stats(false);
  opts.set_xla_gpu_enable_split_k_autotuning(true);

  opts.set_xla_gpu_enable_reduction_epilogue_fusion(true);
  opts.set_xla_gpu_enable_nccl_clique_optimization(false);
  opts.set_xla_gpu_cublas_fallback(true);
  opts.set_xla_gpu_cudnn_gemm_fusion_level(0);
  opts.set_xla_gpu_enable_while_loop_double_buffering(false);
  opts.set_xla_gpu_enable_while_loop_unrolling(
      DebugOptions::WHILE_LOOP_UNROLLING_AUTO_UNROLL);
  opts.set_xla_gpu_ensure_minor_dot_contraction_dims(false);
  opts.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(true);
  opts.set_xla_gpu_llvm_verification_level(0);
  opts.set_xla_gpu_target_config_filename("");
  opts.set_xla_gpu_enable_cub_radix_sort(true);
  opts.set_xla_gpu_enable_cudnn_layer_norm(false);
  opts.set_xla_gpu_threshold_for_windowed_einsum_mib(100000);

  opts.set_xla_gpu_enable_triton_hopper(false);
  opts.set_xla_gpu_experimental_enable_fusion_block_level_rewriter(false);

  opts.set_xla_gpu_enable_llvm_module_compilation_parallelism(false);
  opts.set_xla_gpu_enable_libnvptxcompiler(
      stream_executor::IsLibNvPtxCompilerSupported());
  opts.set_xla_gpu_enable_libnvjitlink(
      stream_executor::IsLibNvJitLinkSupported());

  opts.set_xla_gpu_enable_dot_strength_reduction(true);

  opts.set_xla_gpu_enable_bf16_6way_gemm(false);
  opts.set_xla_gpu_enable_bf16_3way_gemm(false);
  opts.set_xla_gpu_nccl_collective_max_nchannels(0);
  opts.set_xla_gpu_nccl_p2p_max_nchannels(0);

#if GOOGLE_CUDA
  opts.set_xla_gpu_mlir_emitter_level(4);
#else
  opts.set_xla_gpu_mlir_emitter_level(0);
#endif

  opts.set_xla_gpu_multi_streamed_windowed_einsum(false);

  // Minimum combined size of matrices in matrix multiplication to
  // be rewritten to cuBLAS or Triton kernel call.
  // This threshold is a conservative estimate and has been measured
  // to be always beneficial (up to generally several times faster)
  // on V100 and H100 GPUs. See openxla/xla #9319 for details.
  const int64_t kDefaultMinGemmRewriteSize = 100;
  opts.set_xla_gpu_gemm_rewrite_size_threshold(kDefaultMinGemmRewriteSize);

  opts.set_xla_gpu_use_memcpy_local_p2p(false);

  opts.set_xla_reduce_window_rewrite_base_length(16);

  opts.set_xla_gpu_require_complete_aot_autotune_results(false);

  opts.set_xla_gpu_enable_host_memory_offloading(false);

  opts.set_xla_gpu_nccl_terminate_on_error(false);

  opts.set_xla_gpu_shard_autotuning(false);

  opts.set_xla_syntax_sugar_async_ops(false);

  opts.set_xla_gpu_per_fusion_autotune_cache_dir("");

  opts.set_xla_gpu_experimental_autotune_cache_mode(
      DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE);

  opts.set_xla_gpu_autotune_gemm_rtol(0.1f);

  opts.set_xla_enable_command_buffers_during_profiling(false);

  opts.set_xla_gpu_cudnn_gemm_max_plans(5);

  opts.set_xla_gpu_enable_pgle_accuracy_checker(false);

  opts.set_xla_gpu_executable_warn_stuck_timeout_seconds(10);
  opts.set_xla_gpu_executable_terminate_timeout_seconds(30);
  opts.set_xla_gpu_experimental_disable_binary_libraries(false);
  opts.set_xla_experimental_ignore_channel_id(false);
  opts.set_xla_gpu_dot_merger_threshold_mb(32);
  opts.set_xla_enable_fast_math(false);
  opts.set_xla_gpu_experimental_parallel_collective_overlap_limit(1);
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

// A util that does nothing. Used as a no-op flag setter in order to
// soft-deprecate flags.
template <typename T>
static bool noop_flag_setter(T value) {
  return true;
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

  auto uppercase_string_setter_for =
      [debug_options](
          void (DebugOptions::*member_setter)(const std::string& value)) {
        return [debug_options, member_setter](const std::string& value) {
          (debug_options->*member_setter)(absl::AsciiStrToUpper(value));
          return true;
        };
      };

  auto float_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(float)) {
        return [debug_options, member_setter](float value) {
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

  // Custom "sub-parser" lambda for legacy_command_buffer_custom_call_targets.
  auto setter_for_legacy_command_buffer_custom_call_targets =
      [debug_options](std::string comma_separated_values) {
        for (const auto& target : std::vector<std::string>(
                 absl::StrSplit(comma_separated_values, ','))) {
          debug_options->add_legacy_command_buffer_custom_call_targets(target);
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

  // Custom "sub-parser" lambda for xla_partitioning_algorithm.
  auto setter_for_xla_partitioning_algorithm =
      [debug_options](const std::string& value) {
        DebugOptions::PartitioningAlgorithm partitioning_algorithm;
        if (!DebugOptions::PartitioningAlgorithm_Parse(
                value, &partitioning_algorithm)) {
          return false;
        }
        debug_options->set_xla_partitioning_algorithm(partitioning_algorithm);
        return true;
      };

  // Custom "sub-parser" lambda for xla_gpu_graph_level.
  auto setter_for_xla_gpu_graph_level = [debug_options](const int32_t level) {
    debug_options->clear_xla_gpu_enable_command_buffer();
    if (level >= 1) {
      debug_options->add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
    }
    if (level >= 2) {
      debug_options->add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
    }
    if (level >= 3) {
      debug_options->add_xla_gpu_enable_command_buffer(DebugOptions::CUDNN);
    }
    return true;
  };

  auto command_types_to_string =
      [](tsl::protobuf::RepeatedField<int> command_types) -> std::string {
    struct Formatter {
      void operator()(std::string* out, int type) const {
        absl::StrAppend(out, DebugOptions::CommandBufferCmdType_Name(type));
      }
    };
    return absl::StrJoin(command_types, ", ", Formatter());
  };

  // Custom "sub-parser" lambda for xla_gpu_enable_command_buffer.
  auto setter_for_xla_gpu_enable_command_buffer =
      [debug_options](const std::string& input) {
        auto is_command_type = [](absl::string_view value) {
          DebugOptions::CommandBufferCmdType cmd_type;
          return DebugOptions::CommandBufferCmdType_Parse(
              absl::AsciiStrToUpper(value), &cmd_type);
        };

        auto is_add_or_remove_command_type = [&](absl::string_view value) {
          if (absl::StartsWith(value, "+") || absl::StartsWith(value, "-")) {
            return (is_command_type(value.substr(1)));
          }
          return false;
        };

        auto parse_command_type = [](absl::string_view value) {
          DebugOptions::CommandBufferCmdType cmd_type;
          DebugOptions::CommandBufferCmdType_Parse(absl::AsciiStrToUpper(value),
                                                   &cmd_type);
          return cmd_type;
        };

        auto erase_command_type = [](tsl::protobuf::RepeatedField<int>* enabled,
                                     DebugOptions::CommandBufferCmdType type) {
          auto it = enabled->begin();
          while (it != enabled->end()) {
            if (*it == type) {
              it = enabled->erase(it);
            } else {
              it++;
            }
          }
        };

        // Disable command buffers by clearing a set of supported commands.
        if (input.empty()) {
          debug_options->clear_xla_gpu_enable_command_buffer();
          return true;
        }

        std::vector<absl::string_view> values = absl::StrSplit(input, ',');

        // Overwrite a set of supported commands with a flag.
        if (absl::c_all_of(values, is_command_type)) {
          debug_options->clear_xla_gpu_enable_command_buffer();
          for (const absl::string_view value : values) {
            debug_options->add_xla_gpu_enable_command_buffer(
                parse_command_type(value));
          }
          return true;
        }

        // Add or remove a commands from a default set.
        if (absl::c_all_of(values, is_add_or_remove_command_type)) {
          for (const absl::string_view value : values) {
            DebugOptions::CommandBufferCmdType cmd_type =
                parse_command_type(value.substr(1));
            if (absl::StartsWith(value, "+")) {
              debug_options->add_xla_gpu_enable_command_buffer(cmd_type);
            } else if (absl::StartsWith(value, "-")) {
              tsl::protobuf::RepeatedField<int>* enabled =
                  debug_options->mutable_xla_gpu_enable_command_buffer();
              erase_command_type(enabled, cmd_type);
            }
            return true;
          }
        }

        // Return an error if flag value was not recognized as one of the
        // supported modes.
        return false;
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

  auto collective_op_types_to_string =
      [](tsl::protobuf::RepeatedField<int> collective_ops) -> std::string {
    struct Formatter {
      void operator()(std::string* out, int type) const {
        absl::StrAppend(out, DebugOptions::CollectiveOpType_Name(type));
      }
    };
    return absl::StrJoin(collective_ops, ", ", Formatter());
  };

  // Custom parser for `xla_gpu_enable_while_loop_unrolling` flag.
  auto setter_for_xla_gpu_enable_while_loop_unrolling =
      [&debug_options](absl::string_view input) {
        DebugOptions::WhileLoopUnrolling unroll_strategy;
        bool parsed = DebugOptions::WhileLoopUnrolling_Parse(
            absl::AsciiStrToUpper(input), &unroll_strategy);
        if (!parsed) return false;
        debug_options->set_xla_gpu_enable_while_loop_unrolling(unroll_strategy);
        return true;
      };

  // Custom parser for xla_gpu_disable_async_collectives.
  auto setter_for_xla_gpu_disable_async_collectives =
      [debug_options](const absl::string_view& input) {
        auto is_collective_type = [](absl::string_view value) {
          DebugOptions::CollectiveOpType op_type;
          return DebugOptions::CollectiveOpType_Parse(
              absl::AsciiStrToUpper(value), &op_type);
        };

        auto parse_collective_type = [](absl::string_view value) {
          DebugOptions::CollectiveOpType op_type;
          DebugOptions::CollectiveOpType_Parse(absl::AsciiStrToUpper(value),
                                               &op_type);
          return op_type;
        };

        std::vector<absl::string_view> values = absl::StrSplit(input, ',');

        // Overwrite a set of supported commands with a flag.
        if (absl::c_all_of(values, is_collective_type)) {
          debug_options->clear_xla_gpu_disable_async_collectives();
          for (const absl::string_view value : values) {
            debug_options->add_xla_gpu_disable_async_collectives(
                parse_collective_type(value));
          }
          return true;
        }

        // Return an error if flag value was not recognized as one of the
        // supported modes.
        return false;
      };

  // Custom "sub-parser" for xla_gpu_experimental_autotune_cache_mode.
  auto setter_for_xla_gpu_experimental_autotune_cache_mode =
      [debug_options](const std::string& value) {
        DebugOptions::AutotuneCacheMode autotune_cache_mode;
        if (!DebugOptions::AutotuneCacheMode_Parse(value,
                                                   &autotune_cache_mode)) {
          return false;
        }
        debug_options->set_xla_gpu_experimental_autotune_cache_mode(
            autotune_cache_mode);
        return true;
      };

  // Custom "sub-parser" lambda for xla_step_marker_location.
  auto setter_for_xla_step_marker_location = [debug_options](
                                                 const std::string& value) {
    DebugOptions::StepMarkerLocation step_marker_location;
    if (!DebugOptions::StepMarkerLocation_Parse(value, &step_marker_location)) {
      return false;
    }
    debug_options->set_xla_step_marker_location(step_marker_location);
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
      "allow operations to produce NaNs. Ignored when "
      "xla_cpu_enable_fast_math is false."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_fast_math_honor_infs",
      bool_setter_for(&DebugOptions::set_xla_cpu_fast_math_honor_infs),
      debug_options->xla_cpu_fast_math_honor_infs(),
      "When xla_cpu_enable_fast_math is true then this controls whether we "
      "allow operations to produce infinites. Ignored when "
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
      "Disables all HLO passes. Notes that some passes are necessary for "
      "correctness and the invariants that must be satisfied by 'fully "
      "optimized' HLO are different for different devices and may change "
      "over time. The only 'guarantee', such as it is, is that if you compile "
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
      tsl::Flag("xla_cpu_use_thunk_runtime",
                bool_setter_for(&DebugOptions::set_xla_cpu_use_thunk_runtime),
                debug_options->xla_cpu_use_thunk_runtime(),
                "Use Thunk-based runtime for the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_parallel_codegen_split_count",
      int32_setter_for(&DebugOptions::set_xla_cpu_parallel_codegen_split_count),
      debug_options->xla_cpu_parallel_codegen_split_count(),
      "Split LLVM module into at most this many parts before codegen to enable "
      "parallel compilation for the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_concurrency_optimized_scheduler",
      bool_setter_for(
          &DebugOptions::set_xla_cpu_enable_concurrency_optimized_scheduler),
      debug_options->xla_cpu_enable_concurrency_optimized_scheduler(),
      "Use HLO module scheduler that is optimized for extracting concurrency "
      "from an HLO module by trading off extra memory pressure."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_prefer_vector_width",
      int32_setter_for(&DebugOptions::set_xla_cpu_prefer_vector_width),
      debug_options->xla_cpu_prefer_vector_width(),
      "Preferred vector with for the XLA:CPU LLVM backend."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_max_isa",
      uppercase_string_setter_for(&DebugOptions::set_xla_cpu_max_isa),
      debug_options->xla_cpu_max_isa(),
      "Maximum ISA that XLA:CPU LLVM backend will codegen, i.e., it will not "
      "use newer instructions. Available values: SSE4_2, AVX, AVX2, AVX512, "
      "AVX512_VNNI, AVX512_BF16, AMX, and AMX_FP16. (`AMX` will enable both "
      "`AMX_BF16` and `AMX_INT8` instructions.)"));
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
      "on+init; 3 = on+init+reinit; 4 = on+init+reinit+check; "
      "5 = on+init+reinit+check and skip WRONG_RESULT solutions. See also "
      "the related flag xla_gpu_autotune_gemm_rtol. Remark that, setting the "
      "level to 5 only makes sense if you are sure that the reference (first "
      "in the list) solution is numerically CORRECT. Otherwise, the autotuner "
      "might discard many other correct solutions based on the failed "
      "BufferComparator test."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_autotune_max_solutions",
      int64_setter_for(&DebugOptions::set_xla_gpu_autotune_max_solutions),
      debug_options->xla_gpu_autotune_max_solutions(),
      "Maximal number of GEMM solutions to consider for autotuning: 0 means "
      "consider all solutions returned by the GEMM library."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_autotune_gemm_rtol",
      float_setter_for(&DebugOptions::set_xla_gpu_autotune_gemm_rtol),
      debug_options->xla_gpu_autotune_gemm_rtol(),
      "Relative precision for comparing GEMM solutions vs the reference one"));
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
      "Sets compiler fuel, useful for bisecting bugs in passes. Format "
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
      tsl::Flag("xla_dump_large_constants",
                bool_setter_for(&DebugOptions::set_xla_dump_large_constants),
                debug_options->xla_dump_large_constants(),
                "Dumps HLO modules including large constants before and after "
                "optimizations. debug_options are written to the --xla_dump_to "
                "dir, or, if no dir is specified, to stdout. Ignored unless "
                "xla_dump_hlo_as_text is true."));
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
      "xla_gpu_unsafe_pipelined_loop_annotator",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_unsafe_pipelined_loop_annotator),
      debug_options->xla_gpu_unsafe_pipelined_loop_annotator(),
      "If this option is true, then the while loop with rotate right "
      "pattern will be considered a pipelined while loop and the "
      "operations within the pipeline bubbles may be considered no-ops. "
      "Specifically, collective-permute may become a no-op for the iterations "
      "within pipeline bubble. This is an unsafe flag."));
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
      "Overrides normal multi-threaded compilation setting to use this many "
      "threads. Setting to 0 (the default value) means no enforcement."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_llvm_module_compilation_parallelism",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_enable_llvm_module_compilation_parallelism),
      debug_options->xla_gpu_enable_llvm_module_compilation_parallelism(),
      "Decides whether we can do LLVM module compilation in a parallelised "
      "way. If set to false, then it will be single threaded, otherwise the "
      "number of threads depends on the "
      "--xla_gpu_force_compilation_parallelism flag and the thread pool "
      "supplied to GpuCompiler."));

  flag_list->push_back(
      tsl::Flag("xla_gpu_deterministic_ops",
                bool_setter_for(&DebugOptions::set_xla_gpu_deterministic_ops),
                debug_options->xla_gpu_deterministic_ops(),
                "Guarantees run-to-run determinism on GPU."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_exclude_nondeterministic_ops",
      bool_setter_for(&DebugOptions::set_xla_gpu_exclude_nondeterministic_ops),
      debug_options->xla_gpu_exclude_nondeterministic_ops(),
      "Excludes non-deterministic ops from compiled executables."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_disable_async_collectives",
      setter_for_xla_gpu_disable_async_collectives,
      collective_op_types_to_string(
          debug_options->xla_gpu_disable_async_collectives()),
      "This disables a certain set of async collectives and turn them into"
      " synchronous ones. By default, this is empty which indicates enabling"
      " async execution for all collectives. A sample usage is: "
      " --xla_gpu_disable_async_collectives=ALLREDUCE,REDUCESCATTER"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_while_loop_unrolling",
      setter_for_xla_gpu_enable_while_loop_unrolling,
      DebugOptions::WhileLoopUnrolling_Name(
          debug_options->xla_gpu_enable_while_loop_unrolling()),
      "Enables while loop unrolling features. "
      "`WHILE_LOOP_UNROLLING_DOUBLE_BUFFER` unrolls the loop by factor of 2, "
      "`WHILE_LOOP_UNROLLING_FULL_UNROLL` will unroll the entire loop "
      "`WHILE_LOOP_UNROLLING_AUTO_UNROLL` unrolls by a factor of 2, if there is"
      " any collective present within a while loop."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_all_reduce_combine_threshold_bytes",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_all_reduce_combine_threshold_bytes),
      debug_options->xla_gpu_all_reduce_combine_threshold_bytes(),
      "Size threshold (in bytes) for the GPU all-reduce combiner."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_all_gather_combine_threshold_bytes",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_all_gather_combine_threshold_bytes),
      debug_options->xla_gpu_all_gather_combine_threshold_bytes(),
      "Size threshold (in bytes) for the GPU all-gather combiner."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_reduce_scatter_combine_threshold_bytes",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_reduce_scatter_combine_threshold_bytes),
      debug_options->xla_gpu_reduce_scatter_combine_threshold_bytes(),
      "Size threshold (in bytes) for the GPU reduce-scatter combiner."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_all_gather_combine_by_dim",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_all_gather_combine_by_dim),
      debug_options->xla_gpu_enable_all_gather_combine_by_dim(),
      "Combine all-gather ops with the same gather dimension or irrespective "
      "of their dimension."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_reduce_scatter_combine_by_dim",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_reduce_scatter_combine_by_dim),
      debug_options->xla_gpu_enable_reduce_scatter_combine_by_dim(),
      "Combine reduce-scatter ops with the same dimension or irrespective of "
      "their dimension."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_approx_costly_collectives",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_approx_costly_collectives),
      debug_options->xla_gpu_enable_approx_costly_collectives(),
      "Enables more accurate latency approximation of collectives. Used in "
      "`ApproximateLatencyEstimator` scheduler."));
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
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_while_loop_reduce_scatter_code_motion",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_enable_while_loop_reduce_scatter_code_motion),
      debug_options->xla_gpu_enable_while_loop_reduce_scatter_code_motion(),
      "Enable hoisting of reduce-scatter outside while loops."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_collective_inflation_factor",
      int32_setter_for(&DebugOptions::set_xla_gpu_collective_inflation_factor),
      debug_options->xla_gpu_collective_inflation_factor(),
      "Inflation factor for collectives. If set to > 1, each XLA/GPU "
      "collective will execute multiple times (will yield incorrect results)"));

  flag_list->push_back(tsl::Flag(
      "xla_llvm_force_inline_before_split",
      bool_setter_for(&DebugOptions::set_xla_llvm_force_inline_before_split),
      debug_options->xla_llvm_force_inline_before_split(),
      "Decide whether to force inline before llvm module split to get a more "
      "balanced splits for parallel compilation"));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_reassociation_for_converted_ar",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_reassociation_for_converted_ar),
      debug_options->xla_gpu_enable_reassociation_for_converted_ar(),
      "Enable allreduce reassociation on allreduces that are converted to a "
      "wider type. "
      "The reassociated allreduce will be promoted to a wider-typed "
      "allreduce."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_dump_llvmir",
                bool_setter_for(&DebugOptions::set_xla_gpu_dump_llvmir),
                debug_options->xla_gpu_dump_llvmir(), "Dump LLVM IR."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_cudnn_frontend",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_cudnn_frontend),
      debug_options->xla_gpu_enable_cudnn_frontend(),
      "Use the cuDNN frontend API for convolutions when possible."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_cudnn_fmha",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_cudnn_fmha),
      debug_options->xla_gpu_enable_cudnn_fmha(),
      "Use the cuDNN Fused Attention runtime fusion when possible. Note "
      "that dropout support and the development of this feature as a whole is "
      "in progress. Attention with dropout may cause results to diverge with "
      "and without this  flag turned on."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_fused_attention_use_cudnn_rng",
      bool_setter_for(&DebugOptions::set_xla_gpu_fused_attention_use_cudnn_rng),
      debug_options->xla_gpu_fused_attention_use_cudnn_rng(),
      "Use cudnn random number generator for fused attention kernel."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_cudnn_layer_norm",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_cudnn_layer_norm),
      debug_options->xla_gpu_enable_cudnn_layer_norm(),
      "Rewrite layer norm patterns into cuDNN library call."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_cublaslt",
                bool_setter_for(&DebugOptions::set_xla_gpu_enable_cublaslt),
                debug_options->xla_gpu_enable_cublaslt(),
                "Use cuBLASLt for GEMMs when possible."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_graph_level", setter_for_xla_gpu_graph_level, 1,
      "The legacy flag for setting GPU graph level. Use "
      "xla_gpu_enable_command_buffer in new use cases. 0 = off; 1 = capture "
      "fusions and memcpys; 2 = capture gemms; 3 = capture convolutions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_command_buffer", setter_for_xla_gpu_enable_command_buffer,
      command_types_to_string(debug_options->xla_gpu_enable_command_buffer()),
      "The types of the commands that are recorded into command buffers. It"
      " can either be a list of command types or a list of command types with"
      " + and - as prefix, which indicate adding or removing a command type"
      " to/from the default list."));

  flag_list->push_back(
      tsl::Flag("legacy_command_buffer_custom_call_targets",
                setter_for_legacy_command_buffer_custom_call_targets, "",
                "Comma-separated list of custom call targets with legacy "
                "registry API (non FFI API), whose targets supports lowering "
                "to command buffer custom command, i.e., custom call target "
                "supports cuda-graph capturing for CUDA devices."));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_graph_min_graph_size",
      int32_setter_for(&DebugOptions::set_xla_gpu_graph_min_graph_size),
      debug_options->xla_gpu_graph_min_graph_size(),
      "Capture a region as a function to be launched as cuda graph if the "
      "number of moved instructions reaches this threshold."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_graph_enable_concurrent_region",
                bool_setter_for(
                    &DebugOptions::set_xla_gpu_graph_enable_concurrent_region),
                debug_options->xla_gpu_graph_enable_concurrent_region(),
                "Identify concurrent regions in gpu graphs and execute them "
                "concurrently."));
  flag_list->push_back(tsl::Flag(
      "xla_cmd_buffer_trace_cache_size",
      int64_setter_for(&DebugOptions::set_xla_cmd_buffer_trace_cache_size),
      debug_options->xla_cmd_buffer_trace_cache_size(),
      "Set the command buffer trace cache size, increasing the cache size may "
      "sometimes reduces the chances of doing command buffer tracing for "
      "updating command buffer instance."));
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
      "xla_dump_enable_mlir_pretty_form",
      bool_setter_for(&DebugOptions::set_xla_dump_enable_mlir_pretty_form),
      debug_options->xla_dump_enable_mlir_pretty_form(),
      "Enable dumping MLIR using pretty print form. If set to false, the "
      "dumped "
      "MLIR will be in the llvm-parsable format and can be processed by "
      "mlir-opt tools. "
      "Pretty print form is not legal MLIR."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_custom_fusions",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_custom_fusions),
      debug_options->xla_gpu_enable_custom_fusions(),
      "Whether to enable XLA custom fusions"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_custom_fusions_re",
      string_setter_for(&DebugOptions::set_xla_gpu_enable_custom_fusions_re),
      debug_options->xla_gpu_enable_custom_fusions_re(),
      "Limits custom fusion only to fusions which match this regular "
      "expression. Default is all custom fusions registerered in a current "
      "process."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_dynamic_slice_fusion",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_dynamic_slice_fusion),
      debug_options->xla_gpu_enable_dynamic_slice_fusion(),
      "Whether to enable XLA address computation fusion"));
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
      "xla_gpu_enable_nccl_user_buffers",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_nccl_user_buffers),
      debug_options->xla_gpu_enable_nccl_user_buffers(),
      "Enables NCCL User Buffer Registration. collective_memory_size in the "
      "allocator config must also be set to a non-zero value that is large "
      "enough to meet peak collective memory usage."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_temp_buffer_use_separate_color",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_temp_buffer_use_separate_color),
      debug_options->xla_gpu_temp_buffer_use_separate_color(),
      "Enables temp User Buffer Registration. Enable this flag will use a "
      "separate cuda async memory allocator to allocate temp buffer, this will "
      "allocate temp buffer to the fixed address on every iteration"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_nccl_comm_splitting",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_nccl_comm_splitting),
      debug_options->xla_gpu_enable_nccl_comm_splitting(),
      "Enables NCCL communicator splitting which allows sharing NCCL resources "
      "between different NCCL cliques."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_nccl_per_stream_comms",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_nccl_per_stream_comms),
      debug_options->xla_gpu_enable_nccl_per_stream_comms(),
      "A separate NCCL communicator will be created for each stream that a "
      "NCCL collective is executed on. This can lead to higher performance if "
      "NCCL collectives are issued concurrently at the cost of more GPU memory"
      " usage."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_redzone_scratch_max_megabytes",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_redzone_scratch_max_megabytes),
      debug_options->xla_gpu_redzone_scratch_max_megabytes(),
      "Max size (in megabytes) for the GPU redzone scratch allocator."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_redzone_padding_bytes",
      int64_setter_for(&DebugOptions::set_xla_gpu_redzone_padding_bytes),
      debug_options->xla_gpu_redzone_padding_bytes(),
      "Amount of padding the redzone allocator will put on one side of each "
      "buffer it allocates. (So the buffer's total size will be increased by "
      "2x this value.)"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_shape_checks", setter_for_xla_gpu_shape_checks,
      DebugOptions::ShapeChecks_Name(debug_options->xla_gpu_shape_checks()),
      "When to perform shape checks in XLA:GPU."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_strict_dot_conv_math",
      bool_setter_for(&DebugOptions::set_xla_cpu_strict_dot_conv_math),
      debug_options->xla_cpu_strict_dot_conv_math(),
      "By default, XLA:CPU will run fp16 dot/conv as fp32, as this is "
      "generally (much) faster on our hardware. Set this flag to true to "
      "disable this behavior."));
  flag_list->push_back(tsl::Flag(
      "xla_dump_latency_hiding_schedule",
      bool_setter_for(&DebugOptions::set_xla_dump_latency_hiding_schedule),
      debug_options->xla_dump_latency_hiding_schedule(),
      "Dump the schedule from the latency-hiding scheduler."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_mlir_tiling_and_fusion",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_mlir_tiling_and_fusion),
      debug_options->xla_cpu_enable_mlir_tiling_and_fusion(),
      "Enable MLIR tiling and fusion."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_mlir_fusion_outlining",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_mlir_fusion_outlining),
      debug_options->xla_cpu_enable_mlir_fusion_outlining(),
      "Enable MLIR fusion outlining (to improve compile time)."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_custom_matmul_tiling",
      bool_setter_for(&DebugOptions::set_xla_cpu_enable_custom_matmul_tiling),
      debug_options->xla_cpu_enable_custom_matmul_tiling(),
      "Enable custom tiling given by M, K, N parameters."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_matmul_tiling_m_dim",
      int64_setter_for(&DebugOptions::set_xla_cpu_matmul_tiling_m_dim),
      debug_options->xla_cpu_matmul_tiling_m_dim(),
      "Custom tile size for matmul's M dimension."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_matmul_tiling_n_dim",
      int64_setter_for(&DebugOptions::set_xla_cpu_matmul_tiling_n_dim),
      debug_options->xla_cpu_matmul_tiling_n_dim(),
      "Custom tile size for matmul's N dimension."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_matmul_tiling_k_dim",
      int64_setter_for(&DebugOptions::set_xla_cpu_matmul_tiling_k_dim),
      debug_options->xla_cpu_matmul_tiling_k_dim(),
      "Custom tile size for matmul's K dimension."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_enable_experimental_deallocation",
      bool_setter_for(
          &DebugOptions::set_xla_cpu_enable_experimental_deallocation),
      debug_options->xla_cpu_enable_experimental_deallocation(),
      "Enable experimental deallocation."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_latency_hiding_scheduler",
                bool_setter_for(
                    &DebugOptions::set_xla_gpu_enable_latency_hiding_scheduler),
                debug_options->xla_gpu_enable_latency_hiding_scheduler(),
                "Enable latency-hiding scheduler for XLA:GPU"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_analytical_latency_estimator",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_analytical_latency_estimator),
      debug_options->xla_gpu_enable_analytical_latency_estimator(),
      "Enable analytical latency estimator for latency-hiding scheduler for "
      "XLA:GPU"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_pgle_profile_file_or_directory_path",
      string_setter_for(
          &DebugOptions::set_xla_gpu_pgle_profile_file_or_directory_path),
      debug_options->xla_gpu_pgle_profile_file_or_directory_path(),
      "Directory or file for PGLE profiles in XLA:GPU"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_lhs_enable_gpu_async_tracker",
      bool_setter_for(&DebugOptions::set_xla_gpu_lhs_enable_gpu_async_tracker),
      debug_options->xla_gpu_lhs_enable_gpu_async_tracker(),
      "Enable GPU async tracker for latency-hiding scheduler in XLA:GPU"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_memory_limit_slop_factor",
      int32_setter_for(&DebugOptions::set_xla_gpu_memory_limit_slop_factor),
      debug_options->xla_gpu_memory_limit_slop_factor(),
      "Slop factor for memory limits in XLA:GPU"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_highest_priority_async_stream",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_highest_priority_async_stream),
      debug_options->xla_gpu_enable_highest_priority_async_stream(),
      "Enable async stream to have the highest priority."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pipelined_collectives",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_pipelined_collectives),
      debug_options->xla_gpu_enable_pipelined_collectives(),
      "Enable pipelinling of collective instructions (all-reduce, all-gather, "
      "and reduce-scatter)."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pipelined_all_reduce",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_pipelined_all_reduce),
      debug_options->xla_gpu_enable_pipelined_all_reduce(),
      "Enable pipelinling of all-reduce instructions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pipelined_all_gather",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_pipelined_all_gather),
      debug_options->xla_gpu_enable_pipelined_all_gather(),
      "Enable pipelinling of all-gather instructions."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_pipelined_reduce_scatter",
                bool_setter_for(
                    &DebugOptions::set_xla_gpu_enable_pipelined_reduce_scatter),
                debug_options->xla_gpu_enable_pipelined_reduce_scatter(),
                "Enable pipelinling of reduce-scatter instructions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pipelined_p2p",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_pipelined_p2p),
      debug_options->xla_gpu_enable_pipelined_p2p(),
      "Enable pipelinling of P2P instructions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_run_post_layout_collective_pipeliner",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_run_post_layout_collective_pipeliner),
      debug_options->xla_gpu_run_post_layout_collective_pipeliner(),
      "Move collective pipeliner after the post-layout optimization."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_collective_permute_decomposer_threshold",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_collective_permute_decomposer_threshold),
      debug_options->xla_gpu_collective_permute_decomposer_threshold(),
      "Collective permute decomposer threshold."));
  flag_list->push_back(tsl::Flag(
      "xla_partitioning_algorithm", setter_for_xla_partitioning_algorithm,
      DebugOptions::PartitioningAlgorithm_Name(
          debug_options->xla_partitioning_algorithm()),
      "The partitioning algorithm to be used in the PartitionAssignment pass"));
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_triton_gemm",
                bool_setter_for(&DebugOptions::set_xla_gpu_enable_triton_gemm),
                debug_options->xla_gpu_enable_triton_gemm(),
                "Use Triton-based matrix multiplication."));
  flag_list->push_back(tsl::Flag("xla_gpu_enable_triton_softmax_fusion",
                                 noop_flag_setter<bool>, false,
                                 "[Deprecated, do not use]"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_verify_triton_fusion_numerics",
      bool_setter_for(&DebugOptions::set_xla_gpu_verify_triton_fusion_numerics),
      debug_options->xla_gpu_verify_triton_fusion_numerics(),
      "Whether to verify that the numeric results of Triton fusions match the "
      "results of regular emitters."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_cudnn_int8x32_convolution_reordering",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_enable_cudnn_int8x32_convolution_reordering),
      debug_options->xla_gpu_enable_cudnn_int8x32_convolution_reordering(),
      "Enable cuDNN frontend for int8x32 convolutions with reordered filter."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_triton_gemm_any",
                bool_setter_for(&DebugOptions::set_xla_gpu_triton_gemm_any),
                debug_options->xla_gpu_triton_gemm_any(),
                "Use Triton-based matrix multiplication for any GEMM it "
                "supports without filtering only faster ones. To make sure "
                "only triton gemm is chosen by the autotuner run with "
                "`xla_gpu_cublas_fallback` set to false."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_exhaustive_tiling_search",
      bool_setter_for(&DebugOptions::set_xla_gpu_exhaustive_tiling_search),
      debug_options->xla_gpu_exhaustive_tiling_search(),
      "Enable (slow) search for the Triton GEMM fusion tilings."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_priority_fusion",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_priority_fusion),
      debug_options->xla_gpu_enable_priority_fusion(),
      "Enable priority queue for fusion order."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_triton_softmax_priority_fusion",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_triton_softmax_priority_fusion),
      debug_options
          ->xla_gpu_experimental_enable_triton_softmax_priority_fusion(),
      "Enable fusion into Triton Softmax in PriorityFusion pass."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_dump_autotune_results_to",
      string_setter_for(&DebugOptions::set_xla_gpu_dump_autotune_results_to),
      debug_options->xla_gpu_dump_autotune_results_to(),
      "File to write autotune results to. It will be a binary file unless the "
      "name ends with .txt or .textproto. Warning: The results are written at "
      "every compilation, possibly multiple times per process. This only works "
      "on CUDA. In tests, the TEST_UNDECLARED_OUTPUTS_DIR prefix can be used "
      "to write to their output directory."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_load_autotune_results_from",
      string_setter_for(&DebugOptions::set_xla_gpu_load_autotune_results_from),
      debug_options->xla_gpu_load_autotune_results_from(),
      "File to load autotune results from. It will be considered a binary file "
      "unless the name ends with .txt or .textproto. It will be loaded at most "
      "once per process. This only works on CUDA. In tests, the TEST_WORKSPACE "
      "prefix can be used to load files from their data dependencies."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_require_complete_aot_autotune_results",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_require_complete_aot_autotune_results),
      debug_options->xla_gpu_require_complete_aot_autotune_results(),
      "Whether to require complete AOT autotuning results."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_auto_spmd_partitioning_memory_budget_gb",
      int32_setter_for(
          &DebugOptions::set_xla_gpu_auto_spmd_partitioning_memory_budget_gb),
      debug_options->xla_gpu_auto_spmd_partitioning_memory_budget_gb(),
      "Memory budget in GB per device for AutoSharding."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_auto_spmd_partitioning_memory_budget_ratio",
      float_setter_for(
          &DebugOptions::
              set_xla_gpu_auto_spmd_partitioning_memory_budget_ratio),
      debug_options->xla_gpu_auto_spmd_partitioning_memory_budget_ratio(),
      "Enabled when xla_gpu_auto_spmd_partitioning_memory_budget_gb is 0. "
      "The memory budget is set to "
      "xla_gpu_auto_spmd_partitioning_memory_budget_ratio times the estimated "
      "memory usage lower bound."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_triton_gemm_disable_reduced_precision_reduction",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_triton_gemm_disable_reduced_precision_reduction),
      debug_options->xla_gpu_triton_gemm_disable_reduced_precision_reduction(),
      "Forces any reductions during matrix multiplications to use the "
      "accumulator type and not the output type. The precision of the dot "
      "operation may not increase that much if there is output fusion."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_triton_fusion_level",
      int32_setter_for(&DebugOptions::set_xla_gpu_triton_fusion_level),
      debug_options->xla_gpu_triton_fusion_level(),
      "Triton fusion level, higher levels mean more fused operations."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_dump_autotuned_gemm_fusions",
      bool_setter_for(&DebugOptions::set_xla_gpu_dump_autotuned_gemm_fusions),
      debug_options->xla_gpu_dump_autotuned_gemm_fusions(),
      "Dumps autotuned GEMM fusions to the directory specified by "
      "xla_dump_to or stdout. Each fusion is dumped only once, as an optimized "
      "HLO."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_override_gemm_autotuner",
      string_setter_for(&DebugOptions::set_xla_gpu_override_gemm_autotuner),
      debug_options->xla_gpu_override_gemm_autotuner(),
      "Overrides the GEMM autotuner to use the specified "
      "(AutotuneResult::TritonGemmKey) textproto configuration for all Triton "
      "GEMM fusions. (You can get such textprotos from the debug logs of the "
      "GEMM autotuner.) "));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_copy_insertion_use_region_analysis",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_copy_insertion_use_region_analysis),
      debug_options->xla_gpu_copy_insertion_use_region_analysis(),
      "If true, use the new fine-grain region-based live range interference"
      " analysis in the copy insertion optimization pass."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_collect_cost_model_stats",
      bool_setter_for(&DebugOptions::set_xla_gpu_collect_cost_model_stats),
      debug_options->xla_gpu_collect_cost_model_stats(),
      "If true, each fusion instruction will have a cost model runtime "
      "estimate in backend config after compilation."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_split_k_autotuning",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_split_k_autotuning),
      debug_options->xla_gpu_enable_split_k_autotuning(),
      "Enable split_k autotuning for triton gemms."));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_reduction_epilogue_fusion",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_reduction_epilogue_fusion),
      debug_options->xla_gpu_enable_reduction_epilogue_fusion(),
      "Enable fusion for reduction epilogues"));
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_nccl_clique_optimization",
                bool_setter_for(
                    &DebugOptions::set_xla_gpu_enable_nccl_clique_optimization),
                debug_options->xla_gpu_enable_nccl_clique_optimization(),
                "Allow early return when acquiring NCCL cliques"));
  flag_list->push_back(
      tsl::Flag("xla_gpu_cublas_fallback",
                bool_setter_for(&DebugOptions::set_xla_gpu_cublas_fallback),
                debug_options->xla_gpu_cublas_fallback(),
                "Allow GEMM fusion autotuning to fall back to cuBLAS when that "
                "is faster."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_cudnn_gemm_fusion_level",
      int32_setter_for(&DebugOptions::set_xla_gpu_cudnn_gemm_fusion_level),
      debug_options->xla_gpu_cudnn_gemm_fusion_level(),
      "cuDNN GEMM fusion level; higher level corresponds to more kinds of "
      "fused operations."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_mock_custom_calls",
                bool_setter_for(&DebugOptions::set_xla_gpu_mock_custom_calls),
                debug_options->xla_gpu_mock_custom_calls(),
                "Replace custom calls with noop operations."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_while_loop_double_buffering",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_while_loop_double_buffering),
      debug_options->xla_gpu_enable_while_loop_double_buffering(),
      "Enable double buffering for while loop"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_ensure_minor_dot_contraction_dims",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_ensure_minor_dot_contraction_dims),
      debug_options->xla_gpu_ensure_minor_dot_contraction_dims(),
      "Ensure that the contracting dimensions for matmul operands are the most "
      "minor by changing layouts accordingly"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_filter_kernels_spilling_registers_on_autotuning",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_filter_kernels_spilling_registers_on_autotuning),
      debug_options->xla_gpu_filter_kernels_spilling_registers_on_autotuning(),
      "Filter out kernels that spill registers during autotuning"));
  flag_list->push_back(tsl::Flag(
      "xla_debug_buffer_assignment_show_max",
      int64_setter_for(&DebugOptions::set_xla_debug_buffer_assignment_show_max),
      debug_options->xla_debug_buffer_assignment_show_max(),
      "Number of buffers to display when debugging the buffer assignment"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_llvm_verification_level",
      int32_setter_for(&DebugOptions::set_xla_gpu_llvm_verification_level),
      debug_options->xla_gpu_llvm_verification_level(),
      "Sets how often we verify the generated llvm modules. Higher "
      "levels mean more frequent verification. Currently supported: 0, 1."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_target_config_filename",
      string_setter_for(&DebugOptions::set_xla_gpu_target_config_filename),
      debug_options->xla_gpu_target_config_filename(),
      "Filename for GPU TargetConfig. Triggers devicless compilation: attached "
      "device is "
      "ignored, and the proto is queried instead"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_cub_radix_sort",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_cub_radix_sort),
      debug_options->xla_gpu_enable_cub_radix_sort(),
      "Enable radix sort using CUB for simple shapes"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_threshold_for_windowed_einsum_mib",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_threshold_for_windowed_einsum_mib),
      debug_options->xla_gpu_threshold_for_windowed_einsum_mib(),
      "Threshold to enable windowed einsum (collective matmul) in MB."
      "Einsums that have partitioned operand(can be either LHS or RHS) that's "
      "larger than this threshold will be transformed to use windowed einsums."
      "Default is 100000"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_triton_hopper",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_triton_hopper),
      debug_options->xla_gpu_enable_triton_hopper(),
      "Currently used to enable MMA_V3 for Hopper in Triton"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_fusion_block_level_rewriter",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_fusion_block_level_rewriter),
      debug_options->xla_gpu_experimental_enable_fusion_block_level_rewriter(),
      "Enabling this flag will attempt to redirect every fusion possible to "
      "the Triton emitter"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_libnvptxcompiler",
      [debug_options](bool enabled) {
        if (enabled && !stream_executor::IsLibNvPtxCompilerSupported()) {
          // This feature can't be enabled when XLA was built without
          // libnvptxcompiler support.
          return false;
        }
        debug_options->set_xla_gpu_enable_libnvptxcompiler(enabled);
        return true;
      },
      debug_options->xla_gpu_enable_libnvptxcompiler(),
      "Use libnvptxcompiler for PTX-to-GPU-assembly compilation instead of "
      "calling ptxas."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_dot_strength_reduction",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_dot_strength_reduction),
      debug_options->xla_gpu_enable_dot_strength_reduction(),
      "Enable rewriting matmuls with a vector into reductions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_bf16_6way_gemm",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_bf16_6way_gemm),
      debug_options->xla_gpu_enable_bf16_6way_gemm(),
      "Use BF16 6way gemm to compute F32 gemm."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_bf16_3way_gemm",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_bf16_3way_gemm),
      debug_options->xla_gpu_enable_bf16_3way_gemm(),
      "Use BF16 3way gemm to compute F32 gemm."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_nccl_collective_max_nchannels",
                int64_setter_for(
                    &DebugOptions::set_xla_gpu_nccl_collective_max_nchannels),
                debug_options->xla_gpu_nccl_collective_max_nchannels(),
                "Specify the maximum number of channels(SMs) NCCL will use "
                "for collective operations. Default is 0 which is to let "
                "NCCL decide."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_nccl_p2p_max_nchannels",
      int64_setter_for(&DebugOptions::set_xla_gpu_nccl_p2p_max_nchannels),
      debug_options->xla_gpu_nccl_p2p_max_nchannels(),
      "Specify the maximum number of channels(SMs) NCCL will use "
      "for p2p operations. Default is 0 which is to let "
      "NCCL decide."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_mlir_emitter_level",
                int64_setter_for(&DebugOptions::set_xla_gpu_mlir_emitter_level),
                debug_options->xla_gpu_mlir_emitter_level(),
                "Enable new MLIR-based emitters. Level 0 means disabled, "
                "higher levels enable more of the emitters."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_multi_streamed_windowed_einsum",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_multi_streamed_windowed_einsum),
      debug_options->xla_gpu_multi_streamed_windowed_einsum(),
      "Whether to run windowed einsum using multiple compute streams."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_gemm_rewrite_size_threshold",
      int64_setter_for(&DebugOptions::set_xla_gpu_gemm_rewrite_size_threshold),
      debug_options->xla_gpu_gemm_rewrite_size_threshold(),
      "Threshold until which elemental dot emitter is preferred for GEMMs "
      "(minimum combined number of elements of both matrices "
      "in non-batch dimensions to be considered for a rewrite)."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_use_memcpy_local_p2p",
      bool_setter_for(&DebugOptions::set_xla_gpu_use_memcpy_local_p2p),
      debug_options->xla_gpu_use_memcpy_local_p2p(),
      "Whether to use memcpy for local p2p communication."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_dump_autotune_logs_to",
      string_setter_for(&DebugOptions::set_xla_gpu_dump_autotune_logs_to),
      debug_options->xla_gpu_dump_autotune_logs_to(),
      "File to write autotune logs to. It will be a binary file unless the "
      "name ends with .txt or .textproto."));
  flag_list->push_back(tsl::Flag(
      "xla_reduce_window_rewrite_base_length",
      int64_setter_for(
          &DebugOptions::set_xla_reduce_window_rewrite_base_length),
      debug_options->xla_reduce_window_rewrite_base_length(),
      "Base length to rewrite the reduce window to, no rewrite if set to 0."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_host_memory_offloading",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_host_memory_offloading),
      debug_options->xla_gpu_enable_host_memory_offloading(),
      "Whether to trigger host memory offloading on a device."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_nccl_terminate_on_error",
      bool_setter_for(&DebugOptions::set_xla_gpu_nccl_terminate_on_error),
      debug_options->xla_gpu_nccl_terminate_on_error(),
      "If set, then NCCL errors will terminate the process."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_shard_autotuning",
      bool_setter_for(&DebugOptions::set_xla_gpu_shard_autotuning),
      debug_options->xla_gpu_shard_autotuning(),
      "Shard autotuning between participating compiler processes (typically in "
      "multi-host setups) and join the results when it's done."));
  flag_list->push_back(
      tsl::Flag("xla_syntax_sugar_async_ops",
                bool_setter_for(&DebugOptions::set_xla_syntax_sugar_async_ops),
                debug_options->xla_syntax_sugar_async_ops(),
                "Enable syntax sugar for async ops in HLO dumps."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_kernel_cache_file",
                string_setter_for(&DebugOptions::set_xla_gpu_kernel_cache_file),
                debug_options->xla_gpu_kernel_cache_file(),
                "Path to a file to cache compiled kernels. Cached kernels get "
                "reused in further compilations; not yet cached kernels are "
                "compiled as usual and get appended to the cache file whenever "
                "possible."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_per_fusion_autotune_cache_dir",
      string_setter_for(
          &DebugOptions::set_xla_gpu_per_fusion_autotune_cache_dir),
      debug_options->xla_gpu_per_fusion_autotune_cache_dir(),
      "Experimental: Maintain a per-fusion autotune cache in the given "
      "directory. XLA will try to read existing results when they are needed "
      "and write new results when they are determined. The directory must "
      "exist. Cache invalidation has to be handled by the user (e.g. please "
      "use an empty directory if you want to start with an empty cache). XLA "
      "version checks must be done by the user (e.g. if you want to use "
      "separate caches for different versions of XLA, please use different "
      "directories). Default: no cache."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_autotune_cache_mode",
      setter_for_xla_gpu_experimental_autotune_cache_mode,
      DebugOptions::AutotuneCacheMode_Name(
          debug_options->xla_gpu_experimental_autotune_cache_mode()),
      "Experimental: Specify the behavior of per kernel autotuning "
      "cache. Supported modes: read (provides readonly access to "
      "the cache), update (loads if the cache exists, runs autotuning "
      "and dumps the result otherwise). Default: update."));
  flag_list->push_back(tsl::Flag(
      "xla_enable_command_buffers_during_profiling",
      bool_setter_for(
          &DebugOptions::set_xla_enable_command_buffers_during_profiling),
      debug_options->xla_enable_command_buffers_during_profiling(),
      "Experimental: Enable command buffers while a profiling active. "
      "By default, enabling profiling switches from command buffers to "
      "op-by-op mode."));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_cudnn_gemm_max_plans",
      int32_setter_for(&DebugOptions::set_xla_gpu_cudnn_gemm_max_plans),
      debug_options->xla_gpu_cudnn_gemm_max_plans(),
      "Limit for the number of kernel configurations (plans) to use during "
      "autotuning of cuDNN GEMM fusions."));

  flag_list->push_back(tsl::Flag("xla_gpu_enable_triton_gemm_int4",
                                 noop_flag_setter<bool>, true,
                                 "[Deprecated, do not use]"));
  flag_list->push_back(
      tsl::Flag("xla_gpu_async_dot",
                bool_setter_for(&DebugOptions::set_xla_gpu_async_dot),
                debug_options->xla_gpu_async_dot(),
                "Wrap `dot` operations into async computations in an effort to "
                "parallelize matrix operations."));
  flag_list->push_back(tsl::Flag(
      "xla_step_marker_location", setter_for_xla_step_marker_location,
      DebugOptions::StepMarkerLocation_Name(
          debug_options->xla_step_marker_location()),
      "Option to emit a target-specific marker to indicate the start of "
      "a training. The location of the marker (if any) is determined "
      "by the option value of type DebugOptions::StepMarkerLocation."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pgle_accuracy_checker",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_pgle_accuracy_checker),
      debug_options->xla_gpu_enable_pgle_accuracy_checker(),
      "Enables strict PGLE checking. If an FDO profile is specified and "
      "latency hiding scheduler encounters missing instructions in the profile "
      "compilation will halt."));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_executable_warn_stuck_timeout",
      int32_setter_for(
          &DebugOptions::set_xla_gpu_executable_warn_stuck_timeout_seconds),
      debug_options->xla_gpu_executable_warn_stuck_timeout_seconds(),
      "Set timeout for RendezvousSingle stuck warning"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_executable_terminate_timeout",
      int32_setter_for(
          &DebugOptions::set_xla_gpu_executable_terminate_timeout_seconds),
      debug_options->xla_gpu_executable_terminate_timeout_seconds(),
      "Set timeout for RendezvousSingle termination"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_disable_binary_libraries",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_disable_binary_libraries),
      debug_options->xla_gpu_experimental_disable_binary_libraries(),
      "Disable XLA GPU passes that depend on non-open source binary "
      "libraries"));
  flag_list->push_back(tsl::Flag(
      "xla_experimental_ignore_channel_id",
      bool_setter_for(&DebugOptions::set_xla_experimental_ignore_channel_id),
      debug_options->xla_experimental_ignore_channel_id(),
      "Experimental: ignore channel ids for collective operations."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_dot_merger_threshold_mb",
      int32_setter_for(&DebugOptions::set_xla_gpu_dot_merger_threshold_mb),
      debug_options->xla_gpu_dot_merger_threshold_mb(),
      "Dot merger pass threshold to be set in MB."));
  flag_list->push_back(
      tsl::Flag("xla_enable_fast_math",
                bool_setter_for(&DebugOptions::set_xla_enable_fast_math),
                debug_options->xla_enable_fast_math(),
                "Enable optimizations that assume finite math, i.e., no NaN."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_parallel_collective_overlap_limit",
      int32_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_parallel_collective_overlap_limit),
      debug_options->xla_gpu_experimental_parallel_collective_overlap_limit(),
      "This controls how many in-flight collectives "
      "latency hiding scheduler can schedule."));
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
