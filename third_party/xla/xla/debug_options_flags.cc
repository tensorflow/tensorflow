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
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/text_format.h"
#include "xla/debug_options_parsers.h"
#include "xla/parse_flags_from_env.h"
#include "xla/service/collective_utils.h"
#include "xla/stream_executor/cuda/nvjitlink_support.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/cpu_info.h"  // NOLINT

namespace xla {

namespace details {

absl::StatusOr<std::vector<RepeatedFlagModifier>> ParseRepeatedEnumModifiers(
    const absl::string_view flag_value, const absl::string_view add_prefix) {
  std::vector<absl::string_view> values = absl::StrSplit(flag_value, ',');
  std::string prefix = absl::AsciiStrToUpper(add_prefix);
  std::vector<RepeatedFlagModifier> modifiers;
  modifiers.reserve(values.size());
  int incremental_modifiers_count = 0;
  for (absl::string_view value : values) {
    RepeatedFlagModifier modifier;
    value = absl::StripAsciiWhitespace(value);
    if (value.empty()) {
      continue;
    }
    modifier.op = RepeatedFlagModifier::Op::kAdd;
    if (absl::StartsWith(value, "+") || absl::StartsWith(value, "-")) {
      incremental_modifiers_count++;
      if (absl::StartsWith(value, "-")) {
        modifier.op = RepeatedFlagModifier::Op::kRemove;
      }
      value = value.substr(1);
    } else if (modifiers.empty()) {
      modifiers.push_back(
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kClear, ""});
    }
    modifier.value = absl::AsciiStrToUpper(value);
    if (!prefix.empty() && !absl::StartsWith(modifier.value, prefix)) {
      modifier.value = absl::StrCat(prefix, modifier.value);
    }
    modifiers.push_back(modifier);
  }
  // If we have at least one incremental then all values are to be incremental.
  if (incremental_modifiers_count > 0 &&
      incremental_modifiers_count != values.size()) {
    return absl::InvalidArgumentError(
        "All values must be incremental or none of them.");
  }
  return modifiers;
}

}  // namespace details

namespace {

template <typename T>
static auto FindRepeatedFieldValue(google::protobuf::RepeatedField<int>* list, T value) {
  for (auto it = list->begin(); it != list->end(); ++it) {
    if (*it == value) {
      return it;
    }
  }
  return list->end();
}

// Returns a `DebugOptions` setter for repeated enum flag of type `T`.
// NOLINTBEGIN(readability-function-cognitive-complexity)
// We disable the cognitive complexity check here because we need to return a
// lambda function.
template <typename T>
static auto SetterForRepeatedEnum(
    absl::string_view flag_name, absl::string_view enum_prefix,
    bool (*enum_parser)(absl::string_view string_value, T* value),
    google::protobuf::RepeatedField<int>* mutable_array) {
  return [flag_name, enum_prefix, enum_parser,
          mutable_array](const std::string& input) {
    if (input.empty()) {  // Disable all values.
      mutable_array->Clear();
      return true;
    }

    auto value_modifiers =
        details::ParseRepeatedEnumModifiers(input, enum_prefix);
    if (!value_modifiers.ok()) {
      LOG(ERROR) << absl::StreamFormat("Illegal value for %s '%s': %s",
                                       flag_name, input,
                                       value_modifiers.status().message());
      return false;
    }
    for (const auto& mod : *value_modifiers) {
      if (mod.op == details::RepeatedFlagModifier::Op::kClear) {
        mutable_array->Clear();
        continue;
      }
      T value;
      if (!enum_parser(mod.value, &value)) {
        LOG(ERROR) << absl::StreamFormat("Illegal value for --%s '%s'",
                                         flag_name, mod.value);
        return false;
      }
      auto it = FindRepeatedFieldValue(mutable_array, value);
      if (mod.op == details::RepeatedFlagModifier::Op::kAdd &&
          it == mutable_array->end()) {  // Do not add duplicates.
        mutable_array->Add(value);
      } else if (mod.op == details::RepeatedFlagModifier::Op::kRemove &&
                 it != mutable_array->end()) {
        mutable_array->erase(it);
      }
    }
    return true;
  };
}
// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace

inline std::string DefaultMaxIsa() {
  // There are many missing SVE lowerings in LLVM. Limit features to NEON for
  // now. There shouldn't be significant performance impact as most AAarch64
  // CPUs still use 128-bit registers.
  // TODO(penporn): Remove this once SVE is fully supported.
  return tsl::port::IsAarch64CPU() ? "NEON" : "";
}

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
  opts.set_xla_gpu_generate_debug_info(false);
  opts.set_xla_gpu_generate_line_info(false);

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
  opts.set_xla_dump_hlo_as_long_text(true);
  opts.set_xla_dump_large_constants(false);
  opts.set_xla_dump_enable_mlir_pretty_form(true);
  opts.set_xla_dump_full_hlo_config(true);
  opts.set_xla_debug_buffer_assignment_show_max(15);
  opts.set_xla_cpu_use_onednn(false);
  opts.set_xla_cpu_experimental_onednn_custom_call(false);
#ifdef XLA_CPU_USE_ACL
  opts.set_xla_cpu_use_acl(true);
#endif
  opts.set_xla_cpu_use_fusion_emitters(true);
  opts.set_xla_cpu_use_xnnpack(true);
  opts.set_xla_cpu_experimental_xnn_graph_fusion_mode(
      DebugOptions::XNN_GRAPH_FUSION_MODE_DISABLED);
  opts.add_xla_cpu_experimental_ynn_fusion_type(
      DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT);
  opts.set_xla_cpu_parallel_codegen_split_count(32);
  opts.set_xla_cpu_copy_insertion_use_region_analysis(false);
  opts.set_xla_cpu_enable_concurrency_optimized_scheduler(true);
  opts.set_xla_cpu_prefer_vector_width(256);
  opts.set_xla_cpu_max_isa(DefaultMaxIsa());
  opts.set_xla_cpu_generate_unique_c_style_kernel_entry_points(false);
  opts.set_xla_cpu_emitter_verification_level(0);

  opts.set_xla_cpu_enable_fast_math(false);
  opts.set_xla_cpu_enable_platform_dependent_math(true);
  // Disable forms of fast math that have caused users problems in the past.
  opts.set_xla_cpu_fast_math_honor_nans(true);
  opts.set_xla_cpu_fast_math_honor_infs(true);
  opts.set_xla_cpu_fast_math_honor_functions(true);
  opts.set_xla_cpu_fast_math_honor_division(true);

  opts.set_xla_gpu_fused_attention_use_cudnn_rng(false);

  // By default, copy TF's Eigen style min_max behavior with nans.
  opts.set_xla_cpu_enable_fast_min_max(true);

  opts.set_xla_gpu_enable_cublaslt(false);

  opts.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  opts.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  opts.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLASLT);
  opts.add_xla_gpu_enable_command_buffer(DebugOptions::CUSTOM_CALL);
  opts.add_xla_gpu_enable_command_buffer(DebugOptions::CUDNN);
  opts.add_xla_gpu_enable_command_buffer(DebugOptions::DYNAMIC_SLICE_FUSION);
  opts.set_xla_gpu_graph_min_graph_size(5);
  opts.set_xla_gpu_command_buffer_scheduling_mode(DebugOptions::LHS);
  opts.set_xla_gpu_command_buffer_unroll_loops(false);
  opts.set_xla_cmd_buffer_trace_cache_size(16);

  // Despite the name, fast min/max on GPUs does not seem to be any faster, and
  // adds very counter-intuitive "NaN-swallowing" behavior.
  opts.set_xla_gpu_enable_fast_min_max(false);

  opts.set_xla_allow_excess_precision(true);
  opts.set_xla_force_host_platform_device_count(1);
  opts.set_xla_gpu_all_reduce_combine_threshold_bytes(
      kDefaultAllReduceCombineThreshold);
  opts.set_xla_gpu_all_gather_combine_threshold_bytes(
      kDefaultAllGatherCombineThreshold);
  opts.set_xla_gpu_reduce_scatter_combine_threshold_bytes(
      kDefaultReduceScatterCombineThreshold);
  opts.set_xla_gpu_collective_permute_combine_threshold_bytes(
      kDefaultCollectivePermuteCombineThreshold);
  opts.set_xla_gpu_enable_all_gather_combine_by_dim(false);
  opts.set_xla_gpu_enable_reduce_scatter_combine_by_dim(false);
  opts.set_xla_gpu_enable_approx_costly_collectives(false);

  opts.set_xla_gpu_enable_reassociation_for_converted_ar(true);

  opts.set_xla_cpu_enable_xprof_traceme(false);
  opts.set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(false);
  opts.set_xla_multiheap_size_constraint_per_heap(-1);
  opts.set_xla_detailed_logging(true);
  opts.set_xla_enable_dumping(true);
  opts.set_xla_enable_enzyme_comms_opt(false);

  opts.set_xla_gpu_enable_dynamic_slice_fusion(false);
  opts.set_xla_gpu_nccl_termination_timeout_seconds(-1);
  opts.set_xla_gpu_enable_shared_constants(true);
  opts.set_xla_gpu_enable_nccl_user_buffers(false);
  opts.set_xla_gpu_experimental_enable_nccl_symmetric_buffers(false);
  opts.set_xla_gpu_experimental_enable_nvshmem(false);
  opts.set_xla_gpu_enable_nccl_comm_splitting(true);
  opts.set_xla_gpu_nccl_init_max_rank_per_root_ratio(0);

  opts.set_xla_gpu_temp_buffer_use_separate_color(false);
  opts.set_xla_gpu_require_exclusive_lock(false);

  opts.set_xla_gpu_redzone_padding_bytes(8 * 1024 * 1024);
  opts.set_xla_gpu_shape_checks(DebugOptions::RUNTIME);
  opts.set_xla_dump_latency_hiding_schedule(false);
  opts.set_xla_gpu_enable_latency_hiding_scheduler(false);
  opts.set_xla_gpu_enable_analytical_latency_estimator(false);
  opts.set_xla_gpu_enable_analytical_sol_latency_estimator(true);
  auto* sol_estimator_defaults =
      opts.mutable_xla_gpu_analytical_latency_estimator_options();
  sol_estimator_defaults->emplace(kSolNcclOpLaunchUs, "-1");
  sol_estimator_defaults->emplace(kSolNicSpeedGbps, "-1");
  sol_estimator_defaults->emplace(kSolChunkPrepUs, "-1");
  sol_estimator_defaults->emplace(kSolRttUs, "-1");
  sol_estimator_defaults->emplace(kSolChunkSizeBytes, "-1");
  sol_estimator_defaults->emplace(kSolGpusPerNode, "-1");
  opts.set_xla_gpu_pgle_profile_file_or_directory_path("");
  opts.set_xla_gpu_memory_limit_slop_factor(95);
  opts.set_xla_gpu_enable_highest_priority_async_stream(true);

  opts.set_xla_gpu_enable_pipelined_all_reduce(false);
  opts.set_xla_gpu_enable_pipelined_all_gather(false);
  opts.set_xla_gpu_enable_pipelined_reduce_scatter(true);
  opts.set_xla_gpu_enable_pipelined_host_offloading(false);
  opts.set_xla_gpu_enable_pipelined_p2p(false);

  opts.set_xla_gpu_collective_permute_decomposer_threshold(
      std::numeric_limits<int64_t>::max());
  opts.set_xla_gpu_experimental_pipeline_parallelism_opt_level(
      DebugOptions::PIPELINE_PARALLELISM_OPT_LEVEL_DISABLE);

  opts.set_xla_gpu_experimental_collective_cse_distance_threshold(0);

  opts.set_xla_gpu_experimental_enable_subchannel_dequantisation_fusion(false);
  opts.set_xla_partitioning_algorithm(
      DebugOptions::PARTITIONING_ALGORITHM_NOOP);

  opts.set_xla_gpu_enable_triton_gemm(true);
  opts.set_xla_gpu_unsupported_enable_triton_multi_output_fusion(true);
  opts.set_xla_gpu_enable_cudnn_int8x32_convolution_reordering(true);
  opts.set_xla_gpu_triton_gemm_any(true);
  opts.set_xla_gpu_verify_triton_fusion_numerics(false);

  // Moving reduce-scatter out of while loops can increase memory footprint, so
  // turning it off by default.
  opts.set_xla_gpu_enable_while_loop_reduce_scatter_code_motion(false);

  opts.set_xla_gpu_collective_inflation_factor(1);
  opts.set_xla_llvm_force_inline_before_split(true);

  opts.set_xla_gpu_exhaustive_tiling_search(false);

  opts.set_xla_gpu_experimental_enable_triton_heroless_priority_fusion(false);

  opts.set_xla_gpu_auto_spmd_partitioning_memory_budget_gb(0);
  opts.set_xla_gpu_auto_spmd_partitioning_memory_budget_ratio(1.1);

  opts.set_xla_gpu_copy_insertion_use_region_analysis(false);
  opts.set_xla_gpu_collect_cost_model_stats(false);
  opts.set_xla_gpu_enable_split_k_autotuning(true);

  opts.set_xla_gpu_enable_reduction_epilogue_fusion(true);
  opts.set_xla_gpu_cublas_fallback(true);
  opts.set_xla_gpu_cudnn_gemm_fusion_level(0);
  opts.set_xla_gpu_enable_while_loop_double_buffering(false);
  opts.set_xla_gpu_enable_while_loop_unrolling(
      DebugOptions::WHILE_LOOP_UNROLLING_AUTO_UNROLL);
  opts.set_xla_gpu_filter_kernels_spilling_registers_on_autotuning(true);
  opts.set_xla_gpu_fail_ptx_compilation_on_register_spilling(false);
  opts.set_xla_gpu_llvm_verification_level(0);
  opts.set_xla_gpu_target_config_filename("");
  opts.set_xla_gpu_enable_cub_radix_sort(true);
  opts.set_xla_gpu_enable_cudnn_layer_norm(false);
  opts.set_xla_gpu_threshold_for_windowed_einsum_mib(100000);
  opts.set_xla_gpu_operand_bytes_threshold_for_windowed_einsum(-1);

  opts.set_xla_gpu_experimental_enable_fusion_block_level_rewriter(false);

  opts.set_xla_gpu_enable_llvm_module_compilation_parallelism(false);
  opts.set_xla_gpu_default_to_alg_dot_bf16_bf16_f32(false);
  opts.set_xla_gpu_enable_libnvptxcompiler(
      stream_executor::IsLibNvPtxCompilerSupported());
  opts.set_xla_gpu_libnvjitlink_mode(DebugOptions::LIB_NV_JIT_LINK_MODE_AUTO);

  opts.set_xla_gpu_nccl_async_execution(false);
  opts.set_xla_gpu_nccl_blocking_communicators(true);
  opts.set_xla_gpu_nccl_collective_max_nchannels(0);
  opts.set_xla_gpu_nccl_p2p_max_nchannels(0);
  opts.set_xla_gpu_multi_streamed_windowed_einsum(true);

  opts.set_xla_gpu_experimental_stream_annotation(true);
  // Minimum combined size of matrices in matrix multiplication to
  // be rewritten to cuBLAS or Triton kernel call.
  // This threshold is a conservative estimate and has been measured
  // to be always beneficial (up to generally several times faster)
  // on V100 and H100 GPUs. See openxla/xla #9319 for details.
  const int64_t kDefaultMinGemmRewriteSize = 100;
  opts.set_xla_gpu_gemm_rewrite_size_threshold(kDefaultMinGemmRewriteSize);

#ifdef HAS_SUPPORT_FOR_EMBEDDED_LIB_DEVICE
  opts.set_xla_gpu_use_embeded_device_lib(true);
#endif

#ifdef HAS_SUPPORT_FOR_LLD_AS_A_LIBRARY
  opts.set_xla_gpu_use_inprocess_lld(true);
#endif

  opts.set_xla_gpu_use_memcpy_local_p2p(false);

  opts.set_xla_reduce_window_rewrite_base_length(16);

  opts.set_xla_gpu_require_complete_aot_autotune_results(false);

  opts.set_xla_gpu_enable_host_memory_offloading(false);

  opts.set_xla_gpu_nccl_terminate_on_error(false);

  opts.set_xla_gpu_shard_autotuning(true);

  opts.set_xla_syntax_sugar_async_ops(false);

  opts.set_xla_gpu_per_fusion_autotune_cache_dir("");

  opts.set_xla_gpu_experimental_autotune_cache_mode(
      DebugOptions::AUTOTUNE_CACHE_MODE_UPDATE);

  opts.set_xla_gpu_experimental_autotuner_cache_dir("");

  opts.set_xla_gpu_autotune_gemm_rtol(0.1f);

  // TODO(b/355487968): Remove this flag once all data will be presented in
  // xprof with command buffers.
  opts.set_xla_enable_command_buffers_during_profiling(false);

  opts.set_xla_gpu_cudnn_gemm_max_plans(5);

  opts.set_xla_gpu_pgle_accuracy_checker(
      DebugOptions::PGLE_STRICTNESS_LEVEL_WARN);

  opts.set_xla_gpu_executable_embed_debug_info(true);
  opts.set_xla_gpu_executable_warn_stuck_timeout_seconds(10);
  opts.set_xla_gpu_executable_terminate_timeout_seconds(30);

  opts.set_xla_gpu_first_collective_call_warn_stuck_timeout_seconds(20);
  opts.set_xla_gpu_first_collective_call_terminate_timeout_seconds(40);

  opts.set_xla_gpu_experimental_collective_perf_table_path("");
  opts.set_xla_gpu_experimental_matmul_perf_table_path("");
  // TODO(b/366475196): Create XLA GPU without cuDNN, cuBLAS.
  opts.set_xla_gpu_experimental_disable_binary_libraries(false);
  // --xla_ignore_channel_id should be kept false by default while channel ids
  // are load-bearing.
  opts.set_xla_ignore_channel_id(false);
  opts.set_xla_gpu_dot_merger_threshold_mb(32);
  opts.set_xla_enable_fast_math(false);
  opts.set_xla_gpu_experimental_parallel_collective_overlap_limit(1);
  opts.set_xla_pjrt_allow_auto_layout_in_hlo(false);
  opts.set_xla_gpu_enable_scatter_determinism_expander(false);
  opts.set_xla_gpu_unsupported_disable_nested_gemm_fusions(false);
  opts.set_xla_gpu_unsupported_enable_all_reduce_decomposer(false);
  opts.set_xla_gpu_unsupported_enable_ragged_all_to_all_decomposer(false);
  opts.set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel(false);
  opts.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(true);
  opts.set_xla_gpu_experimental_use_autotuner_pass(false);
  opts.set_xla_gpu_experimental_enable_fusion_autotuner(true);
  opts.set_xla_gpu_experimental_allow_unroll_factor_eight(true);
  opts.set_xla_gpu_experimental_pack_dot_operands_along_k_dimension(true);
  opts.set_xla_unsupported_crash_on_hlo_pass_fix_max_iterations(false);
  opts.set_xla_hlo_pass_fix_detect_cycles(false);
  // TODO(b/449025971): Set to true once the issue is fixed.
  opts.set_xla_gpu_experimental_enable_heuristic_collective_combining(false);
  opts.set_xla_unsupported_crash_on_hlo_pass_silent_hlo_change(false);
  opts.set_xla_disable_automatic_host_compute_offload(false);
  opts.set_xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled(
      false);
  opts.set_xla_enable_scoped_logging_timers(true);
  opts.set_xla_unsupported_crash_on_hlo_pass_noop_change(false);
  opts.set_xla_gpu_experimental_enable_split_k_rewrite(false);
  opts.set_xla_gpu_experimental_enable_triton_warp_specialization(false);
  opts.set_xla_detect_unstable_reductions(DebugOptions::DETECTION_MODE_NONE);
  opts.set_xla_detect_unstable_reductions_post_optimizations(
      DebugOptions::DETECTION_MODE_NONE);
  opts.set_xla_gpu_experimental_scaled_dot_with_triton(false);
  opts.set_xla_gpu_experimental_use_raft_select_k(false);
  opts.set_xla_early_exit_with_layouts(false);
  opts.set_xla_gpu_experimental_all_fusions_with_triton(false);

  opts.add_xla_gpu_experimental_autotune_backends(
      DebugOptions::AUTOTUNE_BACKEND_TRITON);
  opts.add_xla_gpu_experimental_autotune_backends(
      DebugOptions::AUTOTUNE_BACKEND_CUBLAS);
  opts.add_xla_gpu_experimental_autotune_backends(
      DebugOptions::AUTOTUNE_BACKEND_CUBLASLT);
  opts.add_xla_gpu_experimental_autotune_backends(
      DebugOptions::AUTOTUNE_BACKEND_CUDNN);
  opts.add_xla_gpu_experimental_autotune_backends(
      DebugOptions::AUTOTUNE_BACKEND_ROCBLAS);
  opts.add_xla_gpu_experimental_autotune_backends(
      DebugOptions::AUTOTUNE_BACKEND_HIPBLASLT);
  opts.add_xla_gpu_experimental_autotune_backends(
      DebugOptions::AUTOTUNE_BACKEND_MIOPEN);

  opts.set_xla_cpu_collective_call_warn_stuck_seconds(20);
  opts.set_xla_cpu_collective_call_terminate_timeout_seconds(40);
  opts.set_xla_cpu_collective_timeout_seconds(30 * 60);

  opts.set_xla_keep_shardings_after_spmd(false);
  opts.set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(false);
  opts.set_xla_gpu_experimental_enable_buffer_saver_on_thunks(false);
  opts.set_xla_gpu_detect_nan(DebugOptions::DETECTION_MODE_NONE);
  opts.set_xla_gpu_detect_inf(DebugOptions::DETECTION_MODE_NONE);
  return opts;
}

static absl::once_flag flags_init;
static DebugOptions* flag_values;
static std::vector<tsl::Flag>* flag_objects;

// Maps pass -> initial fuel values (parsed when AllocateFlags was run).
static absl::flat_hash_map<std::string, int64_t>* const initial_fuel =
    new absl::flat_hash_map<std::string, int64_t>();

// Maps pass -> whether fuel was ever consumed for that pass.
static absl::node_hash_map<std::string, std::atomic<bool>>* const
    fuel_ever_consumed =
        new absl::node_hash_map<std::string, std::atomic<bool>>();

// Maps pass -> remaining fuel.
//
// All threads start off using this global fuel pool, but ResetThreadLocalFuel()
// switches them to a thread-local fuel pool.
static absl::node_hash_map<std::string, std::atomic<int64_t>>* const
    global_fuel = new absl::node_hash_map<std::string, std::atomic<int64_t>>();

// If we're using thread-local fuel, this stores it.
static thread_local std::unique_ptr<
    absl::node_hash_map<std::string, std::atomic<int64_t>>>
    thread_fuel;  // NOLINT (global variable with nontrivial destructor)

// Logs a warning if a pass's fuel was never consumed, on the theory that this
// may be a typo in the flag value.  Called atexit.
static void WarnIfFuelWasNeverConsumed() {
  CHECK_NOTNULL(fuel_ever_consumed);
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

  // Custom "sub-parser" lambda for
  // xla_gpu_analytical_latency_estimator_options.
  auto setter_for_xla_gpu_analytical_latency_estimator_options =
      [debug_options](std::string comma_separated_values) {
        google::protobuf::Map<std::string, std::string>* options_map =
            debug_options
                ->mutable_xla_gpu_analytical_latency_estimator_options();
        parse_xla_backend_extra_options(options_map, comma_separated_values);
        return true;
      };

  // Custom "sub-parser" lambda for
  // xla_gpu_experimental_pipeline_parallelism_opt_level.
  auto setter_for_xla_gpu_experimental_pipeline_parallelism_opt_level =
      [debug_options](const std::string& value) {
        DebugOptions::PipelineParallelismOptLevel level;
        if (!DebugOptions::PipelineParallelismOptLevel_Parse(value, &level)) {
          return false;
        }
        debug_options->set_xla_gpu_experimental_pipeline_parallelism_opt_level(
            level);
        return true;
      };

  // Custom "sub-parser" lambda for `xla_gpu_graph_enable_concurrent_region`.
  auto setter_for_xla_gpu_graph_enable_concurrent_region =
      [debug_options](bool value) {
        if (value) {
          debug_options->set_xla_gpu_command_buffer_scheduling_mode(
              DebugOptions::CONCURRENT);
        } else {
          debug_options->set_xla_gpu_command_buffer_scheduling_mode(
              DebugOptions::SERIALIZE);
        }
        return true;
      };

  // Custom "sub-parser" lambda for `xla_gpu_command_buffer_scheduling_mode`.
  auto setter_for_xla_gpu_command_buffer_scheduling_mode =
      [debug_options](const std::string& value) {
        DebugOptions::CommandBufferSchedulingMode mode;
        if (!DebugOptions::CommandBufferSchedulingMode_Parse(value, &mode)) {
          return false;
        }
        debug_options->set_xla_gpu_command_buffer_scheduling_mode(mode);
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

  auto command_types_to_string =
      [](google::protobuf::RepeatedField<int> command_types) -> std::string {
    struct Formatter {
      void operator()(std::string* out, int type) const {
        absl::StrAppend(out, DebugOptions::CommandBufferCmdType_Name(type));
      }
    };
    return absl::StrJoin(command_types, ", ", Formatter());
  };

  auto autotune_backends_to_string =
      [](google::protobuf::RepeatedField<int> backends) -> std::string {
    struct Formatter {
      void operator()(std::string* out, int type) const {
        absl::StrAppend(out, DebugOptions::AutotuneBackend_Name(type));
      }
    };
    return absl::StrJoin(backends, ", ", Formatter());
  };

  // Custom "sub-parser" for xla_fuel.  Note that ConsumeFuel does not do any
  // locking on the fuel global variables.  This means that it's
  // illegal/undefined behavior to modify this flag value while the compiler is
  // running.
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
      [](google::protobuf::RepeatedField<int> collective_ops) -> std::string {
    struct Formatter {
      void operator()(std::string* out, int type) const {
        absl::StrAppend(out, DebugOptions::CollectiveOpType_Name(type));
      }
    };
    return absl::StrJoin(collective_ops, ", ", Formatter());
  };

  // Custom parser for `xla_cpu_xnn_graph_fusion_mode` flag.
  auto setter_for_xla_cpu_experimental_xnn_graph_fusion_mode =
      [debug_options](absl::string_view input) {
        DebugOptions::XnnGraphFusionMode mode;
        if (!DebugOptions::XnnGraphFusionMode_Parse(
                absl::AsciiStrToUpper(input), &mode)) {
          return false;
        }
        debug_options->set_xla_cpu_experimental_xnn_graph_fusion_mode(mode);
        return true;
      };

  // Custom parser for `xla_gpu_enable_while_loop_unrolling` flag.
  auto setter_for_xla_gpu_enable_while_loop_unrolling =
      [&debug_options](absl::string_view input) {
        DebugOptions::WhileLoopUnrolling unroll_strategy;
        bool parsed = DebugOptions::WhileLoopUnrolling_Parse(
            absl::AsciiStrToUpper(input), &unroll_strategy);
        if (!parsed) {
          return false;
        }
        debug_options->set_xla_gpu_enable_while_loop_unrolling(unroll_strategy);
        return true;
      };

  // Custom parser for xla_gpu_disable_async_collectives.
  auto setter_for_xla_gpu_disable_async_collectives =
      [debug_options](absl::string_view input) {
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
            auto parsed_op = parse_collective_type(value);
            if (parsed_op == DebugOptions::ALLCOLLECTIVES) {
              for (int i = (int)DebugOptions::ALLREDUCE;
                   i < (int)DebugOptions::ALLCOLLECTIVES; i++) {
                debug_options->add_xla_gpu_disable_async_collectives(
                    (DebugOptions::CollectiveOpType)i);
              }
              return true;
            }
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

  // Custom "sub-parser" lambda for xla_gpu_pgle_accuracy_checker.
  auto setter_for_xla_gpu_pgle_accuracy_checker =
      [debug_options](const std::string& value) {
        DebugOptions::PGLEStrictnessLevel strictness_level;
        if (!DebugOptions::PGLEStrictnessLevel_Parse(value,
                                                     &strictness_level)) {
          return false;
        }
        debug_options->set_xla_gpu_pgle_accuracy_checker(strictness_level);
        return true;
      };

  // Custom "sub-parser" for xla_gpu_experimental_autotune_cache_mode.
  auto detection_mode = [](DebugOptions* debug_options,
                           const std::string& value)
      -> std::optional<DebugOptions::DetectionMode> {
    if (value == "none") {
      return DebugOptions::DETECTION_MODE_NONE;
    }
    if (value == "warning") {
      return DebugOptions::DETECTION_MODE_WARNING;
    }
    if (value == "fail") {
      return DebugOptions::DETECTION_MODE_FAIL;
    }
    return std::nullopt;
  };
  auto setter_for_xla_detect_unstable_reductions =
      [debug_options, detection_mode](const std::string& value) {
        if (auto mode = detection_mode(debug_options, value)) {
          debug_options->set_xla_detect_unstable_reductions(mode.value());
          return true;
        }
        return false;
      };

  auto setter_for_xla_detect_unstable_reductions_post_optimizations =
      [debug_options, detection_mode](const std::string& value) {
        if (auto mode = detection_mode(debug_options, value)) {
          debug_options->set_xla_detect_unstable_reductions_post_optimizations(
              mode.value());
          return true;
        }
        return false;
      };

  // Custom "sub-parser" for
  // xla_gpu_experimental_thunk_buffer_debug_filter_by_thunk_id_ranges.
  auto setter_for_thunk_buffer_debug_filter_by_thunk_id =
      [debug_options](const std::string& value) {
        for (const auto& range_str : absl::StrSplit(value, ',')) {
          IntRangeInclusive* range =
              debug_options
                  ->mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
                  ->add_thunk_id_ranges();
          if (!details::ParseIntRangeInclusive(range_str, *range)) {
            return false;
          }
        }
        return true;
      };

  // Custom "sub-parser" for
  // xla_gpu_experimental_thunk_buffer_debug_filter_by_profile_annotation_re.
  auto setter_for_thunk_buffer_debug_filter_by_profile_annotation =
      [debug_options](const std::string& value) {
        for (const auto& regex_str : absl::StrSplit(value, ',')) {
          debug_options
              ->mutable_xla_gpu_experimental_thunk_buffer_debug_filter()
              ->add_profile_annotation_regexes(regex_str);
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
      "xla_cpu_enable_platform_dependent_math",
      bool_setter_for(
          &DebugOptions::set_xla_cpu_enable_platform_dependent_math),
      debug_options->xla_cpu_enable_platform_dependent_math(),
      "Enable platform dependent math in the CPU compiler; this may "
      "produce faster code at the expense of consistent results across CPUs."));
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
      "Enable fast floating point min/max lowering that might not propagate "
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
      "xla_test_add_command_buffer_mode",
      bool_setter_for(&DebugOptions::set_xla_test_add_command_buffer_mode),
      debug_options->xla_test_add_command_buffer_mode(),
      "If true, the test launched with ClientLibraryTestBase will use command "
      "buffer to execute the computation."));
  flag_list->push_back(tsl::Flag(
      "xla_hlo_profile", bool_setter_for(&DebugOptions::set_xla_hlo_profile),
      debug_options->xla_hlo_profile(),
      "Instrument the computation to collect per-HLO cycle counts"));
  flag_list->push_back(tsl::Flag(
      "xla_backend_extra_options", setter_for_xla_backend_extra_options, "",
      "Extra options to pass to a backend; comma-separated list of 'key=val' "
      "strings (=val may be omitted); no whitespace around commas."));
  flag_list->push_back(
      tsl::Flag("xla_cpu_use_onednn",
                bool_setter_for(&DebugOptions::set_xla_cpu_use_onednn),
                debug_options->xla_cpu_use_onednn(),
                "Call oneDNN thunks for matmul and convolution fusions in the "
                "CPU backend."));
  flag_list->push_back(
      tsl::Flag("xla_cpu_experimental_onednn_custom_call",
                bool_setter_for(
                    &DebugOptions::set_xla_cpu_experimental_onednn_custom_call),
                debug_options->xla_cpu_experimental_onednn_custom_call(),
                "Call oneDNN custom call thunks in the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_experimental_onednn_fusion_type",
      SetterForRepeatedEnum<DebugOptions::LibraryFusionType>(
          "xla_cpu_experimental_onednn_fusion_type",
          /*enum_prefix=*/"LIBRARY_FUSION_TYPE_",
          &DebugOptions::LibraryFusionType_Parse,
          debug_options->mutable_xla_cpu_experimental_onednn_fusion_type()),
      "",
      "Comma-separated list of oneDNN fusion types to be enabled; "
      "no whitespace around commas. Two ways to pass values:\n"
      "  1. Exact type names. This overwrites the default setting.\n"
      "  2. '+' or '-' prefix: This adds or removes a fusion type "
      "from the default list. Cannot be mixed with the overwrite "
      "mode. Every item must have the sign prefix.\n"
      "Available fusion types: dot, eltwise, and reduce.\n"
      "The default list is currently empty."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_use_acl", bool_setter_for(&DebugOptions::set_xla_cpu_use_acl),
      debug_options->xla_cpu_use_acl(),
      "Generate calls to ACL (Arm Compute Library) in the CPU backend."));
  flag_list->push_back(
      tsl::Flag("xla_cpu_use_fusion_emitters",
                bool_setter_for(&DebugOptions::set_xla_cpu_use_fusion_emitters),
                debug_options->xla_cpu_use_fusion_emitters(),
                "Use fusion emitters for code generation in the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_use_thunk_runtime",
      [](bool) {
        LOG(WARNING) << "\"xla_cpu_use_thunk_runtime\" is no longer supported "
                        "and will be removed in a future release.";
        return true;
      },
      true, "Deprecated."));
  flag_list->push_back(
      tsl::Flag("xla_cpu_use_xnnpack",
                bool_setter_for(&DebugOptions::set_xla_cpu_use_xnnpack),
                debug_options->xla_cpu_use_xnnpack(),
                "Use XNNPACK for supported operations."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_experimental_xnn_fusion_type",
      SetterForRepeatedEnum<DebugOptions::LibraryFusionType>(
          "xla_cpu_experimental_xnn_fusion_type",
          /*enum_prefix=*/"LIBRARY_FUSION_TYPE_",
          &DebugOptions::LibraryFusionType_Parse,
          debug_options->mutable_xla_cpu_experimental_xnn_fusion_type()),
      "",
      "Comma-separated list of XNN fusion types to be enabled; "
      "no whitespace around commas. Two ways to pass values:\n"
      "  1. Exact type names. This overwrites the default setting.\n"
      "  2. '+' or '-' prefix: This adds or removes a fusion type "
      "from the default list. Cannot be mixed with the overwrite "
      "mode. Every item must have the sign prefix.\n"
      "Available fusion types: dot, eltwise, and reduce.\n"
      "The default list is currently empty."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_experimental_ynn_fusion_type",
      SetterForRepeatedEnum<DebugOptions::LibraryFusionType>(
          "xla_cpu_experimental_ynn_fusion_type",
          /*enum_prefix=*/"LIBRARY_FUSION_TYPE_",
          &DebugOptions::LibraryFusionType_Parse,
          debug_options->mutable_xla_cpu_experimental_ynn_fusion_type()),
      "",
      "Comma-separated list of YNN fusion types to be enabled; "
      "no whitespace around commas. Two ways to pass values:\n"
      "  1. Exact type names. This overwrites the default setting.\n"
      "  2. '+' or '-' prefix: This adds or removes a fusion type "
      "from the default list. Cannot be mixed with the overwrite "
      "mode. Every item must have the sign prefix.\n"
      "The default list is currently empty."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_experimental_xnn_graph_fusion_mode",
      setter_for_xla_cpu_experimental_xnn_graph_fusion_mode,
      DebugOptions::XnnGraphFusionMode_Name(
          debug_options->xla_cpu_experimental_xnn_graph_fusion_mode()),
      "Controls XnnGraphFusion pass. "
      "  `XNN_GRAPH_FUSION_MODE_DISABLED` - default value,\n"
      "  `XNN_GRAPH_FUSION_MODE_GREEDY` - greedy extraction of "
      "XNNPACK-compatible subgraphs starting from root instructions,\n"
      "  `XNN_GRAPH_FUSION_MODE_GREEDY_SLINKY` - same as GREEDY plus "
      "operations that are only supported with slinky,"
      "  `XNN_GRAPH_FUSION_MODE_BYPASS_COST_MODEL` - test-only value for "
      "disabling XNNPACK cost models."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_parallel_codegen_split_count",
      int32_setter_for(&DebugOptions::set_xla_cpu_parallel_codegen_split_count),
      debug_options->xla_cpu_parallel_codegen_split_count(),
      "Split LLVM module into at most this many parts before codegen to enable "
      "parallel compilation for the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_copy_insertion_use_region_analysis",
      bool_setter_for(
          &DebugOptions::set_xla_cpu_copy_insertion_use_region_analysis),
      debug_options->xla_cpu_copy_insertion_use_region_analysis(),
      "Use region based analysis in copy insertion pass."));
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
      "Preferred vector width for the XLA:CPU LLVM backend."));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_max_isa",
      uppercase_string_setter_for(&DebugOptions::set_xla_cpu_max_isa),
      debug_options->xla_cpu_max_isa(),
      "Maximum ISA that XLA:CPU LLVM backend will codegen, i.e., it will not "
      "use newer instructions. Available values: SSE4_2, AVX, AVX2, AVX512, "
      "AVX512_VNNI, AVX512_BF16, AMX, and AMX_FP16. (`AMX` will enable both "
      "`AMX_BF16` and `AMX_INT8` instructions.)"));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_emitter_verification_level",
      int32_setter_for(&DebugOptions::set_xla_cpu_emitter_verification_level),
      debug_options->xla_cpu_emitter_verification_level(),
      "Sets how often we verify the emitted modules. Higher levels mean more "
      "frequent verification. Currently supported: 0, 1."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_crash_on_verification_failures",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_crash_on_verification_failures),
      debug_options->xla_gpu_crash_on_verification_failures(),
      "Crashes the program on extra verification failures, e.g. cuDNN cross "
      "checking failures"));
  flag_list->push_back(tsl::Flag("xla_gpu_strict_conv_algorithm_picker",
                                 noop_flag_setter<bool>, false,
                                 "[Deprecated, do not use]."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_autotune_level",
      int32_setter_for(&DebugOptions::set_xla_gpu_autotune_level),
      debug_options->xla_gpu_autotune_level(),
      "[Stable] Set GEMM and Convolution auto-tuning level. 0 = off; 1 = on; "
      "2 = on+init; 3 = on+init+reinit; 4 = on+init+reinit+check; "
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
  flag_list->push_back(
      tsl::Flag("xla_gpu_generate_debug_info",
                bool_setter_for(&DebugOptions::set_xla_gpu_generate_debug_info),
                debug_options->xla_gpu_generate_debug_info(),
                "Generate debug info for codegened CUDA kernels."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_generate_line_info",
                bool_setter_for(&DebugOptions::set_xla_gpu_generate_line_info),
                debug_options->xla_gpu_generate_line_info(),
                "Generate line info for codegened CUDA kernels."));
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
      "TEST_UNDECLARED_OUTPUTS_DIR. One or more --xla_dump_hlo_as_* flags can "
      "be set to specify the formats of the dumps. For example, if both "
      "--xla_dump_hlo_as_text and --xla_dump_hlo_as_proto are set, then the "
      "HLO modules will be dumped as text and as protos."));
  flag_list->push_back(tsl::Flag(
      "xla_flags_reset", bool_setter_for(&DebugOptions::set_xla_flags_reset),
      debug_options->xla_flags_reset(),
      "Whether to reset XLA_FLAGS next time to parse."));
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
      tsl::Flag("xla_gpu_experimental_dump_fdo_profiles",
                bool_setter_for(
                    &DebugOptions::set_xla_gpu_experimental_dump_fdo_profiles),
                debug_options->xla_gpu_experimental_dump_fdo_profiles(),
                "Dumps FDO profiles as text to the directory specified "
                "by --xla_dump_to."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_dump_gpu_executable",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_dump_gpu_executable),
      debug_options->xla_gpu_experimental_dump_gpu_executable(),
      "Dump the serialized GPU executables to 'gpu_executable_proto' suffixed "
      "files, in the directory specified by `xla_dump_to`. No-op if "
      "`xla_dump_to` isn't set, or during autotuning compilations."));
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
  flag_list->push_back(tsl::Flag(
      "xla_dump_emitter_re",
      string_setter_for(&DebugOptions::set_xla_dump_emitter_re),
      debug_options->xla_dump_emitter_re(),
      "If specified, dumps debug logs (e.g. IR like LLVM or MLIR) before and "
      "after emitters which match this regular expression, in addition to "
      "dumping at the very beginning and end of compilation."));
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

  flag_list->push_back(tsl::Flag(
      "xla_gpu_default_to_alg_dot_bf16_bf16_f32",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_default_to_alg_dot_bf16_bf16_f32),
      debug_options->xla_gpu_default_to_alg_dot_bf16_bf16_f32(),
      "Use the dot precision algorithm `ALG_DOT_BF16_BF16_F32 by default for "
      "f32 dots."));
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
      "[Stable] Size threshold (in bytes) for the GPU all-reduce combiner."));
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
      "[Stable] Size threshold (in bytes) for the GPU reduce-scatter "
      "combiner."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_collective_permute_combine_threshold_bytes",
      int64_setter_for(
          &DebugOptions::
              set_xla_gpu_collective_permute_combine_threshold_bytes),
      debug_options->xla_gpu_collective_permute_combine_threshold_bytes(),
      "Size threshold (in bytes) for the GPU collective-permute combiner."));
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
      "xla_dump_hlo_unoptimized_snapshots",
      bool_setter_for(&DebugOptions::set_xla_dump_hlo_unoptimized_snapshots),
      debug_options->xla_dump_hlo_unoptimized_snapshots(),
      "Every time an HLO module is run, dumps an HloUnoptimizedSnapshot to the "
      "directory specified by --xla_dump_to."));
  flag_list->push_back(tsl::Flag("xla_gpu_enable_cudnn_fmha",
                                 noop_flag_setter<bool>, false,
                                 "[Deprecated, do not use]"));
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
      "xla_gpu_enable_command_buffer",
      SetterForRepeatedEnum<DebugOptions::CommandBufferCmdType>(
          "xla_gpu_enable_command_buffer",
          /*enum_prefix=*/"", &DebugOptions::CommandBufferCmdType_Parse,
          debug_options->mutable_xla_gpu_enable_command_buffer()),
      command_types_to_string(debug_options->xla_gpu_enable_command_buffer()),
      "The types of the commands that are recorded into command buffers. It"
      " can either be a list of command types or a list of command types with"
      " + and - as prefix, which indicate adding or removing a command type"
      " to/from the default list."));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_graph_min_graph_size",
      int32_setter_for(&DebugOptions::set_xla_gpu_graph_min_graph_size),
      debug_options->xla_gpu_graph_min_graph_size(),
      "Capture a region as a function to be launched as cuda graph if the "
      "number of moved instructions reaches this threshold."));

  flag_list->push_back(
      tsl::Flag("xla_gpu_graph_enable_concurrent_region",
                setter_for_xla_gpu_graph_enable_concurrent_region,
                debug_options->xla_gpu_graph_enable_concurrent_region(),
                "[Deprecated, do not use]"));

  flag_list->push_back(
      tsl::Flag("xla_gpu_command_buffer_scheduling_mode",
                setter_for_xla_gpu_command_buffer_scheduling_mode,
                DebugOptions::CommandBufferSchedulingMode_Name(
                    debug_options->xla_gpu_command_buffer_scheduling_mode()),
                "The command buffer scheduling mode for XLA:GPU."));

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
  flag_list->push_back(
      tsl::Flag("xla_dump_full_hlo_config",
                bool_setter_for(&DebugOptions::set_xla_dump_full_hlo_config),
                debug_options->xla_dump_full_hlo_config(),
                "Enable dumping the full HloModuleConfig proto."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_dynamic_slice_fusion",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_dynamic_slice_fusion),
      debug_options->xla_gpu_enable_dynamic_slice_fusion(),
      "[Stable] Whether to enable address computation fusion to optimize "
      "dynamic-slice and dynamic-update-slice operations."));
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
      "xla_gpu_experimental_enable_nccl_symmetric_buffers",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_nccl_symmetric_buffers),
      debug_options->xla_gpu_experimental_enable_nccl_symmetric_buffers(),
      "Enables NCCL symmetric buffer registration."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_aot_compiled_thunks",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_aot_compiled_thunks),
      debug_options->xla_gpu_experimental_aot_compiled_thunks(),
      "Enables an Ahead-of-Time (AOT) compilation flow where the compiled "
      "binary includes the generated Thunks. In contrast, the legacy flow "
      "only compiles up to the HLO optimization stage, before Thunk "
      "generation."));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_nvshmem",
      bool_setter_for(&DebugOptions::set_xla_gpu_experimental_enable_nvshmem),
      debug_options->xla_gpu_experimental_enable_nvshmem(),
      "Enables NVSHMEM."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_temp_buffer_use_separate_color",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_temp_buffer_use_separate_color),
      debug_options->xla_gpu_temp_buffer_use_separate_color(),
      "Enables temp User Buffer Registration. Enable this flag will use a "
      "separate cuda async memory allocator to allocate temp buffer, this will "
      "allocate temp buffer to the fixed address on every iteration"));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_require_exclusive_lock",
      bool_setter_for(&DebugOptions::set_xla_gpu_require_exclusive_lock),
      debug_options->xla_gpu_require_exclusive_lock(),
      "if true, running gpu executable will require exclusive lock on gpu, so "
      "there is no multi thread conlicts on gpu. this can enable some "
      "optimizations that reduce the cost of resource management, e.g, "
      "command "
      "buffer update to ensure correctness when running in multi thread "
      "mode."));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_nccl_comm_splitting",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_nccl_comm_splitting),
      debug_options->xla_gpu_enable_nccl_comm_splitting(),
      "Enables NCCL communicator splitting which allows sharing NCCL resources "
      "between different NCCL cliques."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_nccl_init_max_rank_per_root_ratio",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_nccl_init_max_rank_per_root_ratio),
      debug_options->xla_gpu_nccl_init_max_rank_per_root_ratio(),
      "Maximum number of ranks associated with a root rank to initialize a "
      "NCCL communicator via ncclCommInitRankScalable. "
      "A value of zero will lead to a single root."));
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
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_latency_hiding_scheduler",
                bool_setter_for(
                    &DebugOptions::set_xla_gpu_enable_latency_hiding_scheduler),
                debug_options->xla_gpu_enable_latency_hiding_scheduler(),
                "[Stable] Enable latency-hiding scheduler for XLA:GPU"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_analytical_latency_estimator",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_analytical_latency_estimator),
      debug_options->xla_gpu_enable_analytical_latency_estimator(),
      "Enable analytical latency estimator for latency-hiding scheduler for "
      "XLA:GPU"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_analytical_sol_latency_estimator",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_analytical_sol_latency_estimator),
      debug_options->xla_gpu_enable_analytical_sol_latency_estimator(),
      "Enable analytical Speed-of-Light latency estimator for latency-hiding "
      "scheduler for XLA:GPU, must be used without "
      "xla_gpu_enable_analytical_latency_estimator. It can also benefit from "
      "user-passed options in xla_gpu_analytical_latency_estimator_options"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_analytical_latency_estimator_options",
      setter_for_xla_gpu_analytical_latency_estimator_options, "",
      "Extra platform-specific options to improve analytical latency "
      "estimator precision; comma-separated list of 'key=val' "
      "strings (=val may be omitted); no whitespace around commas."
      "Available options: "
      "--xla_gpu_analytical_latency_estimator_options='nccl_op_launch_us=55,"
      "nic_speed_gbps=40,chunk_prep_us=1,rtt_us=2,gpus_per_node=4,"
      "chunk_size_bytes=1024'"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_pgle_profile_file_or_directory_path",
      string_setter_for(
          &DebugOptions::set_xla_gpu_pgle_profile_file_or_directory_path),
      debug_options->xla_gpu_pgle_profile_file_or_directory_path(),
      "Directory or file for PGLE profiles in XLA:GPU"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_memory_limit_slop_factor",
      int32_setter_for(&DebugOptions::set_xla_gpu_memory_limit_slop_factor),
      debug_options->xla_gpu_memory_limit_slop_factor(),
      "Slop factor for memory limits in XLA:GPU. This flag serves as a "
      "multiplier "
      "applied to the total available memory, creating a threshold that guides "
      "the "
      "Latency Hiding Scheduler (LHS) in balancing memory reduction and "
      "latency "
      "hiding optimizations. This factor effectively establishes a memory "
      "limit "
      "for compiler passes, determining when the scheduler should prioritize: "
      "  1. Memory reduction: When memory usage approaches or exceeds the "
      "calculated "
      "     threshold. "
      "  2. Latency hiding: When memory usage is below the threshold, allowing "
      "for "
      "     more aggressive optimizations that may temporarily increase memory "
      "usage "
      "     but improve overall performance. "
      "By adjusting this factor, users can fine-tune the trade-off between "
      "memory "
      "efficiency and performance optimizations. The default value is 95."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_highest_priority_async_stream",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_highest_priority_async_stream),
      debug_options->xla_gpu_enable_highest_priority_async_stream(),
      "Enable async stream to have the highest priority."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pipelined_all_reduce",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_pipelined_all_reduce),
      debug_options->xla_gpu_enable_pipelined_all_reduce(),
      "[Stable] Enable pipelinling of all-reduce instructions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pipelined_all_gather",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_pipelined_all_gather),
      debug_options->xla_gpu_enable_pipelined_all_gather(),
      "[Stable] Enable pipelinling of all-gather instructions."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_pipelined_reduce_scatter",
                bool_setter_for(
                    &DebugOptions::set_xla_gpu_enable_pipelined_reduce_scatter),
                debug_options->xla_gpu_enable_pipelined_reduce_scatter(),
                "[Stable] Enable pipelinling of reduce-scatter instructions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pipelined_host_offloading",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_pipelined_host_offloading),
      debug_options->xla_gpu_enable_pipelined_host_offloading(),
      "Enable pipelining of host offloading instructions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_pipelined_p2p",
      bool_setter_for(&DebugOptions::set_xla_gpu_enable_pipelined_p2p),
      debug_options->xla_gpu_enable_pipelined_p2p(),
      "Enable pipelinling of P2P instructions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_collective_permute_decomposer_threshold",
      int64_setter_for(
          &DebugOptions::set_xla_gpu_collective_permute_decomposer_threshold),
      debug_options->xla_gpu_collective_permute_decomposer_threshold(),
      "[Stable] Collective permute decomposer threshold."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_pipeline_parallelism_opt_level",
      setter_for_xla_gpu_experimental_pipeline_parallelism_opt_level,
      DebugOptions::PipelineParallelismOptLevel_Name(
          debug_options->xla_gpu_experimental_pipeline_parallelism_opt_level()),
      "Experimental optimizations for SPMD-based pipeline parallelism on "
      "GPU."));
  flag_list->push_back(tsl::Flag(
      "xla_enable_enzyme_comms_opt",
      bool_setter_for(&DebugOptions::set_xla_enable_enzyme_comms_opt),
      debug_options->xla_enable_enzyme_comms_opt(),
      "Enable communication optimization patterns specified in Enzyme. More "
      "details in http://shortn/_jXJ2VFoyMN."));
  flag_list->push_back(tsl::Flag(
      "xla_partitioning_algorithm", setter_for_xla_partitioning_algorithm,
      DebugOptions::PartitioningAlgorithm_Name(
          debug_options->xla_partitioning_algorithm()),
      "The partitioning algorithm to be used in the PartitionAssignment pass"));
  flag_list->push_back(
      tsl::Flag("xla_gpu_enable_triton_gemm",
                bool_setter_for(&DebugOptions::set_xla_gpu_enable_triton_gemm),
                debug_options->xla_gpu_enable_triton_gemm(),
                "[Stable] Whether to use Triton-based matrix multiplication."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsupported_enable_triton_multi_output_fusion",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_unsupported_enable_triton_multi_output_fusion),
      debug_options->xla_gpu_unsupported_enable_triton_multi_output_fusion(),
      "Enable Triton multi-output fusions."));
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
      "[Stable] Search for Triton GEMM tilings exhaustively during autotuning. "
      "This increases the compile time."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_subchannel_dequantisation_fusion",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_subchannel_dequantisation_fusion),
      debug_options
          ->xla_gpu_experimental_enable_subchannel_dequantisation_fusion(),
      "Enable fusion for the subchannel dequantisation sequences like "
      "[x,z]param -> [x,y,z]broadcast -> [x*y,z]bitcast -> multiply -> dot. "
      "Performance can be worse, because some block sizes / split-k > 1 "
      "is not considered for subchannel dequant fusions."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_triton_heroless_priority_fusion",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_triton_heroless_priority_fusion),
      debug_options
          ->xla_gpu_experimental_enable_triton_heroless_priority_fusion(),
      "Enable heroless Triton fusions in the PriorityFusion pass. The pass "
      "will try to make Triton fusions first and foremost where it is "
      "possible."));
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
      "xla_gpu_command_buffer_unroll_loops",
      bool_setter_for(&DebugOptions::set_xla_gpu_command_buffer_unroll_loops),
      debug_options->xla_gpu_command_buffer_unroll_loops(),
      "During command buffer lowering, unroll the loop command if loop has "
      "known loop count."));
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
  flag_list->push_back(tsl::Flag("xla_gpu_enable_nccl_clique_optimization",
                                 noop_flag_setter<bool>, false,
                                 "[Deprecated, do not use]."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_cublas_fallback",
                bool_setter_for(&DebugOptions::set_xla_gpu_cublas_fallback),
                debug_options->xla_gpu_cublas_fallback(),
                "[Stable] Whether to allow GEMM fusion autotuning to fall back "
                "to cuBLAS when it is faster than Triton."));
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
      "[Stable] Enable double buffering for while loop"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_filter_kernels_spilling_registers_on_autotuning",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_filter_kernels_spilling_registers_on_autotuning),
      debug_options->xla_gpu_filter_kernels_spilling_registers_on_autotuning(),
      "Filter out kernels that spill registers during autotuning. Default is "
      "true."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_fail_ptx_compilation_on_register_spilling",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_fail_ptx_compilation_on_register_spilling),
      debug_options->xla_gpu_fail_ptx_compilation_on_register_spilling(),
      "Fails the PTX compilation if a kernel spills registers."));
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
      "xla_gpu_operand_bytes_threshold_for_windowed_einsum",
      int64_setter_for(
          &DebugOptions::
              set_xla_gpu_operand_bytes_threshold_for_windowed_einsum),
      debug_options->xla_gpu_operand_bytes_threshold_for_windowed_einsum(),
      "This controls whether to enable windowed einsum (collective matmul) "
      "based on the sum of sizes of 2 operands if set >= 0."
      "If set >= 0, xla_gpu_threshold_for_windowed_einsum_mib is ignored."
      "Default is -1"));

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
      "xla_gpu_enable_libnvjitlink",
      [debug_options](bool enabled) {
        debug_options->set_xla_gpu_libnvjitlink_mode(
            enabled ? DebugOptions::LIB_NV_JIT_LINK_MODE_ENABLED
                    : DebugOptions::LIB_NV_JIT_LINK_MODE_DISABLED);
        return true;
      },
      stream_executor::IsLibNvJitLinkSupported(),
      "Use libnvjitlink for PTX-to-GPU-assembly compilation instead of "
      "calling ptxas."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_nccl_async_execution",
      bool_setter_for(&DebugOptions::set_xla_gpu_nccl_async_execution),
      debug_options->xla_gpu_nccl_async_execution(),
      "Whether to use asynchronous execution for NCCL communicators"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_nccl_blocking_communicators",
      bool_setter_for(&DebugOptions::set_xla_gpu_nccl_blocking_communicators),
      debug_options->xla_gpu_nccl_blocking_communicators(),
      "Whether to use non-blocking NCCL communicators"));
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
      "xla_gpu_use_embeded_device_lib",
      bool_setter_for(&DebugOptions::set_xla_gpu_use_embeded_device_lib),
      debug_options->xla_gpu_use_embeded_device_lib(),
      "Whether to use embeded bitcode library in codegen."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_use_memcpy_local_p2p",
      bool_setter_for(&DebugOptions::set_xla_gpu_use_memcpy_local_p2p),
      debug_options->xla_gpu_use_memcpy_local_p2p(),
      "Whether to use memcpy for local p2p communication."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_use_inprocess_lld",
                bool_setter_for(&DebugOptions::set_xla_gpu_use_inprocess_lld),
                debug_options->xla_gpu_use_inprocess_lld(),
                "Whether to use lld as a library for the linking."));
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
      "xla_gpu_experimental_autotuner_cache_dir",
      string_setter_for(
          &DebugOptions::set_xla_gpu_experimental_autotuner_cache_dir),
      debug_options->xla_gpu_experimental_autotuner_cache_dir(),
      "Experimental: Specify the directory to read/write autotuner cache to."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_autotune_backends",
      SetterForRepeatedEnum<DebugOptions::AutotuneBackend>(
          "xla_gpu_experimental_autotune_backends",
          /*enum_prefix=*/"AUTOTUNE_BACKEND_",
          &DebugOptions::AutotuneBackend_Parse,
          debug_options->mutable_xla_gpu_experimental_autotune_backends()),
      autotune_backends_to_string(
          debug_options->xla_gpu_experimental_autotune_backends()),
      "Backends to enable for autotuning. Comma-separated (no spaces). "
      "Examples:\n"
      "  'cudnn,triton' (overwrites defaults)\n"
      "  '+cudnn,-cublas' (adds/removes from defaults)\n"
      "Available: cudnn, triton, cublas, cublaslt."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_all_fusions_with_triton",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_all_fusions_with_triton),
      debug_options->xla_gpu_experimental_all_fusions_with_triton(),
      "Experimental: If true, autotune all fusions with block level emitter."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_gemm_autotuner_override_file",
      string_setter_for(
          &DebugOptions::set_xla_gpu_gemm_autotuner_override_file),
      debug_options->xla_gpu_gemm_autotuner_override_file(),
      "A textproto file to override autotune results. See also "
      "`xla_gpu_override_gemm_autotuner` to override with a single config."));
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
      "xla_gpu_pgle_accuracy_checker", setter_for_xla_gpu_pgle_accuracy_checker,
      DebugOptions::PGLEStrictnessLevel_Name(
          debug_options->xla_gpu_pgle_accuracy_checker()),
      "If an FDO profile is specified and latency hiding scheduler encounters "
      "missing instructions in the profile, then the compilation will halt "
      "(ERROR), or a warning will be emitted (WARN), or the checker is "
      "disabled (OFF)"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_executable_embed_debug_info",
      bool_setter_for(&DebugOptions::set_xla_gpu_executable_embed_debug_info),
      debug_options->xla_gpu_executable_embed_debug_info(),
      "Add debug information to the executable such as HLO module, asm_text "
      "etc."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_executable_warn_stuck_timeout",
      int32_setter_for(
          &DebugOptions::set_xla_gpu_executable_warn_stuck_timeout_seconds),
      debug_options->xla_gpu_executable_warn_stuck_timeout_seconds(),
      "Set timeout for Rendezvous stuck warning"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_executable_terminate_timeout",
      int32_setter_for(
          &DebugOptions::set_xla_gpu_executable_terminate_timeout_seconds),
      debug_options->xla_gpu_executable_terminate_timeout_seconds(),
      "Set timeout for Rendezvous termination"));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_first_collective_call_warn_stuck_timeout_seconds",
      int32_setter_for(
          &DebugOptions::
              set_xla_gpu_first_collective_call_warn_stuck_timeout_seconds),
      debug_options->xla_gpu_first_collective_call_warn_stuck_timeout_seconds(),
      "Set timeout for First Collective Call Rendezvous stuck warning"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_first_collective_call_terminate_timeout_seconds",
      int32_setter_for(
          &DebugOptions::
              set_xla_gpu_first_collective_call_terminate_timeout_seconds),
      debug_options->xla_gpu_first_collective_call_terminate_timeout_seconds(),
      "Set timeout for First Collective Call Rendezvous termination"));

  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_disable_binary_libraries",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_disable_binary_libraries),
      debug_options->xla_gpu_experimental_disable_binary_libraries(),
      "Disable XLA GPU passes that depend on non-open source binary "
      "libraries"));
  flag_list->push_back(
      tsl::Flag("xla_ignore_channel_id",
                bool_setter_for(&DebugOptions::set_xla_ignore_channel_id),
                debug_options->xla_ignore_channel_id(),
                "Ignore channel ids for collective operations."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_dot_merger_threshold_mb",
      int32_setter_for(&DebugOptions::set_xla_gpu_dot_merger_threshold_mb),
      debug_options->xla_gpu_dot_merger_threshold_mb(),
      "[Stable] Dot merger pass threshold to be set in MB."));
  flag_list->push_back(
      tsl::Flag("xla_enable_fast_math",
                bool_setter_for(&DebugOptions::set_xla_enable_fast_math),
                debug_options->xla_enable_fast_math(),
                "Enable optimizations that assume finite math, i.e., no NaN."));
  flag_list->push_back(
      tsl::Flag("xla_gpu_experimental_stream_annotation",
                bool_setter_for(
                    &DebugOptions::set_xla_gpu_experimental_stream_annotation),
                debug_options->xla_gpu_experimental_stream_annotation(),
                "Enable the experimental explicit stream annotation support. "
                "If false, the annotations are ignored."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_parallel_collective_overlap_limit",
      int32_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_parallel_collective_overlap_limit),
      debug_options->xla_gpu_experimental_parallel_collective_overlap_limit(),
      "This controls how many in-flight collectives "
      "latency hiding scheduler can schedule."));
  flag_list->push_back(tsl::Flag(
      "xla_pjrt_allow_auto_layout_in_hlo",
      bool_setter_for(&DebugOptions::set_xla_pjrt_allow_auto_layout_in_hlo),
      debug_options->xla_pjrt_allow_auto_layout_in_hlo(),
      "Experimental: Make unset entry computation layout mean auto layout "
      "instead of default layout in HLO when run through PjRT. In other cases "
      "(StableHLO or non-PjRT) the auto layout is already used."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_enable_scatter_determinism_expander",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_enable_scatter_determinism_expander),
      debug_options->xla_gpu_enable_scatter_determinism_expander(),
      "Makes scatter ops deterministic and enables the use of the scatter "
      "determinism expander. This is an optimized pass that rewrites scatter "
      "operations to ensure deterministic behavior with high performance. If "
      "the optimization pass does not support a particular scater op, it will "
      "be made deterministic using a slower implementation. "
      "Note that even when this flag is disabled, scatter operations may still "
      "be deterministic, with the slower implemntation. This is the case when "
      "'xla_gpu_exclude_nondeterministic_ops' is enabled."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsupported_enable_all_reduce_decomposer",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_unsupported_enable_all_reduce_decomposer),
      debug_options->xla_gpu_unsupported_enable_all_reduce_decomposer(),
      "Internal: Enable the AllReduceDecomposer, an unsupported pass that "
      "rewrites small all-reduce operations as a sequence of all-gather and "
      "reduce operations."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsupported_enable_ragged_all_to_all_decomposer",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_unsupported_enable_ragged_all_to_all_decomposer),
      debug_options->xla_gpu_unsupported_enable_ragged_all_to_all_decomposer(),
      "Internal: Enable the RaggedAllToAllDecomposer, an experimental pass "
      "that rewrites ragged-all-to-all as a dense all-to-all operation."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer),  // NOLINT
      debug_options
          ->xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer(),  // NOLINT
      "Internal: Enable the RaggedAllToAllMultiHostDecomposer, an experimental "
      "pass to decompose ragged-all-to-all operation in intra-host and "
      "inter-host parts."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsupported_disable_nested_gemm_fusions",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_unsupported_disable_nested_gemm_fusions),
      debug_options->xla_gpu_unsupported_disable_nested_gemm_fusions(),
      "Enable the new pipeline that does not use nesting at HLO level"));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsupported_override_fast_interconnect_slice_size",
      int64_setter_for(
          &DebugOptions::
              set_xla_gpu_unsupported_override_fast_interconnect_slice_size),
      debug_options
          ->xla_gpu_unsupported_override_fast_interconnect_slice_size(),
      "Internal: Override the number of devices in the fast interconnect "
      "domain. Default is 0, which means the number of devices is not "
      "overridden."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsupported_use_all_reduce_one_shot_kernel",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel),
      debug_options->xla_gpu_unsupported_use_all_reduce_one_shot_kernel(),
      "Internal: Enable the one-shot kernel for single-host all-reduce "
      "operations."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel),
      debug_options
          ->xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(),
      "Internal: Enable the one-shot kernel for single-host ragged-all-to-all "
      "operations."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_alltoall_windowed_einsum",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_alltoall_windowed_einsum),
      debug_options->xla_gpu_experimental_enable_alltoall_windowed_einsum(),
      "Enable windowed einsum rewrite for all-to-all+gemm pattern, "
      "This optimization slices the all-to-all into smaller all-to-alls."
      "It is an experimental feature."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_pack_dot_operands_along_k_dimension",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_pack_dot_operands_along_k_dimension),
      debug_options->xla_gpu_experimental_pack_dot_operands_along_k_dimension(),
      "For sub-byte dot operands, layout them along contracting dimensions."));
  flag_list->push_back(tsl::Flag(
      "xla_unsupported_crash_on_hlo_pass_fix_max_iterations",
      bool_setter_for(
          &DebugOptions::
              set_xla_unsupported_crash_on_hlo_pass_fix_max_iterations),
      debug_options->xla_unsupported_crash_on_hlo_pass_fix_max_iterations(),
      "Crash if HloPassFix can not converge after a fixed number of "
      "iterations."));
  flag_list->push_back(tsl::Flag(
      "xla_hlo_pass_fix_detect_cycles",
      bool_setter_for(&DebugOptions::set_xla_hlo_pass_fix_detect_cycles),
      debug_options->xla_hlo_pass_fix_detect_cycles(),
      "Perform hash-based cycle detection in fixed-point loops."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_heuristic_collective_combining",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_heuristic_collective_combining),
      debug_options
          ->xla_gpu_experimental_enable_heuristic_collective_combining(),
      "Enable heuristic based collective combining."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_collective_cse_distance_threshold",
      int64_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_collective_cse_distance_threshold),
      debug_options->xla_gpu_experimental_collective_cse_distance_threshold(),
      "Set distance threshold for Collective CSE."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_collective_perf_table_path",
      string_setter_for(
          &DebugOptions::set_xla_gpu_experimental_collective_perf_table_path),
      debug_options->xla_gpu_experimental_collective_perf_table_path(),
      "If non empty will interpret this variable as a path for performance "
      "tables for collectives. Expects `xla.gpu.DeviceHloInstructionProfiles` "
      "proto."));
  flag_list->push_back(tsl::Flag(
      "xla_unsupported_crash_on_hlo_pass_noop_change",
      bool_setter_for(
          &DebugOptions::set_xla_unsupported_crash_on_hlo_pass_noop_change),
      debug_options->xla_unsupported_crash_on_hlo_pass_noop_change(),
      "Crash if a pass reports that it did change the HLO but in fact it "
      "did not."));
  flag_list->push_back(tsl::Flag(
      "xla_unsupported_crash_on_hlo_pass_silent_hlo_change",
      bool_setter_for(
          &DebugOptions::
              set_xla_unsupported_crash_on_hlo_pass_silent_hlo_change),
      debug_options->xla_unsupported_crash_on_hlo_pass_silent_hlo_change(),
      "Crash if a pass reports that it did not change the HLO but in fact it "
      "did."));
  flag_list->push_back(tsl::Flag(
      "xla_disable_automatic_host_compute_offload",
      bool_setter_for(
          &DebugOptions::set_xla_disable_automatic_host_compute_offload),
      debug_options->xla_disable_automatic_host_compute_offload(),
      "Return an error if HostOffloader would have automatically offloaded some"
      " compute to the host."));
  flag_list->push_back(tsl::Flag(
      "xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled",
      bool_setter_for(
          &DebugOptions::
              set_xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled),  // NOLINT
      debug_options
          ->xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled(),
      "Allow host-to-host copy when automatic host compute offload is "
      "disabled, i.e. when xla_disable_automatic_host_compute_offload is "
      "set."));
  flag_list->push_back(tsl::Flag(
      "xla_enable_scoped_logging_timers",
      bool_setter_for(&DebugOptions::set_xla_enable_scoped_logging_timers),
      debug_options->xla_enable_scoped_logging_timers(),
      "Do not run scoped logging timers (only supported in some places)."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_matmul_perf_table_path",
      string_setter_for(
          &DebugOptions::set_xla_gpu_experimental_matmul_perf_table_path),
      debug_options->xla_gpu_experimental_matmul_perf_table_path(),
      "If non empty will interpret this variable as a path for performance "
      "tables for matmuls. Expects `xla.gpu.DeviceHloInstructionProfiles` "
      "proto."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_split_k_rewrite",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_enable_split_k_rewrite),
      debug_options->xla_gpu_experimental_enable_split_k_rewrite(),
      "Enable the pass that splits GEMMs that underutilize the GPU load by "
      "splitting the K dimension using a heuristic."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_triton_warp_specialization",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_triton_warp_specialization),
      debug_options->xla_gpu_experimental_enable_triton_warp_specialization(),
      "Enable Triton's auto warp specialization feature where applicable."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_use_autotuner_pass",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_use_autotuner_pass),
      debug_options->xla_gpu_experimental_use_autotuner_pass(),
      "If true, use the AutotunerPass to autotune fusions, instead of the "
      "gemm_fusion_autotuner."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_allow_unroll_factor_eight",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_allow_unroll_factor_eight),
      debug_options->xla_gpu_experimental_allow_unroll_factor_eight(),
      "If true, allows unroll factor 8 on Blackwell architectures."));
  flag_list->push_back(
      tsl::Flag("xla_detect_unstable_reductions",
                setter_for_xla_detect_unstable_reductions,
                DebugOptions::DetectionMode_Name(
                    debug_options->xla_detect_unstable_reductions()),
                "Controls the behavior of the unstable reduction detector pass "
                "that checks for unstable reductions in HLO computations. "
                "Acceptable values are: 'none', 'log', and 'crash'. 'none' is "
                "the default."));
  flag_list->push_back(tsl::Flag(
      "xla_detect_unstable_reductions_post_optimizations",
      setter_for_xla_detect_unstable_reductions_post_optimizations,
      DebugOptions::DetectionMode_Name(
          debug_options->xla_detect_unstable_reductions_post_optimizations()),
      "Controls the behavior of the unstable reduction detector pass "
      "that checks for unstable reductions in HLO computations after "
      "optimizations. Acceptable values are: 'none', 'log', and "
      "'crash'. 'none' is the default."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_use_raft_select_k",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_use_raft_select_k),
      debug_options->xla_gpu_experimental_use_raft_select_k(),
      "If true, use the raft::matrix::select_k implementation of TopK."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_scaled_dot_with_triton",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_scaled_dot_with_triton),
      debug_options->xla_gpu_experimental_scaled_dot_with_triton(),
      "If true, use the Triton emitter for scaled dot."));

  flag_list->push_back(tsl::Flag(
      "xla_cpu_collective_call_warn_stuck_timeout_seconds",
      int32_setter_for(
          &DebugOptions::set_xla_cpu_collective_call_warn_stuck_seconds),
      debug_options->xla_cpu_collective_call_warn_stuck_seconds(),
      "Set timeout for Collective Call Rendezvous stuck warning"));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_collective_call_terminate_timeout_seconds",
      int32_setter_for(
          &DebugOptions::set_xla_cpu_collective_call_terminate_timeout_seconds),
      debug_options->xla_cpu_collective_call_terminate_timeout_seconds(),
      "Set timeout for Collective Call Rendezvous termination"));
  flag_list->push_back(tsl::Flag(
      "xla_cpu_collective_timeout_seconds",
      int32_setter_for(&DebugOptions::set_xla_cpu_collective_timeout_seconds),
      debug_options->xla_cpu_collective_timeout_seconds(),
      "Set timeout for CPU collectives"));
  flag_list->push_back(tsl::Flag(
      "xla_keep_shardings_after_spmd",
      bool_setter_for(&DebugOptions::set_xla_keep_shardings_after_spmd),
      debug_options->xla_keep_shardings_after_spmd(),
      "If true, keep shardings after SPMD."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_checksum_tracing_on_thunks",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_checksum_tracing_on_thunks),
      debug_options->xla_gpu_experimental_enable_checksum_tracing_on_thunks(),
      "Enables an experimental feature to record checksums of selected thunk "
      "inputs/outputs."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_buffer_saver_on_thunks",
      bool_setter_for(
          &DebugOptions::
              set_xla_gpu_experimental_enable_buffer_saver_on_thunks),
      debug_options->xla_gpu_experimental_enable_buffer_saver_on_thunks(),
      "When provided, enables an experimental feature to save results of "
      "selected thunks."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_thunk_buffer_debug_filter_by_thunk_id_ranges",
      setter_for_thunk_buffer_debug_filter_by_thunk_id, "(none)",
      "Limits the thunk buffer debug instrumentation to thunks with IDs "
      "matching one or more ranges defined as a single integer, min:max "
      "(inclusive), or half-open min:/:max."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_thunk_buffer_debug_filter_by_profile_annotation_re",
      setter_for_thunk_buffer_debug_filter_by_profile_annotation, "(none)",
      "Limits the thunk buffer debug instrumentation to thunks with profile "
      "annotations matching one or more regexes passed as comma-separated "
      "string."));
  flag_list->push_back(tsl::Flag(
      "xla_gpu_experimental_enable_fusion_autotuner",
      bool_setter_for(
          &DebugOptions::set_xla_gpu_experimental_enable_fusion_autotuner),
      debug_options->xla_gpu_experimental_enable_fusion_autotuner(),
      "Enable autotuning between the native & triton fusion emitters."));

  auto setter_for_xla_gpu_detect_nan =
      [debug_options, detection_mode](const std::string& value) {
        if (auto mode = detection_mode(debug_options, value)) {
          debug_options->set_xla_gpu_detect_nan(mode.value());
          return true;
        }
        return false;
      };

  flag_list->push_back(tsl::Flag(
      "xla_gpu_detect_nan", setter_for_xla_gpu_detect_nan,
      DebugOptions::DetectionMode_Name(debug_options->xla_gpu_detect_nan()),
      "Controls the behavior of the NaN detector pass that checks for presence "
      "of NaN values in kernel outputs. Acceptable values are: 'none', "
      "'warning', and 'fail'. 'none' is the default. If other than 'none' "
      "value is provided, additional thunks will be added to detect and "
      "warn or fail the execution if NaNs are detected."));
  auto setter_for_xla_gpu_detect_inf =
      [debug_options, detection_mode](const std::string& value) {
        if (auto mode = detection_mode(debug_options, value)) {
          debug_options->set_xla_gpu_detect_inf(mode.value());
          return true;
        }
        return false;
      };
  flag_list->push_back(tsl::Flag(
      "xla_gpu_detect_inf", setter_for_xla_gpu_detect_inf,
      DebugOptions::DetectionMode_Name(debug_options->xla_gpu_detect_inf()),
      "Controls the behavior of the Inf detector pass that checks for presence "
      "of Inf values in kernel outputs. Acceptable values are: 'none', "
      "'warning', and 'fail'. 'none' is the default. If other than 'none' "
      "value is provided, additional thunks will be added to detect and "
      "warn or fail the execution if Infs are detected."));
  flag_list->push_back(tsl::Flag(
      "xla_early_exit_with_layouts",
      bool_setter_for(&DebugOptions::set_xla_early_exit_with_layouts),
      debug_options->xla_early_exit_with_layouts(),
      "If true, exit early from the layout assignment pass after assigning "
      "layouts to entry computations."));
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

void ParseDebugOptionFlagsFromEnv(bool reset_envvar) {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", *flag_objects, reset_envvar);
}

bool ParseFlagsFromDebugOptionsFile(absl::string_view filename) {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  VLOG(2) << "Parsing flags from file: " << filename;
  // Read the file content
  std::string file_content;
  std::ifstream file{std::string(filename)};
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  file_content = buffer.str();
  file.close();
  DebugOptions new_debug_options;
  google::protobuf::TextFormat::Parser parser;
  google::protobuf::TextFormat::ParseInfoTree tree;
  parser.WriteLocationsTo(&tree);
  VLOG(1) << "Debug options file contents: " << file_content;
  if (!parser.ParseFromString(file_content, &new_debug_options)) {
    LOG(ERROR) << "Ill formed debug options file, unable to parse: "
               << filename;
    return false;
  }

  // Read from new_debug_options, and overwrite the flags in debug_options that
  // are actually mentioned in file_contents.
  std::vector<const google::protobuf::FieldDescriptor*> overwritten_fields;
  int field_count = new_debug_options.GetDescriptor()->field_count();
  for (int i = 0; i < field_count; i++) {
    const google::protobuf::FieldDescriptor* field =
        new_debug_options.GetDescriptor()->field(i);
    if (tree.GetLocation(field, field->is_repeated() ? 0 : -1).line != -1) {
      VLOG(2) << "Non default field: " << field->name();
      overwritten_fields.push_back(field);
    }
  }
  flag_values->GetReflection()->SwapFields(flag_values, &new_debug_options,
                                           overwritten_fields);
  return true;
};

void ResetFlagValues() {
  if (flag_values != nullptr) {
    *flag_values = DefaultDebugOptionsIgnoringFlags();
  }
}

void AppendDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                             DebugOptions* debug_options) {
  absl::call_once(flags_init, &AllocateFlags, debug_options);
  flag_list->insert(flag_list->end(), flag_objects->begin(),
                    flag_objects->end());
}

xla::DebugOptions GetDebugOptionsFromFlags() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  if (flag_values->xla_flags_reset()) {
    ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", *flag_objects,
                                     /*reset_envvar=*/true);
  }
  return *flag_values;
}

FlagStatus GetFlagStatus(absl::string_view flag_name) {
  // NOTE: The explicit internal constructor is needed as an explicitly typed
  // variable to avoid a method ambiguity error when compiling with GCC.
  static const absl::NoDestructor<absl::flat_hash_set<std::string>>
      kStableFlags(absl::flat_hash_set<std::string>{
          // go/keep-sorted start
          "xla_gpu_all_reduce_combine_threshold_bytes",
          "xla_gpu_autotune_level",
          "xla_gpu_collective_permute_decomposer_threshold",
          "xla_gpu_cublas_fallback",
          "xla_gpu_dot_merger_threshold_mb",
          "xla_gpu_enable_dynamic_slice_fusion",
          "xla_gpu_enable_latency_hiding_scheduler",
          "xla_gpu_enable_pipelined_all_gather",
          "xla_gpu_enable_pipelined_all_reduce",
          "xla_gpu_enable_pipelined_reduce_scatter",
          "xla_gpu_enable_triton_gemm",
          "xla_gpu_enable_while_loop_double_buffering",
          "xla_gpu_exhaustive_tiling_search",
          "xla_gpu_reduce_scatter_combine_threshold_bytes",
          // go/keep-sorted end
      });
  static const absl::NoDestructor<absl::flat_hash_set<std::string>>
      kDeprecatedFlags(absl::flat_hash_set<std::string>{
          // go/keep-sorted start
          // go/keep-sorted end
      });
  return kStableFlags->contains(flag_name)       ? FlagStatus::kStable
         : kDeprecatedFlags->contains(flag_name) ? FlagStatus::kDeprecated
                                                 : FlagStatus::kExperimental;
}

void ResetThreadLocalFuel() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);

  thread_fuel = std::make_unique<
      absl::node_hash_map<std::string, std::atomic<int64_t>>>();
  CHECK_NOTNULL(initial_fuel);
  for (const auto& kv : *initial_fuel) {
    thread_fuel->emplace(kv.first, kv.second);
  }
}

bool PassFuelIsSet(absl::string_view pass) {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  auto* fuel_pool = thread_fuel ? thread_fuel.get() : global_fuel;
  auto it = fuel_pool->find(pass);
  return it != fuel_pool->end();
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
