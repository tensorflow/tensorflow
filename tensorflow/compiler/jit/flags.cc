/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/flags.h"

#include <mutex>  // NOLINT
#include <vector>

#include "absl/base/call_once.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_graph.h"
#include "tensorflow/compiler/xla/parse_flags_from_env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

BuildXlaOpsPassFlags* build_ops_flags;
MarkForCompilationPassFlags* mark_for_compilation_flags;
XlaDeviceFlags* device_flags;
XlaOpsCommonFlags* ops_flags;
IntroduceFloatingPointJitterPassFlags* jitter_flags;
MlirCommonFlags* mlir_flags;
JitRtFlags* jitrt_flags;
std::vector<Flag>* jitrt_flag_list;

std::vector<Flag>* flag_list;
absl::once_flag flags_init;

bool SetterForXlaAutoJitFlag(const string& value) {
  int32_t opt_level;
  // We need to use the mark_for_compilation_flags directly here instead of
  // going via GetMarkForCompilationPassFlags() to avoid infinite recursion. The
  // latter will try to setup and parse flags, which would bring us back to this
  // setter.
  if (absl::SimpleAtoi(value, &opt_level)) {
    mark_for_compilation_flags->xla_auto_jit_flag
        .optimization_level_single_gpu = opt_level;
    mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_general =
        opt_level;
    return true;
  }

  if (value == "fusible") {
    mark_for_compilation_flags->xla_auto_jit_flag
        .optimization_level_single_gpu = 1;
    mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_general =
        1;
    mark_for_compilation_flags->tf_xla_ops_to_cluster = "FUSIBLE";
    return true;
  }

  absl::string_view value_sv(value);
  if (!absl::ConsumePrefix(&value_sv, "single-gpu(") ||
      !absl::ConsumeSuffix(&value_sv, ")") ||
      !absl::SimpleAtoi(value_sv, &opt_level)) {
    return false;
  }

  mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_single_gpu =
      opt_level;
  return true;
}

void AppendMarkForCompilationPassFlagsInternal(std::vector<Flag>* flag_list) {
  std::vector<Flag> new_flags = {
      Flag("tf_xla_auto_jit", SetterForXlaAutoJitFlag, "0",
           "Control compilation of operators into XLA computations on CPU and "
           "GPU devices.  0 = use ConfigProto setting; -1 = off; 1 = on for "
           "things very likely to be improved; 2 = on for everything; "
           "(experimental) fusible = only for Tensorflow operations that XLA "
           "knows how to fuse.  "
           "If set to single-gpu(<N>) then this resolves to <N> for single-GPU "
           "graphs (graphs that have at least one node placed on a GPU and no "
           "more than one GPU is in use through the entire graph) and 0 "
           "otherwise.  Experimental."),
      Flag("tf_xla_min_cluster_size",
           &mark_for_compilation_flags->tf_xla_min_cluster_size,
           "Minimum number of operators in an XLA compilation. Ignored for "
           "operators placed on an XLA device or operators explicitly marked "
           "for compilation."),
      Flag("tf_xla_max_cluster_size",
           &mark_for_compilation_flags->tf_xla_max_cluster_size,
           "Maximum number of operators in an XLA compilation."),
      Flag(
          "tf_xla_ops_to_cluster",
          &mark_for_compilation_flags->tf_xla_ops_to_cluster,
          "(experimental) "
          "Limit the operations clustered by XLA to these operations. "
          "If multiple, separate them with commas. Shortcuts: "
          " PW: All point-wise operations."
          " RED: All reduction operations."
          " MISC: Mixed operations."
          " PWRED: TF operations that get converted to PW+RED operation in XLA."
          " REDUCEWINDOW: TF operations like MaxPool/AvgPool that get "
          "converted to ReduceWindow in XLA."
          " REDUCEWINDOWPW: Operation that get converted to ReduceWindow + PW "
          "(LRN, LRNGrad)."
          " BN: TF FusedBatchNorm* operations."
          " FUSIBLE: All TF operations that XLA can fuse (All the above). "
          "You can also put any TF operation name, e.g. 'FUSIBLE,MatMul'."),
      Flag("tf_xla_cluster_exclude_ops",
           &mark_for_compilation_flags->tf_xla_cluster_exclude_ops,
           "(experimental) "
           "Exclude the operations from auto-clustering. "
           "If multiple, separate them with commas."
           " Where, Some_other_ops"),
      Flag("tf_xla_clustering_debug",
           &mark_for_compilation_flags->tf_xla_clustering_debug,
           "Dump graphs during XLA compilation."),
      Flag("tf_xla_cpu_global_jit",
           &mark_for_compilation_flags->tf_xla_cpu_global_jit,
           "Enables global JIT compilation for CPU via SessionOptions."),
      Flag("tf_xla_clustering_fuel",
           &mark_for_compilation_flags->tf_xla_clustering_fuel,
           "Places an artificial limit on the number of ops marked as "
           "eligible for clustering."),
      Flag("tf_xla_disable_deadness_safety_checks_for_debugging",
           &mark_for_compilation_flags
                ->tf_xla_disable_deadness_safety_checks_for_debugging,
           "Disable deadness related safety checks when clustering (this is "
           "unsound)."),
      Flag("tf_xla_disable_resource_variable_safety_checks_for_debugging",
           &mark_for_compilation_flags
                ->tf_xla_disable_resource_variable_safety_checks_for_debugging,
           "Disable resource variables related safety checks when clustering "
           "(this is unsound)."),
      Flag("tf_xla_deterministic_cluster_names",
           &mark_for_compilation_flags->tf_xla_deterministic_cluster_names,
           "Causes the function names assigned by auto clustering to be "
           "deterministic from run to run."),
      Flag("tf_xla_persistent_cache_directory",
           &mark_for_compilation_flags->tf_xla_persistent_cache_directory,
           "If non-empty, JIT-compiled executables are saved to and loaded "
           "from the specified file system directory path. Empty by default."),
      Flag("tf_xla_disable_strict_signature_checks",
           &mark_for_compilation_flags->tf_xla_disable_strict_signature_checks,
           "If true, entires loaded into the XLA compile cache will not have "
           "their signatures checked strictly. Defaults to false."),
      Flag("tf_xla_persistent_cache_prefix",
           &mark_for_compilation_flags->tf_xla_persistent_cache_prefix,
           "Specifies the persistance cache prefix. Default is "
           "\"xla_compile_cache\"")};
  flag_list->insert(flag_list->end(), new_flags.begin(), new_flags.end());
}

void AllocateAndParseJitRtFlags() {
  jitrt_flags = new JitRtFlags;
  jitrt_flags->always_specialize = false;
  jitrt_flags->cost_driven_async_parallel_for = false;
  jitrt_flags->log_query_of_death = false;
  jitrt_flags->vectorize = false;
  jitrt_flags->enable_crash_reproducer = false;
  jitrt_flag_list = new std::vector<Flag>({
      Flag("always_specialize", &jitrt_flags->always_specialize, ""),
      Flag("cost_driven_async_parallel_for",
           &jitrt_flags->cost_driven_async_parallel_for, ""),
      Flag("log_query_of_death", &jitrt_flags->log_query_of_death, ""),
      Flag("vectorize", &jitrt_flags->vectorize, ""),
      Flag("enable_crash_reproducer", &jitrt_flags->enable_crash_reproducer,
           ""),
  });
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_JITRT_FLAGS", *jitrt_flag_list);
}

void AllocateAndParseFlags() {
  build_ops_flags = new BuildXlaOpsPassFlags;
  build_ops_flags->tf_xla_enable_lazy_compilation = true;
  build_ops_flags->tf_xla_print_cluster_outputs = false;
  build_ops_flags->tf_xla_check_cluster_input_numerics = false;
  build_ops_flags->tf_xla_check_cluster_output_numerics = false;
  build_ops_flags->tf_xla_disable_constant_folding = false;

  mark_for_compilation_flags = new MarkForCompilationPassFlags;
  mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_single_gpu =
      0;
  mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_general = 0;
  mark_for_compilation_flags->tf_xla_min_cluster_size = 4;
  mark_for_compilation_flags->tf_xla_max_cluster_size =
      std::numeric_limits<int32>::max();
  mark_for_compilation_flags->tf_xla_clustering_debug = false;
  mark_for_compilation_flags->tf_xla_cpu_global_jit = false;
  mark_for_compilation_flags->tf_xla_clustering_fuel =
      std::numeric_limits<int64_t>::max();
  mark_for_compilation_flags
      ->tf_xla_disable_deadness_safety_checks_for_debugging = false;
  mark_for_compilation_flags
      ->tf_xla_disable_resource_variable_safety_checks_for_debugging = false;
  mark_for_compilation_flags->tf_xla_deterministic_cluster_names = false;
  mark_for_compilation_flags->tf_xla_persistent_cache_directory = "";
  mark_for_compilation_flags->tf_xla_disable_strict_signature_checks = false;
  mark_for_compilation_flags->tf_xla_persistent_cache_prefix =
      "xla_compile_cache";

  device_flags = new XlaDeviceFlags;
  device_flags->tf_xla_compile_on_demand = false;
  device_flags->tf_xla_enable_xla_devices = false;

  ops_flags = new XlaOpsCommonFlags;
  ops_flags->tf_xla_always_defer_compilation = false;
  ops_flags->tf_xla_async_compilation = false;

  jitter_flags = new IntroduceFloatingPointJitterPassFlags;
  jitter_flags->jitter_amount = 1e-5;

  // The `enable_mlir_bridge` flag allows the user to explicitly request that
  // their program is (or isn't) compiled using the MLIR-based TF-to-XLA bridge.
  //
  // The `enable_mlir_bridge_is_explicit` variable tracks whether or not the
  // user has made an explicit request. That is, if this variable is set to
  // true, the program honors the user's request as per `enable_mlir_bridge`; if
  // it's set to false, the default behavior is used (which may run either
  // bridge, on a per-graph basis).
  bool enable_mlir_bridge = false;
  bool enable_mlir_bridge_is_explicit = false;
  bool mlir_bridge_safe_mode = false;
  bool enable_mlir_merge_control_flow_pass = true;
  bool enable_mlir_convert_control_to_data_outputs_pass = false;
  auto setter_for_jitter_tensor_names = [](string sequence) {
    jitter_flags->tensor_names = absl::StrSplit(sequence, ',');
    return true;
  };
  // Dump graphs in TFG dialect.
  bool use_tfg_graph_dumper = false;

  flag_list = new std::vector<Flag>(
      {Flag("tf_xla_enable_lazy_compilation",
            &build_ops_flags->tf_xla_enable_lazy_compilation, ""),
       Flag("tf_xla_print_cluster_outputs",
            &build_ops_flags->tf_xla_print_cluster_outputs,
            "If true then insert Print nodes to print out values produced by "
            "XLA clusters."),
       Flag("tf_xla_check_cluster_input_numerics",
            &build_ops_flags->tf_xla_check_cluster_input_numerics,
            "If true then insert CheckNumerics nodes to check all cluster "
            "inputs."),
       Flag("tf_xla_check_cluster_output_numerics",
            &build_ops_flags->tf_xla_check_cluster_output_numerics,
            "If true then insert CheckNumerics nodes to check all cluster "
            "outputs."),
       Flag("tf_xla_disable_constant_folding",
            &build_ops_flags->tf_xla_disable_constant_folding,
            "If true then disables constant folding on TF graph before XLA "
            "compilation."),

       Flag("tf_xla_compile_on_demand", &device_flags->tf_xla_compile_on_demand,
            "Switch a device into 'on-demand' mode, where instead of "
            "autoclustering ops are compiled one by one just-in-time."),

       Flag("tf_xla_enable_xla_devices",
            &device_flags->tf_xla_enable_xla_devices,
            "Generate XLA_* devices, where placing a computation on such a "
            "device"
            "forces compilation by XLA. Deprecated."),

       Flag("tf_xla_always_defer_compilation",
            &ops_flags->tf_xla_always_defer_compilation, ""),
       Flag("tf_xla_async_compilation", &ops_flags->tf_xla_async_compilation,
            "When lazy compilation is enabled, asynchronous compilation starts "
            "the cluster compilation in the background, and the fallback path "
            "is executed until the compilation has finished."),

       Flag("tf_introduce_floating_point_jitter_to_tensors",
            setter_for_jitter_tensor_names, "",
            "The Tensors to add the jitter to.  The tensors are named in the "
            "TensorId format of <node name>:<output idx>."),
       Flag("tf_introduce_floating_point_jitter_amount",
            &jitter_flags->jitter_amount,
            "The amount of jitter to introduce.  This amount is added to each "
            "element in the tensors named in `tensor_names."),

       Flag("tf_mlir_enable_mlir_bridge", &enable_mlir_bridge,
            "Enables experimental MLIR-Based TensorFlow Compiler Bridge.",
            &enable_mlir_bridge_is_explicit),
       Flag("tf_mlir_enable_merge_control_flow_pass",
            &enable_mlir_merge_control_flow_pass,
            "Enables MergeControlFlow pass for MLIR-Based TensorFlow Compiler "
            "Bridge."),
       Flag("tf_mlir_enable_convert_control_to_data_outputs_pass",
            &enable_mlir_convert_control_to_data_outputs_pass,
            "Enables `tf-executor-convert-control-to-data-outputs` pass for "
            "MLIR-Based TensorFlow Compiler Bridge."),
       Flag(
           "tf_mlir_bridge_safe_mode", &mlir_bridge_safe_mode,
           "When tf_mlir_enable_mlir_bridge is true, this field can enable "
           "the MLIR bridge's safe mode. When the MLIR bridge is in safe mode, "
           "it only runs for graphs that use features MLIR bridge currently "
           "supports."),
       Flag("tf_dump_graphs_in_tfg", &use_tfg_graph_dumper,
            "When tf_dump_graphs_in_tfg is true, graphs after transformations "
            "are dumped in MLIR TFG dialect and not in GraphDef")});

  AppendMarkForCompilationPassFlagsInternal(flag_list);
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_XLA_FLAGS", *flag_list);

  mlir_flags = new MlirCommonFlags;
  if (!enable_mlir_bridge_is_explicit) {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        (mlir_bridge_safe_mode)
            ? ConfigProto::Experimental::
                  MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED
            : ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED;
  } else if (enable_mlir_bridge) {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        (mlir_bridge_safe_mode)
            ? ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED
            : ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
  } else {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED;
  }
  mlir_flags->tf_mlir_enable_merge_control_flow_pass =
      enable_mlir_merge_control_flow_pass;
  mlir_flags->tf_mlir_enable_convert_control_to_data_outputs_pass =
      enable_mlir_convert_control_to_data_outputs_pass;

  if (use_tfg_graph_dumper) {
    UseMlirForGraphDump(MlirDumpConfig{}.elide_large_attributes().emit_dialect(
        MlirDumpConfig::Dialect::kTFG));
  }

  AllocateAndParseJitRtFlags();
}

void ResetFlags() {
  delete build_ops_flags;
  delete mark_for_compilation_flags;
  delete device_flags;
  delete ops_flags;
  delete jitter_flags;
  delete mlir_flags;
  delete flag_list;
  delete jitrt_flags;
  delete jitrt_flag_list;
  AllocateAndParseFlags();
}

}  // namespace

bool SetXlaAutoJitFlagFromFlagString(const string& value) {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  return SetterForXlaAutoJitFlag(value);
}

BuildXlaOpsPassFlags* GetBuildXlaOpsPassFlags() {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  return build_ops_flags;
}

MarkForCompilationPassFlags* GetMarkForCompilationPassFlags() {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  return mark_for_compilation_flags;
}

XlaDeviceFlags* GetXlaDeviceFlags() {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  return device_flags;
}

const XlaOpsCommonFlags& GetXlaOpsCommonFlags() {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  return *ops_flags;
}

const IntroduceFloatingPointJitterPassFlags&
GetIntroduceFloatingPointJitterPassFlags() {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  return *jitter_flags;
}

MlirCommonFlags* GetMlirCommonFlags() {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  return mlir_flags;
}

void ResetJitCompilerFlags() { ResetFlags(); }

const JitRtFlags& GetJitRtFlags() {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  return *jitrt_flags;
}

ConfigProto::Experimental::MlirBridgeRollout GetMlirBridgeRolloutState(
    std::optional<const ConfigProto> config_proto) {
  // TF1 graphs that do not override Sessions's ConfigProto and TF2 graphs
  // can enable/disable the graph via tf_mlir_enable_mlir_bridge.
  auto tf_mlir_enable_mlir_bridge =
      GetMlirCommonFlags()->tf_mlir_enable_mlir_bridge;
  if (tf_mlir_enable_mlir_bridge !=
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED) {
    return tf_mlir_enable_mlir_bridge;
  }

  // If a ConfigProto was not passed in, we can assume the caller is
  // checking if TF2 graph should have the bridge enabled / disabled. In that
  // case, we have already checked tf_mlir_enable_mlir_bridge so it is safe to
  // return UNSPECIFIED here.
  if (!config_proto.has_value()) {
    return ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED;
  }

  // TF1 graphs that do override Session's ConfigProto and set
  // ConfigProto's enable_mlir_bridge or mlir_bridge_rollout fields will not
  // update tf_mlir_enable_mlir_bridge so check their values.

  // ConfigProto's enable_mlir_bridge defaults to false so only respect it
  // when it is true.
  if (config_proto.value().experimental().enable_mlir_bridge()) {
    return ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
  }
  return config_proto.value().experimental().mlir_bridge_rollout();
}

void AppendMarkForCompilationPassFlags(std::vector<Flag>* flag_list) {
  absl::call_once(flags_init, &AllocateAndParseFlags);
  AppendMarkForCompilationPassFlagsInternal(flag_list);
}

static std::atomic<bool> xla_compilation_disabled(false);

void DisableXlaCompilation() { xla_compilation_disabled = true; }

bool FailOnXlaCompilation() { return xla_compilation_disabled; }

}  // namespace tensorflow
