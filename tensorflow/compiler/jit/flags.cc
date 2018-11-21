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

#include <mutex>  // NOLINT

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/xla/parse_flags_from_env.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

BuildXlaOpsPassFlags* build_ops_flags;
DumpGraphFlags* dump_graph_flags;
MarkForCompilationPassFlags* mark_for_compilation_flags;
XlaDeviceFlags* device_flags;
XlaOpsCommonFlags* ops_flags;

std::vector<Flag>* flag_list;
std::once_flag flags_init;

void AppendDumpGraphFlagsInternal(std::vector<Flag>* flag_list) {
  std::vector<Flag> new_flags = {
      Flag("tf_dump_graph_prefix", &dump_graph_flags->tf_dump_graph_prefix,
           "Path prefix to which graphs dumped during debugging should be "
           "written."),
  };
  flag_list->insert(flag_list->end(), new_flags.begin(), new_flags.end());
}

void AppendMarkForCompilationPassFlagsInternal(std::vector<Flag>* flag_list) {
  std::vector<Flag> new_flags = {
      Flag("tf_xla_auto_jit", &mark_for_compilation_flags->tf_xla_auto_jit,
           "Control compilation of operators into XLA computations on CPU and "
           "GPU devices.  0 = use ConfigProto setting; -1 = off; 1 = on for "
           "things very likely to be improved; 2 = on for everything.  "
           "Experimental."),
      Flag("tf_xla_min_cluster_size",
           &mark_for_compilation_flags->tf_xla_min_cluster_size,
           "Minimum number of operators in an XLA compilation. Ignored for "
           "operators placed on an XLA device or operators explicitly marked "
           "for compilation."),
      Flag("tf_xla_max_cluster_size",
           &mark_for_compilation_flags->tf_xla_max_cluster_size,
           "Maximum number of operators in an XLA compilation."),
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
      Flag("tf_xla_fusion_only",
           &mark_for_compilation_flags->tf_xla_fusion_only,
           "enable fusion of element-wise operations only using XLA when "
           "global_jit_level is ON*.")};
  flag_list->insert(flag_list->end(), new_flags.begin(), new_flags.end());
}

void AllocateAndParseFlags() {
  build_ops_flags = new BuildXlaOpsPassFlags;
  build_ops_flags->tf_xla_enable_lazy_compilation = true;

  dump_graph_flags = new DumpGraphFlags;
  dump_graph_flags->tf_dump_graph_prefix = "/tmp/";

  mark_for_compilation_flags = new MarkForCompilationPassFlags;
  mark_for_compilation_flags->tf_xla_auto_jit = 0;
  mark_for_compilation_flags->tf_xla_min_cluster_size = 2;
  mark_for_compilation_flags->tf_xla_max_cluster_size =
      std::numeric_limits<int32>::max();
  mark_for_compilation_flags->tf_xla_clustering_debug = false;
  mark_for_compilation_flags->tf_xla_cpu_global_jit = false;
  mark_for_compilation_flags->tf_xla_clustering_fuel =
      std::numeric_limits<int64>::max();
  mark_for_compilation_flags->tf_xla_fusion_only = false;

  device_flags = new XlaDeviceFlags;
  device_flags->tf_xla_compile_on_demand = false;

  ops_flags = new XlaOpsCommonFlags;
  ops_flags->tf_xla_always_defer_compilation = false;

  flag_list = new std::vector<Flag>({
      Flag("tf_xla_enable_lazy_compilation",
           &build_ops_flags->tf_xla_enable_lazy_compilation, ""),

      Flag("tf_xla_compile_on_demand", &device_flags->tf_xla_compile_on_demand,
           "Switch a device into 'on-demand' mode, where instead of "
           "autoclustering ops are compiled one by one just-in-time."),

      Flag("tf_xla_always_defer_compilation",
           &ops_flags->tf_xla_always_defer_compilation, ""),
  });
  AppendDumpGraphFlagsInternal(flag_list);
  AppendMarkForCompilationPassFlagsInternal(flag_list);
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_XLA_FLAGS", *flag_list);
}

}  // namespace

const BuildXlaOpsPassFlags& GetBuildXlaOpsPassFlags() {
  std::call_once(flags_init, &AllocateAndParseFlags);
  return *build_ops_flags;
}

DumpGraphFlags* GetDumpGraphFlags() {
  std::call_once(flags_init, &AllocateAndParseFlags);
  return dump_graph_flags;
}

MarkForCompilationPassFlags* GetMarkForCompilationPassFlags() {
  std::call_once(flags_init, &AllocateAndParseFlags);
  return mark_for_compilation_flags;
}

XlaDeviceFlags* GetXlaDeviceFlags() {
  std::call_once(flags_init, &AllocateAndParseFlags);
  return device_flags;
}

const XlaOpsCommonFlags& GetXlaOpsCommonFlags() {
  std::call_once(flags_init, &AllocateAndParseFlags);
  return *ops_flags;
}

void AppendMarkForCompilationPassFlags(std::vector<Flag>* flag_list) {
  std::call_once(flags_init, &AllocateAndParseFlags);
  AppendMarkForCompilationPassFlagsInternal(flag_list);
}

void AppendDumpGraphFlags(std::vector<Flag>* flag_list) {
  std::call_once(flags_init, &AllocateAndParseFlags);
  AppendDumpGraphFlagsInternal(flag_list);
}

}  // namespace tensorflow
