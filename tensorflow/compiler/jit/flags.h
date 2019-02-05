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

#ifndef TENSORFLOW_COMPILER_JIT_FLAGS_H_
#define TENSORFLOW_COMPILER_JIT_FLAGS_H_

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {

// Flags associated with the XLA bridge's mark_for_compilation_pass module.
struct MarkForCompilationPassFlags {
  // Control compilation of operators into XLA computations on CPU and GPU
  // devices.  0 = use ConfigProto setting; -1 = off; 1 = on for things very
  // likely to be improved; 2 = on for everything.
  //
  // Experimental.
  int32 tf_xla_auto_jit;

  // Minimum number of operators in an XLA compilation. Ignored for operators
  // placed on an XLA device or operators explicitly marked for compilation.
  int32 tf_xla_min_cluster_size;

  // Maximum number of operators in an XLA compilation.
  int32 tf_xla_max_cluster_size;

  // Dump graphs during XLA compilation.
  bool tf_xla_clustering_debug;

  // Enables global JIT compilation for CPU via SessionOptions.
  bool tf_xla_cpu_global_jit;

  // "Compiler fuel" for clustering.  Only this many ops will be marked as
  // eligible for clustering.
  int64 tf_xla_clustering_fuel;

  // tf_xla_fusion_only is effective only when global_jit_level is set to ON*
  // and overrides its behavior. If true, enable fusion of element-wise
  // operations only using XLA.
  bool tf_xla_fusion_only;

  // If tf_xla_disable_deadness_safety_checks_for_debugging is set to true then
  // we do not do deadness related safety checks.  This is unsound in general,
  // but can be used as a debugging aid.
  bool tf_xla_disable_deadness_safety_checks_for_debugging;
};

// Flags associated with the XLA bridge's xla_device module.
struct XlaDeviceFlags {
  // Switch the CPU device into "on-demand" mode, where instead of
  // autoclustering ops are compiled one by one just-in-time.
  // Enabling this mode by a legacy flag is a temporary mechanism. When this
  // feature is battle-tested, we will switch this to be a session option.
  bool tf_xla_compile_on_demand;
};

// Flags common to the _Xla* ops and their kernels.
struct XlaOpsCommonFlags {
  // If true, _XlaCompile always refuses to compile the cluster, which means the
  // XLA clusters always run in the TF executor.  Defaults to false.
  bool tf_xla_always_defer_compilation;
};

// Flags for the build_xla_ops pass.
struct BuildXlaOpsPassFlags {
  // Enables lazy compilation for TF/XLA (only when auto-clustering) if true.
  // Defaults to true.
  bool tf_xla_enable_lazy_compilation;
};

// Flags for the XLA bridge's dump_graph module.
struct DumpGraphFlags {
  // Path prefix to which graphs dumped during debugging should be written.
  string tf_dump_graph_prefix;
};

// Return a pointer to the DumpGraphFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.

// Getters for flags structs defined above.  The first call to any of these
// parses TF_XLA_FLAGS for all of them.  Those functions which return a pointer
// always return the same pointer.
MarkForCompilationPassFlags* GetMarkForCompilationPassFlags();
const BuildXlaOpsPassFlags& GetBuildXlaOpsPassFlags();
XlaDeviceFlags* GetXlaDeviceFlags();
const XlaOpsCommonFlags& GetXlaOpsCommonFlags();
DumpGraphFlags* GetDumpGraphFlags();

// Appends the flag definitions associated with
// MarkForCompilationPassFlags/DumpGraphFlags to `flag_list`.
//
// Has the side-effect of parsing TF_XLA_FLAGS if that hasn't happened yet.
void AppendMarkForCompilationPassFlags(
    std::vector<tensorflow::Flag>* flag_list);
void AppendDumpGraphFlags(std::vector<tensorflow::Flag>* flag_list);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_FLAGS_H_
