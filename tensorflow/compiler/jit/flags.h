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

struct XlaAutoJitFlag {
  // Control compilation of operators into XLA computations on CPU and GPU
  // devices.  0 = use ConfigProto setting; -1 = off; 1 = on for things very
  // likely to be improved; 2 = on for everything.
  //
  // If all non-CPU ops in the graph being optimized are placed on a single GPU
  // and there is at least one node placed on that GPU then
  // `optimization_level_single_gpu` applies.  Otherwise
  // `optimization_level_general` applies.
  //
  // Experimental.
  int32 optimization_level_single_gpu;
  int32 optimization_level_general;
};

// Sets the xla_auto_jit_flag based on the given flag sting. Supported syntax
// is:
// <number>: sets general and single_gpu setting to the provided number.
// single-gpu(<number>): sets the single_gpu setting to the provided number.
bool SetXlaAutoJitFlagFromFlagString(const string& value);

// Flags associated with the XLA bridge's mark_for_compilation_pass module.
struct MarkForCompilationPassFlags {
  XlaAutoJitFlag xla_auto_jit_flag;

  // Minimum number of operators in an XLA compilation. Ignored for operators
  // placed on an XLA device or operators explicitly marked for compilation.
  int32 tf_xla_min_cluster_size;

  // Maximum number of operators in an XLA compilation.
  int32 tf_xla_max_cluster_size;

  // If non-empty, limit XLA clustering to the following TF operations.
  string tf_xla_ops_to_cluster;

  // Dump graphs during XLA compilation.
  bool tf_xla_clustering_debug;

  // Enables global JIT compilation for CPU via SessionOptions.
  bool tf_xla_cpu_global_jit;

  // "Compiler fuel" for clustering.  Only this many ops will be marked as
  // eligible for clustering.
  int64 tf_xla_clustering_fuel;

  // If tf_xla_disable_deadness_safety_checks_for_debugging is set to true then
  // we do not do deadness related safety checks.  This is unsound in general,
  // but can be used as a debugging aid.
  bool tf_xla_disable_deadness_safety_checks_for_debugging;

  // If tf_xla_disable_resource_variable_safety_checks_for_debugging is set to
  // true then we do not do safety checks to preserve TensorFlow's resource
  // variable concurrency semantics.  This is unsound in general, but can be
  // used as a debugging aid.
  bool tf_xla_disable_resource_variable_safety_checks_for_debugging;
};

// Flags associated with the XLA bridge's xla_device module.
struct XlaDeviceFlags {
  // Switch the CPU device into "on-demand" mode, where instead of
  // autoclustering ops are compiled one by one just-in-time.
  // Enabling this mode by a legacy flag is a temporary mechanism. When this
  // feature is battle-tested, we will switch this to be a session option.
  bool tf_xla_compile_on_demand;

  // Enables "XLA" devices if this flag is set.
  bool tf_xla_enable_xla_devices;
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

  // If true then insert Print nodes to print out values produced by XLA
  // clusters.  Useful for debugging.
  bool tf_xla_print_cluster_outputs;

  // If true, insert CheckNumerics nodes for every floating point typed input to
  // an XLA cluster.
  bool tf_xla_check_cluster_input_numerics;

  // If true, insert CheckNumerics nodes for every floating point typed output
  // from an XLA cluster.
  bool tf_xla_check_cluster_output_numerics;

  // Disables all constant folding. The primary use for this is for testing to
  // guarantee that tests are run on XLA and not on TF's CPU implementation.
  bool tf_xla_disable_constant_folding;
};

// Flags for the IntroduceFloatingPointJitter pass.
struct IntroduceFloatingPointJitterPassFlags {
  // The amount of jitter to introduce.  This amount is added to each element in
  // the tensors named in `tensor_names.
  float jitter_amount;

  // The Tensors to add the jitter to.  The tensors are named in the TensorId
  // format of <node name>:<output idx>.
  std::vector<string> tensor_names;
};

// Flags for common MLIR configurations.
struct MlirCommonFlags {
  bool tf_mlir_enable_mlir_bridge;
};

// Return a pointer to the DumpGraphFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.

// Getters for flags structs defined above.  The first call to any of these
// parses TF_XLA_FLAGS for all of them.  Those functions which return a pointer
// always return the same pointer.
MarkForCompilationPassFlags* GetMarkForCompilationPassFlags();
BuildXlaOpsPassFlags* GetBuildXlaOpsPassFlags();
XlaDeviceFlags* GetXlaDeviceFlags();
const XlaOpsCommonFlags& GetXlaOpsCommonFlags();

const IntroduceFloatingPointJitterPassFlags&
GetIntroduceFloatingPointJitterPassFlags();

MlirCommonFlags* GetMlirCommonFlags();

// Appends the flag definitions associated with
// MarkForCompilationPassFlags/DumpGraphFlags to `flag_list`.
//
// Has the side-effect of parsing TF_XLA_FLAGS if that hasn't happened yet.
void AppendMarkForCompilationPassFlags(
    std::vector<tensorflow::Flag>* flag_list);

// Makes all future calls to `IsXlaEnabled()` return `true`.
//
// Should only be called when XLA is linked in.
void SetXlaIsEnabled();

// Returns whether XLA is enabled.
bool IsXlaEnabled();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_FLAGS_H_
