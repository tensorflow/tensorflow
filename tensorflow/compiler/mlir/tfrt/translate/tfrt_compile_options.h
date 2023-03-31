/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_TFRT_COMPILE_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_TFRT_COMPILE_OPTIONS_H_

#include <iosfwd>
#include <ostream>
#include <string>
#include <vector>

#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

enum class TfrtDeviceInfraTarget {
  kCpu,             // CPU only, no device support.
  kTpurt,           // Target TPURT dialect and kernels.
  kTfFallback,      // Target TPU kernels in TF Fallback.
  kBridgeFallback,  // TPU support but choose kTpurt or kTfFallback depending on
                    // whether the graph has unsupported feature in Bridge.
  kGpu,             // Target GPU specific compiler passes and runtime
                    // initializations.
};

std::ostream& operator<<(std::ostream& os, TfrtDeviceInfraTarget device_target);

struct TfrtCompileOptions {
  // TODO(tfrt-devs): Ideally, compiler should make the decision where
  // to place the variable.
  std::string variable_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  std::string default_device = "/job:localhost/replica:0/task:0/device:CPU:0";

  // Enable compiler optimization in TFRT dialect.
  bool enable_optimizer = true;

  // If true, run grappler passes before compiling.
  bool enable_grappler = true;

  // Graph rewrite options that will be applied on GraphDef before converting to
  // MLIR.
  GraphOptions graph_options;

  // Force data format for all layout sensitive operations, eg. setting it to
  // "NHWC" will changes all data format in the graph to "NHWC" by inserting
  // or removing related tf.Transpose op. Currently the supported formats are
  // "NHWC" and "NCHW".
  //
  // TODO(tfrt-devs): Ideally compiler should figure out whether the
  // data format should be changed, instead of controlled by users.
  std::string force_data_format;

  // The target device infrastructure to use. This will trigger target specific
  // compiler passes and runtime initialization.
  TfrtDeviceInfraTarget device_target = TfrtDeviceInfraTarget::kCpu;

  // If true, use the fused TPU compile_and_execute kernel, which performs all
  // TPU inference related operations, e.g. core selection, h2d/d2h transfers,
  // compile and execute.
  bool tpu_fuse_ops = false;

  // If true, resource gather ops in the device graph are moved to host graphs
  // in order to saved TPU memory usage. This option is experimental.
  bool tpu_move_resource_gather_to_host = false;

  // The threshold in bytes that controls whether a resource gather op on TPU
  // should be moved to host. A negative value means there is no threshold. This
  // option is experimental.
  int64_t tpu_gather_table_width_threshold_bytes = -1;

  // If true, fallback executeops that produce inputs to tpu program will use
  // tpu host allocator. This options is experimental.
  bool use_tpu_host_allocator_for_inputs = false;

  // To allow unpadded batch for TPU execution.
  enum class TpuAllowUnpaddedBatch {
    // Disable this feature.
    kDisabled,
    // Enable this feature when in-graph batching is detected.
    kAuto,
    // Force to enable this feature.
    kEnforced,
  };
  TpuAllowUnpaddedBatch tpu_allow_unpadded_batch =
      TpuAllowUnpaddedBatch::kDisabled;

  // If true, the compiler will try to hoist invariant ops (e.g., const ops and
  // their non-side-effecting consumers) to loading phase, which avoids the
  // runtime cost during later running.
  // TODO(tfrt-devs): Set the default value to true after testing as it is
  // supposed to be turned on by default.
  bool hoist_invariant_ops = false;

  // If true, the compiler will try to sink in the invariant ops (e.g. const
  // ops, var handle ops, etc.) to the nested function (e.g. batch function) to
  // facilitate invariant ops hoisting.
  // TODO(tfrt-devs): Set the default value to true after testing as it is
  // supposed to be turned on by default.
  bool sink_in_invariant_ops = false;

  // If true, tf.While's iterations will be parallelized on a best-effort
  // basis. This is currently experimental.
  bool enable_while_parallel_iterations = false;

  // A set of flags to control auto-fusion: automatic clustering of Tensorflow
  // operations and compiling outlined regions using MLIR based compilation
  // stack.
  //
  // WARNING: These flags are experimental and are intended for manual testing
  // of different auto-fusion strategies. They will be removed in the future.

  // A list of Tensorflow operations that are supported by auto-fusion
  // clustering and compilation (e.g. tf.Tanh).
  std::vector<std::string> auto_fusion_oplist;

  // Minimum size of the cluster to be compiled at runtime.
  int auto_fusion_min_cluster_size = 2;

  // The cost threshold to decide whether a sequence of operations is cheap, and
  // then whether it can be executed inline. If the cost is smaller than the
  // threshold, it will be considered as cheap operations. Since the cost must
  // be positive integers, setting the threshold to 1 makes all operations
  // expensive.
  uint64_t cost_threshold = 1;

  // The threshold to decie whether an inline execution sequence is too large
  // even if the operations forms a sequential data dependency as it may occupy
  // the CPU core for too long. In that case, they are broken into multiple
  // sequences. The default is -1 which means no limit.
  int64_t upper_cost_threshold = -1;

  // If true, streams with inter data depenedencies will be preferred to be
  // merged for inline execution.
  bool merge_inter_dependent_streams = true;

  // Whether to enable the DecomposeResourceOpsPass.
  bool decompose_resource_ops = true;

  // Whether to compile to sync TFRT dialect.
  bool compile_to_sync_tfrt_dialect = false;

  // Whether to use bridge for GPU.
  // TODO(b/260915352): Remove the flag and default to using bridge.
  bool use_bridge_for_gpu = false;
};

std::ostream& operator<<(std::ostream& os, const TfrtCompileOptions& options);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_TFRT_COMPILE_OPTIONS_H_
