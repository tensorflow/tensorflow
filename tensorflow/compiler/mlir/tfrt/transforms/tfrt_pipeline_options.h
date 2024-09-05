/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_PIPELINE_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_PIPELINE_OPTIONS_H_

#include <cstdint>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"

namespace tensorflow {

struct TfrtPipelineOptions
    : public mlir::PassPipelineOptions<TfrtPipelineOptions> {
  Option<std::string> saved_model_dir{*this, "saved-model-dir",
                                      llvm::cl::desc(""), llvm::cl::init("")};
  Option<std::string> default_device{
      *this, "default-device", llvm::cl::desc("default device assignment"),
      llvm::cl::init("/job:localhost/replica:0/task:0/device:CPU:0")};
  Option<bool> enable_optimizer{
      *this, "enable-optimizer",
      llvm::cl::desc("run optimization passes on corert dialect"),
      llvm::cl::init(false)};
  Option<bool> decompose_resource_ops{
      *this, "decompose-resource-ops",
      llvm::cl::desc("decompose composite resource ops into ReadVariableOp and "
                     "non-resource ops. This is currently used in TFRT "
                     "savedmodel pipeline."),
      llvm::cl::init(false)};
  Option<std::string> force_data_format{
      *this, "force-data-format",
      llvm::cl::desc("force data format for all layout sensitive operations")};
  // TODO(tfrt-devs): consider making compiler to figure out whether to fold
  // transpose or not instead of exposing the specific option.
  Option<bool> skip_fold_transpose_in_ops{
      *this, "skip-fold-transpose-in-ops",
      llvm::cl::desc("Skip folding transpose operands in Ops which can support "
                     "different layouts.")};
  Option<bool> target_tpurt{*this, "target-tpurt",
                            llvm::cl::desc("target TPURT dialect if true"),
                            llvm::cl::init(false)};
  Option<bool> tpu_use_core_selector{
      *this, "tpu-use-core-selector",
      llvm::cl::desc("If true, use ServingCoreSelector to pick TPU core. "
                     "Otherwise, use the assigned core. Currently we use "
                     "core selector for Servo serving use cases."),
      llvm::cl::init(true)};
  Option<bool> tpu_use_bundled_transfer{
      *this, "tpu-use-bundled-transfer",
      llvm::cl::desc("If true, use BundledTransferToTpuOp to transfer "
                     "variables and input tensors to TPU."),
      llvm::cl::init(true)};
  Option<bool> tpu_lower_to_fallback{
      *this, "tpu-lower-to-fallback",
      llvm::cl::desc("If true, lower an TF op that's placed on TPU device "
                     "to be executed by tfrt_fallback.execute."),
      llvm::cl::init(true)};
  Option<bool> tpu_fuse_ops{
      *this, "tpu-fuse-ops",
      llvm::cl::desc("If true, use the TPU fused compile_and_execute kernel"),
      llvm::cl::init(false)};
  // TODO(b/194081364): remove this option once we unify servo TPU serving
  // result transfer behavior.
  Option<bool> tpu_transfer_result_to_host{
      *this, "tpu-transfer-result-to-host",
      llvm::cl::desc("If true, transfer the result of tpurt.execute from TPU "
                     "to host."),
      llvm::cl::init(true)};
  Option<bool> use_tpu_host_allocator_for_inputs{
      *this, "use-tpu-host-allocator-for-inputs",
      llvm::cl::desc("If true, fallback executeops that produce inputs to tpu "
                     "program will use tpu host allocator."),
      llvm::cl::init(false)};
  Option<TfrtCompileOptions::TpuAllowUnpaddedBatch> tpu_allow_unpadded_batch{
      *this, "tpu-allow-unpadded-batch",
      llvm::cl::desc("To allow unpadded batch for TPU execution."),
      llvm::cl::values(
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kDisabled,
                     "disabled", "Disable this feature."),
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kAuto, "auto",
                     "Enable this feature when in-graph batching is detected."),
          clEnumValN(TfrtCompileOptions::TpuAllowUnpaddedBatch::kEnforced,
                     "enforced", "Force to enable this feature.")),
      llvm::cl::init(TfrtCompileOptions::TpuAllowUnpaddedBatch::kDisabled)};

  Option<bool> target_gpu{
      *this, "target-gpu",
      llvm::cl::desc("If true, target GPU compiler passes."),
      llvm::cl::init(false)};

  // TODO(b/294895431): Remove the flag and default to the fused op.
  Option<bool> use_gpu_compile_and_execute_op{
      *this, "use-gpu-compile-and-execute-op",
      llvm::cl::desc("If true, gpurt.compile_and_execute is used for GPU"),
      llvm::cl::init(false)};

  Option<bool> enable_while_parallel_iterations{
      *this, "enable-while-parallel-iterations",
      llvm::cl::desc("If true, tf.While op will be parallelized. This is "
                     "currently experimental."),
      llvm::cl::init(false)};

  Option<bool> hoist_invariant_ops{
      *this, "hoist-invariant-ops",
      llvm::cl::desc("If true, invariant ops in savedmodels will be hoisted "
                     "out to run during loading."),
      llvm::cl::init(false)};

  Option<bool> fuse_get_resource_ops_in_hoisting{
      *this, "fuse-get-resource-ops-in-hoisting",
      llvm::cl::desc("If true, get_resource_op will be fused during hoisting"),
      llvm::cl::init(true)};

  Option<bool> sink_in_invariant_ops{
      *this, "sink-in-invariant-ops",
      llvm::cl::desc("If true, sink the selected invariant ops in to the "
                     "nested functions to facilitate invariant ops hoisting."),
      llvm::cl::init(false)};

  Option<uint64_t> cost_threshold{
      *this, "tfrt-cost-threshold",
      llvm::cl::desc(
          "The cost threshold to decide whether a sequence of operations is "
          "cheap, and then whether it can be executed inline."),
      llvm::cl::init(1)};

  Option<int64_t> min_num_batch_threads{
      *this, "tfrt-min-num-batch-threads",
      llvm::cl::desc("The minimum number of batch threads"), llvm::cl::init(1)};

  Option<int64_t> min_max_enqueued_batches{
      *this, "tfrt-min-max-enqueued-batches",
      llvm::cl::desc(
          "The minimum of the maximum number of outstanding enqueued batches"),
      llvm::cl::init(1)};

  Option<std::string> batch_padding_policy{
      *this, "tfrt-batch-padding-policy",
      llvm::cl::desc("The policy used when padding (or splitting) batches."),
      llvm::cl::init("")};

  Option<bool> merge_inter_dependent_streams{
      *this, "tfrt-merge-inter-dependent-streams",
      llvm::cl::desc("If true, streams with inter data depenedencies will be "
                     "preferred to be merged for inline execution."),
      llvm::cl::init(false)};
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_TFRT_PIPELINE_OPTIONS_H_
