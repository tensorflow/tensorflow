/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_COMMON_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_COMMON_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {
namespace tpu {

// Abstract base class for TpuCompileOpKernel implementation.
class TPUCompileOpKernelCommon {
 public:
  TPUCompileOpKernelCommon(const std::string& mlir_module,
                           const tpu::TPUCompileMetadataProto metadata,
                           int num_computations)
      : metadata_(metadata),
        use_mlir_(true),
        mlir_module_(mlir_module),
        num_computations_(num_computations) {}

  TPUCompileOpKernelCommon(const NameAttrList& function,
                           const tpu::TPUCompileMetadataProto metadata,
                           int num_computations)
      : metadata_(metadata),
        use_mlir_(false),
        function_(function),
        num_computations_(num_computations) {}

  virtual ~TPUCompileOpKernelCommon() = default;

  virtual void Compute(OpKernelContext* ctx) = 0;

  // Computes shapes for each argument. Uses both the static shape from the
  // metadata, and the dynamic shapes where the static shape is not
  // defined. There must be one dynamic_shape for each argument with a
  // partially defined shape, in index order.
  static Status ComputeArgumentShapes(
      const tpu::TPUCompileMetadataProto& metadata,
      const std::vector<TensorShape>& dynamic_shapes,
      std::vector<TensorShape>* arg_shapes);

  // Performs shape inference on `computation`, filling shape_info with operator
  // shapes. The shapes of the _Arg nodes are taken from `arg_shapes`.
  static Status RunShapeInferenceOnComputation(
      const tpu::TPUCompileMetadataProto& metadata,
      const std::vector<PartialTensorShape>& arg_shapes, Graph* graph,
      FunctionLibraryRuntime* flr, GraphShapeInfo* shape_info);

 protected:
  // Sleeps for `kSleepSeconds` seconds to give time for TPUCompileOp to finish
  // before terminating peacefully.
  static void ExitCountdown(OpKernelContext* ctx,
                            std::shared_ptr<std::atomic<bool>> done);

  // Converts the `dynamic_shapes` arguments to the compile operator into
  // TensorShapes.
  static Status GetDynamicShapes(OpKernelContext* ctx,
                                 std::vector<TensorShape>* shapes);

  // Adds TPU_REPLICATED_CORE device assignments to the _Arg and _Retval
  // nodes in `graph', using the sharding/index assignments in
  // `arg_core_mapping` and `retval_core_mapping`. The mappings are maps from
  // original argument/return index to (sharding, per-core argument/return
  // index) pairs. Node attributes, such as device assignments, are not
  // preserved on function argument and return values nodes, so we must recreate
  // them the compilation metadata.
  static Status AssignDevicesToArgsAndRetvals(
      absl::Span<const tpu::ShardingAndIndex> arg_core_mapping,
      absl::Span<const tpu::ShardingAndIndex> retval_core_mapping,
      Graph* graph);

  // Optimizes `graph`, given the argument descriptions in `metadata` and
  // `arg_shapes`.
  static Status OptimizeGraph(const tpu::TPUCompileMetadataProto& metadata,
                              const std::vector<PartialTensorShape>& arg_shapes,
                              std::unique_ptr<Graph>* graph,
                              FunctionLibraryRuntime* flr,
                              FunctionLibraryDefinition* fld);

  // Converts a TF Function into XLA HLO, stores generated HLO module and
  // accompanying metadata in CompilationResult.
  Status CompileTFFunctionToHlo(
      const FunctionLibraryDefinition& flib_def, int graph_def_version,
      const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
      const std::vector<TensorShape>& arg_shapes,
      const OpInputList& guaranteed_constants, const NameAttrList& function,
      std::function<Status(ResourceMgr*)> populate_resource_manager_fn,
      xla::CompileOnlyClient* client,
      std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
      std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
      XlaCompiler::CompilationResult* compilation_result);

  // Gets information regarding how input arguments are sharded across multiple
  // cores.
  Status GetShardingInfo(
      absl::Span<const TensorShape> arg_shapes,
      const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
      std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
      std::vector<std::vector<xla::Shape>>* per_core_arg_shapes);

  // Populates the mapping from return value to ShardingAndIndex.
  Status AssignReturnValueToCore(
      std::vector<tpu::ShardingAndIndex>* retval_core_mapping);

  // Populates the arguments, core mapping and per core argument shape for the
  // computation.
  Status BuildComputationArgumentDescriptions(
      const std::vector<TensorShape>& arg_shapes,
      const OpInputList& guaranteed_constants, const XlaCompiler& compiler,
      std::vector<XlaCompiler::Argument>* args,
      std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
      std::vector<std::vector<xla::Shape>>* per_core_arg_shapes);

  const tpu::TPUCompileMetadataProto metadata_;

  // Whether to compile given MLIR module in `mlir_module` instead of
  // TensorFlow function referenced in `function_`.
  bool use_mlir_;

  // Function containing the computation to compile.
  NameAttrList function_;

  // A serialized MLIR ModuleOp.
  std::string mlir_module_;

  // Number of different programs to compile. This maps to number of cores in
  // each replica.
  int num_computations_;

 private:
  TPUCompileOpKernelCommon(const TPUCompileOpKernelCommon&) = delete;
  TPUCompileOpKernelCommon& operator=(const TPUCompileOpKernelCommon&) = delete;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_COMMON_H_
