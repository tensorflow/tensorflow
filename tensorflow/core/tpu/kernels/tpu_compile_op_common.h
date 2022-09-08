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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_COMMON_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_COMMON_H_

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"

namespace tensorflow {
namespace tpu {
// Forward declaration, defined below.
class TpuCompileOpKernelCommon;

// A base factory class for creating a `TpuCompileOpKernelImpl` variant.
// By design, the actual factory can only be set once.
class CompileOpImplFactory {
 public:
  virtual ~CompileOpImplFactory() = default;

  virtual stream_executor::port::StatusOr<
      std::unique_ptr<TpuCompileOpKernelCommon>>
  CreateNonMlirImpl(OpKernelConstruction* ctx) = 0;

  virtual stream_executor::port::StatusOr<
      std::unique_ptr<TpuCompileOpKernelCommon>>
  CreateMlirImpl(OpKernelConstruction* ctx) = 0;

  static CompileOpImplFactory* Get();
  static void Register(CompileOpImplFactory* factory);

 private:
  static CompileOpImplFactory* factory_;
};

// Abstract base class for TpuCompileOpKernel implementation.
class TpuCompileOpKernelCommon {
 public:
  TpuCompileOpKernelCommon(const std::string& mlir_module,
                           const tpu::TPUCompileMetadataProto metadata,
                           int num_computations, bool return_hlo_protos,
                           bool unload_cache_on_session_close)
      : metadata_(metadata),
        use_mlir_(true),
        mlir_module_(mlir_module),
        num_computations_(num_computations),
        return_hlo_protos_(return_hlo_protos),
        unload_cache_entry_on_session_close_(unload_cache_on_session_close),
        persistent_cache_(nullptr) {
    mlir_module_fingerprint_ = tensorflow::Fingerprint64(mlir_module_);
  }

  TpuCompileOpKernelCommon(
      const NameAttrList& function, const tpu::TPUCompileMetadataProto metadata,
      int num_computations, bool return_hlo_protos,
      bool unload_cache_on_session_close,
      std::unique_ptr<TpuPersistentCompilationCacheInterface> persistent_cache)
      : metadata_(metadata),
        use_mlir_(false),
        function_(function),
        num_computations_(num_computations),
        return_hlo_protos_(return_hlo_protos),
        unload_cache_entry_on_session_close_(unload_cache_on_session_close),
        persistent_cache_(std::move(persistent_cache)) {}

  virtual ~TpuCompileOpKernelCommon() = default;

  void Compute(OpKernelContext* ctx);

  // Lowers Mlir or TF Function computation into HLO IR and using XLA compiler
  // compiles into TPU programs ready for execution.
  virtual Status Compile(
      const absl::variant<MlirToHloArgs, FunctionToHloArgs>& computation,
      const XLA_TpuMeshState* mesh_state,
      const std::vector<TensorShape>& arg_shapes,
      const TpuCompilationCacheKey* key,
      TpuProgramGroupInterface* tpu_program_group) = 0;

  // Performs shape inference on `computation`, filling shape_info with operator
  // shapes. The shapes of the _Arg nodes are taken from `arg_shapes`.
  static Status RunShapeInferenceOnComputation(
      const tpu::TPUCompileMetadataProto& metadata,
      const std::vector<PartialTensorShape>& arg_shapes, Graph* graph,
      FunctionLibraryRuntime* flr, GraphShapeInfo* shape_info);

 protected:
  Status ComputeInternal(OpKernelContext* ctx);

  // Compile TPU program locally and populate the host compilation cache.
  Status CompileLocallyAndFillHostCache(
      FunctionLibraryRuntime* flib_runtime,
      const SessionMetadata* session_metadata,
      const TpuMeshStateInterface* mesh_state,
      const std::vector<TensorShape>& dynamic_shapes,
      const OpInputList& guaranteed_constants,
      const tpu::TpuCompilationCacheKey& key,
      TpuProgramGroupInterface* tpu_program_group);

  Status CompileLocallyAndFillHostCacheInternal(
      FunctionLibraryRuntime* flib_runtime,
      const SessionMetadata* session_metadata,
      const TpuMeshStateInterface* mesh_state,
      const std::vector<TensorShape>& dynamic_shapes,
      const OpInputList& guaranteed_constants,
      const tpu::TpuCompilationCacheKey& key,
      TpuProgramGroupInterface* tpu_program_group);

  // Lookup from persistent compilation cache and populate both host cache and
  // persistent cache.
  virtual Status LookupPersistentCompilationCacheAndFillCaches(
      FunctionLibraryRuntime* flib_runtime,
      const SessionMetadata* session_metadata,
      const TpuMeshStateInterface* mesh_state,
      const std::vector<TensorShape>& dynamic_shapes,
      const OpInputList& guaranteed_constants,
      TpuPersistentCompilationCacheInterface* persistent_cache,
      const tpu::TpuCompilationCacheKey& key,
      TpuProgramGroupInterface* tpu_program_group) {
    LOG(FATAL) << "Lookup from a persistent cache is NOT supported.";
  }

  // Sleeps for `kSleepSeconds` seconds to give time for TPUCompileOp to finish
  // before terminating peacefully.
  static void ExitCountdown(Env* env, std::shared_ptr<std::atomic<bool>> done);

  // Converts the `dynamic_shapes` arguments to the compile operator into
  // TensorShapes.
  static Status GetDynamicShapes(OpKernelContext* ctx,
                                 std::vector<TensorShape>* shapes);

  tpu::TPUCompileMetadataProto metadata_;

  // Whether to compile given MLIR module in `mlir_module` instead of
  // TensorFlow function referenced in `function_`.
  bool use_mlir_;

  // Function containing the computation to compile.
  NameAttrList function_;

  // A serialized MLIR ModuleOp.
  std::string mlir_module_;
  // Fingerprint of the MLIR Module created once on construction to avoid paying
  // the cost on each invocation.
  uint64 mlir_module_fingerprint_ = 0;

  // Number of different programs to compile. This maps to number of cores in
  // each replica.
  int num_computations_;

  // A flag to populate HLO protos field in CompilationResultProto. The HLO
  // metadata could be large so default to not populating it unless explicitly
  // requested.
  bool return_hlo_protos_;

  // If enabled, DirectSession::Close will unload cache entries created during
  // the lifetime of the session.
  bool unload_cache_entry_on_session_close_;

  // Persistent cache for compiled TPU program for inference.
  std::unique_ptr<TpuPersistentCompilationCacheInterface> persistent_cache_;

  Status RegisterXLAFingerprints(const std::vector<TensorShape>& arg_shapes,
                                 TpuProgramGroupInterface* tpu_program_group,
                                 uint64 fingerprint);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TpuCompileOpKernelCommon);
};
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_COMMON_H_
