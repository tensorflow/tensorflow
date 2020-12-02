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
#include "tensorflow/core/tpu/kernels/tpu_compile_op_impl.h"

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/tpu/kernels/tpu_compile.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

namespace tensorflow {
namespace tpu {
using stream_executor::port::StatusOr;

Status TpuCompileOpKernelImpl::Compile(
    const absl::variant<MlirToHloArgs, FunctionToHloArgs>& computation,
    const XLA_TpuMeshState* mesh_state,
    const std::vector<TensorShape>& arg_shapes,
    TpuProgramGroupInterface* tpu_program_group) {
  TF_ASSIGN_OR_RETURN(
      TpuCompilationRequestProto compilation_request,
      CreateTpuCompilationRequest(computation, metadata_, arg_shapes));

  return TpuProgramGroup::CompileAndBuild(compilation_request, mesh_state,
                                          tpu_program_group);
}

class TpuCompileOpImplFactory : public CompileOpImplFactory {
 public:
  StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>> CreateNonMlirImpl(
      OpKernelConstruction* ctx) override {
    NameAttrList function_name;
    TPUCompileMetadataProto metadata;
    TF_RETURN_IF_ERROR(CompileOpMetadataFromContext(ctx, &metadata,
                                                    &function_name,
                                                    /*mlir_module=*/nullptr));
    VLOG(1) << "Create tensorflow::tpu::TpuCompileOpKernelImpl";
    return {std::make_unique<TpuCompileOpKernelImpl>(
        function_name, metadata, metadata.num_cores_per_replica(),
        /*return_hlo_protos=*/false,
        /*unload_cache_on_session_close=*/false)};
  }

  StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>> CreateMlirImpl(
      OpKernelConstruction* ctx) override {
    TPUCompileMetadataProto metadata;
    std::string mlir_module;
    TF_RETURN_IF_ERROR(CompileOpMetadataFromContext(
        ctx, &metadata, /*function_name=*/nullptr, &mlir_module));
    VLOG(1) << "Create tensorflow::tpu::TpuCompileOpKernelImpl";
    return {std::make_unique<TpuCompileOpKernelImpl>(
        mlir_module, metadata, metadata.num_cores_per_replica(),
        /*return_hlo_protos=*/false,
        /*unload_cache_on_session_close=*/false)};
  }
};

#if defined(LIBTPU_ON_GCE)
REGISTER_MODULE_INITIALIZER(tpu_compile_op_impl_factory, {
  VLOG(1) << "register TpuCompileOpImplFactory()";
  CompileOpImplFactory::Register(new TpuCompileOpImplFactory());
});
#endif  // LIBTPU_ON_GCE
}  // namespace tpu
}  // namespace tensorflow
