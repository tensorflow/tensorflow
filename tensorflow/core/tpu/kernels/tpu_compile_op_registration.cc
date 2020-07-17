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
#include <string>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_common.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_impl.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_impl_factory.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"

namespace tensorflow {
namespace tpu {
using ::stream_executor::port::StatusOr;
StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>> CreateTpuCompileOpImpl(
    OpKernelConstruction* ctx) {
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

StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>> CreateTpuCompileOpMlirImpl(
    OpKernelConstruction* ctx) {
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
}  // namespace tpu
}  // namespace tensorflow
