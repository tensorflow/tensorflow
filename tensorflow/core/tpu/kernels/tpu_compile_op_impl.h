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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_H_

#include <string>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_common.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

namespace tensorflow {
namespace tpu {

// Base class for TpuCompileOp and TpuCompileMlirOp.
// Depends on whether it is given a computation in the form of serialized MLIR
// module or a Tensorflow function, TpuCompileOpKernelImpl converts computation
// into XLA HLO and then into a TPU execuable binary.
class TpuCompileOpKernelImpl : public TpuCompileOpKernelCommon {
 public:
  TpuCompileOpKernelImpl(const std::string& mlir_module,
                         const tpu::TPUCompileMetadataProto& metadata,
                         int num_computations, bool return_hlo_protos,
                         bool unload_cache_on_session_close)
      : TpuCompileOpKernelCommon(mlir_module, metadata, num_computations,
                                 return_hlo_protos,
                                 unload_cache_on_session_close) {}

  TpuCompileOpKernelImpl(const NameAttrList& function,
                         const tpu::TPUCompileMetadataProto& metadata,
                         int num_computations, bool return_hlo_protos,
                         bool unload_cache_on_session_close)
      : TpuCompileOpKernelCommon(
            function, metadata, num_computations, return_hlo_protos,
            unload_cache_on_session_close, /*persistent_cache=*/nullptr) {}

  Status Compile(
      const absl::variant<MlirToHloArgs, FunctionToHloArgs>& computation,
      const XLA_TpuMeshState* mesh_state,
      const std::vector<TensorShape>& arg_shapes,
      const TpuCompilationCacheKey* key,
      TpuProgramGroupInterface* tpu_program_group) override;
};
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_H_
