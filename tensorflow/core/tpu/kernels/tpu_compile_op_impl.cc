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
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"

namespace tensorflow {
namespace tpu {
Status TpuCompileOpKernelImpl::Compile(
    const std::variant<MlirToHloArgs, FunctionToHloArgs>& computation,
    const XLA_TpuMeshState* mesh_state,
    const std::vector<TensorShape>& arg_shapes,
    TpuProgramGroupInterface* tpu_program_group) {
  TF_ASSIGN_OR_RETURN(
      TpuCompilationRequestProto compilation_request,
      CreateTpuCompilationRequest(computation, metadata_, arg_shapes));

  return TpuProgramGroup::CompileAndBuild(compilation_request, mesh_state,
                                          tpu_program_group);
}
}  // namespace tpu
}  // namespace tensorflow
