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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_FACTORY_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_FACTORY_H_

#include <functional>
#include <memory>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_common.h"

namespace tensorflow {
namespace tpu {

typedef std::function<stream_executor::port::StatusOr<
    std::unique_ptr<TpuCompileOpKernelCommon>>(OpKernelConstruction*)>
    TpuCompileOpImplCreateFn;

// Creates the callback for creating `TpuCompileOpImpl` instance.
stream_executor::port::StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>>
CreateTpuCompileOpImpl(OpKernelConstruction* ctx);

// Creates the callback for creating Mlir `TpuCompileOpImpl` instance.
stream_executor::port::StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>>
CreateTpuCompileOpMlirImpl(OpKernelConstruction* ctx);

// Gets the callback for creating default `TpuCompileOpImpl` instance.
TpuCompileOpImplCreateFn* GetTpuCompileOpCreateFn();

// Gets the callback for creating Mlir `TpuCompileOpImpl` instance.
TpuCompileOpImplCreateFn* GetTpuCompileOpMlirCreateFn();

// Sets the callback for creating default `TpuCompileOpImpl` instance.
void SetTpuCompileOpCreateFn(TpuCompileOpImplCreateFn fn);

// Sets the callback for creating Mlir `TpuCompileOpImpl` instance.
void SetTpuCompileOpMlirCreateFn(TpuCompileOpImplCreateFn fn);
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_IMPL_FACTORY_H_
