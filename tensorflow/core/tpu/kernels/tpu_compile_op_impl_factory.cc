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
#include "tensorflow/core/tpu/kernels/tpu_compile_op_impl_factory.h"

namespace tensorflow {
namespace tpu {
namespace {
static TpuCompileOpImplCreateFn* tpu_compile_op_impl_creation_fn =
    new TpuCompileOpImplCreateFn(CreateTpuCompileOpImpl);
static TpuCompileOpImplCreateFn* tpu_compile_op_mlir_impl_creation_fn =
    new TpuCompileOpImplCreateFn(CreateTpuCompileOpMlirImpl);
}  // namespace

TpuCompileOpImplCreateFn* GetTpuCompileOpCreateFn() {
  return tpu_compile_op_impl_creation_fn;
}

TpuCompileOpImplCreateFn* GetTpuCompileOpMlirCreateFn() {
  return tpu_compile_op_mlir_impl_creation_fn;
}

void SetTpuCompileOpCreateFn(TpuCompileOpImplCreateFn fn) {
  VLOG(1) << "SetTpuCompileOpCreateFn.";
  delete tpu_compile_op_impl_creation_fn;
  tpu_compile_op_impl_creation_fn = new TpuCompileOpImplCreateFn(fn);
}

void SetTpuCompileOpMlirCreateFn(TpuCompileOpImplCreateFn fn) {
  VLOG(1) << "SetTpuCompileOpMlirCreateFn.";
  delete tpu_compile_op_mlir_impl_creation_fn;
  tpu_compile_op_mlir_impl_creation_fn = new TpuCompileOpImplCreateFn(fn);
}
}  // namespace tpu
}  // namespace tensorflow
