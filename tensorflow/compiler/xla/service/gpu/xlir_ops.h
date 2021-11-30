/*
 * Copyright 2021 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MLIR op definitions for xlir_ops library
//
// This file declares the 'xlir' dialect as well as the operators that make up
// the xlir_ops library.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XLIR_OPS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XLIR_OPS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/tensor/opdefs/host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/opdefs/tensor.h"  // from @tf_runtime
#include "tfrt/tensor/opdefs/tensor_shape.h"  // from @tf_runtime

namespace xla {
namespace gpu {

// Dialect for XLIR operations.
class XlirDialect : public mlir::Dialect {
 public:
  static llvm::StringRef getDialectNamespace() { return "xlir"; }
  explicit XlirDialect(mlir::MLIRContext* context);
};

// GPU module data container to be stored in TFRT's resource context and picked
// up by xlir.module.load.
struct GpuModuleData {
  llvm::StringRef blob;

  struct ConstantInfo {
    llvm::StringRef symbol_name;
    llvm::ArrayRef<uint8_t> content;
  };
  std::vector<ConstantInfo> constants;
};

}  // namespace gpu
}  // namespace xla

// TableGen'd declarations
#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/service/gpu/xlir_opdefs.h.inc"

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XLIR_OPS_H_
