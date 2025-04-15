/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_HOST_RUNTIME_TFRT_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_HOST_RUNTIME_TFRT_OPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_HOST_RUNTIME_TFRT_OPS_H_
