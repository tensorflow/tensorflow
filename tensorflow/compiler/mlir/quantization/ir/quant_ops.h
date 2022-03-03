/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Quantization Ops and Types

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_IR_QUANT_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_IR_QUANT_OPS_H_

#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/ir/quant_ops_dialect.h.inc"

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/quantization/ir/quant_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_IR_QUANT_OPS_H_
