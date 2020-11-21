/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *     ==============================================================================*/

// This file defines structures used in the LMHLO_GPU dialect.

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_LHLO_GPU_OPS_STRUCTS_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_LHLO_GPU_OPS_STRUCTS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

// Order matters, this .inc header is not self-contained, and relies on the
// #includes above.
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops_structs.h.inc"

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_LHLO_GPU_OPS_STRUCTS_H_
