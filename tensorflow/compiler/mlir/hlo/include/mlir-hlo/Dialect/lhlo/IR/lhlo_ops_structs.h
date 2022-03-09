/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file defines structures used in LMHLO dialect.

#ifndef MLIR_HLO_DIALECT_LHLO_IR_LHLO_OPS_STRUCTS_H
#define MLIR_HLO_DIALECT_LHLO_IR_LHLO_OPS_STRUCTS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

// Order matters, this .inc header is not self-contained, and relies on the
// #includes above.
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops_structs.h.inc"

#endif  // MLIR_HLO_DIALECT_LHLO_IR_LHLO_OPS_STRUCTS_H
