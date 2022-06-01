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

// This file defines enums used in MHLO and LMHLO.
#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_BASE_ENUMS_H
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_BASE_ENUMS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

// Order matters, this .inc header is not self-contained, and relies on the
// #includes above.

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h.inc"

#endif  // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_BASE_ENUMS_H
