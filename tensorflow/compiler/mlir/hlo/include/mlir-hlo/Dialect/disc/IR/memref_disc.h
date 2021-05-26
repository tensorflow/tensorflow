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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MEMREF_DISC_IR_MEMREF_DISC_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MEMREF_DISC_IR_MEMREF_DISC_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"

class Location;
class OpBuilder;

//===----------------------------------------------------------------------===//
// MemRefDisc Dialect
//===----------------------------------------------------------------------===//
// MemRefDisc Dialect is an expansion for MemRefDialect for DISC. This is
// a temporary workaround for optimizing the index calculations during codegen.
// Refer to
// https://llvm.discourse.group/t/add-an-expanded-load-store-op-in-memref-dialect/3503/26
// for more informations.
//
// TODO: Re-visit this with solutions like the explicit linearize/delinearize
// op definition.

#include "mlir-hlo/Dialect/disc/IR/memref_disc_ops_dialect.h.inc"

//===----------------------------------------------------------------------===//
// MemRefDisc Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/disc/IR/memref_disc_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MEMREF_DISC_IR_MEMREF_DISC_H_
