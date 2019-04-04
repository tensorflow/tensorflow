//===- LLVMIntrinsics.h - declarative builders for LLVM dialect -*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef LINALG1_LLVMINTRINSICS_H_
#define LINALG1_LLVMINTRINSICS_H_

#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/LLVMIR/LLVMDialect.h"

// Expose some LLVM IR instructions to declarative builders.
namespace intrinsics {
using undef = mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::UndefOp>;
using insertvalue =
    mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::InsertValueOp>;
using extractvalue =
    mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::ExtractValueOp>;
using constant = mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::ConstantOp>;
using add = mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::AddOp>;
using sub = mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::SubOp>;
using mul = mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::MulOp>;
using load = mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::LoadOp>;
using store = mlir::edsc::intrinsics::OperationBuilder<mlir::LLVM::StoreOp>;
using gep = mlir::edsc::intrinsics::ValueBuilder<mlir::LLVM::GEPOp>;
} // end namespace intrinsics

#endif // LINALG1_LLVMINTRINSICS_H_
