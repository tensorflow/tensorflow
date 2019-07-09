//===- Conversion.cpp - Linalg to LLVM conversion driver ------------------===//
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

// RUN: %p/conversion | FileCheck %s

#include "TestHarness.h"

#include "linalg3/ConvertToLLVMDialect.h"

#include "linalg1/Common.h"
#include "linalg1/Dialect.h"
#include "linalg2/Intrinsics.h"
#include "linalg3/Ops.h"
#include "linalg3/Transforms.h"
#include "mlir/IR/OpImplementation.h"

using llvm::StringRef;

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace linalg;
using namespace linalg::common;
using namespace linalg::intrinsics;

FuncOp makeFunctionWithAMatmulOp(Module module, StringRef name) {
  MLIRContext *context = module.getContext();
  auto dynamic2DMemRefType = floatMemRefType<2>(context);
  mlir::FuncOp f = linalg::common::makeFunction(
      module, name,
      {dynamic2DMemRefType, dynamic2DMemRefType, dynamic2DMemRefType}, {});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  // clang-format off
  ValueHandle
    M = dim(f.getArgument(0), 0),
    N = dim(f.getArgument(2), 1),
    K = dim(f.getArgument(0), 1),
    rM = range(constant_index(0), M, constant_index(1)),
    rN = range(constant_index(0), N, constant_index(1)),
    rK = range(constant_index(0), K, constant_index(1)),
    vA = view(f.getArgument(0), {rM, rK}),
    vB = view(f.getArgument(1), {rK, rN}),
    vC = view(f.getArgument(2), {rM, rN});
  matmul(vA, vB, vC);
  ret();
  // clang-format on

  return f;
}

TEST_FUNC(foo) {
  MLIRContext context;
  OwningModuleRef module = Module::create(&context);
  mlir::FuncOp f = makeFunctionWithAMatmulOp(*module, "matmul_as_loops");
  lowerToLoops(f);

  convertLinalg3ToLLVM(*module);

  // clang-format off
  // CHECK:      {{.*}} = llvm.extractvalue {{.*}}[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.mul {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.add {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.mul {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.add {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  // CHECK-NEXT: {{.*}} = llvm.load {{.*}} : !llvm<"float*">
  // CHECK:      {{.*}} = llvm.extractvalue {{.*}}[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.mul {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.add {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.mul {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.add {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  // CHECK-NEXT: {{.*}} = llvm.load {{.*}} : !llvm<"float*">
  // CHECK:      %159 = llvm.extractvalue {{.*}}[1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.mul {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.add {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[3, 1] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.mul {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.add {{.*}}, {{.*}} : !llvm.i64
  // CHECK-NEXT: {{.*}} = llvm.extractvalue {{.*}}[0] : !llvm<"{ float*, i64, [2 x i64], [2 x i64] }">
  // CHECK-NEXT: {{.*}} = llvm.getelementptr {{.*}}[{{.*}}] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  // CHECK-NEXT: llvm.store {{.*}}, {{.*}} : !llvm<"float*">
  // clang-format on
  module->print(llvm::outs());
}

int main() {
  mlir::registerDialect<linalg::LinalgDialect>();
  RUN_TESTS();
  return 0;
}
