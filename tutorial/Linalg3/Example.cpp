//===- Example.cpp - Our running example ----------------------------------===//
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

// RUN: %p/test | FileCheck %s

#include "TestHarness.h"
#include "linalg1/Common.h"
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

Function *makeFunctionWithAMatmulOp(Module &module, StringRef name) {
  MLIRContext *context = module.getContext();
  auto dynamic2DMemRefType = floatMemRefType<2>(context);
  mlir::Function *f = linalg::common::makeFunction(
      module, name,
      {dynamic2DMemRefType, dynamic2DMemRefType, dynamic2DMemRefType}, {});

  ScopedContext scope(f);
  // clang-format off
  ValueHandle
    M = dim(f->getArgument(0), 0),
    N = dim(f->getArgument(2), 1),
    K = dim(f->getArgument(0), 1),
    rM = range(constant_index(0), M, constant_index(1)),
    rN = range(constant_index(0), N, constant_index(1)),
    rK = range(constant_index(0), K, constant_index(1)),
    vA = view(f->getArgument(0), {rM, rK}),
    vB = view(f->getArgument(1), {rK, rN}),
    vC = view(f->getArgument(2), {rM, rN});
  matmul(vA, vB, vC);
  ret();
  // clang-format on

  return f;
}

TEST_FUNC(matmul_as_matvec) {
  MLIRContext context;
  Module module(&context);
  mlir::Function *f = makeFunctionWithAMatmulOp(module, "matmul_as_matvec");
  lowerToFinerGrainedTensorContraction(f);
  // clang-format off
  // CHECK-LABEL: func @matmul_as_matvec(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  //       CHECK: %[[N:.*]] = dim %arg2, 1 : memref<?x?xf32>
  //       CHECK: affine.for %i0 = 0 to (d0) -> (d0)(%[[N]]) {
  //  CHECK-NEXT:   %[[vB:.*]] = linalg.slice %{{.*}}[*, %i0] { dim : 1 } : !linalg<"view<f32>">
  //  CHECK-NEXT:   %[[vC:.*]] = linalg.slice %{{.*}}[*, %i0] { dim : 1 } : !linalg<"view<f32>">
  //  CHECK-NEXT:   linalg.matvec {%{{.*}}, %[[vB]]} -> {%[[vC]]}
  // clang-format on
  cleanupAndPrintFunction(f);
}

TEST_FUNC(matmul_as_dot) {
  MLIRContext context;
  Module module(&context);
  mlir::Function *f = makeFunctionWithAMatmulOp(module, "matmul_as_dot");
  lowerToFinerGrainedTensorContraction(f);
  lowerToFinerGrainedTensorContraction(f);
  // clang-format off
  // CHECK-LABEL: func @matmul_as_dot(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  //       CHECK: %[[M:.*]] = dim %arg0, 0 : memref<?x?xf32>
  //       CHECK: %[[N:.*]] = dim %arg2, 1 : memref<?x?xf32>
  //       CHECK: affine.for %i0 = 0 to (d0) -> (d0)(%[[N]]) {
  //  CHECK-NEXT:   %[[vB:.*]] = linalg.slice {{.*}}[*, %i0] { dim : 1 } : !linalg<"view<f32>">
  //  CHECK-NEXT:   %[[sC:.*]]  = linalg.slice {{.*}}[*, %i0] { dim : 1 } : !linalg<"view<f32>">
  //  CHECK-NEXT:   affine.for %i1 = 0 to (d0) -> (d0)(%[[M]]) {
  //  CHECK-NEXT:     %[[vA:.*]] = linalg.slice {{.*}}[%i1, *] { dim : 0 } : !linalg<"view<f32>">
  //  CHECK-NEXT:     %[[vC:.*]] = linalg.slice %[[sC]][%i1] { dim : 0 } : !linalg<"view<0xf32>">
  //  CHECK-NEXT:     linalg.dot {%[[vA]], %[[vB]]} -> {%[[vC]]}
  // clang-format on
  cleanupAndPrintFunction(f);
}

int main() {
  RUN_TESTS();
  return 0;
}
