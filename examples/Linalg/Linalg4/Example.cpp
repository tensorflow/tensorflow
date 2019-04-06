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
#include "linalg1/Dialect.h"
#include "linalg2/Intrinsics.h"
#include "linalg3/Ops.h"
#include "linalg4/Transforms.h"
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

TEST_FUNC(matmul_tiled_loops) {
  MLIRContext context;
  Module module(&context);
  mlir::Function *f = makeFunctionWithAMatmulOp(module, "matmul_tiled_loops");
  lowerToTiledLoops(f, {8, 9});
  PassManager pm;
  pm.addPass(createLowerLinalgLoadStorePass());
  if (succeeded(pm.run(f->getModule())))
    cleanupAndPrintFunction(f);

  // clang-format off
  // CHECK-LABEL: func @matmul_tiled_loops(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  //       CHECK: %[[M:.*]] = dim %arg0, 0 : memref<?x?xf32>
  //       CHECK: %[[N:.*]] = dim %arg2, 1 : memref<?x?xf32>
  //       CHECK: %[[K:.*]] = dim %arg0, 1 : memref<?x?xf32>
  //       CHECK: affine.for %i0 = 0 to (d0) -> (d0)(%[[M]]) step 8 {
  //       CHECK:   affine.for %i1 = 0 to (d0) -> (d0)(%[[N]]) step 9 {
  //       CHECK:     affine.for %i2 = 0 to (d0) -> (d0)(%[[K]]) {
  //       CHECK:       affine.for %i3 = max (d0)[s0] -> (s0, d0)(%i0)[%{{.*}}] to min (d0)[s0] -> (s0, d0 + 8)(%i0)[%[[M]]] {
  //       CHECK:         affine.for %i4 = max (d0)[s0] -> (s0, d0)(%i1)[%{{.*}}] to min (d0)[s0] -> (s0, d0 + 9)(%i1)[%[[N]]] {
  //  CHECK-NEXT:           %{{.*}} = cmpi "eq", %i2, %{{.*}} : index
  //  CHECK-NEXT:           %[[I3:.*]] = affine.apply (d0) -> (d0)(%i3)
  //  CHECK-NEXT:           %[[I4:.*]] = affine.apply (d0) -> (d0)(%i4)
  //  CHECK-NEXT:           %{{.*}} = load %arg2[%[[I3]], %[[I4]]] : memref<?x?xf32>
  //  CHECK-NEXT:           %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : f32
  //  CHECK-NEXT:           %[[I2:.*]] = affine.apply (d0) -> (d0)(%i2)
  //  CHECK-NEXT:           %{{.*}} = load %arg1[%[[I2]], %[[I4]]] : memref<?x?xf32>
  //  CHECK-NEXT:           %{{.*}} = load %arg0[%[[I3]], %[[I2]]] : memref<?x?xf32>
  //  CHECK-NEXT:           %{{.*}} = mulf %10, %9 : f32
  //  CHECK-NEXT:           %{{.*}} = addf %7, %11 : f32
  //  CHECK-NEXT:           store %{{.*}}, %arg2[%[[I3]], %[[I4]]] : memref<?x?xf32>
  // clang-format on
}

TEST_FUNC(matmul_tiled_views) {
  MLIRContext context;
  Module module(&context);
  mlir::Function *f = makeFunctionWithAMatmulOp(module, "matmul_tiled_views");
  FuncBuilder b(f);
  lowerToTiledViews(f, {b.create<ConstantIndexOp>(f->getLoc(), 8),
                        b.create<ConstantIndexOp>(f->getLoc(), 9)});
  composeSliceOps(f);

  // clang-format off
  // CHECK-LABEL: func @matmul_tiled_views(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  //       CHECK: %[[M:.*]] = dim %arg0, 0 : memref<?x?xf32>
  //       CHECK: %[[N:.*]] = dim %arg2, 1 : memref<?x?xf32>
  //       CHECK: %[[K:.*]] = dim %arg0, 1 : memref<?x?xf32>
  //       CHECK: affine.for %i0 = 0 to (d0) -> (d0)(%[[M]]) step 8 {
  //  CHECK-NEXT:   affine.for %i1 = 0 to (d0) -> (d0)(%[[N]]) step 9 {
  //  CHECK-NEXT:     %[[i0min:.*]] = affine.apply (d0) -> (d0)(%i0)
  //  CHECK-NEXT:     %[[i0max:.*]] = affine.apply (d0) -> (d0 + 8)(%i0)
  //  CHECK-NEXT:     %[[ri0:.*]] = linalg.range %[[i0min]]:%[[i0max]]:{{.*}} : !linalg.range
  //       CHECK:     %[[rK:.*]] = linalg.range %{{.*}}:%{{.*}}:%{{.*}} : !linalg.range
  //       CHECK:     %[[vA:.*]] = linalg.view %arg0[%[[ri0]], %[[rK]]] : !linalg<"view<?x?xf32>">
  //       CHECK:     %[[i1min:.*]] = affine.apply (d0) -> (d0)(%i1)
  //  CHECK-NEXT:     %[[i1max:.*]] = affine.apply (d0) -> (d0 + 9)(%i1)
  //  CHECK-NEXT:     %[[ri1:.*]] = linalg.range %[[i1min]]:%[[i1max]]:%{{.*}} : !linalg.range
  //  CHECK-NEXT:     %[[vB:.*]]  = linalg.view %arg1[%10, %13] : !linalg<"view<?x?xf32>">
  //  CHECK-NEXT:     %[[vC:.*]]  = linalg.view %arg2[%5, %13] : !linalg<"view<?x?xf32>">
  //  CHECK-NEXT:     linalg.matmul(%[[vA]], %[[vB]], %[[vC]]) : !linalg<"view<?x?xf32>">
  // clang-format on
  cleanupAndPrintFunction(f);
}

TEST_FUNC(matmul_tiled_views_as_loops) {
  MLIRContext context;
  Module module(&context);
  mlir::Function *f =
      makeFunctionWithAMatmulOp(module, "matmul_tiled_views_as_loops");
  FuncBuilder b(f);
  lowerToTiledViews(f, {b.create<ConstantIndexOp>(f->getLoc(), 8),
                        b.create<ConstantIndexOp>(f->getLoc(), 9)});
  composeSliceOps(f);
  lowerToLoops(f);
  // This cannot lower below linalg.load and linalg.store due to lost
  // information related to loop bounds and tiling. There are multiple ways to
  // attack the problem, the best one is an IR change.

  // clang-format off
  // CHECK-LABEL: func @matmul_tiled_views_as_loops(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  //       CHECK: %[[M:.*]] = dim %arg0, 0 : memref<?x?xf32>
  //       CHECK: %[[N:.*]] = dim %arg2, 1 : memref<?x?xf32>
  //       CHECK: %[[K:.*]] = dim %arg0, 1 : memref<?x?xf32>
  //       CHECK: affine.for %i0 = 0 to (d0) -> (d0)(%[[M]]) step 8 {
  //  CHECK-NEXT:   affine.for %i1 = 0 to (d0) -> (d0)(%[[N]]) step 9 {
  //  CHECK-NEXT:     %[[i0min:.*]] = affine.apply (d0) -> (d0)(%i0)
  //  CHECK-NEXT:     %[[i0max:.*]] = affine.apply (d0) -> (d0 + 8)(%i0)
  //  CHECK-NEXT:     %[[ri0:.*]] = linalg.range %[[i0min]]:%[[i0max]]:{{.*}} : !linalg.range
  //       CHECK:     %[[rK:.*]] = linalg.range %{{.*}}:%{{.*}}:%{{.*}} : !linalg.range
  //       CHECK:     %[[vA:.*]] = linalg.view %arg0[%[[ri0]], %[[rK]]] : !linalg<"view<?x?xf32>">
  //       CHECK:     %[[i1min:.*]] = affine.apply (d0) -> (d0)(%i1)
  //  CHECK-NEXT:     %[[i1max:.*]] = affine.apply (d0) -> (d0 + 9)(%i1)
  //  CHECK-NEXT:     %[[ri1:.*]] = linalg.range %[[i1min]]:%[[i1max]]:%{{.*}} : !linalg.range
  //  CHECK-NEXT:     %[[vB:.*]]  = linalg.view %arg1[%10, %13] : !linalg<"view<?x?xf32>">
  //  CHECK-NEXT:     %[[vC:.*]]  = linalg.view %arg2[%5, %13] : !linalg<"view<?x?xf32>">
  //  CHECK-NEXT:     affine.for %i2 = (d0) -> (d0)(%i0) to (d0) -> (d0)(%[[i0max]]) {
  //  CHECK-NEXT:       affine.for %i3 = (d0) -> (d0)(%i1) to (d0) -> (d0)(%[[i1max]]) {
  //  CHECK-NEXT:         affine.for %i4 = 0 to (d0) -> (d0)(%[[K]]) {
  //  CHECK-NEXT:           %{{.*}} = cmpi "eq", %i4, %c0 : index
  //  CHECK-NEXT:           %{{.*}} = linalg.load %[[vC]][%i2, %i3] : !linalg<"view<?x?xf32>">
  //  CHECK-NEXT:           %{{.*}} = select %{{.*}}, %cst, %{{.*}} : f32
  //  CHECK-NEXT:           %{{.*}} = linalg.load %[[vB]][%i4, %i3] : !linalg<"view<?x?xf32>">
  //  CHECK-NEXT:           %{{.*}} = linalg.load %[[vA]][%i2, %i4] : !linalg<"view<?x?xf32>">
  //  CHECK-NEXT:           %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
  //  CHECK-NEXT:           %{{.*}} = addf %{{.*}}, %{{.*}} : f32
  //  CHECK-NEXT:           linalg.store %{{.*}}, %[[vC]][%i2, %i3] : !linalg<"view<?x?xf32>">
  // clang-format on
  cleanupAndPrintFunction(f);
}

int main() {
  mlir::registerDialect<linalg::LinalgDialect>();
  RUN_TESTS();
  return 0;
}
