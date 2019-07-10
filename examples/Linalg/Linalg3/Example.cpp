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
#include "linalg3/Transforms.h"
#include "mlir/IR/OpImplementation.h"

using llvm::StringRef;

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace linalg;
using namespace linalg::common;
using namespace linalg::intrinsics;

FuncOp makeFunctionWithAMatmulOp(ModuleOp module, StringRef name) {
  MLIRContext *context = module.getContext();
  auto dynamic2DMemRefType = floatMemRefType<2>(context);
  mlir::FuncOp f = linalg::common::makeFunction(
      module, name,
      {dynamic2DMemRefType, dynamic2DMemRefType, dynamic2DMemRefType}, {});

  mlir::OpBuilder builder(f.getBody());
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

TEST_FUNC(matmul_as_matvec) {
  MLIRContext context;
  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
  mlir::FuncOp f = makeFunctionWithAMatmulOp(module, "matmul_as_matvec");
  lowerToFinerGrainedTensorContraction(f);
  composeSliceOps(f);
  // clang-format off
  // CHECK-LABEL: func @matmul_as_matvec(%{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>) {
  //       CHECK: %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
  //       CHECK: %[[vA:.*]] = linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  //       CHECK: affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[N]]) {
  //       CHECK:   %[[vB:.*]] = linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
  //       CHECK:   %[[vC:.*]] = linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
  //       CHECK:   linalg.matvec(%[[vA]], %[[vB]], %[[vC]]) : !linalg.view<?xf32>
  // clang-format on
  cleanupAndPrintFunction(f);
}

TEST_FUNC(matmul_as_dot) {
  MLIRContext context;
  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
  mlir::FuncOp f = makeFunctionWithAMatmulOp(module, "matmul_as_dot");
  lowerToFinerGrainedTensorContraction(f);
  lowerToFinerGrainedTensorContraction(f);
  composeSliceOps(f);
  // clang-format off
  // CHECK-LABEL: func @matmul_as_dot(%{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>) {
  //       CHECK: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32>
  //       CHECK: %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
  //       CHECK: affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[N]]) {
  //       CHECK:   %[[vB:.*]] = linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
  //  CHECK-NEXT:   affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[M]]) {
  //       CHECK:     %[[vA:.*]] = linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>, index, !linalg.range, !linalg.view<?xf32>
  //  CHECK-NEXT:     %[[vC:.*]] = linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>, index, index, !linalg.view<f32>
  //  CHECK-NEXT:     linalg.dot(%[[vA]], %[[vB]], %[[vC]]) : !linalg.view<f32>
  // clang-format on
  cleanupAndPrintFunction(f);
}

TEST_FUNC(matmul_as_loops) {
  MLIRContext context;
  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
  mlir::FuncOp f = makeFunctionWithAMatmulOp(module, "matmul_as_loops");
  lowerToLoops(f);
  composeSliceOps(f);
  // clang-format off
  // CHECK-LABEL: func @matmul_as_loops(%{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>) {
  //       CHECK: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32>
  //       CHECK: %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
  //       CHECK: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
  //       CHECK: %[[rM:.*]] = linalg.range %{{.*}}:%[[M]]:%{{.*}} : !linalg.range
  //       CHECK: %[[rN:.*]] = linalg.range %{{.*}}:%[[N]]:%{{.*}} : !linalg.range
  //       CHECK: %[[rK:.*]] = linalg.range %{{.*}}:%[[K]]:%{{.*}} : !linalg.range
  //       CHECK: %[[vA:.*]] = linalg.view %{{.*}}[%[[rM]], %[[rK]]] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  //       CHECK: %[[vB:.*]] = linalg.view %{{.*}}[%[[rK]], %[[rN]]] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  //       CHECK: %[[vC:.*]] = linalg.view %{{.*}}[%[[rM]], %[[rN]]] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  //       CHECK: affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[M]]) {
  //       CHECK:   affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[N]]) {
  //       CHECK:     affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[K]]) {
  //       CHECK:       %{{.*}} = cmpi "eq", %{{.*}} : index
  //       CHECK:       %{{.*}} = linalg.load %[[vC]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
  //       CHECK:       %{{.*}} = select {{.*}} : f32
  //       CHECK:       %{{.*}} = linalg.load %[[vB]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
  //       CHECK:       %{{.*}} = linalg.load %[[vA]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
  //       CHECK:       %{{.*}} = mulf {{.*}} : f32
  //       CHECK:       %{{.*}} = addf {{.*}} : f32
  //       CHECK:       linalg.store {{.*}}[%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
  // clang-format on
  cleanupAndPrintFunction(f);
}

TEST_FUNC(matmul_as_matvec_as_loops) {
  MLIRContext context;
  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
  mlir::FuncOp f =
      makeFunctionWithAMatmulOp(module, "matmul_as_matvec_as_loops");
  lowerToFinerGrainedTensorContraction(f);
  lowerToLoops(f);
  composeSliceOps(f);
  // clang-format off
  // CHECK-LABEL: func @matmul_as_matvec_as_loops(%{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>) {
  //       CHECK: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32>
  //       CHECK: %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
  //       CHECK: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
  //       CHECK: %[[vA:.*]] = linalg.view %{{.*}}[{{.*}}, {{.*}}] : memref<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  //       CHECK: affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[N]]) {
  //       CHECK:   %[[vB:.*]] = linalg.view %{{.*}}[{{.*}}, {{.*}}] : memref<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
  //       CHECK:   %[[vC:.*]] = linalg.view %{{.*}}[{{.*}}, {{.*}}] : memref<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
  //       CHECK:   affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[M]]) {
  //       CHECK:     affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[K]]) {
  //       CHECK:        %{{.*}} = cmpi "eq", %{{.*}}, %{{.*}} : index
  //       CHECK:        %[[C:.*]] = linalg.load %[[vC]][%{{.*}}] : !linalg.view<?xf32>
  //       CHECK:        %[[C2:.*]] = select %{{.*}}, %{{.*}}, %[[C]] : f32
  //       CHECK:        %[[B:.*]] = linalg.load %[[vB]][%{{.*}}] : !linalg.view<?xf32>
  //       CHECK:        %[[A:.*]] = linalg.load %[[vA]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
  //       CHECK:        %{{.*}} = mulf %[[A]], %[[B]] : f32
  //       CHECK:        %{{.*}} = addf %[[C2]], %{{.*}} : f32
  //       CHECK:        linalg.store %{{.*}}, %{{.*}}[%{{.*}}] : !linalg.view<?xf32>
  // clang-format on
  cleanupAndPrintFunction(f);
}

TEST_FUNC(matmul_as_matvec_as_affine) {
  MLIRContext context;
  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
  mlir::FuncOp f =
      makeFunctionWithAMatmulOp(module, "matmul_as_matvec_as_affine");
  lowerToFinerGrainedTensorContraction(f);
  composeSliceOps(f);
  lowerToLoops(f);
  PassManager pm;
  pm.addPass(createLowerLinalgLoadStorePass());
  if (succeeded(pm.run(f.getModule())))
    cleanupAndPrintFunction(f);

  // clang-format off
  // CHECK-LABEL: func @matmul_as_matvec_as_affine(%{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>, %{{.*}}: memref<?x?xf32>) {
  //       CHECK: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32>
  //       CHECK: %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
  //       CHECK: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32>
  //       CHECK: affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[N]]) {
  //   CHECK-NOT: {{.*}} = linalg.
  //       CHECK:   affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[M]]) {
  //       CHECK:     affine.for %{{.*}} = 0 to (d0) -> (d0)(%[[K]]) {
  //       CHECK:       %{{.*}} = cmpi "eq", %{{.*}}, %{{.*}} : index
  //       CHECK:       %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
  //       CHECK:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : f32
  //   CHECK-NOT: {{.*}} = linalg.
  //       CHECK:       %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
  //       CHECK:       %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
  //       CHECK:       %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
  //       CHECK:       %{{.*}} = addf %{{.*}}, %{{.*}} : f32
  //   CHECK-NOT: {{.*}} = linalg.
  //       CHECK:       store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
  // clang-format on
}

int main() {
  mlir::registerDialect<linalg::LinalgDialect>();
  RUN_TESTS();
  return 0;
}
