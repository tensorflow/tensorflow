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
#include "linalg1/Intrinsics.h"
#include "linalg1/Ops.h"
#include "linalg1/Types.h"
#include "mlir/IR/Function.h"

using namespace linalg;
using namespace linalg::common;
using namespace linalg::intrinsics;
using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

TEST_FUNC(view_op) {
  MLIRContext context;
  Module module(&context);
  auto indexType = mlir::IndexType::get(&context);
  Function *f =
      makeFunction(module, "view_op", {indexType, indexType, indexType}, {});

  ScopedContext scope(f);

  // Let's be lazy and define some custom ops that prevent DCE.
  CustomOperation<OperationHandle> some_consumer("some_consumer");

  // clang-format off
  ValueHandle M(f->getArgument(0)), N(f->getArgument(1)),
    A0 = alloc(floatMemRefType<0>(&context)),
    A1 = alloc(floatMemRefType<1>(&context), ArrayRef<ValueHandle>{M}),
    A2 = alloc(floatMemRefType<2>(&context), ArrayRef<ValueHandle>{M, N}),
    r0 = range(constant_index(3), constant_index(17), constant_index(1)),
    v0 = view(A0),
    v1 = view(A1, ArrayRef<ValueHandle>{r0}),
    v2 = view(A2, ArrayRef<ValueHandle>{r0, r0});
  some_consumer(ArrayRef<ValueHandle>{v0, v1, v2});
  ret();
  // CHECK-LABEL: func @view_op(%arg0: index, %arg1: index, %arg2: index) {
  //       CHECK:   %[[R:.*]] = linalg.range %{{.*}}:%{{.*}}:%{{.*}} : !linalg<"range">
  //  CHECK-NEXT:  {{.*}} = linalg.view {{.*}}[] : !linalg<"view<0xf32>">
  //  CHECK-NEXT:  {{.*}} = linalg.view {{.*}}[%[[R]]] : !linalg<"view<f32>">
  //  CHECK-NEXT:  {{.*}} = linalg.view {{.*}}[%[[R]], %[[R]]] : !linalg<"view<f32xf32>">
  // clang-format on

  cleanupAndPrintFunction(f);
}

TEST_FUNC(slice_op) {
  MLIRContext context;
  Module module(&context);
  auto indexType = mlir::IndexType::get(&context);
  Function *f =
      makeFunction(module, "slice_op", {indexType, indexType, indexType}, {});

  ScopedContext scope(f);

  // Let's be lazy and define some custom op that prevents DCE.
  CustomOperation<OperationHandle> some_consumer("some_consumer");

  // clang-format off
  ValueHandle M(f->getArgument(0)), N(f->getArgument(1)),
      A = alloc(floatMemRefType<2>(&context), {M, N}),
      r1 = range(constant_index(3), constant_index(17), constant_index(1)),
      r2 = range(constant_index(0), N, constant_index(1));
  ViewOp vA = view(A, {r1, r2}).getValue()->getDefiningOp()->cast<ViewOp>();
  IndexHandle i, j;
  LoopNestRangeBuilder({&i, &j}, vA.getRanges())({
    some_consumer(slice(vA, i, 1)),
    some_consumer(slice(slice(vA, j, 0), i, 0)),
  });
  ret();
  // CHECK-LABEL: func @slice_op(%arg0: index, %arg1: index, %arg2: index) {
  //       CHECK: %[[ALLOC:.*]] = alloc(%arg0, %arg1) : memref<?x?xf32>
  //  CHECK-NEXT: %[[R1:.*]] = linalg.range {{.*}}:{{.*}}:{{.*}} : !linalg<"range">
  //  CHECK-NEXT: %[[R2:.*]] = linalg.range {{.*}}:%arg1:{{.*}} : !linalg<"range">
  //  CHECK-NEXT: %[[V:.*]] = linalg.view %0[%[[R1]], %[[R2]]] : !linalg<"view<f32xf32>">
  //  CHECK-NEXT: for %i0 = 3 to 17 {
  //  CHECK-NEXT:   for %i1 = 0 to (d0) -> (d0)(%arg1) {
  //  CHECK-NEXT:     %[[S1:.*]] = linalg.slice %[[V]][*, %i0] { dim : 1 } : !linalg<"view<f32>">
  //  CHECK-NEXT:     "some_consumer"(%[[S1]]) : (!linalg<"view<f32>">) -> ()
  //  CHECK-NEXT:     %[[S2:.*]] = linalg.slice %[[V]][%i1, *] { dim : 0 } : !linalg<"view<f32>">
  //  CHECK-NEXT:     %[[S3:.*]] = linalg.slice %[[S2]][%i0] { dim : 0 } : !linalg<"view<0xf32>">
  //  CHECK-NEXT:     "some_consumer"(%[[S3]]) : (!linalg<"view<0xf32>">) -> ()
  // clang-format on

  cleanupAndPrintFunction(f);
}

int main() {
  RUN_TESTS();
  return 0;
}
