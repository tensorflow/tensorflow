// RUN: xla-runtime-opt %s --xla-math-legalization \
// RUN:   | FileCheck %s

// RUN: xla-runtime-opt %s --xla-math-legalization=enable-approximations=0 \
// RUN:   | FileCheck --check-prefix=NO-APPROX %s

// CHECK-LABEL: func @log1p(
//   CHECK-DAG:   %[[C1:.*]] = llvm.mlir.constant(1.0
//       CHECK:   %[[P1:.*]] = llvm.fadd %[[C1]]
//       CHECK:   %[[RET:.*]] = llvm.intr.log(%[[P1]])
//       CHECK:   return %[[RET]]

// NO-APPROX-LABEL: func @log1p(
//       NO-APPROX:   call @log1pf

func.func @log1p(%arg0: f32) -> f32 {
  %0 = math.log1p %arg0 : f32
  func.return %0 : f32
}