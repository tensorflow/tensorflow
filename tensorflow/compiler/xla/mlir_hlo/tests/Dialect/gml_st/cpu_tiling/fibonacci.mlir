// RUN: mlir-hlo-opt %s --gml-st-cpu-tiling-pipeline | FileCheck %s

func.func @fuse_fibonacci(%init : tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64

  %0 = linalg.fill ins(%c0 : i64) outs(%init : tensor<?xi64>) -> tensor<?xi64>
  %1 = linalg.fill ins(%c1 : i64) outs(%init : tensor<?xi64>) -> tensor<?xi64>
  %2 = linalg.map { arith.addi } ins(%0, %1 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %3 = linalg.map { arith.addi } ins(%1, %2 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %4 = linalg.map { arith.addi } ins(%2, %3 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %5 = linalg.map { arith.addi } ins(%3, %4 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %6 = linalg.map { arith.addi } ins(%4, %5 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %7 = linalg.map { arith.addi } ins(%5, %6 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %8 = linalg.map { arith.addi } ins(%6, %7 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %9 = linalg.map { arith.addi } ins(%7, %8 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %10 = linalg.map { arith.addi } ins(%8, %9 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %11 = linalg.map { arith.addi } ins(%9, %10 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %12 = linalg.map { arith.addi } ins(%10, %11 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %13 = linalg.map { arith.addi } ins(%11, %12 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %14 = linalg.map { arith.addi } ins(%12, %13 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %15 = linalg.map { arith.addi } ins(%13, %14 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %16 = linalg.map { arith.addi } ins(%14, %15 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %17 = linalg.map { arith.addi } ins(%15, %16 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %18 = linalg.map { arith.addi } ins(%16, %17 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %19 = linalg.map { arith.addi } ins(%17, %18 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %20 = linalg.map { arith.addi } ins(%18, %19 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %21 = linalg.map { arith.addi } ins(%19, %20 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %22 = linalg.map { arith.addi } ins(%20, %21 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %23 = linalg.map { arith.addi } ins(%21, %22 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %24 = linalg.map { arith.addi } ins(%22, %23 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %25 = linalg.map { arith.addi } ins(%23, %24 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %26 = linalg.map { arith.addi } ins(%24, %25 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %27 = linalg.map { arith.addi } ins(%25, %26 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %28 = linalg.map { arith.addi } ins(%26, %27 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %29 = linalg.map { arith.addi } ins(%27, %28 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %30 = linalg.map { arith.addi } ins(%28, %29 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %31 = linalg.map { arith.addi } ins(%29, %30 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %32 = linalg.map { arith.addi } ins(%30, %31 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %33 = linalg.map { arith.addi } ins(%31, %32 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %34 = linalg.map { arith.addi } ins(%32, %33 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %35 = linalg.map { arith.addi } ins(%33, %34 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %36 = linalg.map { arith.addi } ins(%34, %35 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %37 = linalg.map { arith.addi } ins(%35, %36 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %38 = linalg.map { arith.addi } ins(%36, %37 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %39 = linalg.map { arith.addi } ins(%37, %38 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  func.return %39 : tensor<?xi64>
}
// CHECK-LABEL: @fuse_fibonacci
// CHECK-DAG: %[[SCALAR_RESULT:.*]] = arith.constant 63245986 : i64
// CHECK-DAG: %[[VECTOR_RESULT:.*]] = arith.constant dense<63245986> : vector<8xi64>

// CHECK:     scf.for
// CHECK:       %[[VEC:.*]] = vector.transfer_write %[[VECTOR_RESULT]]
// CHECK:       scf.yield %[[VEC]]

// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         tensor.insert %[[SCALAR_RESULT]]
// CHECK:       tensor.insert_slice
