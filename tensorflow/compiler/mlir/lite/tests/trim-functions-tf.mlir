// RUN: litert-opt -tfl-trim-funcs-tf="trim-funcs-allowlist=bar,foobar" %s | FileCheck %s

func.func @foo(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  func.return %arg0 : tensor<1x4xf32>
}

func.func @bar(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  func.return %arg0 : tensor<2x4xf32>
}

func.func @foobar(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  func.return %arg0 : tensor<1x4xf32>
}

// CHECK-DAG: func @main
// CHECK-DAG: func @foobar
// CHECK-NOT: func @foo
// CHECK-NOT: func @bar
