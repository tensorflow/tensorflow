// RUN: tf-opt -tfl-trim-funcs-tf -tfl-trim-funcs-whitelist="bar,foobar" %s | FileCheck %s --dump-input-on-failure

func @foo(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  return %arg0 : tensor<1x4xf32>
}

func @bar(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  return %arg0 : tensor<2x4xf32>
}

func @foobar(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  return %arg0 : tensor<1x4xf32>
}

// CHECK-DAG: func @main
// CHECK-DAG: func @foobar
// CHECK-NOT: func @foo
// CHECK-NOT: func @bar
