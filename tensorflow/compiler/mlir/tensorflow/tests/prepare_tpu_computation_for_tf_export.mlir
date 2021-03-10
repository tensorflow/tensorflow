// RUN: tf-opt %s -split-input-file -verify-diagnostics -prepare-tpu-computation-for-tf-export | FileCheck %s

// CHECK-LABEL: @main
func @main(%arg0: tensor<128x10xf32> {mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"}, %arg1: tensor<10x1024xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<128x1024xf32> {mhlo.sharding = ""}) -> (tensor<10x1024xf32>, tensor<128x1024xf32>) {

  // CHECK: %[[SHARDED_ARG0:.*]] = "tf.XlaSharding"(%arg0) {_XlaSharding = "\08\03\1A\02\01\02\22\02\00\01", sharding = "\08\03\1A\02\01\02\22\02\00\01"}
  // CHECK: %[[SHARDED_ARG1:.*]] = "tf.XlaSharding"(%arg1) {_XlaSharding = "\08\01\1A\01\01\22\01\00", sharding = "\08\01\1A\01\01\22\01\00"}

  // CHECK: "tf.Identity"(%[[SHARDED_ARG1]])
  %0 = "tf.Identity"(%arg1) : (tensor<10x1024xf32>) -> tensor<10x1024xf32>

  // CHECK: "tf.Identity"(%arg2)
  %1 = "tf.Identity"(%arg2) : (tensor<128x1024xf32>) -> tensor<128x1024xf32>
  return %0, %1 : tensor<10x1024xf32>, tensor<128x1024xf32>
}

// -----

// CHECK-NOT: tf.aliasing_output
func @main(%arg0: tensor<2xf32> {tf.aliasing_output = 0 : i64}) -> (tensor<2xf32>) {
  return %arg0 : tensor<2xf32>
}
