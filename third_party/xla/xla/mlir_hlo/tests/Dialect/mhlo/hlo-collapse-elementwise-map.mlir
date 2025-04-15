// RUN: mlir-hlo-opt %s --split-input-file --mhlo-collapse-elementwise-map | \
// RUN: FileCheck %s

// CHECK-LABEL: @map_of_elementwise
func.func @map_of_elementwise(%arg: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "mhlo.map"(%arg, %arg1) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %add = mhlo.add %a, %b : tensor<f32>
    %multiply = mhlo.multiply %add, %b : tensor<f32>
    "mhlo.return"(%multiply) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-NOT: mhlo.map
// CHECK: %[[ADD:.*]] =  mhlo.add %arg0, %arg1 : tensor<?xf32>
// CHECK: %[[MUL:.*]] =  mhlo.multiply %[[ADD]], %arg1 : tensor<?xf32>
// CHECK: return %[[MUL]]

// -----

// CHECK-LABEL: @map_not_collapse
func.func @map_not_collapse(%arg: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<i1>) -> tensor<?xf32> {
  // CHECK: mhlo.map
  %0 = "mhlo.map"(%arg, %arg1) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %0 = "mhlo.if"(%arg2) ({
        "mhlo.return"(%a) : (tensor<f32>) -> ()
      }, {
        "mhlo.return"(%b) : (tensor<f32>) -> ()
      }) : (tensor<i1>) -> tensor<f32>
    "mhlo.return"(%0) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
