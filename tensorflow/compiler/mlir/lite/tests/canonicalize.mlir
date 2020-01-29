// RUN: tf-opt -pass-pipeline='func(canonicalize)' %s | FileCheck %s --dump-input-on-failure

// Checks that tfl.reshape should be removed if its output's only user is
// another tfl.reshape
func @reshape_removeAdjacent(tensor<4x4x4xf32>) -> tensor<64xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  return %1 : tensor<64xf32>

// CHECK-LABEL: func @reshape_removeAdjacent
// CHECK:  %cst = constant dense<64> : tensor<1xi32>
// CHECK:  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  return
}

// Checks that tfl.reshape should be removed if its output has more than one
// user but all users are tfl.reshape
func @reshape_removeAdjacentWithMultipleUse(tensor<4x4x4xf32>) -> tensor<64xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  %2 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  %3 = addf %1, %2 : tensor<64xf32>
  return %3 : tensor<64xf32>

// CHECK-LABEL: func @reshape_removeAdjacentWithMultipleUse
// CHECK:  %cst = constant dense<64> : tensor<1xi32>
// CHECK:  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  %1 = "tfl.reshape"(%arg0, %cst) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  %2 = addf %0, %1
// CHECK:  return %2
}

// Checks that tfl.reshape should be kept if its output has more than one
// user and not all users are tfl.reshape
func @reshape_keepAdjacentWithMultipleUse(tensor<4x4x4xf32>) -> (tensor<16x4xf32>, tensor<64xf32>) {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  return %0, %1 : tensor<16x4xf32>, tensor<64xf32>

// CHECK-LABEL: func @reshape_keepAdjacentWithMultipleUse
// CHECK:  %cst = constant dense<[16, 4]> : tensor<2xi32>
// CHECK:  %cst_0 = constant dense<64> : tensor<1xi32>
// CHECK:  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
// CHECK:  %1 = "tfl.reshape"(%arg0, %cst_0) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  return %0, %1
}

// Checks that tfl.reshape should be removed if its output type is the same
// as its input type and both are static.
func @reshape_removeIdentity(tensor<4x4x4xf32>) -> tensor<4x4x4xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %cst = constant dense<[4, 4, 4]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<4x4x4xf32>, tensor<3xi32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>

// CHECK-LABEL: func @reshape_removeIdentity
// CHECK:  return %arg0 : tensor<4x4x4xf32>
}

// Checks that tfl.reshape shouldn't be removed if either output type or input
// type are dynamic.
func @reshape_not_removeIdentity(%arg0: tensor<?xf32>, %arg1: tensor<3xi32>) -> tensor<?xf32> {
  %0 = "tfl.reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<3xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

// CHECK-LABEL: func @reshape_not_removeIdentity
// CHECK-NEXT: "tfl.reshape"
}

// -----

// CHECK-LABEL: @RemoveRedundantUnpackPack
func @RemoveRedundantUnpackPack(%arg0: tensor<2x5xf32>) -> tensor<2x5xf32> {
  %0:2 = "tfl.unpack"(%arg0) {axis = 0 : i32, num = 2 : i32} : (tensor<2x5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  %1 = "tfl.pack"(%0#0, %0#1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<5xf32>, tensor<5xf32>) -> (tensor<2x5xf32>)
  return %1: tensor<2x5xf32>
  // CHECK-NOT: pack
  // CHECK: return %arg0 : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: @RemoveRedundantPack
func @RemoveRedundantPack(%arg0: tensor<2x5xf32>) -> (tensor<2x5xf32>, tensor<5xf32>) {
  %0:2 = "tfl.unpack"(%arg0) {axis = 0 : i32, num = 2 : i32} : (tensor<2x5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  %1 = "tfl.pack"(%0#0, %0#1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<5xf32>, tensor<5xf32>) -> (tensor<2x5xf32>)
  return %1, %0#0: tensor<2x5xf32>, tensor<5xf32>
  // CHECK: %[[UNPACK:.*]]:2 = "tfl.unpack"
  // CHECK-NOT: pack
  // CHECK: return %arg0, %[[UNPACK]]#0 : tensor<2x5xf32>, tensor<5xf32>
}
