// RUN: tfg-transforms-opt --tfg-cf-sink %s | FileCheck %s

// CHECK: tfg.func @test(%[[ARG0:.*]]: tensor<i32>,
// CHECK:                %[[ARG1:.*]]: tensor<i32>,
// CHECK:                %[[ARG2:.*]]: tensor<i1>)
// CHECK:  {
// CHECK:   %[[IF:.*]], %{{.*}} = IfRegion %[[ARG2]] then  {
// CHECK:     %[[ADDV2:.*]], %{{.*}} = AddV2(%[[ARG0]], %[[ARG1]])
// CHECK:     yield(%[[ADDV2]])
// CHECK:   } else {
// CHECK:     %[[SUB:.*]], %{{.*}} = Sub(%[[ARG0]], %[[ARG1]])
// CHECK:     yield(%[[SUB]])
// CHECK:   }
// CHECK:   return(%[[IF]])
// CHECK: }
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>, %cond: tensor<i1>) -> (tensor<i32>) {
  %AddV2, %ctl = AddV2(%arg0, %arg1) {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  %Sub, %ctl_0 = Sub(%arg0, %arg1) {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  %IfRegion, %ctl_2 = IfRegion %cond then {
    yield(%AddV2) : tensor<i32>
  } else {
    yield(%Sub) : tensor<i32>
  } : (tensor<i1>) -> (tensor<i32>)
  return(%IfRegion) : tensor<i32>
}
