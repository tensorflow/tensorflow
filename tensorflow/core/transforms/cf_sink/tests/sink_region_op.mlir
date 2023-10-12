// RUN: tfg-transforms-opt --tfg-cf-sink %s | not FileCheck %s

// CHECK: tfg.func @test(%[[A0:.*]]: tensor<i32> {tfg.name = "[[A0]]"},
// CHECK:                %[[A1:.*]]: tensor<i32> {tfg.name = "[[A1]]"},
// CHECK:                %[[INDEX:.*]]: tensor<i32> {tfg.name = "[[INDEX]]"},
// CHECK:                %[[COND:.*]]: tensor<i1> {tfg.name = "[[COND]]"})
// CHECK:  {
// CHECK:   %[[IF:.*]], %{{.*}} = StatelessIfRegion %[[COND]] then  {
// CHECK:     %[[CASE:.*]], %{{.*}} = StatelessCaseRegion %[[INDEX]]  {
// CHECK:       %[[CASE0:.*]], %{{.*}} = StatelessCaseRegion %[[INDEX]]  {
// CHECK:         %[[ADD:.*]], %{{.*}} = AddV2(%[[A0]], %[[A1]])
// CHECK:         yield(%[[ADD]])
// CHECK:       },  {
// CHECK:         %[[SUB:.*]], %{{.*}} = Sub(%[[A0]], %[[A1]])
// CHECK:         yield(%[[SUB]])
// CHECK:       }
// CHECK:       yield(%[[CASE0]])
// CHECK:     },  {
// CHECK:       %[[LESS:.*]], %{{.*}} = Less(%[[A0]], %[[A1]])
// CHECK:       %[[IF0:.*]], %{{.*}} = StatelessIfRegion %[[LESS]] then  {
// CHECK:         yield(%[[A0]])
// CHECK:       } else {
// CHECK:         yield(%[[A1]])
// CHECK:       }
// CHECK:       yield(%[[IF0]])
// CHECK:     }
// CHECK:     yield(%[[CASE]])
// CHECK:   } else {
// CHECK:     yield(%[[INDEX]])
// CHECK:   }
// CHECK:   return(%[[IF]])
// CHECK: }
tfg.func @test(%a0: tensor<i32> {tfg.name = "a0"},
               %a1: tensor<i32> {tfg.name = "a1"},
               %branch_index: tensor<i32> {tfg.name = "branch_index"},
               %cond: tensor<i1> {tfg.name = "cond"}) -> (tensor<i32>)
{
  %Add, %ctl_Add = AddV2(%a0, %a1) {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  %Sub, %ctl_Sub = Sub(%a0, %a1) {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)

  %Case, %ctl_Case = StatelessCaseRegion %branch_index {
    yield(%Add) : tensor<i32>
  }, {
    yield(%Sub) : tensor<i32>
  } : (tensor<i32>) -> (tensor<i32>)

  %Less, %ctl_Less = Less(%a0, %a1) {T = i32} : (tensor<i32>, tensor<i32>) -> (tensor<i1>)
  %If_0, %ctl_If_0 = StatelessIfRegion %Less then {
    yield(%a0) : tensor<i32>
  } else {
    yield(%a1) : tensor<i32>
  } : (tensor<i1>) -> (tensor<i32>)

  %Case_0, %ctl_Case_0 = StatelessCaseRegion %branch_index {
    yield(%Case) : tensor<i32>
  }, {
    yield(%If_0) : tensor<i32>
  } : (tensor<i32>) -> (tensor<i32>)

  %If_1, %ctl_If_1 = StatelessIfRegion %cond then {
    yield(%Case_0) : tensor<i32>
  } else {
    yield(%branch_index) : tensor<i32>
  } : (tensor<i1>) -> (tensor<i32>)
  return(%If_1) : tensor<i32>
}
