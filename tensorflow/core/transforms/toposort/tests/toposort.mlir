// RUN: tfg-transforms-opt %s --pass-pipeline="tfg.graph(tfg-toposort), tfg.func(tfg-toposort)" | FileCheck %s

// Sort graphs topologically

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK-NEXT: %[[PLACEHOLDER0:.*]], %{{.*}} = placeholder name("placeholder0")
  // CHECK-NEXT: %[[PLACEHOLDER1:.*]], %{{.*}} = placeholder name("placeholder1")
  // CHECK-NEXT: AddV2(%[[PLACEHOLDER0]], %[[PLACEHOLDER1]]) name("add")
  %placeholder, %ctl = placeholder name("placeholder0") : () -> (tensor<*xi32>)
  %AddV2, %ctl_0 = AddV2(%placeholder, %placeholder_1) name("add") : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  %placeholder_1, %ctl_2 = placeholder name("placeholder1") : () -> (tensor<*xi32>)
}

// empty graph
// CHECK-LABEL: graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1, bad_consumers = [1, 2, 5, 12]> {
}

// This graph has cycles
// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1, bad_consumers = [1, 2, 5, 12]> {
  // CHECK-NEXT: %[[MERGE:.*]], %{{.*}} = Merge(%[[NEXT:.*]]) name("merge")
  // CHECK-NEXT: %[[PLACEHOLDER1:.*]], %{{.*}} = placeholder name("placeholder1")
  // CHECK-NEXT: %[[PLACEHOLDER0:.*]], %{{.*}} = placeholder name("placeholder0")
  // CHECK-NEXT: %[[ADD:.*]], %{{.*}} = AddV2(%[[PLACEHOLDER0]], %[[MERGE]]) name("add")
  // CHECK-NEXT: %[[NEXT]], %{{.*}} = NextIteration(%[[ADD]]) name("next")
  %AddV2, %ctl_0 = AddV2(%placeholder, %Merge) name("add") : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  %Merge, %ctl_25 = Merge(%NextIteration) name("merge") : (tensor<*xi32>) -> (tensor<*xi32>)
  %NextIteration, %ctl_1 = NextIteration(%AddV2) name("next") : (tensor<*xi32>) -> (tensor<*xi32>)
  %placeholder_2, %ctl_3 = placeholder name("placeholder1") : () -> (tensor<*xi32>)
  %placeholder, %ctl = placeholder name("placeholder0") : () -> (tensor<*xi32>)
}

// CHECK-LABEL: tfg.func @foo
// CHECK-SAME: %[[ARG0:.*]]: tensor
// CHECK-NEXT: %[[ARG1:.*]]: tensor
tfg.func @foo(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> (tensor<*xf32>) {
  // CHECK:      %[[OP2:.*]], %{{.*}} = Op2(%[[ARG0]], %[[ARG1]]) name("b")
  // CHECK-NEXT: %[[OP1:.*]], %{{.*}} = Op1(%[[OP2]], %[[ARG1]]) name("a")
  // CHECK-NEXT: return(%[[OP1]])
  %Op1, %ctl = Op1(%Op2, %arg1) name("a") : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  %Op2, %ctl_0 = Op2(%arg0, %arg1) name("b") : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%Op1) : tensor<*xf32>
}

// This graph has cycles
// CHECK-LABEL: tfg.func @cyclic
tfg.func @cyclic() -> (tensor<*xi32>) {
  // CHECK:      %[[MERGE:.*]], %{{.*}} = Merge(%[[NEXT:.*]]) name("merge")
  // CHECK-NEXT: %[[PLACEHOLDER1:.*]], %{{.*}} = placeholder name("placeholder1")
  // CHECK-NEXT: %[[PLACEHOLDER0:.*]], %{{.*}} = placeholder name("placeholder0")
  // CHECK-NEXT: %[[ADD:.*]], %{{.*}} = AddV2(%[[PLACEHOLDER0]], %[[MERGE]]) name("add")
  // CHECK-NEXT: %[[NEXT]], %{{.*}} = NextIteration(%[[ADD]]) name("next")
  // CHECK-NEXT: return(%[[PLACEHOLDER0]])
  %AddV2, %ctl_0 = AddV2(%placeholder, %Merge) name("add") : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  %Merge, %ctl_25 = Merge(%NextIteration) name("merge") : (tensor<*xi32>) -> (tensor<*xi32>)
  %NextIteration, %ctl_1 = NextIteration(%AddV2) name("next") : (tensor<*xi32>) -> (tensor<*xi32>)
  %placeholder_2, %ctl_3 = placeholder name("placeholder1") : () -> (tensor<*xi32>)
  %placeholder, %ctl = placeholder name("placeholder0") : () -> (tensor<*xi32>)
  return(%placeholder) : tensor<*xi32>
}
