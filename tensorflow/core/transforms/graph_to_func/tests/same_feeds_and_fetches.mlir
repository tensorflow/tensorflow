// RUN: tfg-transforms-opt -pass-pipeline='tfg-lift-graph-to-func{feeds=Placeholder1 fetches=Placeholder1}' %s | FileCheck %s

// CHECK:   tfg.func @_mlir_lifted_graph(%Placeholder1_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder1", 0 : index], tfg.name = "Placeholder1_0"}
// CHECK-NEXT: -> (tensor<*xf32> {tfg.name = "Placeholder1_0"})
// CHECK:   tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>
tfg.graph #tf_type.version<producer = 34, min_consumer = 5> {
  %Placeholder, %ctl_0 = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[PLACEHOLDER1:.*]], {{.*}} Placeholder name("Placeholder2")
  %Placeholder_1, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[ADD1:.*]], {{.*}} = Add(%Placeholder1_0, %[[PLACEHOLDER1]]) name("SomeAdd1")
  %add1, %ctl2 = Add(%Placeholder, %Placeholder_1) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: Add(%[[ADD1]], %[[PLACEHOLDER1]]) name("SomeAdd2")
  %add2, %ctl3 = Add(%add1, %Placeholder_1) name("SomeAdd2") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: return(%Placeholder1_0)
}
