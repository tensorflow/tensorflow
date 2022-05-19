// RUN: tfg-transforms-opt -pass-pipeline='tfg-lift-graph-to-func{feeds=Placeholder1,Placeholder2 fetches=SomeAdd3 control_rets=SomeAdd4}' %s | FileCheck %s
// RUN: tfg-transforms-opt -pass-pipeline='tfg-lift-graph-to-func{feeds=Placeholder1,Placeholder2,Placeholder1 fetches=SomeAdd3,SomeAdd3 control_rets=SomeAdd4,SomeAdd4}' %s | FileCheck %s

// Test that we can lift the graph into a function by using the provided feeds
// as function arguments and the provided fetch as function results.

// CHECK:   tfg.func @_mlir_lifted_graph(%Placeholder1_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder1", 0 : index], tfg.name = "Placeholder1_0"},
// CHECK-NEXT:                           %Placeholder2_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder2", 0 : index], tfg.name = "Placeholder2_0"})
// CHECK:   tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>

// Feeds substitution:
// CHECK: Add(%Placeholder1_0, %Placeholder2_0) name("SomeAdd1")

// Fetch:
// CHECK: %[[FETCH:.*]], %ctl_6 = Add{{.*}} name("SomeAdd3")
// CHECK: %ctl_[[CTL:.*]] = Add{{.*}} name("SomeAdd4")
// CHECK: return(%[[FETCH]]) [%ctl_[[CTL]]]

tfg.graph #tf_type.version<producer = 34, min_consumer = 5> {
  %Placeholder, %ctl_0 = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  %Placeholder_1, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  %add1, %ctl2 = Add(%Placeholder, %Placeholder_1) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  %add2, %ctl3 = Add(%add1, %Placeholder_1) name("SomeAdd2") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  %add3, %ctl4 = Add(%Placeholder, %add2) name("SomeAdd3") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  %add4, %ctl5 = Add(%add3, %add1) name("SomeAdd4") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  %add5, %ctl6 = Add(%Placeholder, %add4) name("SomeAdd5") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
}
