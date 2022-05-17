// RUN: tfg-transforms-opt --tfg-lower-func-to-graph -verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK: tfg.graph #tf_type.version<producer = 34, min_consumer = 5>
tfg.func @_mlir_lifted_graph(%Placeholder13A0: tensor<*xf32> {tfg.name = "Placeholder1:0"},
                             %Placeholder23A0: tensor<*xf32> {tfg.name = "Placeholder2:0"})
    -> (tensor<*xf32> {tfg.name = "SomeAdd3:0"})
 attributes {tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>} {
  // CHECK: %[[PLACEHOLDER1:.*]], %[[CTRL0:.*]] = Placeholder name("Placeholder1")
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[PLACEHOLDER2:.*]], %[[CTRL1:.*]] = Placeholder name("Placeholder2")
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[SOMEADD1:.*]], %[[CTRL2:.*]] = Add(%[[PLACEHOLDER1]], %[[PLACEHOLDER2]]) name("SomeAdd1")
  %Add, %ctl_2 = Add(%Placeholder13A0, %Placeholder23A0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK-NOT: return
  return(%Add) [%ctl_2] : tensor<*xf32>
}

// -----

// expected-error@+1 {{lifted arg can't find the associated operation: Unknown:0}}
tfg.func @_mlir_lifted_graph(%Placeholder13A0: tensor<*xf32> {tfg.name = "Unknown:0"},
                             %Placeholder23A0: tensor<*xf32> {tfg.name = "Placeholder2:0"})
     -> (tensor<*xf32> {tfg.name = "SomeAdd3:0"})
 attributes {tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>} {
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  %Add, %ctl_2 = Add(%Placeholder13A0, %Placeholder23A0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%Add) [%ctl_2] : tensor<*xf32>
}

// -----

// expected-error@+1 {{result index out of bound: seeing index 3 from arg: Placeholder1:3, but op only has 2 results}}
tfg.func @_mlir_lifted_graph(%Placeholder13A0: tensor<*xf32> {tfg.name = "Placeholder1:3"},
                             %Placeholder23A0: tensor<*xf32> {tfg.name = "Placeholder2:0"})
     -> (tensor<*xf32> {tfg.name = "SomeAdd3:0"})
 attributes {tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>} {
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  %Add, %ctl_2 = Add(%Placeholder13A0, %Placeholder23A0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%Add) [%ctl_2] : tensor<*xf32>
}

// -----

// expected-error@+1 {{Invalid result index format: Placeholder1:a}}
tfg.func @_mlir_lifted_graph(%Placeholder13A0: tensor<*xf32> {tfg.name = "Placeholder1:a"},
                             %Placeholder23A0: tensor<*xf32> {tfg.name = "Placeholder2:0"})
     -> (tensor<*xf32> {tfg.name = "SomeAdd3:0"})
 attributes {tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>} {
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  %Add, %ctl_2 = Add(%Placeholder13A0, %Placeholder23A0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%Add) [%ctl_2] : tensor<*xf32>
}
