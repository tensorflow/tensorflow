// RUN: tfg-transforms-opt --tfg-lower-func-to-graph -verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK: tfg.graph #tf_type.version<producer = 34, min_consumer = 5>
tfg.func @_mlir_lifted_graph(%Placeholder1_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder1", 0 : index], tfg.name = "Placeholder1_0"},
                             %Placeholder2_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder2", 0 : index], tfg.name = "Placeholder2_0"})
    -> (tensor<*xf32> {tfg.name = "SomeAdd3_0"})
 attributes {tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>} {
  // CHECK: %[[PLACEHOLDER1:.*]], %[[CTRL0:.*]] = Placeholder name("Placeholder1")
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[PLACEHOLDER2:.*]], %[[CTRL1:.*]] = Placeholder name("Placeholder2")
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[SOMEADD1:.*]], %[[CTRL2:.*]] = Add(%[[PLACEHOLDER1]], %[[PLACEHOLDER2]]) name("SomeAdd1")
  %Add, %ctl_2 = Add(%Placeholder1_0, %Placeholder2_0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK-NOT: return
  return(%Add) [%ctl_2] : tensor<*xf32>
}

// -----

// expected-error@+1 {{lifted graph func is missing version attribute}}
tfg.func @_mlir_lifted_graph(%Placeholder1_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder1", 0 : index], tfg.name = "Placeholder1_0"},
                             %Placeholder2_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder2", 0 : index], tfg.name = "Placeholder2_0"})
     -> (tensor<*xf32> {tfg.name = "SomeAdd3_0"}) {
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  %Add, %ctl_2 = Add(%Placeholder1_0, %Placeholder2_0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%Add) [%ctl_2] : tensor<*xf32>
}

// -----

// expected-error@+1 {{lifted arg can't find the associated operation: Unknown}}
tfg.func @_mlir_lifted_graph(%Placeholder1_0: tensor<*xf32> {tfg.lifted_value_attr = ["Unknown", 0 : index], tfg.name = "Unknown_0"},
                             %Placeholder2_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder2", 0 : index], tfg.name = "Placeholder2_0"})
     -> (tensor<*xf32> {tfg.name = "SomeAdd3_0"})
 attributes {tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>} {
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  %Add, %ctl_2 = Add(%Placeholder1_0, %Placeholder2_0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%Add) [%ctl_2] : tensor<*xf32>
}

// -----

// expected-error@+1 {{result index out of bound: seeing index 3 from lifted_value_attr of arg #0, but op only has 2 results}}
tfg.func @_mlir_lifted_graph(%Placeholder1_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder1", 3 : index], tfg.name = "Placeholder1_3"},
                             %Placeholder2_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder2", 0 : index], tfg.name = "Placeholder2_0"})
     -> (tensor<*xf32> {tfg.name = "SomeAdd3_0"})
 attributes {tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>} {
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  %Add, %ctl_2 = Add(%Placeholder1_0, %Placeholder2_0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%Add) [%ctl_2] : tensor<*xf32>
}

// -----

// expected-error@+1 {{arg #0 is missing tfg.lifted_value_attr, can't be lowered}}
tfg.func @_mlir_lifted_graph(%Placeholder1_0: tensor<*xf32> {tfg.name = "Placeholder1_0"},
                             %Placeholder2_0: tensor<*xf32> {tfg.lifted_value_attr = ["Placeholder2", 0 : index], tfg.name = "Placeholder2_0"})
     -> (tensor<*xf32> {tfg.name = "SomeAdd3_0"})
 attributes {tfg.lifted_graph_version = #tf_type.version<producer = 34, min_consumer = 5>} {
  %Placeholder, %ctl = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  %Placeholder_0, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  %Add, %ctl_2 = Add(%Placeholder1_0, %Placeholder2_0) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%Add) [%ctl_2] : tensor<*xf32>
}
