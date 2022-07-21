// RUN: tfg-transforms-opt -pass-pipeline='tfg-lift-graph-to-func{feeds=Placeholder1,Placeholder2 fetches=SomeAdd3 control_rets=SomeAdd4},tfg-lower-func-to-graph' %s | FileCheck %s
// RUN: tfg-transforms-opt -pass-pipeline='tfg-lift-graph-to-func{fetches=SomeAdd3 control_rets=SomeAdd4},tfg-lower-func-to-graph' %s | FileCheck %s
// RUN: tfg-transforms-opt -pass-pipeline='tfg-lift-graph-to-func{feeds=Placeholder1,Placeholder2 control_rets=SomeAdd4},tfg-lower-func-to-graph' %s | FileCheck %s
// RUN: tfg-transforms-opt -pass-pipeline='tfg-lift-graph-to-func{feeds=Placeholder1,Placeholder2},tfg-lower-func-to-graph' %s | FileCheck %s

// CHECK: tfg.graph #tf_type.version<producer = 34, min_consumer = 5>
tfg.graph #tf_type.version<producer = 34, min_consumer = 5> {
  // CHECK: %[[PLACEHOLDER1:.*]], %[[CTRL0:.*]] = Placeholder name("Placeholder1")
  %Placeholder, %ctl_0 = Placeholder name("Placeholder1") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[PLACEHOLDER2:.*]], %[[CTRL1:.*]] = Placeholder name("Placeholder2")
  %Placeholder_1, %ctl_1 = Placeholder name("Placeholder2") {dtype = i32} : () -> (tensor<*xf32>)
  // CHECK: %[[SOMEADD1:.*]], %[[CTRL2:.*]] = Add(%[[PLACEHOLDER1]], %[[PLACEHOLDER2]]) name("SomeAdd1")
  %add1, %ctl2 = Add(%Placeholder, %Placeholder_1) name("SomeAdd1") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: %[[SOMEADD2:.*]], %[[CTRL3:.*]] = Add(%[[SOMEADD1]], %[[PLACEHOLDER2]]) name("SomeAdd2")
  %add2, %ctl3 = Add(%add1, %Placeholder_1) name("SomeAdd2") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: %[[SOMEADD3:.*]], %[[CTRL4:.*]] = Add(%[[PLACEHOLDER1]], %[[SOMEADD2]]) name("SomeAdd3")
  %add3, %ctl4 = Add(%Placeholder, %add2) name("SomeAdd3") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: %[[SOMEADD4:.*]], %[[CTRL5:.*]] = Add(%[[SOMEADD3]], %[[SOMEADD1]]) name("SomeAdd4")
  %add4, %ctl5 = Add(%add3, %add1) name("SomeAdd4") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: %[[SOMEADD5:.*]], %[[CTRL6:.*]] = Add(%[[PLACEHOLDER1]], %[[SOMEADD4]]) name("SomeAdd5")
  %add5, %ctl6 = Add(%Placeholder, %add4) name("SomeAdd5") {T = i32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK-NOT: return
}
