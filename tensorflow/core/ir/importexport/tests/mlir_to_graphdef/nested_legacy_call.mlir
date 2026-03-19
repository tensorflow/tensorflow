// RUN: tfg-translate -mlir-to-graphdef %s | FileCheck %s

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: name: "foo1"
  %Foo:2, %ctl = Foo name("foo1") : () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: name: "id1"
  // CHECK-NEXT: op: "Identity"
  // CHECK-NEXT: input: "foo1"
  %Identity, %ctl_0 = Identity(%Foo#0) name("id1") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
  // CHECK: name: "id2"
  // CHECK-NEXT: op: "Identity"
  // CHECK-NEXT: input: "foo1:1"
  %Identity_1, %ctl_2 = Identity(%Foo#1) name("id2") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
}

// CHECK: library

// CHECK: name: "Foo"
tfg.func @Foo() -> (tensor<*xi32> {tfg.name = "ret1"},
                    tensor<*xi32> {tfg.name = "ret2"}) {
  // CHECK: name: "bar"
  %Bar:2, %ctl = Bar name("bar") : () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: value: "bar:ret1:0"
  // CHECK: value: "bar:ret2:0"
  return(%Bar#0, %Bar#1) : tensor<*xi32>, tensor<*xi32>
}

// CHECK: name: "Bar"
tfg.func @Bar() -> (tensor<*xi32> {tfg.name = "ret1"},
                    tensor<*xi32> {tfg.name = "ret2"}) {
  // CHECK: name: "const"
  %Const, %ctl = Const name("const") {value = dense<0> : tensor<i32>, dtype = i32} : () -> (tensor<*xi32>)
  // CHECK: value: "const:output:0"
  // CHECK: value: "const:output:0"
  return(%Const, %Const) : tensor<*xi32>, tensor<*xi32>
}
