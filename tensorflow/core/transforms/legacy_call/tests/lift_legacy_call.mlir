// RUN: tfg-transforms-opt --tfg-lift-legacy-call %s | FileCheck %s

tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK: Foo {tfg.legacy_call = @Foo}
  %Foo, %ctl = Foo : () -> (tensor<i1>)
}

tfg.func @Foo() -> (tensor<i1>) {
  // CHECK-NOT: tfg.legacy_call
  %Const, %ctl = Const {dtype = i1, value = dense<0> : tensor<i1>} : () -> (tensor<i1>)
  return(%Const) : tensor<i1>
}