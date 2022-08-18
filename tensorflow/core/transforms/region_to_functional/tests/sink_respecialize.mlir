// RUN: tfg-transforms-opt -pass-pipeline='tfg-functional-to-region,tfg.func(tfg-cf-sink),tfg-region-to-functional,tfg-prepare-attrs-export,tfg-shape-inference' %s | FileCheck %s

// In this test case, `@then` has an unused argument `%b`. Sinking `Add` into 
// `@else` does not cause the signature to visibly change, so the function is
// not specialized during outlining. However, the argument now refers to a 
// different argument, so when shape inference is run, a type mismatch occurs.

tfg.func @then(%a: tensor<?xi32> {tfg.name = "a"},
               %b: tensor<?xi32> {tfg.name = "b"})
    -> (tensor<?xi32> {tfg.name = "c"}) {
  return(%a) : tensor<?xi32>
}

tfg.func @else(%a: tensor<?xi32> {tfg.name = "a"},
               %b: tensor<?xi32> {tfg.name = "b"})
    -> (tensor<?xi32> {tfg.name = "c"}) {
  return(%b) : tensor<?xi32>
}

// CHECK-LABEL: tfg.func @test_respecialize
tfg.func @test_respecialize(%cond: tensor<i1> {tfg.name = "cond"},
                            %arg: tensor<*xi32> {tfg.name = "arg"})
    -> (tensor<*xi32> {tfg.name = "ret"}) {
  %Const, %ctlConst = Const name("const") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<*xi32>)
  %b, %ctlA = Add(%Const, %arg) name("add") : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  %If, %ctlIf = If(%cond, %Const, %b) name("if") {
    then_branch = #tf_type.func<@then, {}>,
    else_branch = #tf_type.func<@else, {}>
  } : (tensor<i1>, tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  return(%If) : tensor<*xi32>
}

// CHECK: tfg.func @then_tfg_region_specialized_if_0

// CHECK: tfg.func @else_tfg_region_specialized_if_1