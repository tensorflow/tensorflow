// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

tfg.func @then(%arg: tensor<*xi32>) -> (tensor<*xi32>) {
  return(%arg) : tensor<*xi32>
}

// CHECK-LABEL: tfg.func @test_missing_function
tfg.func @test_missing_function(%cond: tensor<*xi1>, %arg: tensor<*xi32>) -> (tensor<*xi32>) {
  // CHECK: If(
  %If, %ctlIf = If(%cond, %arg) {
    then_branch = #tf_type.func<@then, {}>,
    else_branch = #tf_type.func<@else, {}>
  } : (tensor<*xi1>, tensor<*xi32>) -> (tensor<*xi32>)
  return(%If) : tensor<*xi32>
}
