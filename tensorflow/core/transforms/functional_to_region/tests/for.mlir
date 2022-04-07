// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @body
tfg.func @body(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
  %A, %ctl = A(%arg0, %arg1) : (tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
  return(%A) : tensor<*xf32>
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[START:.*]], %[[CTL:.*]] = Start
  %Start, %ctl = Start : () -> (tensor<i32>)
  // CHECK-NEXT: %[[LIMIT:.*]], %[[CTL_0:.*]] = Limit
  %Limit, %ctl_0 = Limit : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DELTA:.*]], %[[CTL_1:.*]] = Delta
  %Delta, %ctl_1 = Delta : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DATA:.*]], %[[CTL_2:.*]] = Data
  %Data, %ctl_2 = Data : () -> (tensor<*xf32>)
  // CHECK-NEXT: %[[FOR:.*]], %[[CTL_5:.*]] = ForRegion(%[[DATA]]) from %[[START]] to %[[LIMIT]] by %[[DELTA]]  {
  // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<{{.*}}>,
  // CHECK-SAME:      %[[ARG2:.*]]: !tf_type.control, %[[ARG3:.*]]: !tf_type.control):
  // CHECK-NEXT:   %[[A:.*]], %[[CTL_6:.*]] = A(%[[ARG0]], %[[ARG1]])
  // CHECK-NEXT:   yield(%[[A]])
  // CHECK-NEXT: } {_some_attr, body_attrs = {}
  %For, %ctl_3 = For(%Start, %Limit, %Delta, %Data)
                 {T = [f32], _some_attr, body = #tf_type.func<@body, {}>}
                 : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
}
