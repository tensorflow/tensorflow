// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Check that `ForRegion` is correctly converted to functional form, with names
// assigned to the body index argument.

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[START:.*]], %[[CTL:.*]] = Start name("[[IDX:.*]]")
  %Start, %ctl = Start name("idx") : () -> (tensor<i32>)
  // CHECK-NEXT: %[[LIMIT:.*]], %[[CTL_0:.*]] = Limit
  %Limit, %ctl_0 = Limit : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DELTA:.*]], %[[CTL_1:.*]] = Delta name("[[DTA:.*]]")
  %Delta, %ctl_1 = Delta name("dta") : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DATA:.*]], %[[CTL_2:.*]] = Data name("[[DATA_NAME:.*]]")
  %Data, %ctl_2 = Data name("data") : () -> (tensor<*xf32>)
  // CHECK-NEXT: %[[FOR:.*]], %[[CTL_3:.*]] = For(%[[START]], %[[LIMIT]], %[[DELTA]], %[[DATA]])
  // CHECK-SAME: {body = #tf_type.func<@[[BODY_FUNC:.*]], {}>}
  // CHECK-SAME: (tensor<i32>, tensor<i32>, tensor<i32>, tensor<{{.*}}>) -> (tensor<{{.*}}>)
  %outs, %ctl_3 = ForRegion(%Data) from %Start to %Limit by %Delta {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<*xf32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    %A, %ctl_4 = A(%arg0, %arg1) : (tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
    yield(%A) : tensor<*xf32>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)

  // CHECK: %{{.*}}, %[[CTL_4:.*]] = For(%[[START]], %[[LIMIT]], %[[DELTA]], %[[START]])
  // CHECK: body = #tf_type.func<@[[BODY_FUNC0:.*]], {}>
  %outs_0, %ctl_4 = ForRegion(%Start) from %Start to %Limit by %Delta {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    yield(%arg0) : tensor<i32>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)

  // CHECK: Sink [%[[CTL_3]], %[[CTL_4]]]
  %ctl_5 = Sink [%ctl_3, %ctl_4]
}

// CHECK: tfg.func @[[BODY_FUNC]](%[[ARG0:.*]]: tensor<i32> {tfg.name = "[[IDX]]_tfg_result_0", tfg.regenerate_output_shapes},
// CHECK:                         %[[DATA_NAME]]_tfg_result_0: tensor<{{.*}}> {tfg.name = "[[DATA_NAME]]_tfg_result_0", tfg.regenerate_output_shapes})
// CHECK:   %[[A:.*]], %[[CTL:.*]] = A(%[[ARG0]], %[[DATA_NAME]]_tfg_result_0)
// CHECK:   return(%[[A]])

// Test that the index when used as a result has a name.
// CHECK: tfg.func @[[BODY_FUNC0]]
// CHECK-SAME: tensor<i32> {tfg.name = "[[IDX]]_tfg_result_0"
// CHECK: -> (tensor<i32> {tfg.name = "[[IDX]]_tfg_result_0_1", tfg.regenerate_output_shapes})
